"""Chain modes: our later submissions (2026-11-13 to 2026-11-15), rebuilt from stored test maps.

Each submission is a recipe, data/chain/<name>.json (data/chain/index.json lists them in documentation order):
  archives   the Mask2Former test maps, one archive per fold model and map set (NEW{k}_{j} = fold k, set j; uint16 levels,
             value / 510; 180 members, one per test image), with the model file and the input size and views that made it;
  unet       the U-Net test maps: 'package' (the package's fold-2 maps, full.npz / local.npz of the GPU stage) or
             'fold5mean' (the per-pixel mean of the five cross-validation fold models' maps, uint8);
  weight     the blend weight w of the Mask2Former map: blend = w MF + (1 - w)/2 full + (1 - w)/2 local;
  selector   the instance filter fitted for this map (the 68 P-combo features, its own cutoff), a file in data/chain/;
  trees      (ms125m4 only) a LightGBM, an XGBoost and a CatBoost model of the same 68 features (native files in
             data/chain/, sha256, the library versions that saved them); the filter probability is then the plain mean
             of the selector's and the three models' probabilities, kept at the recipe's cutoff (the selector file's
             own cutoff is not used);
  regressor  the mask-repair regressor: 'package' (weights/regressor.pkl) or the one refitted on this map's candidates;
  repair_trees (ms125m4r4 only) a LightGBM, an XGBoost and a CatBoost regressor of the same 64 repair features, fitted on
             the same rows as the regressor (scikit-learn-API pickles in data/chain/, sha256, the library versions that
             saved them); the per-mask threshold is then the snap of the plain mean of the four regressors' predictions
             (the regressor first), on the same grid with the same tie rule (pcand.load_regressor);
  csv        the sha256 and row count of the submitted CSV.
The CSV stage is the package's own (pipeline.image_stages) with the MF map = the level mean over every archive and the
blend with weight w; everything else (SHIP_R3, features, selector, P-cand repair, RLE) is unchanged. The arithmetic is
the chain CSV worker's that built these CSVs (tests/data/chain_csv_reference.py): integer level sum / (510 n) in float32,
and the blend in the package's insertion order with u = (1 - w) / 2.

The stored maps and the model files are not in this repository (about 3.8 GB): they are the release files listed, with
bytes and sha256, in data/chain/release_files.json. Every file is found by its name under the given roots and checked
before use. This is CPU work; the CSV of every recipe is checked against its pinned sha256.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / 'data'
CHAIN = DATA / 'chain'
UNET_FILES = {'fold5mean': dict(full='unet_fold5mean_full.npz', local='unet_fold5mean_local.npz'),
              'package': dict(full='unet_package_full.npz', local='unet_package_local.npz')}
DEN = 510


def names():
    """The recipe names, in documentation order."""
    return json.loads((CHAIN / 'index.json').read_text())


def load_recipe(name):
    if name not in names():raise ValueError('Unknown chain recipe: ' + repr(name))
    r = json.loads((CHAIN / f'{name}.json').read_text())
    if r['name'] != name or r['unet'] not in UNET_FILES or not 0 < r['weight'] < 1:raise ValueError('Malformed chain recipe: ' + name)
    expected = [f'NEW{k}_{j}' for k in range(5) for j in range(int(r['sets']))]
    if [a['name'] for a in r['archives']] != expected:raise ValueError('Chain recipe archives must be 5 folds x its map sets: ' + name)
    return r


def release_files():
    """{release file name: {bytes, sha256}}: every stored map and model file a recipe can read."""
    return json.loads((CHAIN / 'release_files.json').read_text())


def unet_files(recipe):
    return UNET_FILES[recipe['unet']]


REPO_FINAL = Path(__file__).resolve().parents[3] / 'weights/final'   # the final submission's files, in plain git (parts)


def repo_release(name):
    """The repository's own copy of a recipe's stored maps (weights/final, reassembled and hash-checked by
    repo_weights.prepare), or None when the repository does not hold all of them (then attach the release files)."""
    from .repo_weights import MANIFEST, prepare
    if not (REPO_FINAL / MANIFEST).is_file():return None
    r = load_recipe(name); held = json.loads((REPO_FINAL / MANIFEST).read_text())['files']
    if not all(f in held for f in [a['file'] for a in r['archives']] + list(unet_files(r).values())):return None
    return prepare(REPO_FINAL)


def selector(recipe):
    """The recipe's hash-checked selector (refit.load_selector), with the P-combo order and the recipe's cutoff. A recipe
    with trees gets the mean4 selector: the same state plus the three hash-checked tree models (refit.apply_selector)."""
    from . import refit
    s = refit.load_selector(CHAIN / recipe['selector']['file'], recipe['selector']['sha256'])
    if 'trees' in recipe:
        if s['feature_order'] != refit.PCOMBO_FEATURES or recipe['cutoff'] not in refit.CHAIN_CUTOFF_GRID:raise ValueError('Chain selector differs from its recipe')
        return refit.check(dict(s, threshold=recipe['cutoff'], mean4=trees(recipe['trees'])))
    if s['feature_order'] != refit.PCOMBO_FEATURES or s['threshold'] != recipe['cutoff']:raise ValueError('Chain selector differs from its recipe')
    return s


def trees(spec):
    """{'lgbm', 'xgb', 'cat': probability function of a 68-column float matrix} from the recipe's native model files,
    each hash-checked, loaded only under the library versions that saved them (as the CSV kernel that pinned the CSV read
    them: LightGBM Booster.predict, XGBoost and CatBoost predict_proba[:, 1])."""
    # One OpenMP thread, as in the CSV kernel that pinned the CSV; set before the libraries load (on macOS, LightGBM's
    # OpenMP runtime crashed when torch's was already loaded and this was unset). The probabilities do not depend on it.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import catboost, lightgbm, xgboost
    for lib in (lightgbm, xgboost, catboost):
        if lib.__version__ != spec['versions'][lib.__name__]:raise ValueError(f"{lib.__name__} must be {spec['versions'][lib.__name__]} (installed {lib.__version__})")
    path = {}
    for a in ('lgbm', 'xgb', 'cat'):
        p = CHAIN / spec[a]['file']
        if not spec[a]['sha256'] or _sha(p) != spec[a]['sha256']:raise ValueError(f'{p.name}: tree model hash differs from the recipe')
        path[a] = str(p)
    lg = lightgbm.Booster(model_file=path['lgbm'])
    xg = xgboost.XGBClassifier(); xg.load_model(path['xgb'])
    ca = catboost.CatBoostClassifier(); ca.load_model(path['cat'])
    return dict(lgbm=lg.predict, xgb=lambda x: xg.predict_proba(x)[:, 1], cat=lambda x: ca.predict_proba(x)[:, 1])


def repair_trees(spec):
    """[lgbm, xgb, cat]: the recipe's three extra repair regressors (pickled LGBMRegressor, XGBRegressor and
    CatBoostRegressor), each hash-checked, loaded only under the library versions that saved them."""
    import pickle
    os.environ.setdefault('OMP_NUM_THREADS', '1')   # as in chain.trees: one OpenMP thread, set before the libraries load
    import catboost, lightgbm, xgboost
    for lib in (lightgbm, xgboost, catboost):
        if lib.__version__ != spec['versions'][lib.__name__]:raise ValueError(f"{lib.__name__} must be {spec['versions'][lib.__name__]} (installed {lib.__version__})")
    out = []
    for a in ('lgbm', 'xgb', 'cat'):
        p = CHAIN / spec[a]['file']
        if not spec[a]['sha256'] or _sha(p) != spec[a]['sha256']:raise ValueError(f'{p.name}: repair model hash differs from the recipe')
        out.append(pickle.loads(p.read_bytes()))
    return out


def repair(recipe, reg):
    """The recipe's P-cand predictor (pcand.load_regressor of reg = regressor(recipe, ...)), with its repair_trees if any."""
    from . import pcand
    return pcand.load_regressor(*reg, members=repair_trees(recipe['repair_trees']) if 'repair_trees' in recipe else ())


def regressor(recipe, weights_root):
    """(path, sha256) of the recipe's P-cand regressor, for pcand.load_regressor."""
    if recipe['regressor'] == 'package':
        from . import pcand
        return pcand.deployed(weights_root)
    return CHAIN / recipe['regressor']['file'], recipe['regressor']['sha256']


def chain_mf(maps, names):
    """The MF probability map (float32) of one image: the level mean over the named uint16 archives."""
    from .arithmetic import mf_levels_mean
    return mf_levels_mean(*(maps[n] for n in names))


def weighted_blend(full, local, mf, w):
    """blend_with_mf with the weights (w, (1 - w)/2, (1 - w)/2): the same checks, insertion order and Python sum
    (w = 0.25 is blend_with_mf bit for bit)."""
    if full.dtype != np.uint8 or local.dtype != np.uint8:raise ValueError('U-Net maps must be uint8')
    if mf.dtype != np.float32 or not full.shape == local.shape == mf.shape:raise ValueError('Map shapes differ')
    u = (1 - w) / 2
    return sum(np.float32(x) * m for x, m in ((w, mf), (u, full.astype(np.float32) / 255), (u, local.astype(np.float32) / 255)))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):h.update(b)
    return h.hexdigest()


def find(name, roots):
    """The single file called name under the first root that has one."""
    for root in roots:
        root = Path(root)
        found = sorted(root.rglob(name)) if root.is_dir() else []
        if len(found) > 1:raise ValueError(f'More than one {name} under {root}')
        if found:return found[0]
    raise FileNotFoundError(f'{name} not found under {[str(r) for r in roots]}: attach the release files (data/chain/release_files.json)')


def stored_files(recipe, roots, strict=True):
    """{archive name (NEW{k}_{j}) or 'full' / 'local': path} of a recipe's maps. strict: bytes and sha256 must equal the
    release record (else ValueError); strict=False (maps re-inferred from the models) records them instead."""
    pins = release_files(); out = {}; seen = {}
    wanted = [(a['name'], a['file']) for a in recipe['archives']] + list(unet_files(recipe).items())
    for key, name in wanted:
        p = find(name, roots); out[key] = p
        if strict:
            if p.stat().st_size != pins[name]['bytes'] or _sha(p) != pins[name]['sha256']:raise ValueError(f'{name}: size or hash differs from the release record')
        else:seen[key] = dict(file=name, sha256=_sha(p), equal_to_release=_sha(p) == pins[name]['sha256'])
    return (out, seen) if not strict else out


def _open(paths, keys):
    out = {}
    for n, p in paths.items():
        z = np.load(p, allow_pickle=False)
        if sorted(z.files) != keys:raise ValueError('Map key coverage differs: ' + n)
        out[n] = z
    return out


def _check(a):
    for n, q in a.items():
        unet = n in ('full', 'local')
        if q.shape != (2048, 2048) or q.dtype != (np.uint8 if unet else np.uint16) or q.max() > (255 if unet else DEN):raise ValueError('Malformed probability map: ' + n)


def _setup(recipe_name, weights):
    from . import pcand as P
    from .assets import resolve_assets
    from .quality import load_quality
    import cv2
    cv2.setNumThreads(1)
    r = load_recipe(recipe_name); root, _ = resolve_assets(weights)
    model = load_quality(root / 'quality_model.json'); sel = selector(r); reg = regressor(r, root)
    return r, model, sel, reg, P


def image_rows(recipe_name, images, roots, keys, weights=None):
    """{image key: the kept masks' RLE strings} for some test images (the CSV rows of those images, in order)."""
    from .pipeline import prepare_images, read_image, test_keys, image_stages
    import tempfile
    r, model, sel, reg, P = _setup(recipe_name, weights)
    with tempfile.TemporaryDirectory() as tmp:files = prepare_images(images, Path(tmp))
    maps = _open(stored_files(r, roots), test_keys()); names = [a['name'] for a in r['archives']]; out = {}
    for key in keys:
        a = {n: z[key] for n, z in maps.items()}; _check(a)
        v = image_stages(read_image(files[key]), a, model, pcand=repair(r, reg), selector=sel, chain=dict(names=names, weight=r['weight']))
        out[key] = [x for x, k in zip(v['rles'], v['keep']) if k]
    return out


def run_chain_csv(images, weights, release, output, recipe, strict=True, image_set=None):
    """CPU-only: the submission CSV of one recipe from its maps under release (a root or a list of roots, searched in
    order). strict=False reads re-inferred maps (same file names) and records their hashes instead of refusing them."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1', None):raise RuntimeError('Use a CPU runtime for formatting')
    from .csv_io import write_rows
    from .audit import identity
    from .pipeline import prepare_images, read_image, test_keys, image_stages, save
    from .validate import validate_submission
    start = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if release is None:
        release = repo_release(recipe)
        if release is None:raise FileNotFoundError(f'The repository does not hold the maps of {recipe}: attach the release files and pass their folder')
    roots = [release] if isinstance(release, (str, Path)) else list(release)
    r, model, sel, reg, P = _setup(recipe, weights); predictor = repair(r, reg)
    files = prepare_images(images, output / 'test_images', image_set=image_set)
    found = stored_files(r, roots, strict); paths, seen = (found, None) if strict else found
    maps = _open(paths, test_keys() if image_set is None else sorted(files)); names = [a['name'] for a in r['archives']]
    rows = {}; keeps = {}; per = {}
    for i, (key, path) in enumerate(sorted(files.items())):
        a = {n: z[key] for n, z in maps.items()}; _check(a)
        v = image_stages(read_image(path), a, model, pcand=predictor, selector=sel, chain=dict(names=names, weight=r['weight']))
        rows[key] = v['rles']; keeps[key] = v['keep']
        per[key] = dict(candidates=v['candidates'], kept=int(np.sum(v['keep'])), stages=v['stages'], pcand={k: v['pcand'][k] for k in ('kept', 'fallback', 'changed', 'added', 'removed')})
        if (i + 1) % 30 == 0:print(json.dumps(dict(stage='csv', recipe=recipe, done=i + 1, total=len(files), elapsed_seconds=time.monotonic() - start)), flush=True)
    counts = write_rows(output / 'submission.csv', sorted(files), rows, keeps)
    check = validate_submission(output / 'submission.csv', test_keys() if image_set is None else sorted(files))
    csv = identity(output / 'submission.csv')
    result = dict(status='complete', recipe=recipe, images=len(files), candidates=sum(x['candidates'] for x in per.values()), kept=sum(counts.values()), csv=csv,
                  expected_csv_sha256=r['csv']['sha256'], matches_expected=csv['sha256'] == r['csv']['sha256'], validation=check,
                  weight=r['weight'], cutoff=r['cutoff'], selector_sha256=r['selector']['sha256'], regressor_sha256=reg[1], unet=r['unet'],
                  **({'trees': {a: r['trees'][a]['sha256'] for a in ('lgbm', 'xgb', 'cat')}} if 'trees' in r else {}),
                  **({'repair_trees': {a: r['repair_trees'][a]['sha256'] for a in ('lgbm', 'xgb', 'cat')}} if 'repair_trees' in r else {}),
                  maps='stored (hash-checked)' if strict else dict(rule='re-inferred, hashes recorded', archives=seen),
                  pcand_totals={k: sum(x['pcand'][k] for x in per.values()) for k in ('kept', 'fallback', 'changed', 'added', 'removed')},
                  per_image=per, model_fits=0, submission_api_calls=0, elapsed_seconds=time.monotonic() - start)
    if image_set is not None:result.update(image_set=image_set, expected_csv_sha256=None, matches_expected=None)
    save(output / 'csv_result.json', result)
    return result
