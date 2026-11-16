"""Kaggle-CPU-only test stage of chain2 (order CH2): the 180-image submission CSV of pcombo and of each weight arm w<NN> from
the deployment selectors fitted in the same kernel by case/worker.py, with the package's own CSV code (branch
repro-package-20261111). Adapted from chain/csv_worker.py (CH1).

usage: csv_worker.py <package_root> <images> <weights_root> <maps_root> <new_maps_root> <out_dir> <ship_dir> <chain_inputs.json>

pcombo  regression: run_csv(selector='pcombo') must rebuild the shipped CSV 35abbeed... byte for byte, and the deployment
        selector fitted here for the pcombo arm must equal the pinned package selector (fd25d491...) byte for byte.
w<NN>   the test MF map = the level mean of every set's fold models' test maps: sum_{k,j} NEW{k}_{j} / (510 * 5 n) in float32
        (= the mean over sets of each set's 5-model level mean; with one set it is chain CH1 arm new's test map bit for bit);
        blend = w * MF + (1 - w)/2 * full + (1 - w)/2 * local (the package blend_with_mf's insertion order and Python sum;
        w = 0.25 is that function bit for bit and is asserted equal on every image).
Everything that reads the MF map (SHIP_R3 candidates via the blend, 15 context, 6 neighbour and 8 MF-rich features, P-cand
repair on the blend) reads the arm's map and blend (pipeline.image_stages with the MF variant 'chain2' and the blend
weight installed at run time; the package is not edited). P-cand uses the package's deployment regressor b4dcd7d1....
cmb     chain2s, only when chain_inputs.json has 'combined': the combined-map arm. Test MF map = the level mean of the 5 fold
        archives' test maps, sum_k CMB{k} / (510 * 5) in float32 (the package's mf_levels_mean, MF variant 'chain2s_cmb'); blend
        weight 1.0: blend_with_mf is replaced by mf_blend, which returns the MF map (no U-Net terms); instancing SHIP_R3 with the
        archive's thr (combined_stage, passed as image_stages' instance_stage for this arm only); selector, P-cand as w<NN>.
Checks per CSV: the package validate_submission (header, decodes, no empty mask, no overlapping pixels, unique contiguous ids,
known image names) plus 180 expected images; difference from the pcombo CSV.

chainF (order INT1) additions, each only when its key is in chain_inputs.json (absent = chain2m's test stage exactly):
  unet         chain2u's U-Net test maps (header of chain2u/csv_worker.py; functions UnetMean .. unet_digest copied verbatim).
  test_unet    (--test-unet fold5mean, ported after F0) the weight arms' and cmb's U-Net test maps = the 5-fold means
               (FOLD5MEAN_SHA256, staged files used unchanged); a view swapped by --full/--local keeps the set mean.
  pcand_refit  for each listed arm, after its frozen CSV: <arm>_refit = the same arm (selector, maps, weight, levels path) with
               the deployment regressor refit in this kernel (case/out/deploy/<arm>_regressor.pkl, sha256 = deploy_fit.json),
               written next to the frozen one (out/csv/<arm>_refit/submission.csv); pcand's csv_pcand.py 'refit' variant.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

if len(sys.argv) == 9:                               # the kernel call; tests import the helpers with the package on PYTHONPATH
    ROOT, IMAGES, WEIGHTS, MAPS, NEWMAPS, OUT, SHIP, INPUTS = (Path(a) for a in sys.argv[1:9])
    sys.path.insert(0, str(ROOT))
import numpy as np

PCOMBO_CSV_SHA = '35abbeed91108e40ddbd18c094c901d116441e363b10c31aa8092988d8e8f6e6'
PCOMBO_SELECTOR_SHA = 'fd25d491d828aaac1bdd8c07d120edbe32ac24d8f5980599d00190f7573b389c'
MAP_SET_SHA = 'd7aa6a41fafc26b07ce89b306b884f6667ebb68b86289ea4745ee44c0b0d195f'
VARIANT = 'chain2'
DEN = 510
CMB_VARIANT = 'chain2s_cmb'        # chain2s: the combined-map arm's MF variant
CMB_ARM = 'cmb'
DEN = 510
START = time.monotonic()
_BASE = {}


def now(): return subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S %Z'], text=True).strip()
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); os.replace(tmp, p)


def arms(pins):
    """[(arm, weight)] in ascending weight (the arm names of case/mt.py: w<round(100 w):02d>)."""
    return [(f'w{int(round(w * 100)):02d}', float(w)) for w in sorted(pins['weights'])]


def level_tag(levels):
    """chain2m: case/levels.level_tag (arm-name suffix; '' for None or [0.5])."""
    if levels is None or list(levels) == [0.5]: return ''
    return ''.join(f'l{int(round(t * 100)):02d}' for t in levels)


def level_arms(pins):
    """chain2m: [(arm, weight, levels or None)] in case/mt.py's layout order: without 'levels' = arms(pins) with None; with
    level sets, w25 stays plain and every other weight gets one arm per level set."""
    if 'levels' not in pins: return [(a, w, None) for a, w in arms(pins)]
    out = []
    for a, w in arms(pins):
        if a == 'w25': out.append((a, w, None)); continue
        for s in pins['levels']: out.append((a + level_tag(s), w, [float(x) for x in s]))
    return out


def resolve_keep(p, tau, pairs):
    """chain2m: case/levels.resolve_keep (the same greedy rule as the CV): p >= tau, then highest p first (ties lower index),
    a candidate sharing a pixel with a kept one is dropped. Kept indices, ascending."""
    p = np.asarray(p, float); n = len(p); nb = [[] for _ in range(n)]
    for i, j in pairs: nb[i].append(j); nb[j].append(i)
    alive = np.zeros(n, bool)
    for i in sorted(range(n), key=lambda i: (-p[i], i)):
        if not any(alive[j] for j in nb[i]): alive[i] = True
    return np.flatnonzero(alive & (p >= tau)).tolist()


def overlap_pairs(masks):
    """chain2m: case/levels.overlap_pairs."""
    boxes = []
    for m in masks:
        ys = np.flatnonzero(m.any(1)); xs = np.flatnonzero(m.any(0))
        boxes.append((ys[0], ys[-1], xs[0], xs[-1]) if len(ys) else None)
    out = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = boxes[i], boxes[j]
            if a is None or b is None or a[0] > b[1] or b[0] > a[1] or a[2] > b[3] or b[2] > a[3]: continue
            y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
            if (masks[i][y0:y1 + 1, x0:x1 + 1] & masks[j][y0:y1 + 1, x0:x1 + 1]).any(): out.append((i, j))
    return out


def load_level_selector(path, sha256, role, threshold):
    """chain2m: a multi-level arm's selector (the 68 P-combo columns + 'level'); refit.check knows only 68-column orders."""
    from filseg.reproduction import refit
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256: raise ValueError('Selector file hash differs from the pinned value')
    m = json.loads(data)
    if m.get('role') != role or list(m.get('feature_order', [])) != list(refit.PCOMBO_FEATURES) + ['level'] or m.get('threshold') != threshold:
        raise ValueError('Wrong chain2m level selector recipe')
    d = len(m['feature_order'])
    if len(m['scaler_mean']) != d or len(m['scaler_scale']) != d or len(m['coef'][0]) != d: raise ValueError('Selector shape')
    return m


def level_image_stages(raw, maps, model, levels, pcand, selector, mf_variant):
    """chain2m: pipeline.image_stages (package @ 3f6524cd) for the path the weight arms use (MF variant, selector with context +
    neighbour + MF-rich features, P-cand; no P-nat, no views), with the candidates = the union, level by level, of
    SHIP_R3.with_(thr=t) instances of the blend. Per level: the overlap check of the package and the neighbour / MF-rich
    features on that level's candidates. One level: the package code exactly (refit.check + apply_selector on 68 columns).
    More levels: 'level' joins the selector rows; keep = p >= cutoff, then the greedy overlap rule (resolve_keep)."""
    from filseg.reproduction.arithmetic import blend_with_mf
    from filseg.reproduction.mfmaps import variant_mf
    from filseg.reproduction.instancing import iter_instances, SHIP_R3
    from filseg.reproduction.quality import apply_quality
    from filseg.reproduction.audit import array_sha
    from filseg.reproduction.pipeline import fileless_sha
    from filseg.reproduction import refit
    from filseg.reproduction.context_features import context_features
    from filseg.reproduction.neighbour import neighbour_features
    from filseg.reproduction.mfrich_features import image_features
    from filseg.reproduction import pcand as P
    from pycocotools import mask as mu
    multi = len(levels) > 1
    mf = variant_mf(maps, mf_variant)
    blend = blend_with_mf(maps['full'], maps['local'], mf)
    masks, lev, neighbour, mfrich = [], [], [], []
    labels = np.zeros((len(levels),) + raw.shape, np.int32)
    for li, t in enumerate(levels):
        ms = [np.asarray(m, bool) for m in iter_instances(blend, SHIP_R3.with_(thr=t))]
        occupied = np.zeros(raw.shape, bool)
        for i, m in enumerate(ms):
            if not m.any() or (occupied & m).any(): raise ValueError('Empty or overlapping SHIP_R3 instance')
            occupied |= m; labels[li][m] = i + 1
        neighbour += neighbour_features(ms, blend); mfrich += image_features(ms, mf); masks += ms; lev += [float(t)] * len(ms)
    labels = labels[0] if not multi else labels
    rows, x, probability, keep = apply_quality(masks, maps['full'], maps['local'], raw, model)
    extra = {}
    cmaps = {'full': maps['full'].astype(np.float32) / 255, 'local': maps['local'].astype(np.float32) / 255, 'm2f_public': mf}
    context = [context_features(m, cmaps, raw) for m in masks]
    pairs = overlap_pairs(masks) if multi else []
    if not multi:
        refit.check(selector)
        _, selected, new_keep = refit.apply_selector(rows, context, selector, neighbour, mfrich, None)
    else:
        srows = [dict(r, level=t) for r, t in zip(rows, lev)]
        _, selected, _ = refit.apply_selector(srows, context, selector, neighbour, mfrich, None)
        new_keep = np.zeros(len(masks), bool); new_keep[resolve_keep(selected, selector['threshold'], pairs)] = True
    extra['selector'] = dict(lr39_keep=keep.tolist(), probability=selected.tolist(), keep=new_keep.tolist(), context=context, neighbour=neighbour, mfrich=mfrich)
    extra['selector_stage'] = array_sha(selected)
    keep = new_keep
    kept = [int(j) for j in np.flatnonzero(keep)]; taus = []
    if kept:
        xp = []
        for j in kept:
            row = {**{n: float(rows[j][n]) for n in P.FEATURES[:39]}, **context_features(masks[j], cmaps, raw), **P.window_features(masks[j], blend)}
            xp.append([row[n] for n in P.FEATURES])
        xp = np.asarray(xp, float)
        if not np.isfinite(xp).all(): raise ValueError('Nonfinite P-cand features')
        taus = [float(t) for t in pcand(xp)]
        if len(taus) != len(kept) or any(t not in P.GRID for t in taus): raise ValueError('P-cand thresholds must be grid values, one per kept candidate')
    repaired, stats = P.repair_taus([masks[j] for j in kept], blend, taus)
    occupied = np.zeros(raw.shape, bool)
    for m in repaired:
        if not m.any() or (occupied & m).any(): raise ValueError('Empty or overlapping P-cand mask')
        occupied |= m
    masks = list(masks)
    for j, m in zip(kept, repaired): masks[j] = m
    extra['pcand'] = dict(stats, taus=taus)
    rles = []
    for mask in masks:
        r = mu.encode(np.asfortranarray(mask, dtype=np.uint8))
        if not np.array_equal(mu.decode(r).astype(bool), mask): raise ValueError('RLE roundtrip failed')
        rles.append(r['counts'].decode())
    stages = {'MF_mean': array_sha(mf), 'candidate_blend': array_sha(blend), 'SHIP_R3': array_sha(labels), 'features39': array_sha(x),
              'LR39_probability': array_sha(probability), 'keep': array_sha(keep), 'RLE': fileless_sha(rles)}
    stages['selector_probability'] = extra.pop('selector_stage')
    return dict(stages=stages, rows=rows, features=x, scores=probability, keep=keep, rles=rles, candidates=len(masks), levels=lev,
                overlap=[list(p_) for p_ in pairs], **extra)


def test_sources(pins):
    """Archive names of the test maps in the per-image map dict: NEW{k}_{j} = fold-k model of map set j."""
    return [f'NEW{k}_{j}' for k in range(5) for j in range(int(pins['sets']))]


def chain_mf(maps, names):
    """The weight arms' MF probability map (float32) for one image: the package's mf_levels_mean over every set's fold-model maps."""
    from filseg.reproduction.arithmetic import mf_levels_mean
    return mf_levels_mean(*(maps[n] for n in names))


def weighted_blend(w, check_against=None):
    """blend_with_mf with the weights (w, (1 - w)/2, (1 - w)/2): the same checks, insertion order and Python sum. With
    check_against (the package function), every call asserts bit equality (used for w = 0.25)."""
    u = (1 - w) / 2

    def blend(full, local, mf):
        if full.dtype != np.uint8 or local.dtype != np.uint8: raise ValueError('U-Net maps must be uint8')
        if mf.dtype != np.float32 or not full.shape == local.shape == mf.shape: raise ValueError('Map shapes differ')
        p = sum(np.float32(x) * m for x, m in ((w, mf), (u, full.astype(np.float32) / 255), (u, local.astype(np.float32) / 255)))
        if check_against is not None: assert np.array_equal(p, check_against(full, local, mf)), 'W25_BLEND_DIFFERS'
        return p
    blend.weight = w
    return blend


def _bases():
    from filseg.reproduction import mfmaps, arithmetic
    _BASE.setdefault('variant_mf', mfmaps.variant_mf); _BASE.setdefault('blend_with_mf', arithmetic.blend_with_mf)
    return _BASE


def install(names, roles, cmb_names=None):
    """Teach the package the chain2 MF variant (mfmaps.variant_mf, read at call time by pipeline.image_stages) and the weight
    arms' selector roles (refit.check: the 68 P-combo features, the fitted cutoff). The package files are not edited.
    chain2s: cmb_names (the 5 combined test archives) also installs the variant CMB_VARIANT = their level mean."""
    from filseg.reproduction import mfmaps, refit
    base = _bases()['variant_mf']

    def variant_mf(maps, variant):
        if cmb_names is not None and variant == CMB_VARIANT: return chain_mf(maps, cmb_names)
        return chain_mf(maps, names) if variant == VARIANT else base(maps, variant)
    variant_mf.chain = True; variant_mf.base = base
    mfmaps.variant_mf = variant_mf
    for arm, (role, threshold) in roles.items():
        refit.MFMAP_ROLES['chain2_' + arm] = role; refit.MFMAP_THRESHOLDS['chain2_' + arm] = threshold


def mf_blend(full, local, mf):
    """chain2s: the combined arm's blend (weight 1.0): blend_with_mf's input checks, then the MF map itself, MF * float32(1.0)
    (bit for bit the MF map; no U-Net terms)."""
    if full.dtype != np.uint8 or local.dtype != np.uint8: raise ValueError('U-Net maps must be uint8')
    if mf.dtype != np.float32 or not full.shape == local.shape == mf.shape: raise ValueError('Map shapes differ')
    return mf * np.float32(1.0)


mf_blend.weight = 1.0


def set_weight(w):
    """The blend weight read by pipeline.image_stages (it imports arithmetic.blend_with_mf at call time). chain2s: w = 1.0
    (the combined arm only; the weight arms are in (0, 1)) installs mf_blend."""
    from filseg.reproduction import arithmetic
    base = _bases()['blend_with_mf']
    arithmetic.blend_with_mf = mf_blend if w == 1.0 else weighted_blend(w, base if w == .25 else None)


def restore():
    from filseg.reproduction import mfmaps, arithmetic
    if 'variant_mf' in _BASE: mfmaps.variant_mf = _BASE['variant_mf']
    if 'blend_with_mf' in _BASE: arithmetic.blend_with_mf = _BASE['blend_with_mf']


def load_new_test(root, pins, keys):
    """NEW{k}_{j}: the 180 test maps of set j's fold-k model, file identity pinned in chain_inputs.json; uint16 [2048, 2048], <= 510."""
    out = {}
    for k in range(5):
        assert len(pins['test'][str(k)]) == int(pins['sets'])
        for j, row in enumerate(pins['test'][str(k)]):
            p = [x for x in Path(root).rglob(row['file'])]
            assert len(p) == 1, ('NEW_TEST_FILE', row['file'], len(p)); p = p[0]
            b = p.read_bytes(); assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256'], ('NEW_TEST_IDENTITY', k, j)
            z = np.load(p, allow_pickle=False); assert sorted(z.files) == sorted(keys), ('NEW_TEST_KEYS', k, j)
            out[f'NEW{k}_{j}'] = z
    return out


class UnetMean:
    """chain2u: per-image uint8 mean of m U-Net test archives (m = 5 fold models x n sets): rint(level sum / m)."""
    def __init__(self, archives): self.archives = list(archives)

    def __getitem__(self, key):
        a = [z[key] for z in self.archives]
        if any(x.dtype != np.uint8 or x.shape != a[0].shape for x in a): raise ValueError('U-Net arrays must be matching uint8 maps')
        return np.rint(sum(x.astype(np.int64) for x in a) / len(a)).astype(np.uint8)


def load_unet_test(root, pins, keys):
    """{view: UnetMean over the view's 5 n staged test archives} for the swapped views (pins['unet']); {} without U-Net sets."""
    out = {}
    for view, u in sorted((pins.get('unet') or {}).items()):
        zs = []
        for k in range(5):
            assert len(u['test'][str(k)]) == len(u['sets']), ('UNET_TEST_SETS', view, k)
            for j, row in enumerate(u['test'][str(k)]):
                p = [x for x in Path(root).rglob(row['file'])]
                assert len(p) == 1, ('UNET_TEST_FILE', row['file'], len(p)); b = p[0].read_bytes()
                assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256'], ('UNET_TEST_IDENTITY', view, k, j)
                z = np.load(p[0], allow_pickle=False); assert sorted(z.files) == sorted(keys), ('UNET_TEST_KEYS', view, k, j); zs.append(z)
        out[view] = UnetMean(zs)
    return out


def check_stored_equivalent(pins, inferred):
    """A stored-equivalent view's test archives are the package map file itself (inference_result.json sha256)."""
    for view, u in (pins.get('unet') or {}).items():
        if u['stored_equivalent']:
            assert all(r['sha256'] == inferred['networks'][view]['sha256'] for k in range(5) for r in u['test'][str(k)]), ('UNET_NOT_PACKAGE_MAP', view)


def arm_maps(maps, unet):
    """The weight arms' U-Net map dict: chain2's package maps with the swapped views replaced."""
    return maps if not unet else dict(maps, **unet)


def unet_digest(unet, maps, keys):
    """Descriptive: per swapped view, the sha256 of the per-image averaged-map hashes and the images equal to the package map."""
    out = {}
    for view, src in unet.items():
        hs = {}; equal = 0
        for key in sorted(keys):
            a = src[key]; hs[key] = hashlib.sha256(a.tobytes()).hexdigest(); equal += int(np.array_equal(a, maps[view][key]))
        out[view] = dict(map_sha256=hashlib.sha256(json.dumps(hs).encode()).hexdigest(), images_equal_to_package_map=equal, archives=len(src.archives))
    return out


FOLD5MEAN_SHA256 = dict(full='ff0c7a809993de4ac99681ff83ea0175118ec274c4dae75cf065469e42695fb9',      # unet_refit/out/fold5mean (UN1), the
                        local='c62d6e2aa7273c4e8c57a994e11e47737c13f9e0a2a2a4c9696f479e479cfccf')      # submissions' U-Net test maps since 12:40


def load_test_unet(root, pins, keys):
    """chainF (--test-unet fold5mean, ported after F0): the 5-fold-mean U-Net test maps, staged files used unchanged (chain3's /
    pcand ship's one-source rule): exactly one file of the pinned name under root, bytes + sha256 as pinned and equal to
    FOLD5MEAN_SHA256, keys = the 180 test stems. {} without the flag."""
    t = pins.get('test_unet')
    if not t: return {}
    assert t['name'] == 'fold5mean', ('TEST_UNET', t['name'])
    out = {}
    for view in ('full', 'local'):
        row = t[view]; p = [x for x in Path(root).rglob(row['file'])]
        assert len(p) == 1, ('TEST_UNET_FILE', row['file'], len(p)); b = p[0].read_bytes()
        assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256'] == FOLD5MEAN_SHA256[view], ('TEST_UNET_IDENTITY', view)
        z = np.load(p[0], allow_pickle=False); assert sorted(z.files) == sorted(keys), ('TEST_UNET_KEYS', view); out[view] = z
    return out


def test_maps(maps, unet, test_unet):
    """The weight arms' U-Net test-map dict: the package maps, then --test-unet's fold5mean for every view, then --full/--local's
    set means (a swapped view keeps chain2u's rule). Flag absent and no sets: the package maps (chain2m exactly)."""
    if not (unet or test_unet): return maps
    out = dict(maps); out.update(test_unet); out.update(unet); return out


def test_unet_record(pins, test_unet):
    """Recorded in csv_check.json and the call's results: which U-Net test maps the weight arms read, with their sha256."""
    t = pins.get('test_unet') or {}
    return dict(name=t.get('name'), sha256={v: t[v]['sha256'] for v in ('full', 'local')}, files={v: t[v]['file'] for v in ('full', 'local')},
                views=sorted(test_unet)) if t else None


def refit_arms(pins):
    """chainF: the arms whose P-cand v1 was refit in the kernel ([] without --pcand-refit)."""
    return list((pins.get('pcand_refit') or {}).get('arms', []))


def refit_regressor(ship_dir, arm):
    """chainF: (path, sha256) of the arm's refit deployment regressor, checked against deploy_fit.json (worker.pc_deployment_fit)."""
    deploy = Path(ship_dir).parent / 'deploy'; fit = json.loads((Path(ship_dir).parent / 'deploy_fit.json').read_text())[arm]
    p = deploy / f'{arm}_regressor.pkl'; h = sha_file(p); assert h == fit['regressor_sha256'], ('DEPLOY_REGRESSOR_IDENTITY', arm)
    return p, h


RULE_KEYS = ('thr', 'min_area', 'close_px', 'link_gap', 'link_angle')


def combined_arm(pins):
    """chain2s: None without 'combined' in chain_inputs.json (the call is chain2m exactly); else the cmb arm's spec."""
    if 'combined' not in pins: return None
    assert 'rule' not in pins and 'levels' not in pins, 'COMBINED_WITH_RULE_OR_LEVELS'
    c = pins['combined']; thr = float(c['thr']); assert 0 < thr < 1, ('COMBINED_THR', thr)
    return dict(arm=CMB_ARM, weight=1.0, thr=thr, name=c.get('name'), names=[f'CMB{k}' for k in range(5)], variant=CMB_VARIANT)


def combined_stage(thr):
    """chain2s: the combined arm's instance stage, iter_instances(p, SHIP_R3.with_(thr=thr)) of the package's instancing module
    (chain2r's instance_stage pattern; passed to image_stages for this arm only)."""
    from filseg.reproduction.instancing import iter_instances, SHIP_R3
    rule = SHIP_R3.with_(thr=float(thr))
    stage = lambda probability: iter_instances(probability, rule)
    stage.rule = rule
    return stage


def load_combined_test(root, pins, keys):
    """chain2s: CMB{k} = the combiner's 180 test maps of base fold k (one archive per fold), file identity pinned in
    chain_inputs.json['combined']['test']; keys = the 180 test stems (uint16 levels <= 510 are checked per image in build_arm)."""
    out = {}
    for k in range(5):
        rows = pins['combined']['test'][str(k)]; assert len(rows) == 1, ('CMB_TEST_PINS', k); row = rows[0]
        p = [x for x in Path(root).rglob(row['file'])]
        assert len(p) == 1, ('CMB_TEST_FILE', row['file'], len(p)); p = p[0]
        b = p.read_bytes(); assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256'], ('CMB_TEST_IDENTITY', k)
        z = np.load(p, allow_pickle=False); assert sorted(z.files) == sorted(keys), ('CMB_TEST_KEYS', k)
        out[f'CMB{k}'] = z
    return out


def instance_stage(pins):
    """chain2r (order PP1): None without 'rule' in chain_inputs.json (image_stages' default SHIP_R3, chain2 exactly); else
    iter_instances(p, rule) of the package's own instancing module, rule = SHIP_R3 with the pinned fields."""
    if 'rule' not in pins: return None
    from filseg.reproduction.instancing import iter_instances, SHIP_R3
    f = pins['rule']['fields']; assert sorted(f) == sorted(RULE_KEYS), ('RULE_FIELDS', f)
    rule = SHIP_R3.with_(**f)
    text = ','.join(f'{k}={getattr(rule, k):g}' if isinstance(getattr(rule, k), float) else f'{k}={getattr(rule, k)}' for k in RULE_KEYS)
    assert text == pins['rule']['text'], ('RULE_SPEC', text, pins['rule']['text'])
    stage = lambda probability: iter_instances(probability, rule)
    stage.rule = rule
    return stage


def build_arm(arm, w, keys, files, maps, newz, model, predictor_factory, selector, out_dir, instance_stage=None, levels=None, mf_variant=VARIANT):
    """One CSV with the package's image_stages (MF variant chain2, blend weight w), as run_csv builds it. chain2r: the
    weight arm's instancing rule is passed as image_stages' instance_stage (None = SHIP_R3)."""
    from filseg.reproduction.pipeline import image_stages, read_image
    from filseg.reproduction.csv_io import write_rows
    set_weight(w)
    predictor = predictor_factory(); rows = {}; keeps = {}; per = {}; t0 = time.monotonic()
    for i, key in enumerate(sorted(files)):
        raw = read_image(files[key]); a = {n: z[key] for n, z in maps.items()}; a.update({n: z[key] for n, z in newz.items()})
        for n, q in a.items():
            if q.shape != (2048, 2048) or q.dtype != (np.uint8 if n in ('full', 'local') else np.uint16) or q.max() > (255 if n in ('full', 'local') else DEN):
                raise ValueError('Malformed probability map: ' + n)
        if levels is not None:                       # chain2m: the level path (one level: the package code exactly)
            assert instance_stage is None, 'LEVELS_WITH_RULE'
            v = level_image_stages(raw, a, model, levels, pcand=predictor, selector=selector, mf_variant=VARIANT)
        else:
            v = image_stages(raw, a, model, pcand=predictor, selector=selector, mf_variant=mf_variant) if instance_stage is None else \
                image_stages(raw, a, model, instance_stage=instance_stage, pcand=predictor, selector=selector, mf_variant=mf_variant)
        rows[key] = v['rles']; keeps[key] = v['keep']
        per[key] = dict(candidates=v['candidates'], kept=int(np.sum(v['keep'])), mf_sha=v['stages']['MF_mean'], blend_sha=v['stages']['candidate_blend'],
                        neighbour=v['selector'].get('neighbour'), mfrich=v['selector'].get('mfrich'), pcand=dict(v['pcand'], taus=v['pcand']['taus']))
        if levels is not None and len(levels) > 1:
            kl = [t for t, k in zip(v['levels'], v['keep']) if k]
            per[key]['levels'] = dict(candidates={f'{t:g}': v['levels'].count(t) for t in levels}, kept={f'{t:g}': kl.count(t) for t in levels},
                                      overlapping_pairs=len(v['overlap']), over_cutoff=int(sum(np.asarray(v['selector']['probability']) >= selector['threshold'])))
        if (i + 1) % 30 == 0: print(json.dumps(dict(stage='csv_' + arm, done=i + 1, total=len(files), elapsed_seconds=time.monotonic() - t0)), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True); csv = out_dir / 'submission.csv'
    counts = write_rows(csv, sorted(files), rows, keeps)
    lvl = None
    if levels is not None and len(levels) > 1:
        lvl = {k: {f'{t:g}': sum(x['levels'][k][f'{t:g}'] for x in per.values()) for t in levels} for k in ('candidates', 'kept')}
        lvl.update(overlapping_pairs=sum(x['levels']['overlapping_pairs'] for x in per.values()), over_cutoff=sum(x['levels']['over_cutoff'] for x in per.values()))
    return csv, dict(weight=w, levels=levels, level_counts=lvl, candidates=sum(x['candidates'] for x in per.values()), kept=sum(counts.values()), images_with_rows=sum(1 for c in counts.values() if c),
                     pcand_totals={k: sum(x['pcand'][k] for x in per.values()) for k in ('kept', 'fallback', 'changed', 'added', 'removed')},
                     mf_map_sha256=hashlib.sha256(json.dumps({k: per[k]['mf_sha'] for k in sorted(per)}).encode()).hexdigest(),
                     seconds=time.monotonic() - t0), per


def validation(csv, keys):
    """The package's competition-format check plus the order's: 180 images, ids unique, no overlap, every row decodes."""
    from filseg.reproduction.validate import validate_submission
    v = validate_submission(csv, keys)
    v['images_expected_180'] = len(keys) == 180 and v['expected_files'] == 180 and v['files_with_rows'] + v['files_without_rows'] == 180
    v['order_checks_passed'] = bool(v['passed'] and v['images_expected_180'] and v['ids_unique'] and not v['overlapping_pixel_images'] and v['decoded_rows'] == v['rows'])
    return v


def shift(per, dist):
    """Descriptive: test medians of the 6 neighbour and 8 MF-rich features vs the arm's training [q05, q95] (ship_fit.json)."""
    out = {}
    for block in ('neighbour', 'mfrich'):
        rows = [r for x in per.values() for r in (x[block] or [])]
        for n in (rows[0] if rows else {}):
            med = float(np.median([r[n] for r in rows])); d = dist[n]
            out[n] = dict(test_median=med, train_q05=d['q05'], train_q95=d['q95'], inside=bool(d['q05'] <= med <= d['q95']))
    return out


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import cv2; cv2.setNumThreads(1)
    from filseg.reproduction.pipeline import prepare_images, test_keys
    from filseg.reproduction.quality import load_quality
    from filseg.reproduction.assets import resolve_assets
    from filseg.reproduction.mfmaps import map_set_sha256, MAP_SETS
    from filseg.reproduction.validate import compare_submissions
    from filseg.reproduction.audit import verify
    from filseg.reproduction import refit, pcand as PC
    OUT.mkdir(parents=True, exist_ok=True); keys = test_keys(); pins = json.loads(INPUTS.read_text())
    inferred = json.loads((MAPS / 'inference_result.json').read_text())
    ms = map_set_sha256({n: inferred['networks'][n]['sha256'] for n in MAP_SETS['package']}); assert ms == MAP_SET_SHA, 'MAP_SET_IDENTITY'
    fit = json.loads((SHIP / 'ship_fit.json').read_text())['arms']; AW = level_arms(pins); CB = combined_arm(pins)
    assert sorted(fit) == sorted(['pcombo'] + [a for a, _, _ in AW] + ([CB['arm']] if CB else [])), ('SHIP_ARMS', sorted(fit))
    result = dict(status='running', created_at=now(), label=pins['label'], fold_sources=pins['folds'], weights=pins['weights'], map_sets=pins['sets'],
                  test_sources=test_sources(pins), map_set_sha256=ms, arms={})
    # ---- pcombo: the package's own CSV path (regression) and the deployment refit vs the pinned selector
    pc_dir = OUT / 'pcombo'
    subprocess.run([sys.executable, '-u', '-c', 'from filseg.reproduction.pipeline import run_csv;run_csv(' + ','.join(repr(str(x)) for x in (IMAGES, WEIGHTS, MAPS, pc_dir))
                    + ",pcand=True,selector='pcombo')"], check=True, env=dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=''))
    pcr = json.loads((pc_dir / 'csv_result.json').read_text()); pc_csv = pc_dir / 'submission.csv'
    shipped_pcombo = SHIP / 'pcombo_selector.json'
    result['pcombo'] = dict(csv_sha256=sha_file(pc_csv), expected_csv_sha256=PCOMBO_CSV_SHA, byte_identical=sha_file(pc_csv) == PCOMBO_CSV_SHA, kept=pcr['kept'], candidates=pcr['candidates'],
                            deployment_refit_sha256=sha_file(shipped_pcombo), pinned_selector_sha256=PCOMBO_SELECTOR_SHA,
                            deployment_refit_equals_pinned=sha_file(shipped_pcombo) == PCOMBO_SELECTOR_SHA == refit.PINNED['pcombo'][1],
                            validation=validation(pc_csv, keys))
    save(OUT / 'csv_check.json', result)
    print(json.dumps(dict(stage='pcombo_regression', **{k: result['pcombo'][k] for k in ('byte_identical', 'deployment_refit_equals_pinned')})), flush=True)
    # ---- weight arms
    root, _ = resolve_assets(WEIGHTS); model = load_quality(root / 'quality_model.json'); regressor = PC.deployed(root)
    files = prepare_images(IMAGES, OUT / 'unused_test_images')
    maps = {}
    for n in ('full', 'local'):
        p = MAPS / (n + '.npz'); verify(p, inferred['networks'][n]); maps[n] = np.load(p, allow_pickle=False)
    newz = load_new_test(NEWMAPS, pins, keys)
    unet = load_unet_test(NEWMAPS, pins, keys); check_stored_equivalent(pins, inferred)                                  # chain2u
    tun = load_test_unet(NEWMAPS, pins, keys); amaps = test_maps(maps, unet, tun)                                      # chainF: --test-unet fold5mean
    if tun: result['test_unet'] = test_unet_record(pins, tun); save(OUT / 'csv_check.json', result)
    if unet:
        result['unet'] = dict(rule='uint8 rint(level sum / (5 n)) over every set\'s fold-model test maps', sets={v: pins['unet'][v]['sets'] for v in unet},
                              stored_equivalent={v: pins['unet'][v]['stored_equivalent'] for v in unet}, **unet_digest(unet, maps, keys))
        for v in unet:
            if pins['unet'][v]['stored_equivalent']: assert result['unet'][v]['images_equal_to_package_map'] == len(keys), ('UNET_STORED_EQUIVALENT_MAP', v)
        save(OUT / 'csv_check.json', result)
    RA = refit_arms(pins); assert all(a in [x for x, _, _ in AW] + ([CB['arm']] if CB else []) for a in RA), ('PCAND_ARMS', RA)
    refit_ok = (SHIP.parent / 'deploy_fit.json').is_file() and all((SHIP.parent / 'deploy' / f'{a}_regressor.pkl').is_file() for a in RA)
    if RA and not refit_ok: result['pcand_refit'] = dict(skipped=True, reason='no refit regressors in case/out (worker pcand_refit_failure.json)')
    names = test_sources(pins); assert set(names) == set(newz)
    roles = {}
    for arm in [a for a, _, _ in AW] + ([CB['arm']] if CB else []):
        sel_path = SHIP / f'{arm}_selector.json'; assert sha_file(sel_path) == fit[arm]['selector_sha256'], ('SHIP_SELECTOR_IDENTITY', arm)
        rec = json.loads(sel_path.read_text()); roles[arm] = (rec['role'], rec['threshold'])
    install(names, roles, CB['names'] if CB else None); stage = instance_stage(pins)
    if stage is not None: result['instancing_rule'] = pins['rule']
    for arm, w, levels in AW:
        sel_path = SHIP / f'{arm}_selector.json'
        if levels is not None and len(levels) > 1:      # chain2m: 68 + level columns
            selector = load_level_selector(sel_path, sha_file(sel_path), roles[arm][0], fit[arm]['threshold'])
        else:
            selector = refit.load_selector(sel_path, sha_file(sel_path))
            assert selector['feature_order'] == refit.PCOMBO_FEATURES
        assert selector['threshold'] == fit[arm]['threshold']
        csv, stats, per = build_arm(arm, w, keys, files, amaps, newz, model, lambda: PC.load_regressor(*regressor), selector, OUT / arm, stage, levels)
        result['arms'][arm] = dict(stats, csv_sha256=sha_file(csv), selector_sha256=fit[arm]['selector_sha256'], cutoff=fit[arm]['threshold'], role=roles[arm][0],
                                   pcand_regressor_sha256=regressor[1], validation=validation(csv, keys), difference_vs_pcombo=compare_submissions(csv, pc_csv, keys),
                                   feature_shift_descriptive=shift(per, fit[arm]['feature_distribution']))
        save(OUT / 'csv_check.json', result)
        if arm in RA and refit_ok:                      # chainF: the refit P-cand variant of the same arm, next to the frozen CSV
            rp, rh = refit_regressor(SHIP, arm); a = arm + '_refit'
            csv2, stats2, per2 = build_arm(a, w, keys, files, amaps, newz, model, lambda p_=str(rp), h_=rh: PC.load_regressor(p_, h_), selector, OUT / a, stage, levels)
            result['arms'][a] = dict(stats2, csv_sha256=sha_file(csv2), selector_sha256=fit[arm]['selector_sha256'], cutoff=fit[arm]['threshold'], role=roles[arm][0],
                                     pcand_regressor_sha256=rh, refit_of=arm, validation=validation(csv2, keys), difference_vs_pcombo=compare_submissions(csv2, pc_csv, keys),
                                     difference_vs_frozen=compare_submissions(csv2, csv, keys))
            save(OUT / 'csv_check.json', result)
    if CB is not None:                                   # chain2s: the combined-map arm (MF = combiner map, blend weight 1, its own thr)
        assert stage is None, 'COMBINED_WITH_RULE'
        arm = CB['arm']; sel_path = SHIP / f'{arm}_selector.json'; cz = load_combined_test(NEWMAPS, pins, keys); assert sorted(cz) == CB['names']
        selector = refit.load_selector(sel_path, sha_file(sel_path)); assert selector['feature_order'] == refit.PCOMBO_FEATURES and selector['threshold'] == fit[arm]['threshold']
        cstage = combined_stage(CB['thr'])
        csv, stats, per = build_arm(arm, CB['weight'], keys, files, amaps, cz, model, lambda: PC.load_regressor(*regressor), selector, OUT / arm, cstage, None, mf_variant=CB['variant'])
        result['arms'][arm] = dict(stats, csv_sha256=sha_file(csv), selector_sha256=fit[arm]['selector_sha256'], cutoff=fit[arm]['threshold'], role=roles[arm][0],
                                   pcand_regressor_sha256=regressor[1], validation=validation(csv, keys), difference_vs_pcombo=compare_submissions(csv, pc_csv, keys),
                                   feature_shift_descriptive=shift(per, fit[arm]['feature_distribution']),
                                   combined=dict(name=CB['name'], thr=CB['thr'], rule=dict(thr=cstage.rule.thr), test_sources=CB['names'], mf_variant=CB['variant']))
        save(OUT / 'csv_check.json', result)
        if arm in RA and refit_ok:                      # chainF: the refit P-cand variant of the cmb arm
            rp, rh = refit_regressor(SHIP, arm); a = arm + '_refit'
            csv2, stats2, per2 = build_arm(a, CB['weight'], keys, files, amaps, cz, model, lambda p_=str(rp), h_=rh: PC.load_regressor(p_, h_), selector, OUT / a, cstage, None, mf_variant=CB['variant'])
            result['arms'][a] = dict(stats2, csv_sha256=sha_file(csv2), selector_sha256=fit[arm]['selector_sha256'], cutoff=fit[arm]['threshold'], role=roles[arm][0],
                                     pcand_regressor_sha256=rh, refit_of=arm, validation=validation(csv2, keys), difference_vs_pcombo=compare_submissions(csv2, pc_csv, keys),
                                     difference_vs_frozen=compare_submissions(csv2, csv, keys))
            save(OUT / 'csv_check.json', result)
    restore()
    result.update(status='complete', all_csv_valid=bool(result['pcombo']['validation']['order_checks_passed'] and all(a['validation']['order_checks_passed'] for a in result['arms'].values())),
                  regression_passed=bool(result['pcombo']['byte_identical'] and result['pcombo']['deployment_refit_equals_pinned']),
                  elapsed_seconds=time.monotonic() - START, finished_at=now(), segmentation_inference_calls=0, submission_calls=0)
    save(OUT / 'csv_check.json', result)
    print(json.dumps(dict(stage='csv_complete', regression=result['regression_passed'], valid=result['all_csv_valid'],
                          **{a: dict(sha=v['csv_sha256'][:12], kept=v['kept']) for a, v in result['arms'].items()})), flush=True)


if __name__ == '__main__': main()
