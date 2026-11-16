"""Kaggle-CPU-only worker of chain2 (order CH2): pcombo with every OOF Mask2Former map replaced by the per-image average of
one or more new MF map sets, under several MF blend weights (arms w<NN>), plus the deployment selectors for the test CSVs.

Derived from chain/case/worker.py (CH1; itself mfcomb's, mfhires's, mftta's and pcombo's). Phase A on the stored maps, the
100 outer v1 fits, the pcombo selector and pass 2 are pcombo's code path unchanged (inc, inc_pnat, inc_pcand_v1 and pcombo
are exact replays; checked locally by finish.py against pcombo's records). The new maps come from the call's private
dataset (chain_inputs.json pins the files and the weights). Arm w<NN>: MF map = level sum of the image's n set maps /
(510 n); mixture = w * MF + (1 - w) / 2 * (full + local) (mt.mix_w; w = 0.25 is the shipped mixture bit for bit, so w25
with one set is chain CH1 arm new). For each arm the candidates, labels and all 68 selector + 64 v1 features are rebuilt
from that MF map and mixture; pcombo's selector recipe is refit (same inner folds, nested cutoff); P-cand v1 uses pcombo's
100 outer v1 regressors unchanged (fitted on the stored-map bank) on the arm's features; the repair runs on the arm's mixture.
Ship stage (test CSVs): per selector arm, the cutoff = median of its 100 nested cutoffs and one StandardScaler + LR fit on
all 707 images of that arm's bank (fitted twice, byte-identical), written in the package's selector format.
chain2s (only when chain_inputs.json has 'combined'): one more arm, cmb, whose MF map is a combiner's OOF probability map
(own private dataset, one archive per base fold; map / 510), mixture = that map * float32(1.0) (no U-Net terms), instancing
SHIP_R3.with_(thr=<archive thr>) for this arm only; everything downstream as a weight arm (68 features on that map and
mixture, the selector recipe refit on its own candidates with the nested cutoff, P-cand v1 on its mixture).
Ship stage (test CSVs): per selector arm, the cutoff = median of its 100 nested cutoffs and one StandardScaler + LR fit on
all 707 images of that arm's bank (fitted twice, byte-identical), written in the package's selector format.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import json, sys, time, tarfile, hashlib, subprocess, functools, multiprocessing as mp
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import numpy as np, cv2
from scipy.special import expit
from pycocotools import mask as mu
from load_inputs import prepare, unique, checked, read, sha, counts as count6
from bank_loader import native_original
from quality_features import features_for_mask
from filseg.instancing import iter_instances, SHIP_R3
from pq_metric import annotation_rles
from make_time_folds import temporal_groups, assign_groups
from contract import mix
from context_features import FEATURES as CTX_FEATURES, context_features
from rules import assert_split, pq
from pnat import repair as pnat_repair
from pcand import GRID, TAU, WINDOW_FEATURES, snap, choose_target, candidate_scores, window_features, prepare as prepare_repair, repair_taus
from count_plan import count_coefficients, threshold_sweep
import presel as P
import mr as S
import mt as M
import mf_features as MF
import neigh as NB
import levels as LV
import pt as T                   # chainF (order INT1): pcand's pure helpers (byte copy of pcand/case/pt.py)

G = {}
PCAND_ARMS = []                 # chainF (order INT1): arms whose P-cand v1 is refit in this kernel ([] = chain2m exactly)
UNET_SETS = {}                  # chain2u (order CH3): view -> U-Net set names swapped into the weight arms ({} = chain2 exactly)
RULE = SHIP_R3                   # chain2r (order PP1): the weight arms' instancing rule; SHIP_R3 unless chain_inputs.json has 'rule'


def set_rule(spec):
    """spec = chain_inputs.json['rule'] (dict(text, fields)) or None. Set before the pool forks (workers inherit RULE)."""
    global RULE
    if spec is None: RULE = SHIP_R3; return
    import rulespec
    RULE = SHIP_R3.with_(**spec['fields']); assert rulespec.rule_text(RULE) == spec['text'], ('RULE_SPEC', spec)


def set_pcand(arms):
    """chainF: arms = chain_inputs.json['pcand_refit']['arms'] or None. Each must be a weight arm of the call, at most once, and
    not a multi-level arm (the C5 LR39 row rule is not defined for overlapping level versions). Set before the pool forks."""
    arms = list(arms or [])
    assert len(set(arms)) == len(arms) and all(a in M.NEW_VARIANTS and not M.multi_level(a) for a in arms), ('PCAND_ARMS', arms, M.NEW_VARIANTS)
    PCAND_ARMS[:] = arms


def configure(weights, levelsets=None, combined=None, *, require_base=True):
    """The call's arm layout (mt.configure) mirrored into this module's globals (forked pool workers inherit them).
    chain2m: levelsets (chain_inputs.json 'levels') = None for chain2's layout."""
    global ARMS, PASS2_ARMS, SELECTOR_ARMS, CONTRASTS, VARIANT
    M.configure(weights, levelsets, combined, require_base=require_base)
    ARMS, PASS2_ARMS, SELECTOR_ARMS, CONTRASTS, VARIANT = M.ARMS, M.PASS2_ARMS, M.SELECTOR_ARMS, M.CONTRASTS, M.VARIANT


configure(M.DEFAULT_WEIGHTS)


def rule_of(variant):
    """The instancing rule of a map variant: SHIP_R3 for the stored maps, RULE for the weight arms (chain2r), and
    chain2s: SHIP_R3.with_(thr=t) for the combined arm (its own thr; no other arm is affected)."""
    if variant == 'stored': return SHIP_R3
    if M.COMBINED is not None and variant == M.COMBINED_ARM: return SHIP_R3.with_(thr=M.COMBINED['thr'])
    return RULE


def remote(): return Path('/kaggle/input').is_dir()
def now(): return subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S %Z'], text=True).strip()


def save(path, value):
    p = Path(path); p.parent.mkdir(exist_ok=True, parents=True); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False, separators=(',', ':'), default=native)); os.replace(tmp, p)


def native(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)


def progress(**kw):
    row = dict(**kw, elapsed_seconds=time.monotonic() - START, date=now()); save('out/progress.json', row); print(json.dumps(row), flush=True)


def project(m, x):
    return expit(((x - np.asarray(m['scaler_mean'])) / np.asarray(m['scaler_scale'])) @ np.asarray(m['coef'])[0] + m['intercept'][0])


def packed(m): return hashlib.sha256(np.packbits(m).tobytes()).hexdigest()
def encode(ms): return [mu.encode(np.asfortranarray(m, dtype=np.uint8)) for m in ms]


def ious(gts, pred):
    return [np.asarray(mu.iou(g, pred, [0] * len(pred))).reshape(len(g), len(pred)) if pred else np.zeros((len(g), 0)) for g in gts]


def total_counts(qs): return sum((count6(q) for q in qs), start=np.zeros(6))


# ---------------------------------------------------------------- remote inputs (pcombo's inventory, unchanged) + the Stage-A archives
# pcombo/worker.py inventory(), verbatim
def inventory():
    spec, annotations, models, outer, folds, images, query = prepare(); files = sorted(folds)
    cfg = read(H / 'support/confirmation_protocol.json'); pins = read(H / 'support/confirmation_inputs.json')
    root = unique('input_identity.json', 'filament-mfqualityconfirm-input-1109').parent
    assert read(root / 'input_identity.json') == pins
    prov = {}
    for k in (3, 4):
        spec[k] = {}; valid = sorted(f for f in files if folds[f] == k)
        for view in ('full', 'local'):
            row, m = native_original(cfg, pins, root, k, view)
            assert sorted(m['train_files']) == sorted(set(files) - set(valid)) and sorted(m['validation_files']) == valid and not m['smoke_subset']
            spec[k][view] = row
            prov[f'fold{k}_{view}'] = dict(source='confirmation original manifest', train=len(m['train_files']), validation=len(m['validation_files']),
                                         overlap=len(set(m['train_files']) & set(m['validation_files'])), validation_equals_fold=True, smoke_subset=False)
        row = pins['MF'][str(k)]; p = checked(root / row['filename'], row); m = read(checked(root / row['manifest_filename'], row['manifest']))
        assert m['fold'] == k and m['epochs'] == 12 and m['epochs_completed'] == 12 and m['init_policy'] == 'public_COCO_only'
        assert m['domain_warmstart'] is False and m['initial_state_sha256'] == cfg['expected_MF_initial_state_sha256']
        assert sorted(m['train_files']) == sorted(set(files) - set(valid)) and sorted(m['validation_files']) == valid
        spec[k]['m2f_public'] = dict(row, local=str(p))
        prov[f'fold{k}_m2f_public'] = dict(source='MF remaining-fold result manifest', train=len(m['train_files']), validation=len(m['validation_files']),
                                         overlap=len(set(m['train_files']) & set(m['validation_files'])), validation_equals_fold=True, epochs=12)
        for row in spec[k].values():
            with np.load(row['local'], allow_pickle=False) as z: assert sorted(z.files) == valid
    mfp = read(H / 'frozen/evidence/mf_inputs.json'); data = unique('input_identity.json', 'filament-crossfoldquality-input-1109').parent
    for k in (0, 1):
        valid = sorted(f for f in files if folds[f] == k)
        for view in ('full', 'local'):
            mr = mfp['original'][f'f{k}_{view}_manifest']; m = read(checked(data / mr['filename'], mr))
            assert sorted(m['validation_files']) == valid and sorted(m['train_files']) == sorted(set(files) - set(valid)) and not m['smoke_subset']
            prov[f'fold{k}_{view}'] = dict(source='crossfold original manifest', train=len(m['train_files']), validation=len(m['validation_files']),
                                         overlap=len(set(m['train_files']) & set(m['validation_files'])), validation_equals_fold=True, smoke_subset=False)
        row = mfp['MF'][str(k)]; m = read(checked(data / row['manifest_filename'], row['manifest']))
        assert m['fold'] == k and sorted(m['train_files']) == sorted(set(files) - set(valid)) and sorted(m['validation_files']) == valid
        prov[f'fold{k}_m2f_public'] = dict(source='MF crossfold result manifest', train=len(m['train_files']), validation=len(m['validation_files']),
                                         overlap=len(set(m['train_files']) & set(m['validation_files'])), validation_equals_fold=True, epochs=m['epochs'])
    ev = read(H / 'evidence/fold2_map_provenance.json'); valid = sorted(f for f in files if folds[f] == 2)
    for view in ('full', 'local'):
        e = ev[view]; m = read(H / e['manifest_file']); assert sha(H / e['manifest_file']) == e['manifest_sha256']
        assert spec[2][view]['sha256'] == e['dataset_map']['sha256'] and spec[2][view]['md5'] == e['dataset_map']['md5']
        assert sorted(m['validation_files']) == valid and sorted(m['train_files']) == sorted(set(files) - set(valid)) and not m['smoke_subset']
        prov[f'fold2_{view}'] = dict(source='oof_bank5 phaseA manifest beside the identical-md5 Drive map', train=len(m['train_files']),
                                     validation=len(m['validation_files']), overlap=len(set(m['train_files']) & set(m['validation_files'])),
                                     validation_equals_fold=True, smoke_subset=False)
    e = ev['m2f_public']; dp = unique('protocol.json', 'filament-archm2f-oof-1108'); assert sha(dp) == e['protocol_sha256'] == sha(H / e['protocol_file'])
    m = read(dp); assert spec[2]['m2f_public']['sha256'] == e['dataset_map']['sha256'] and m['fold'] == 2
    assert sorted(m['validation_files']) == valid and sorted(m['train_files']) == sorted(set(files) - set(valid))
    prov['fold2_m2f_public'] = dict(source='archm2f dataset protocol.json', train=len(m['train_files']), validation=len(m['validation_files']),
                                    overlap=len(set(m['train_files']) & set(m['validation_files'])), validation_equals_fold=True, epochs=m['epochs'])
    assert len(prov) == 15 and all(v['overlap'] == 0 and v['validation_equals_fold'] and v['train'] == 707 - v['validation'] for v in prov.values())
    raw = checked(unique('raw_inputs.tar.gz', 'filament-h1-radial-prepare-cpu-1104'), read(H / 'frozen/evidence/mf_inputs.json')['raw'])
    with tarfile.open(raw) as tar:
        for m in tar:
            n = Path(m.name).name
            if m.isfile() and '/train_images/' in '/' + m.name and n.endswith('.jpeg') and n[:-5] in files and not (images / n).exists():
                (images / n).write_bytes(tar.extractfile(m).read())
    assert len(list(images.glob('*.jpeg'))) == 707
    save('out/map_provenance.json', dict(all_15_maps_oof_for_their_base_fold=True, verified_before_any_fit=True,
                                         maps_used=['full', 'local', 'm2f_public'], maps=prov, date=now()))
    return spec, annotations, models, outer, folds, images, query


def inputs_inventory(folds):
    """The new MF map sets (private dataset pinned by chain_inputs.json): per fold k, n archives (one per map set) of the fold-k
    images, each written by a model that did not train on fold k. Keys = the base fold, uint16 [2048, 2048], levels <= 510
    (the package MF format), file identity (bytes, md5, sha256) asserted. Also returns the call's blend weights."""
    pins = read(H / 'chain_inputs.json'); files = sorted(folds); out = {'new': {}}; rows = {}
    src = M.fold_sources(pins['folds']); n = M.n_sets(src); weights = M.parse_weights(pins['weights'], not pins.get('no_w25'))
    for k in range(5):
        valid = sorted(f for f in files if folds[f] == k); paths = []; rows[f'fold{k}'] = []
        assert len(pins['oof'][str(k)]) == n
        for j, row in enumerate(pins['oof'][str(k)]):
            assert row['run_id'] == src[k][j], ('INPUT_RUN', k, j)
            p = checked(unique(row['file'], pins['dataset_scope']), row)
            with np.load(p, allow_pickle=False) as z:
                assert sorted(z.files) == valid, ('INPUT_KEYS', k, j)
                for f in valid:
                    a = z[f]; assert a.dtype == np.uint16 and a.shape == (2048, 2048) and int(a.max()) <= M.DENOMINATOR, ('INPUT_FORMAT', k, j, f)
            paths.append(str(p)); rows[f'fold{k}'].append(dict({x: row[x] for x in ('file', 'bytes', 'md5', 'sha256')}, run_id=src[k][j], path=str(p)))
        out['new'][k] = dict(locals=paths, denominator=M.DENOMINATOR)
    rows_out = dict(label=pins['label'], folds=pins['folds'], sets=n, weights=weights, archives=rows, format_checked=True)
    if 'rule' in pins: rows_out['rule'] = pins['rule']      # chain2r: only when a rule was given (no flag = chain2's record)
    if 'levels' in pins:                                     # chain2m: only when --levels was given
        assert 'rule' not in pins, 'LEVELS_WITH_RULE'; rows_out['levels'] = pins['levels']
    if 'combined' in pins:                                   # chain2s: only when --combined was given
        assert 'rule' not in pins and 'levels' not in pins, 'COMBINED_WITH_RULE_OR_LEVELS'
        c = pins['combined']; out['cmb'] = {}; crows = {}
        for k in range(5):
            valid = sorted(f for f in files if folds[f] == k); assert len(c['oof'][str(k)]) == 1, ('COMBINED_PINS', k); row = c['oof'][str(k)][0]
            p = checked(unique(row['file'], c['dataset_scope']), row)
            with np.load(p, allow_pickle=False) as z:
                assert sorted(z.files) == valid, ('COMBINED_KEYS', k)
                for f in valid:
                    a = z[f]; assert a.dtype == np.uint16 and a.shape == (2048, 2048) and int(a.max()) <= M.DENOMINATOR, ('COMBINED_FORMAT', k, f)
            out['cmb'][k] = dict(locals=[str(p)], denominator=M.DENOMINATOR)
            crows[f'fold{k}'] = [dict({x: row[x] for x in ('file', 'bytes', 'md5', 'sha256')}, path=str(p))]
        rows_out['combined'] = dict(arm=M.COMBINED_ARM, name=c.get('name'), thr=float(c['thr']), dataset=c.get('dataset'), archives=crows)
    if pins.get('no_w25'): rows_out['no_w25'] = True                        # chainF: only when --no-w25 was given
    if 'pcand_refit' in pins: rows_out['pcand_refit'] = pins['pcand_refit']      # chainF: only when --pcand-refit was given
    if pins.get('unet'):                                   # chain2u (order CH3): U-Net OOF archives of the swapped views
        rows_out['unet'] = {}
        for view, u in sorted(pins['unet'].items()):
            assert view in M.UNET_VIEWS and u['sets'] == M.parse_set_spec('+'.join(u['sets'])), ('UNET_SETS', view)
            out['unet_' + view] = {}; urows = {}
            for k in range(5):
                valid = sorted(f for f in files if folds[f] == k); paths = []; urows[f'fold{k}'] = []
                assert len(u['oof'][str(k)]) == len(u['sets']), ('UNET_SET_COUNT', view, k)
                for j, row in enumerate(u['oof'][str(k)]):
                    assert row['set'] == u['sets'][j] and row['fold'] == k, ('UNET_ROW', view, k, j)
                    p = checked(unique(row['file'], pins['dataset_scope']), row)
                    with np.load(p, allow_pickle=False) as z:
                        assert sorted(z.files) == valid, ('UNET_KEYS', view, k, j)       # the fold-k images = the fold-k model's held-out fold
                        for f in valid:
                            a = z[f]; assert a.dtype == np.uint8 and a.shape == (2048, 2048), ('UNET_FORMAT', view, k, j, f)
                    paths.append(str(p)); urows[f'fold{k}'].append(dict({x: row[x] for x in ('file', 'bytes', 'md5', 'sha256')}, set=row['set'], path=str(p)))
                out['unet_' + view][k] = dict(locals=paths, denominator=M.UNET_DENOMINATOR)
            rows_out['unet'][view] = dict(sets=u['sets'], stored_equivalent=bool(u['stored_equivalent']), archives=urows)
    return out, rows_out


# ---------------------------------------------------------------- per-image work (forked pool)
def init(state):
    cv2.setNumThreads(1); G.update(state)
    G['z'] = {k: {n: np.load(v['local'], allow_pickle=False) for n, v in rows.items() if n != 'yolo_roi'} for k, rows in G['spec'].items()}
    G['znew'] = {v: {k: [np.load(p, allow_pickle=False) for p in r['locals']] for k, r in rows.items()} for v, rows in G.get('newspec', {}).items()}


def variant_sources(f, variant):
    """The image's uint16 maps of every new map set (fold k = the image's base fold). chain2s: the combined arm reads its
    one combiner map instead."""
    k = G['folds'][f]; src = 'cmb' if M.COMBINED is not None and variant == M.COMBINED_ARM else 'new'
    return dict(new=[z[f] for z in G['znew'][src][k]])


def unet_maps(f):
    """chain2u: the swapped U-Net views of image f (fold-k archives of every set, k = the image's base fold), averaged."""
    k = G['folds'][f]
    return {v: M.mean_unet([z[f] for z in G['znew']['unet_' + v][k]]) for v in M.UNET_VIEWS if 'unet_' + v in G.get('znew', {})}


def branch_maps(f, variant='stored'):
    k = G['folds'][f]
    maps = {n: z[f].astype(np.float32) / G['spec'][k][n]['denominator'] for n, z in G['z'][k].items()}
    if variant != 'stored':
        assert G['spec'][k]['m2f_public']['denominator'] == M.DENOMINATOR
        maps['m2f_public'] = M.variant_mf(variant, variant_sources(f, variant))
        for v, p in unet_maps(f).items():
            assert G['spec'][k][v]['denominator'] == M.UNET_DENOMINATOR
            maps[v] = p
    assert set(maps) == {'full', 'local', 'm2f_public'}
    assert all(p.shape == (2048, 2048) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in maps.values())
    return maps


def gt_sets(f):
    gts = [annotation_rles(e.annotations) for e in G['annotations'][f]]
    return [g for g in gts if g]


def raw_image(f):
    raw = cv2.imread(str(G['images'] / (f + '.jpeg')), 0); assert raw.shape == (2048, 2048) and raw.dtype == np.uint8
    return raw


def decode(f, variant='stored'):
    rles, hashes = (G['rles'][f], G['hashes'][f]) if variant == 'stored' else (G['rles_v'][variant][f], G['hashes_v'][variant][f])
    masks = [np.ascontiguousarray(mu.decode(r).astype(bool)) for r in rles]
    assert [packed(m) for m in masks] == hashes
    return masks


# ---------------------------------------------------------------- phase A: candidates, 78 features (pcombo's prediction side)
def candidate_rows(f, variant='stored'):
    """Candidates of the incumbent mixture of the variant's maps and their 64 v1 + 6 neighbour + 8 MF features. Reads no annotation."""
    raw = raw_image(f)
    maps = branch_maps(f, variant); probs = dict(full=maps['full'], local=maps['local'], blend=(maps['full'] + maps['local']) * .5)
    p = M.mixture(maps, variant)
    if variant != 'stored' and M.LEVELS.get(variant) is not None: return level_candidate_rows(f, variant, raw, maps, probs, p)
    masks = [np.asarray(m, bool) for m in iter_instances(p, rule_of(variant))]
    nf = NB.neighbour_features(masks, p)
    rows = []; mfr = MF.image_features(masks, maps['m2f_public'])
    for m, g6, r8 in zip(masks, nf, mfr):
        q = features_for_mask(m, probs, raw); c = context_features(m, maps, raw); w = window_features(m, p)
        row = {**{n: float(q[n]) for n in G['order39']}, **c, **w, **g6, **r8}; assert len(row) == 78; rows.append(row)
    return masks, p, rows


def level_candidate_rows(f, variant, raw, maps, probs, p):
    """chain2m (order MT1): the union, level by level in the arm's order, of SHIP_R3.with_(thr=t) candidates of the mixture p.
    Per level the 64 v1 + 6 neighbour + 8 MF features exactly as candidate_rows computes them for one rule (the neighbour and
    MF-rich features read that level's candidates only); with more than one level each row also has 'level' = t."""
    levels = M.LEVELS[variant]; masks, rows = [], []
    for t in levels:
        ms = [np.asarray(m, bool) for m in iter_instances(p, SHIP_R3.with_(thr=t))]
        nf = NB.neighbour_features(ms, p); mfr = MF.image_features(ms, maps['m2f_public'])
        for m, g6, r8 in zip(ms, nf, mfr):
            q = features_for_mask(m, probs, raw); c = context_features(m, maps, raw); w = window_features(m, p)
            row = {**{n: float(q[n]) for n in G['order39']}, **c, **w, **g6, **r8}; assert len(row) == 78
            if len(levels) > 1: row['level'] = float(t)
            rows.append(row)
        masks += ms
    return masks, p, rows


def candidate_levels(variant, rows):
    """The level of every candidate of a level arm (one-level arms: its single level)."""
    levels = M.LEVELS[variant]
    return [r['level'] for r in rows] if len(levels) > 1 else [levels[0]] * len(rows)


def prediction_side(f):
    """pcombo's prediction side on the stored maps: candidates, features and every saved selector's keep pattern."""
    masks, p, rows = candidate_rows(f, 'stored')
    x = np.asarray([[r[n] for n in G['order39']] for r in rows], float).reshape(len(rows), 39)
    keeps = {(s, k): tuple(np.flatnonzero(project(m, x) >= m['threshold']).tolist()) for (s, k), m in G['models'].items()}
    return masks, p, rows, keeps


def image_pass(f):
    """pcombo's phase A on the stored maps (unchanged)."""
    masks, p, rows, keeps = prediction_side(f)
    gts = gt_sets(f); patterns = sorted(set(keeps.values())); pid = {key: i for i, key in enumerate(patterns)}
    scores = []; tau_path_equal = True; pnat_out = {}
    for key in patterns:
        seeds = [masks[j] for j in key]
        if not key: scores.append([]); continue
        pr = prepare_repair(seeds, p); Sc = np.zeros((len(key), len(GRID)))
        for ti, t in enumerate(GRID):
            out, _ = repair_taus(seeds, p, [t] * len(key), pr)
            Sc[:, ti] = candidate_scores(ious(gts, encode(out)), len(key))
            if t == TAU:
                ref, stats = pnat_repair(seeds, p); pnat_out[key] = (ref, stats)
                tau_path_equal &= all(np.array_equal(a, b) for a, b in zip(out, ref))
        scores.append(Sc.tolist())
    rle = encode(masks); qraw = ious(gts, rle)
    res = dict(n=len(masks), rle=rle, hashes=[packed(m) for m in masks], features=rows, patterns=[list(k) for k in patterns],
               pattern_of={f'{s}:{k}': pid[key] for (s, k), key in keeps.items()}, scores=scores, tau_path_equal=bool(tau_path_equal),
               n_gt_sets=len(gts), raw_iou=[q.tolist() for q in qraw])
    inc = {}; qpid = {}
    for s, assignment in G['outer'].items():
        key = keeps[(s, assignment[f])]; qpid[s] = pid[key]
        inc[s] = total_counts([q[:, np.asarray(key, dtype=int)] for q in qraw]).tolist()
    pn = {}; dup = {}
    for i in sorted(set(qpid.values())):
        key = patterns[i]; seeds = [masks[j] for j in key]; ki = np.asarray(key, dtype=int)
        ref, stats = pnat_out[key] if key in pnat_out else pnat_repair(seeds, p)
        rr = encode(ref); qs = ious(gts, rr)
        pn[i] = dict(counts=total_counts(qs).tolist(), stats=stats)
        dup[i] = dict(inc=P.duplicate_counts([rle[j] for j in key], [q[:, ki] for q in qraw]), inc_pnat=P.duplicate_counts(rr, qs))
    res.update(inc_counts=inc, query_pattern=qpid, pnat=pn, dup=dup)
    return f, res


def map_change(f, variant):
    """GT-free: how the variant's MF map and mixture differ from the stored ones on image f (probability units)."""
    a = branch_maps(f, 'stored'); b = branch_maps(f, variant)
    d = np.abs(b['m2f_public'] - a['m2f_public']); pa = mix(a, 'incumbent'); pb = M.mixture(b, variant)
    out = dict(mf_mean_abs=float(d.mean()), mf_max_abs=float(d.max()), mf_flip_at_half=float(((a['m2f_public'] >= .5) != (b['m2f_public'] >= .5)).mean()),
               mf_ge_half_stored=float((a['m2f_public'] >= .5).mean()), mf_ge_half_variant=float((b['m2f_public'] >= .5).mean()),
               mixture_mean_abs=float(np.abs(pb - pa).mean()), mixture_flip_at_half=float(((pa >= .5) != (pb >= .5)).mean()))
    for v in M.UNET_VIEWS:                                 # chain2u: only for a swapped view (chain2's keys otherwise)
        if 'unet_' + v in G.get('znew', {}):
            out[f'{v}_mean_abs'] = float(np.abs(b[v] - a[v]).mean()); out[f'{v}_flip_at_half'] = float(((a[v] >= .5) != (b[v] >= .5)).mean())
    return out


def image_pass_variant(variant, f):
    """Phase A for a map variant: candidates, the 78 features, raw IoU with every annotation set (labels). No saved-selector work."""
    masks, p, rows = candidate_rows(f, variant)
    gts = gt_sets(f); rle = encode(masks); qraw = ious(gts, rle)
    out = dict(n=len(masks), rle=rle, hashes=[packed(m) for m in masks], features=rows, n_gt_sets=len(gts), raw_iou=[q.tolist() for q in qraw],
               map_change=map_change(f, variant))
    if M.LEVELS.get(variant) is not None:          # chain2m: levels, overlapping pairs, nesting across levels
        lev = candidate_levels(variant, rows); pairs = LV.overlap_pairs(masks)
        out.update(level=lev, overlap=[list(x) for x in pairs], nesting=LV.nesting(masks, lev, M.LEVELS[variant], pairs),
                   areas=[int(m.sum()) for m in masks])
    if variant in PCAND_ARMS: out.update(pc_tables(masks, p, rows, gts))   # chainF: pcand's phase A on the same candidates
    return f, out


# ---------------------------------------------------------------- chainF: the P-cand v1 refit of pcand (order PC1) on an arm's own bank
# Copied from pcand/case/pworker.py (keep_patterns, image_pass_arm's tables, fit_task_arm, deployment_fit, pass2_arm 'refit');
# tests/test_pcand_refit.py checks each against pworker.py itself. The global-tau CSV and pass 2 are not computed (not ordered).
DEPLOY_SCOPE = 'filament-repro-weights-1111'
PC_RECIPE = 'HistGradientBoostingRegressor(random_state=42), default parameters (pcand_tau v1)'


def deployed_selector(pin):
    """The package's full-707 LR39 (role deployment_full707_original_only, 0.35): the keep rule of b4dcd7d1's training rows."""
    path = unique('quality_model.json', DEPLOY_SCOPE); assert sha(path) == pin['quality_model_sha256'], 'DEPLOYED_SELECTOR_IDENTITY'
    obj = read(path); sel = [m for m in obj.get('models', [obj]) if m.get('role') == pin['role']]; assert len(sel) == 1
    m = sel[0]; assert m['classes'] == [0, 1] and len(m['feature_order']) == 39 and m['threshold'] == pin['threshold']
    return m


def pc_keep_patterns(rows):
    """GT-free: the keep pattern of each saved C5 selector (s, k) and of the deployed selector (pworker.keep_patterns)."""
    x = np.asarray([[r[n] for n in G['order39']] for r in rows], float).reshape(len(rows), 39)
    keeps = {f'{s}:{k}': tuple(np.flatnonzero(project(m, x) >= m['threshold']).tolist()) for (s, k), m in G['models'].items()}
    dep = G['deployed']; xd = np.asarray([[r[n] for n in dep['feature_order']] for r in rows], float).reshape(len(rows), 39)
    keeps['deploy'] = tuple(np.flatnonzero(T.keep_of(dep, xd)[0]).tolist()) if len(rows) else ()
    return keeps


def pc_tables(masks, p, rows, gts):
    """pworker.image_pass_arm's additions: distinct keep patterns, pattern_of, the 9-tau table of every pattern."""
    keeps = pc_keep_patterns(rows); patterns = sorted(set(keeps.values())); pid = {key: i for i, key in enumerate(patterns)}
    tabs = [T.tau_table([masks[j] for j in key], p, gts) for key in patterns]
    return dict(patterns=[list(k) for k in patterns], pattern_of={key: pid[val] for key, val in keeps.items()}, scores=[t['scores'] for t in tabs],
                counts=[t['counts'] for t in tabs], tau_path_equal=bool(all(t['tau_path_equal'] for t in tabs)))


def pc_fit_task(vsk):
    """pworker.fit_task_arm: outer v1 refit (s, k) on arm v's bank (rows = the C5 (s, k) keeps of the outer-training images)."""
    v, s, k = vsk; AV = G['AV'][v]; files = G['files']; names = G['names_v1']; assignment = G['outer'][s]
    train = sorted(f for f in files if assignment[f] != k); valid = sorted(f for f in files if assignment[f] == k)
    assert_split(train, valid, G['groups'])
    assert hashlib.sha256(json.dumps(train, separators=(',', ':')).encode()).hexdigest() == G['models'][s, k]['train_identity_sha256'], ('TRAIN_IDENTITY', s, k)
    key = f'{s}:{k}'
    X, y, excluded = T.training_rows(AV, train, key, names); m = make_regressor().fit(X, y)
    pred = {f: ([snap(x) for x in m.predict(T.all_rows(AV, f, names))] if AV[f]['n'] else []) for f in valid}
    pooled = T.pool_counts({f: AV[f]['counts'] for f in train}, {f: AV[f]['pattern_of'] for f in train}, key, train); gt = T.choose_global(pooled)
    stats = dict(arm=v, partition_seed=s, outer=k, train_images=len(train), query_images=len(valid), rows=len(y), excluded_unmatched=excluded,
                 target_histogram={str(t): int((y == t).sum()) for t in GRID}, target_mean=float(y.mean()), n_iter=int(m.n_iter_),
                 global_tau=gt, training_pooled_pq=[pq(c) for c in pooled])
    return vsk, dict(stats=stats, pred=pred)


def pc_deployment_fit(v, AV, files, names):
    """pworker.deployment_fit: rows = candidates kept by the deployed LR39 (in-sample) on all images; one regressor fitted
    twice; payload = pickle of {model, features, recipe, training} (b4dcd7d1's layout, read by the package's pcand.load_regressor)."""
    import pickle
    X, y, excluded = T.training_rows(AV, files, 'deploy', names); assert len(X) > 0 and np.isfinite(X).all()
    m1 = make_regressor().fit(X, y); b1 = pickle.dumps(m1, protocol=5); m2 = make_regressor().fit(X, y); b2 = pickle.dumps(m2, protocol=5)
    deterministic = bool(np.array_equal(m1.predict(X), m2.predict(X))); assert deterministic, 'REGRESSOR_NOT_DETERMINISTIC'
    payload = pickle.dumps(dict(model=m1, features=list(names), recipe=PC_RECIPE,
                                training=f'all {len(files)} training images, chain2 arm {v} bank; keep = deployed full707 LR39 (in-sample); targets as pcand_tau'), protocol=5)
    path = Path(f'out/deploy/{v}_regressor.pkl'); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    pooled = T.pool_counts({f: AV[f]['counts'] for f in files}, {f: AV[f]['pattern_of'] for f in files}, 'deploy', files)
    kept = int(sum(len(AV[f]['patterns'][AV[f]['pattern_of']['deploy']]) for f in files))
    return dict(arm=v, training_images=len(files), candidates=int(sum(AV[f]['n'] for f in files)), kept_candidates=kept, rows=int(len(y)), excluded_unmatched=excluded,
                target_histogram={str(t): int((y == t).sum()) for t in GRID}, target_mean=float(y.mean()), n_iter=int(m1.n_iter_),
                regressor_sha256=hashlib.sha256(payload).hexdigest(), model_pickle_sha256=hashlib.sha256(b1).hexdigest(), refit_pickle_identical=bool(b1 == b2),
                deterministic=deterministic, global_tau=T.choose_global(pooled), pooled_pq=[pq(c) for c in pooled], features=list(names), file=str(path))


def pc_pass2_refit(f):
    """pworker.pass2_arm for the 'refit' variant: the arm's own pass-2 keeps (keeps2) repaired with the refit taus."""
    gts = gt_sets(f); seeds = G['partition_seeds']; rows = {}; info = {s: dict(dup={}, stats={}) for s in seeds}
    for v in PCAND_ARMS:
        p = M.mixture(branch_maps(f, v), v); masks = decode(f, v); prep = {}; cache = {}
        for s in seeds:
            key = tuple(G['keeps2'][f][s][v]); taus = tuple(G['taus_refit'][v][f][s][j] for j in key); ck = (key, taus)
            if ck not in cache:
                sd = [masks[j] for j in key]
                if key and key not in prep: prep[key] = prepare_repair(sd, p)
                out, st = repair_taus(sd, p, list(taus), prep.get(key))
                rl = encode(out); qs = ious(gts, rl)
                cache[ck] = dict(counts=total_counts(qs).tolist(), stats=st, dup=P.duplicate_counts(rl, qs))
            rows.setdefault(v, {})[s] = cache[ck]['counts']; info[s]['dup'][v] = cache[ck]['dup']; info[s]['stats'][v] = cache[ck]['stats']
    return f, rows, info


def pcand_refit_phase(AV, keeps2, E, per, files, folds, groups, outer, seeds, cfg, query, state, names_v1):
    """After the chain arms: per arm v in PCAND_ARMS the 100 outer refits, the deployment regressor (out/deploy/<v>_regressor.pkl,
    read by csv_worker.py for the <v>_refit CSV), pass 2 with the refit taus, and pcand's report (arms pcombo, <v>_raw,
    <v>_frozen = the chain arm, <v>_refit; contrasts of pt.contrasts_of without the global-tau ones; cohorts of pt.cohorts)."""
    vs = list(PCAND_ARMS); t0 = time.monotonic(); fitres = {}; jobs = [(v, s, k) for v in vs for s in seeds for k in range(5)]
    with mp.get_context('fork').Pool(4) as pool:
        for i, (key, r) in enumerate(pool.imap(pc_fit_task, jobs)):
            fitres[key] = r
            if (i + 1) % 20 == 0: progress(stage='pcand_refits', done=i + 1, total=len(jobs))
    refit = {v: {f: {s: fitres[v, s, outer[s][f]]['pred'][f] for s in seeds} for f in files} for v in vs}
    deploy = {}
    for v in vs: deploy[v] = pc_deployment_fit(v, AV[v], files, names_v1); progress(stage=f'pcand_deploy_fit_{v}', sha=deploy[v]['regressor_sha256'][:12])
    save('out/pcand_fits.json', {v: dict(outer=[fitres[v, s, k]['stats'] for s in seeds for k in range(5)], recipe=PC_RECIPE + f'; refit on chain arm {v} bank, rows = C5 (s, k) keeps', features=names_v1) for v in vs})
    save('out/deploy_fit.json', deploy)
    G.update(taus_refit=refit, keeps2=keeps2, partition_seeds=seeds)
    E2 = pool_map(pc_pass2_refit, files, 'pass2_pcand_refit', state)
    arms = ['pcombo'] + [T.arm(v, x) for v in vs for x in ('raw', 'frozen', 'refit')]
    contrasts = [c for c in T.contrasts_of(vs) if 'gtau' not in c[0]]
    p2 = {s: dict(pcombo=per[s]['pcombo']) for s in seeds}
    for s in seeds:
        for v in vs:
            p2[s][T.arm(v, 'raw')] = [E[f][1][s]['raw_counts'][v] for f in files]; p2[s][T.arm(v, 'frozen')] = per[s][v]
            p2[s][T.arm(v, 'refit')] = [E2[f][0][v][s] for f in files]
    save('out/pcand_counts.json', dict(files=files, arms=arms, per_partition_counts=p2, note='pcombo and <v>_frozen = the chain arms of this kernel; <v>_raw = their keeps unrepaired'))
    save('out/pcand_taus.json', dict(refit={v: {f: {str(s): refit[v][f][s] for s in seeds} for f in files} for v in vs}))
    reports = {name: T.summarize(p2, fs, files, groups, cfg, arms, contrasts) for name, fs in T.cohorts(files, folds, query).items()}
    dup_zero = all(E2[f][1][s]['dup'][v][k] == 0 for v in vs for s in seeds for f in files for k in ('overlapping_prediction_pairs', 'gt_matched_by_more_than_one_prediction'))
    result = dict(status='complete', arms_requested=vs, arms=arms, contrasts=[n for n, _, _ in contrasts], contrasts_spec=contrasts, subsets=reports,
                  report={v: {n: T.report_row(reports, n) for n, a, _ in contrasts if a.startswith(v + '_')} for v in vs}, duplicate_zero=bool(dup_zero),
                  tau_path_equal=bool(all(AV[v][f]['tau_path_equal'] for v in vs for f in files)),
                  deploy={v: {k: x for k, x in deploy[v].items() if k != 'features'} for v in vs}, v1_refits=len(fitres), deployment_fits=2 * len(vs),
                  per_partition_counts=p2, elapsed_seconds=time.monotonic() - t0, created_at=now())
    save('out/pcand_refit.json', {k: x for k, x in result.items() if k != 'per_partition_counts'})
    progress(stage='pcand_refit_complete', seconds=result['elapsed_seconds'], **{f'{v}_refit_minus_frozen': reports['all707']['contrasts'][f'{v}_refit_minus_{v}_frozen']['delta'] for v in vs})
    return result


# ---------------------------------------------------------------- phase B: v1 regressor, outer fits (pcombo), predictions on every variant
def make_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=42)


def training_rows(A, train, s, k, names):
    X, y, excluded = [], [], 0
    for f in train:
        i = A[f]['pattern_of'][f'{s}:{k}']; key = A[f]['patterns'][i]
        for r, j in enumerate(key):
            t = choose_target(A[f]['scores'][i][r])
            if t is None: excluded += 1; continue
            X.append([A[f]['features'][j][n] for n in names]); y.append(t)
    return np.asarray(X, float).reshape(len(X), len(names)), np.asarray(y, float), excluded


def all_rows(A, f, names):
    return np.asarray([[r[n] for n in names] for r in A[f]['features']], float).reshape(A[f]['n'], len(names))


def fit_task(sk):
    """pcombo's outer v1 fit (s, k) on the stored-map bank; the fitted regressor also predicts every candidate of the query
    images of fold k in each map variant (it is not refit on the variants)."""
    s, k = sk; A = G['A']; files = G['files']; names = G['names_v1']; assignment = G['outer'][s]
    train = sorted(f for f in files if assignment[f] != k); valid = sorted(f for f in files if assignment[f] == k)
    assert_split(train, valid, G['groups'])
    assert hashlib.sha256(json.dumps(train, separators=(',', ':')).encode()).hexdigest() == G['models'][s, k]['train_identity_sha256']
    X, y, excluded = training_rows(A, train, s, k, names); m = make_regressor().fit(X, y)
    stats = dict(partition_seed=s, outer=k, train_images=len(train), train_identity_sha256=G['models'][s, k]['train_identity_sha256'], rows=len(y),
                 excluded_unmatched=excluded, target_histogram={str(t): int((y == t).sum()) for t in GRID}, target_mean=float(y.mean()), n_iter=int(m.n_iter_))
    pred = {v: {} for v in ['stored'] + M.NEW_VARIANTS}
    for v in pred:
        AV = A if v == 'stored' else G['AV'][v]
        for f in valid:
            pred[v][f] = [snap(x) for x in m.predict(all_rows(AV, f, names))] if AV[f]['n'] else []
    return (s, k), dict(stats=stats, pred=pred)


# ---------------------------------------------------------------- phase D: selectors (pcombo's nested recipe)
def nested_select(bank, plan, ngt, train, inner, overlap=None):
    """chain2m: overlap (per image, the overlapping candidate pairs of a level arm) -> the sweep scores the greedy survivors only
    (levels.resolve_probabilities: kept(t) = survivors with p >= t, the rule applied after the cutoff)."""
    prob = np.full(len(bank['X']), np.nan); inner_fits = []
    for j in range(3):
        tr = [f for f in train if inner[f] != j]; va = [f for f in train if inner[f] == j]
        ids, y = P.label_rows(bank, tr); m = P.fit_selector(bank['X'][ids], y); q = P.candidate_ids(bank, va)
        if len(q): prob[q] = m.predict_proba(bank['X'][q])[:, 1]
        inner_fits.append(dict(inner=j, expanded_rows=len(ids), iterations=m[1].n_iter_.tolist()))
    if overlap is not None: prob = LV.resolve_probabilities(prob, bank['records'], train, overlap)
    tau, sweep = threshold_sweep(plan, ngt, bank['records'], train, prob, P.THRESHOLD_GRID)
    ids, y = P.label_rows(bank, train); final = P.fit_selector(bank['X'][ids], y)
    return tau, sweep, dict(P.selector_state(final), expanded_rows=len(ids), threshold=tau, inner_fits=inner_fits)


# ---------------------------------------------------------------- ship stage: deployment selectors for the test CSVs
PCOMBO_ROLE = 'deployment_refit_pcombo_full707_mf_mixture'
PCOMBO_CUTOFF = 0.35
RECIPE = 'StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=2000, random_state=42), presel.fit_selector'
LABELS = 'per (candidate, nonempty annotation set): IoU > 0.5 with any GT of the set'
PCOMBO_BANK = 'MF-mixture SHIP_R3 candidates of the OOF maps; 39 quality + 15 context + 6 neighbour (neigh.py on the mixture p) + 8 MF-rich features (mf_features.py on the OOF MF map); pcombo worker.prediction_side'


def bank_text(arm):
    if arm != 'pcombo' and M.COMBINED is not None and arm == M.COMBINED_ARM:     # chain2s
        return (f'MF-mixture candidates of the combined OOF map (chain2s arm {arm}: MF = the combiner probability map {M.COMBINED.get("name")} / 510; '
                f'mixture = MF * float32(1.0), no U-Net terms; instancing SHIP_R3 with thr={M.COMBINED["thr"]:g}); 39 quality + 15 context + 6 neighbour + 8 MF-rich '
                f'features on that MF map and mixture; chain2s worker.candidate_rows(f, "{arm}")')
    t = bank_text_chain2(arm)
    if arm != 'pcombo' and M.multi_level(arm):
        return t + f'; chain2m levels {",".join(f"{x:g}" for x in M.LEVELS[arm])} (union of SHIP_R3 with thr = each level; feature level; greedy overlap resolution after the cutoff)'
    if arm == 'pcombo' or RULE == SHIP_R3: return t
    import rulespec
    return t + f'; instancing rule {rulespec.rule_text(RULE)} in place of SHIP_R3 (chain2r)'


def bank_text_chain2(arm):
    if arm == 'pcombo': return PCOMBO_BANK
    w = M.WEIGHT[arm]
    return (f'MF-mixture SHIP_R3 candidates of the averaged new OOF MF maps (chain2 arm {arm}: MF = level sum of the set maps / (510 n); '
            f'mixture w = {w:.2f} MF + {(1 - w) / 2:.3f} full + {(1 - w) / 2:.3f} local); 39 quality + 15 context + 6 neighbour + 8 MF-rich features on that MF map and mixture; '
            f'chain2 worker.candidate_rows(f, "{arm}")') + ''.join(f'; U-Net {v} = {"+".join(UNET_SETS[v])} (chain2u: level sum of the set maps / (255 n))' for v in M.UNET_VIEWS if v in UNET_SETS)


def ship_role(arm): return PCOMBO_ROLE if arm == 'pcombo' else f'deployment_refit_chain2_{arm}_full707_mf_mixture'


def selector_record(model, arm, threshold, feature_order, expanded_rows, candidates, images):
    """The deployable selector state in the package's format (refit.load_selector reads it); deterministic JSON bytes.
    For pcombo the bytes must equal the pinned package selector (fd25d491...)."""
    state = P.selector_state(model)
    rec = dict(role=ship_role(arm), feature_order=list(feature_order), classes=state['classes'], scaler_mean=state['scaler_mean'],
               scaler_scale=state['scaler_scale'], coef=state['coef'], intercept=state['intercept'], threshold=threshold,
               iterations=state['iterations'], training=dict(images=images, candidates=candidates, expanded_rows=expanded_rows,
               recipe=RECIPE, labels=LABELS, bank=bank_text(arm)))
    return rec, (json.dumps(rec, indent=1) + '\n').encode()


def ship_cutoff(thresholds):
    """The median of an arm's 100 nested cutoffs (the rule every ship lane used), as a Python float."""
    import statistics
    assert len(thresholds) == 100
    return float(statistics.median(thresholds))


def ship(banks, selectors, files, names, folds, overlaps=None):
    """Per selector arm: cutoff = median of its 100 nested cutoffs; one fit on all 707 images of the arm's bank (fitted twice,
    byte-identical); in-sample keep; training distributions of the 6 neighbour + 8 MF-rich features (test shift reference)."""
    out = {}; Path('out/ship').mkdir(parents=True, exist_ok=True)
    for arm in SELECTOR_ARMS:
        b = banks[arm][0]; cut = [x['threshold'] for x in selectors if x['arm'] == arm]; tau = ship_cutoff(cut)
        if arm == 'pcombo': assert PCOMBO_CUTOFF is None or tau == PCOMBO_CUTOFF, ('PCOMBO_CUTOFF', tau)
        ids, y = P.label_rows(b, files); cand = int(len(P.candidate_ids(b, files)))
        an = M.arm_feature_names(names, arm)
        rec1, b1 = selector_record(P.fit_selector(b['X'][ids], y), arm, tau, an, len(ids), cand, len(files))
        rec2, b2 = selector_record(P.fit_selector(b['X'][ids], y), arm, tau, an, len(ids), cand, len(files))
        assert b1 == b2, ('SELECTOR_NOT_BYTE_IDENTICAL', arm)
        Path(f'out/ship/{arm}_selector.json').write_bytes(b1)
        X = b['X']; keep = P.project(rec1, X) >= tau if len(X) else np.zeros(0, bool)
        if overlaps is not None and overlaps.get(arm) is not None and len(X):      # chain2m: the greedy rule, as on test
            keep = LV.resolve_probabilities(P.project(rec1, X), b['records'], files, overlaps[arm]) >= tau
        out[arm] = dict(selector_sha256=hashlib.sha256(b1).hexdigest(), role=rec1['role'], threshold=tau, two_fits_byte_identical=True,
                        cutoff_histogram={str(t): int(sum(x == t for x in cut)) for t in sorted(set(cut))}, sorted_cutoffs_50_51=sorted(cut)[49:51],
                        training_images=len(files), candidates=int(len(X)), expanded_rows=int(len(ids)), positive_rows=int(np.sum(y)), iterations=rec1['iterations'],
                        in_sample_kept=int(keep.sum()),
                        feature_distribution={n: quantiles(X[:, an.index(n)]) for n in list(NB.FEATURES) + list(MF.FEATURES) + (['level'] if 'level' in an else [])})
    save('out/ship/ship_fit.json', dict(arms=out, features=names, features_by_arm={a: M.arm_feature_names(names, a) for a in SELECTOR_ARMS}, created_at=now()))
    progress(stage='ship_fits', **{a: v['selector_sha256'][:12] for a, v in out.items()})
    return out


# ---------------------------------------------------------------- phase E: pass 2 (P-cand v1 after every keep) on every image
def pass2_image(f):
    gts = gt_sets(f); ctx = {}; gt_areas = [[float(a) for a in mu.area(g)] for g in gts]
    for v in sorted({VARIANT[a] for a in PASS2_ARMS}):
        AA = G['A'] if v == 'stored' else G['AV'][v]
        ctx[v] = dict(p=M.mixture(branch_maps(f, v), v), masks=decode(f, v), prep={}, qraw=[P._mat(q, AA[f]['n']) for q in AA[f]['raw_iou']],
                      areas=[float(a) for a in mu.area(AA[f]['rle'])] if AA[f]['n'] else [])
    keeps = G['keeps2'][f]; pred = G['pred_outer'][f]
    cache = {}; rows = {a: {} for a in PASS2_ARMS}; info = {}
    for s in sorted(keeps):
        cks = {}
        for arm in PASS2_ARMS:
            v = VARIANT[arm]; c = ctx[v]
            key = tuple(keeps[s][arm]); taus = tuple(pred[v][s][j] for j in key); ck = (v, key, taus); cks[arm] = ck
            if ck not in cache:
                seeds = [c['masks'][j] for j in key]
                if key and key not in c['prep']: c['prep'][key] = prepare_repair(seeds, c['p'])
                out, st = repair_taus(seeds, c['p'], list(taus), c['prep'].get(key))
                rl = encode(out); qs = ious(gts, rl); ki = np.asarray(key, dtype=int); qk = [q[:, ki] for q in c['qraw']]
                cache[ck] = dict(counts=total_counts(qs).tolist(), stats=st, dup=P.duplicate_counts(rl, qs),
                                 raw_counts=total_counts(qk).tolist(),
                                 bins=M.size_counts(qs, gt_areas, [float(a) for a in mu.area(rl)] if rl else []),
                                 raw_bins=M.size_counts(qk, gt_areas, [c['areas'][j] for j in key]))
            rows[arm][s] = cache[ck]['counts']
        info[s] = dict(dup={a: cache[cks[a]]['dup'] for a in PASS2_ARMS}, stats={a: cache[cks[a]]['stats'] for a in PASS2_ARMS},
                       raw_counts={a: cache[cks[a]]['raw_counts'] for a in PASS2_ARMS}, bins={a: cache[cks[a]]['bins'] for a in PASS2_ARMS},
                       raw_bins={a: cache[cks[a]]['raw_bins'] for a in PASS2_ARMS})
    return f, rows, info, len(cache)


# ---------------------------------------------------------------- summaries
def summarize(per, query, files, groups, cfg, arms):
    """pcombo's summarize (CI95 from the same bootstrap draws for every arm)."""
    ix = [files.index(f) for f in query]; seeds = cfg['partition_seeds']
    values = {s: {a: np.asarray(per[s][a])[ix] for a in arms} for s in seeds}
    means = {a: np.mean([values[s][a] for s in seeds], axis=0) for a in arms}
    gid = {f: min(g) for g in groups for f in g}; gids = sorted({gid[f] for f in query}); indices = [[i for i, f in enumerate(query) if gid[f] == g] for g in gids]
    mats = {a: np.stack([c[ix].sum(0) for ix in indices]) for a, c in means.items()}; rng = np.random.default_rng(cfg['bootstrap']['seed']); boots = {a: [] for a in arms}
    for _ in range(cfg['bootstrap']['draws']):
        ids = rng.integers(len(gids), size=len(gids))
        for a in arms: boots[a].append(pq(mats[a][ids].sum(0)))
    scores = {a: pq(c.sum(0)) for a, c in means.items()}; partition_pq = {a: [pq(values[s][a].sum(0)) for s in seeds] for a in arms}; contrasts = {}
    for name, a, b in CONTRASTS:
        ds = np.asarray(partition_pq[a]) - partition_pq[b]
        contrasts[name] = dict(delta=scores[a] - scores[b], mean_paired_partition_delta=float(ds.mean()), positive_of20=int((ds > 0).sum()), partition_deltas=ds.tolist(),
                               **S.intervals(np.asarray(boots[a]) - boots[b]))
    return dict(images=len(query), groups=len(gids), scores=scores, mean_counts={a: c.sum(0).tolist() for a, c in means.items()}, partition_pq=partition_pq, contrasts=contrasts)


def level_descriptives(AV, keeps2, files, seeds, cohorts):
    """chain2m, per level arm and cohort (all707 / without fold 2): candidates per level, overlapping pairs, nesting (candidates
    with a parent, identical to the parent), kept per level (partition mean)."""
    out = {}
    for a, levels in M.LEVELS.items():
        if levels is None: continue
        AA = AV[M.VARIANT[a]]; out[a] = {}
        for c, fs in (('all707', files), ('without_fold2', [f for f in files if f not in set(cohorts['fold2'])])):
            lv = lambda f, j: AA[f]['level'][j]
            out[a][c] = dict(images=len(fs), candidates={f'{t:g}': int(sum(sum(1 for x in AA[f]['level'] if x == t) for f in fs)) for t in levels},
                             overlapping_pairs=int(sum(len(AA[f]['overlap']) for f in fs)),
                             with_parent=int(sum(sum(1 for q in AA[f]['nesting']['parent'] if q >= 0) for f in fs)),
                             identical_to_parent=int(sum(sum(AA[f]['nesting']['identical_to_parent']) for f in fs)),
                             kept_per_level_partition_mean={f'{t:g}': float(np.mean([sum(1 for f in fs for j in keeps2[f][s][a] if lv(f, j) == t) for s in seeds])) for t in levels})
    return out


def quantiles(v):
    return dict(zip(['min', 'q05', 'q25', 'median', 'q75', 'q95', 'max'], np.quantile(v, [0, .05, .25, .5, .75, .95, 1]).tolist()), mean=float(np.mean(v))) if len(v) else {}


def pool_map(fn, items, stage, state, every=20):
    out = {}
    with mp.get_context('fork').Pool(4, initializer=init, initargs=(state,)) as pool:
        for i, (key, *rest) in enumerate(pool.imap(fn, items)):
            out[key] = rest[0] if len(rest) == 1 else rest
            if (i + 1) % every == 0 or i + 1 == len(items): progress(stage=stage, done=i + 1, total=len(items))
    return out


def main():
    global START
    START = time.monotonic(); assert remote(), 'REMOTE_CPU_ONLY'; cfg = read('protocol.json')
    assert cfg['score_started'] is False and tuple(cfg['tau_grid_v1']) == GRID and cfg['window_features_v1'] == WINDOW_FEATURES and cfg['context_features'] == CTX_FEATURES
    assert cfg['threshold_grid'] == P.THRESHOLD_GRID and cfg['fixed_arms'] == M.FIXED_ARMS and cfg['gate_contrast'] == M.GATE_CONTRAST
    assert cfg['mf_features'] == MF.FEATURES and cfg['neighbour_features'] == NB.FEATURES and cfg['replay_contrasts'] == M.REPLAY_CONTRASTS
    replay = read('out/replay_result.json'); assert replay['full_pipeline_replay_passed'] and replay['exact_historical_per_image_six_counts']
    for n, digest in cfg['repair_code_sha256'].items(): assert sha(H / n) == digest, ('REPAIR_CODE_IDENTITY', n)
    for n, digest in cfg['feature_code_sha256'].items(): assert sha(H / n) == digest, ('FEATURE_CODE_IDENTITY', n)
    spec, annotations, models, outer, folds, images, query = inventory(); files = sorted(folds); groups = temporal_groups(files, 24); seeds = cfg['partition_seeds']
    n_exp = cfg['expected_images']; assert len(files) == n_exp['all'] and len(query) == n_exp['query425']
    cohorts = S.cohorts(files, folds, query); assert len(cohorts['combined282']) == n_exp['query282']
    newspec, input_rows = inputs_inventory(folds)
    if input_rows.get('no_w25'): configure(input_rows['weights'], input_rows.get('levels'), input_rows.get('combined'), require_base=False)   # chainF: --no-w25
    elif 'combined' in input_rows: configure(input_rows['weights'], input_rows.get('levels'), input_rows['combined']); assert M.GATE_CONTRAST in [c[0] for c in CONTRASTS]    # chain2s
    else: configure(input_rows['weights'], input_rows.get('levels')); assert M.GATE_CONTRAST in [c[0] for c in CONTRASTS]     # chain2m's call exactly
    set_rule(input_rows.get('rule'))
    set_pcand((input_rows.get('pcand_refit') or {}).get('arms'))            # chainF: [] without --pcand-refit
    UNET_SETS.clear()
    for view, u in input_rows.get('unet', {}).items():    # chain2u: a stored-equivalent set must be the stored OOF maps byte for byte
        UNET_SETS[view] = list(u['sets'])
        if u['stored_equivalent']:
            assert all(row['sha256'] == spec[k][view]['sha256'] for k in range(5) for row in u['archives'][f'fold{k}']), ('UNET_NOT_STORED_EQUIVALENT', view)
    query_set = set(files)
    orders = {tuple(m['feature_order']) for m in models.values()}; assert len(orders) == 1; order39 = list(next(iter(orders)))
    names_v1 = order39 + CTX_FEATURES + WINDOW_FEATURES; assert names_v1 == cfg['features_v1'] and len(names_v1) == 64
    names = M.feature_names(order39, CTX_FEATURES); assert names == cfg['features'] and len(names) == 68
    inner = {}
    for s in seeds:
        for k in range(5):
            train = sorted(f for f in files if outer[s][f] != k); innergroups = [g for g in groups if g[0] in train]
            inner[s, k] = assign_groups(innergroups, 3, s + 10000 + k); assert set(inner[s, k]) == set(train)
    names_of = {a: M.arm_feature_names(names, a) for a in SELECTOR_ARMS}
    progress(stage='provenance_verified_before_fit', weights=M.WEIGHTS, arms=ARMS, map_sets=input_rows['sets'], levels=M.LEVELS, unet_sets=dict(UNET_SETS))
    state = dict(spec=spec, newspec=newspec, annotations=annotations, folds=folds, images=images, models=models, outer=outer, order39=order39)
    if PCAND_ARMS:                                       # chainF: the deployed LR39 (pcand_ship's deployment row rule)
        state['deployed'] = deployed_selector(input_rows['pcand_refit']['deployed_selector']); assert sorted(state['deployed']['feature_order']) == sorted(order39)
    cohort_of = {f: 'combined282' if folds[f] in (3, 4) else 'combined425' for f in files}
    # ---- phase A: stored maps (pcombo), then the four map variants
    A = pool_map(image_pass, files, 'image_pass_stored', state)
    AV = {v: pool_map(functools.partial(image_pass_variant, v), files, f'image_pass_{v}', state) for v in M.NEW_VARIANTS}
    inc_exact = all(np.array_equal(A[f]['inc_counts'][s], replay['per_image_counts'][f]['incumbent'][str(s)]) for f in query for s in seeds)
    tau_path_equal = all(A[f]['tau_path_equal'] for f in files)
    finite = {}
    for v, AA in [('stored', A)] + list(AV.items()):
        rows = [r for f in files for r in AA[f]['features']]
        x = np.asarray([[r[n] for n in sorted(set(names_v1) | set(names_of.get(v, names)))] for r in rows], float); finite[v] = bool(np.isfinite(x).all())
    assert all(finite.values()), ('FEATURES_UNDEFINED', finite)
    dup = {c: {a: {} for a in ARMS} for c in ('combined425', 'combined282')}
    for a in ('inc', 'inc_pnat'):
        for f in files:
            for s in seeds: dup[cohort_of[f]][a] = P.add_counts(dup[cohort_of[f]][a], A[f]['dup'][A[f]['query_pattern'][s]][a])
    progress(stage='image_pass_complete', inc_exact=inc_exact, tau_path_equal=tau_path_equal, candidates={v: int(sum(AA[f]['n'] for f in files)) for v, AA in [('stored', A)] + list(AV.items())})
    # ---- phase B: the 100 outer v1 regressors (pcombo), predictions on every variant
    G.update(A=A, AV=AV, files=files, names_v1=names_v1, outer=outer, groups=groups, models=models, query_set=query_set)
    fitres = {}
    with mp.get_context('fork').Pool(4) as pool:
        for i, (sk, r) in enumerate(pool.imap(fit_task, [(s, k) for s in seeds for k in range(5)])):
            fitres[sk] = r
            if (i + 1) % 20 == 0: progress(stage='v1_fits', done=i + 1, total=100)
    pred_outer = {v: {f: {s: fitres[s, outer[s][f]]['pred'][v][f] for s in seeds} for f in files} for v in ['stored'] + M.NEW_VARIANTS}
    save('out/fits.json', dict(v1=[fitres[s, k]['stats'] for s in seeds for k in range(5)],
                               recipe='HistGradientBoostingRegressor(random_state=42), default parameters (pcand_tau v1); fitted on the stored-map bank only', features=names_v1))
    progress(stage='v1_fits_complete')
    # ---- phase D: selectors (pcombo's recipe on 68 features; one bank per map variant)
    data = dict(pcombo=A, **{a: AV[VARIANT[a]] for a in SELECTOR_ARMS if a != 'pcombo'})
    rows_of = {a: {f: all_rows(data[a], f, names_of[a]) for f in files} for a in SELECTOR_ARMS}
    OV = {a: ({f: [tuple(x) for x in data[a][f]['overlap']] for f in files} if a != 'pcombo' and M.LEVELS.get(a) is not None else None) for a in SELECTOR_ARMS}
    banks = {}
    for a in SELECTOR_ARMS:
        raw_ious = {f: [P._mat(q, data[a][f]['n']) for q in data[a][f]['raw_iou']] for f in files}
        b = P.build_bank(files, rows_of[a], raw_ious); plan, ngt = count_coefficients(b['records'], len(b['X'])); banks[a] = (b, plan, ngt)
    keeps2 = {f: {} for f in files}; selectors = []; t_fit = 0; isolation = 0
    for s in seeds:
        for k in range(5):
            train = sorted(f for f in files if outer[s][f] != k); valid = sorted(f for f in files if outer[s][f] == k)
            assert_split(train, valid, groups); assert set(inner[s, k]) == set(train)
            assert hashlib.sha256(json.dumps(train, separators=(',', ':')).encode()).hexdigest() == models[s, k]['train_identity_sha256']; isolation += 1
            fits = {a: nested_select(*banks[a], train, inner[s, k], OV[a]) for a in SELECTOR_ARMS}; t_fit += 4 * len(SELECTOR_ARMS)
            for arm in SELECTOR_ARMS:
                tau, sw, st = fits[arm]
                selectors.append(dict(st, arm=arm, partition_seed=s, outer=k, train_images=len(train),
                                      candidates=int(len(P.candidate_ids(banks[arm][0], train))), sweep_pq=[r['pq'] for r in sw]))
            for f in valid:
                kp = dict(inc_pcand_v1=tuple(A[f]['patterns'][A[f]['query_pattern'][s]]))
                for arm in SELECTOR_ARMS:
                    tau, _, st = fits[arm]
                    if not data[arm][f]['n']: kp[arm] = ()
                    elif OV[arm] is None: kp[arm] = tuple(np.flatnonzero(P.project(st, rows_of[arm][f]) >= tau).tolist())
                    else: kp[arm] = tuple(LV.resolve_keep(P.project(st, rows_of[arm][f]), tau, OV[arm][f]))      # chain2m: greedy after the cutoff
                keeps2[f][s] = kp
        progress(stage='selectors', done=len(selectors) // (5 * len(SELECTOR_ARMS)), total=20, fits=t_fit)
    assert t_fit == 4 * 5 * 20 * len(SELECTOR_ARMS) and isolation == 100 and all(len(keeps2[f]) == 20 for f in files)
    save('out/selectors.json', dict(models=selectors, fit_count=t_fit, features=names, features_by_arm=names_of, threshold_grid=P.THRESHOLD_GRID))
    ship_result = ship(banks, selectors, files, names, folds, OV)
    # ---- phase E: pass 2
    G.update(keeps2=keeps2, pred_outer={f: {v: pred_outer[v][f] for v in pred_outer} for f in files}, rles={f: A[f]['rle'] for f in files}, hashes={f: A[f]['hashes'] for f in files},
             rles_v={v: {f: AV[v][f]['rle'] for f in files} for v in M.NEW_VARIANTS}, hashes_v={v: {f: AV[v][f]['hashes'] for f in files} for v in M.NEW_VARIANTS})
    E = pool_map(pass2_image, files, 'pass2', state)
    per = {s: dict(inc=[A[f]['inc_counts'][s] for f in files], inc_pnat=[A[f]['pnat'][A[f]['query_pattern'][s]]['counts'] for f in files],
                   **{a: [E[f][0][a][s] for f in files] for a in PASS2_ARMS}) for s in seeds}
    for a in PASS2_ARMS:
        for f in files:
            for s in seeds: dup[cohort_of[f]][a] = P.add_counts(dup[cohort_of[f]][a], E[f][1][s]['dup'][a])
    save('out/counts.json', dict(files=files, folds={f: folds[f] for f in files}, arms=ARMS, per_partition_counts=per))
    save('out/keeps.json', dict(files=files, keeps={f: {str(s): {a: list(v) for a, v in keeps2[f][s].items()} for s in seeds} for f in files},
                                v1_outer_taus={f: {str(s): pred_outer['stored'][f][s] for s in seeds} for f in files},
                                v1_outer_taus_variants={v: {f: {str(s): pred_outer[v][f][s] for s in seeds} for f in files} for v in M.NEW_VARIANTS}))
    reports = {name: summarize(per, fs, files, groups, cfg, ARMS) for name, fs in cohorts.items()}
    replayed = {name: dict(value=reports[name]['scores']['inc'], target=v, passed=abs(reports[name]['scores']['inc'] - v) < (5e-11 if name == 'fold2' else 1e-12))
                for name, v in cfg['replay_targets'].items()}
    replay_passed = bool(inc_exact and all(r['passed'] for r in replayed.values()))
    dup_zero = all(dup[c][a]['overlapping_prediction_pairs'] == 0 and dup[c][a]['gt_matched_by_more_than_one_prediction'] == 0 for c in dup for a in ARMS)
    integrity_passed = bool(tau_path_equal and all(finite.values()) and dup_zero and isolation == 100)
    inputs_passed = bool(input_rows['format_checked'])
    # ---- descriptives (none choose anything)
    nv = {a: (A if VARIANT[a] == 'stored' else AV[VARIANT[a]]) for a in ARMS}
    def by_cohort(c):
        fs = cohorts[c]; mc_ = reports[c]['mean_counts']
        dec = {a: dict(iou_sum=mc_[a][0], tp=mc_[a][1], fp=mc_[a][2], fn=mc_[a][3], mean_matched_iou=mc_[a][0] / mc_[a][1] if mc_[a][1] else 0.) for a in ARMS}
        diff = lambda a, b: {k_: dec[a][k_] - dec[b][k_] for k_ in ('iou_sum', 'tp', 'fp', 'fn', 'mean_matched_iou')}
        kept = {a: float(np.mean([sum(len(keeps2[f][s][a]) for f in fs) for s in seeds])) for a in PASS2_ARMS}
        cand = {v: int(sum(AA[f]['n'] for f in fs)) for v, AA in [('stored', A)] + list(AV.items())}
        smean = lambda arm, key: float(np.mean([sum(E[f][1][s]['stats'][arm][key] for f in fs) for s in seeds]))
        mch = {v: {k_: quantiles([AV[v][f]['map_change'][k_] for f in fs]) for k_ in AV[v][fs[0]]['map_change']} for v in M.NEW_VARIANTS}
        dist = {v: S.feature_distribution([r for f in fs for r in AA[f]['features']], list(MF.FEATURES) + list(NB.FEATURES))
                for v, AA in [('stored', A)] + list(AV.items())}
        rawpq = {a: [pq(np.sum([E[f][1][s]['raw_counts'][a] for f in fs], axis=0)) for s in seeds] for a in PASS2_ARMS}
        binmean = lambda key, a: {cat: {bn: float(np.mean([sum(E[f][1][s][key][a][cat][bn] for f in fs) for s in seeds])) for bn in M.SIZE_BINS} for cat in ('tp', 'fp', 'fn')}
        before_repair = dict(pq_partition_mean={a: float(np.mean(v)) for a, v in rawpq.items()},
                             paired_partition_deltas={n: (np.asarray(rawpq[x]) - rawpq[y]).tolist() for n, x, y in CONTRASTS if x in PASS2_ARMS and y in PASS2_ARMS},
                             note='selector keeps (or the saved C5 keeps for inc_pcand_v1) scored before the P-cand repair')
        size_bins = dict(after_repair={a: binmean('bins', a) for a in PASS2_ARMS}, before_repair={a: binmean('raw_bins', a) for a in PASS2_ARMS},
                         note='partition means of sums over images and annotation sets; TP/FN by GT instance area, FP by predicted mask area; bins <256, 256-1024, >1024 px; match IoU > 0.5')
        return dict(images=len(fs), candidates=cand, before_repair=before_repair, size_bins=size_bins, kept_per_partition_mean=kept, keep_rate_per_partition_mean={a: kept[a] / max(1, cand[VARIANT[a]]) for a in PASS2_ARMS},
                    counts_decomposition_partition_mean=dec, count_changes={n: diff(a, b) for n, a, b in CONTRASTS},
                    repair_profile_per_partition={a: {k_: smean(a, k_) for k_ in ('kept', 'fallback', 'changed', 'added', 'removed')} for a in PASS2_ARMS},
                    map_change=mch, feature_distribution=dist,
                    v1_tau_mean_on_kept={a: float(np.mean([t for f in fs for s in seeds for t in [pred_outer[VARIANT[a]][f][s][j] for j in keeps2[f][s][a]]] or [0.])) for a in PASS2_ARMS})
    sel = {a: [x for x in selectors if x['arm'] == a] for a in SELECTOR_ARMS}
    cut = {a: [x['threshold'] for x in sel[a]] for a in SELECTOR_ARMS}
    coef = {a: np.asarray([x['coef'][0] for x in sel[a]]) for a in SELECTOR_ARMS}
    descriptive = dict(by_cohort={c: by_cohort(c) for c in ('all707', 'combined425', 'combined282')},
                       cutoff={a: dict(quantiles(cut[a]), histogram={str(t): int(sum(x == t for x in cut[a])) for t in sorted(set(cut[a]))}) for a in SELECTOR_ARMS},
                       standardized_coefficients_mean={a: dict(zip(names_of[a], coef[a].mean(0).tolist())) for a in SELECTOR_ARMS},
                       levels=level_descriptives(AV, keeps2, files, seeds, cohorts),
                       mf_feature_coefficients={a: {n: dict(mean=float(coef[a][:, names.index(n)].mean()), sd=float(coef[a][:, names.index(n)].std())) for n in list(MF.FEATURES) + S.MF_CONTEXT}
                                                for a in SELECTOR_ARMS},
                       lbfgs_iterations_max={a: int(max(max(x['iterations']) for x in sel[a])) for a in SELECTOR_ARMS})
    dup_result = dict(stage='complete', cohorts=dup, zero_on_707=bool(dup_zero), date=now(),
                      note='Sums over images x 20 partitions per cohort. Pairs = predicted masks of one image and arm sharing a pixel; GT matched twice = GT rows with IoU > 0.5 to more than one prediction.')
    save('out/duplicate_audit.json', dup_result)
    save('out/map_change.json', {v: {f: AV[v][f]['map_change'] for f in files} for v in M.NEW_VARIANTS})
    result = dict(status='complete', replay_passed=replay_passed, replay=replayed, inc_exact_per_image_counts_425=bool(inc_exact),
                  pcombo_replay='counts, keeps, v1 taus, selector states, scores and contrast statistics of inc, inc_pnat, inc_pcand_v1 and pcombo compared locally in finish.py (blocking)',
                  inputs_passed=inputs_passed, inputs=input_rows, uniform_tau_path_equals_pnat_repair=bool(tau_path_equal), features_finite=finite,
                  duplicate_zero_on_707=bool(dup_zero), integrity_passed=integrity_passed,
                  query_group_isolation=dict(v1_outer_fits_checked=100, selector_fits_checked=isolation, selector_arms=len(SELECTOR_ARMS), inner_folds_within_outer_training=True),
                  subsets=reports, descriptive=descriptive, duplicate_audit=dup_result,
                  v1_regressor_fits=100, selector_fits=t_fit, segmentation_fits=0, segmentation_inference_calls=0, submission_calls=0, test_images_used=0,
                  partition_seeds=seeds, images=len(files), features=names, features_v1=names_v1, protocol_sha256=sha('protocol.json'),
                  elapsed_seconds=time.monotonic() - START, created_at=now(),
                  limitations=['Development data reused: the 707 images are those on which pcombo, mfrich and pneigh were measured',
                               'v1 regressors are pcombo\'s, fitted on the stored-map candidates, and applied unchanged to the variant candidates',
                               'CV MF maps average one out-of-fold model per map set; the test pipeline reads the level mean of every set\'s fold models\' test maps',
                               'The blend weights are compared on the same 707 images that chose nothing else here; the best weight is a selection on this CV'],
                  label=input_rows['label'], fold_sources=input_rows['folds'], map_sets=input_rows['sets'], weights=M.WEIGHTS, weight_arms=M.WEIGHT,
                  arm_contrasts=M.ARM_CONTRASTS, ship=ship_result, levels=M.LEVELS, levelsets=M.LEVELSETS)
    if M.COMBINED is not None: result['combined'] = M.COMBINED          # chain2s: only with the option
    result['gate'] = M.gate707(result)
    result['kernel_verdict'] = M.final_verdict(result['gate'], replay_passed and integrity_passed and inputs_passed)
    result['verdict_note'] = 'kernel_verdict covers the kernel checks only; the reported verdict also requires the local audit (exact pcombo replay)'
    save('out/science_result.json', result)
    save('out/decision.json', dict(kernel_verdict=result['kernel_verdict'], gate=result['gate'], adopt=False, test_inference=False, submission_calls=0, protocol_sha256=sha('protocol.json')))
    progress(stage='complete', gate=result['gate'], kernel_verdict=result['kernel_verdict'])
    if PCAND_ARMS:                                       # chainF: after every chain output is saved; a failure here leaves them intact
        try: pcand_refit_phase(AV, keeps2, E, per, files, folds, groups, outer, seeds, cfg, query, state, names_v1)
        except Exception as e:
            import traceback
            save('out/pcand_refit_failure.json', dict(type=type(e).__name__, reason=str(e)[:2000], traceback=traceback.format_exc(), date=now()))
            progress(stage='pcand_refit_failed', reason=str(e)[:300])


if __name__ == '__main__': main()
