"""chainF (order INT1) additions (each only when its key is in chain_inputs.json; absent = chain2m's audit exactly):
  unet         the replays against a chain2 / chain CH1 reference are skipped unless every swapped view is stored-equivalent
               (the arms then differ by construction); a descriptive cross-call contrast vs a validated chain2 call with the
               same fold tokens (vs_chain2_same_mf, each arm vs the same arm) when one exists locally.
  pcand_refit  the refit CSVs are validated with the others (out/<label>/submission_<arm>_refit.csv); pcand_refit.json must be
               complete with zero duplicates; when a validated pcand (PC1) run exists on the reused chain2 label for an arm
               (pcand/stage/<run>/pcand_inputs.json + results/<run>_<arm>.json), the arm's refit must equal it exactly
               (raw/frozen/refit per-partition counts, pcombo counts, refit taus, deployment regressor sha256, the shared
               contrasts' statistics in every cohort, the refit CSV sha256) -- blocking.

Below: chain2r's original header.

Local audit and report of one chain2r call (order PP1; chain2 = order CH2; small JSON and the CSVs; never fits or predicts). Adapted from
chain/finish.py (CH1).

usage: finish.py <label>
reads   out/<label>/case/out/*.json (pulled CV results), out/<label>/case/out/ship/*.json, out/<label>/out/csv/... (CSVs + checks)
        read only, when present: a validated chain CH1 result with the same single-set fold tokens (chain/results/<ref>.json
        and chain/out/<ref>/...), the reference that arm w25 must reproduce exactly
writes  results/<label>.json            CV contrasts of every weight arm vs pcombo and vs w25 (combined707 PQ, CI95, positives/20,
                                        fold means), the exact pcombo replay, the w25 reference replay, ship cutoffs/selectors, CSV checks
        out/<label>/submission_<arm>.csv + out/<label>/csv_validation.json
Blocking: pcombo CV replay exact against pcombo's records; recomputation of every reported PQ / delta / interval from
counts.json (< 1e-12); pcombo CSV byte-identical to 35abbeed...; every CSV valid (re-checked here with the package's
validate_submission); when a CH1 reference exists, arm w25 = its arm new exactly (per-partition counts, keeps, v1 taus,
100 selectors, scores, contrast statistics, deployment selector numbers, CSV bytes).
"""
import hashlib, json, shutil, subprocess, sys, types
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
CASE = H / 'case'
CHAIN = H.parent / 'chain'                       # chain CH1 (read only): the w25 reference
PCAND = H.parent / 'pcand'                       # pcand PC1 (read only): the refit reference
CHAIN2 = H.parent / 'chain2'                     # chain2 (read only): the reused stage's validated call, the replay reference / baseline
SHIP_TEXT = 'thr=0.5,min_area=256,close_px=21,link_gap=60,link_angle=45'
PKG_SRC = H.parent / 'pkg/src'
sys.path[:0] = [str(CASE / 'frozen'), str(CASE / 'frozen/eval'), str(CASE / 'support'), str(CASE), str(PKG_SRC)]
try: import torch  # noqa: F401
except ImportError: sys.modules['torch'] = types.ModuleType('torch')   # the package's MF modules import torch; validation never runs a network
from make_time_folds import temporal_groups
import mr as R
import mt as M

SEL_FIELDS = ('threshold', 'coef', 'intercept', 'scaler_mean', 'scaler_scale', 'sweep_pq', 'expanded_rows', 'candidates', 'train_images')
STAT_FIELDS = ('delta', 'mean_paired_partition_delta', 'positive_of20', 'partition_deltas', 'paired_group_CI95')
SHIP_FIELDS = ('feature_order', 'classes', 'scaler_mean', 'scaler_scale', 'coef', 'intercept', 'threshold', 'iterations')
REPLAY_ARMS = ['inc', 'inc_pnat', 'inc_pcand_v1', 'pcombo']
PCOMBO_CSV_SHA = '35abbeed91108e40ddbd18c094c901d116441e363b10c31aa8092988d8e8f6e6'
REFIT_ARMS = []                          # chainF: set by finish() from chain_inputs.json
FOLD5MEAN_SHA = dict(full='ff0c7a809993de4ac99681ff83ea0175118ec274c4dae75cf065469e42695fb9', local='c62d6e2aa7273c4e8c57a994e11e47737c13f9e0a2a2a4c9696f479e479cfccf')
PCOMBO_ALL707 = 0.42957651412886416      # reference/pcombo_science_result.json subsets.all707.scores.pcombo


def read(p): return json.loads(Path(p).read_text())
def now(): return subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S %Z'], text=True).strip()
def pq(c): d = c[1] + .5 * (c[2] + c[3]); return float(c[0] / d) if d else 0.
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selector_reproduction(mine, ref, arm, ref_arm=None):
    ref = {(x['partition_seed'], x['outer']): x for x in ref if x['arm'] == (ref_arm or arm)}
    mine = [x for x in mine if x['arm'] == arm]; keys = {(x['partition_seed'], x['outer']) for x in mine}
    differing = [dict(partition_seed=x['partition_seed'], outer=x['outer'], fields=[n for n in SEL_FIELDS if x[n] != ref[x['partition_seed'], x['outer']][n]])
                 for x in mine if (x['partition_seed'], x['outer']) in ref]
    differing = [d for d in differing if d['fields']]
    return dict(arm=arm, reference_arm=ref_arm or arm, compared=len(keys & set(ref)), differing_count=len(differing), differing=differing[:20],
                exact=bool(len(ref) == 100 and keys == set(ref) and len(mine) == 100 and not differing))


def pcombo_replay(r, per, kj, sel, fits, files, seeds):
    """chain finish.pcombo_replay against the pcombo lane's records (case/reference, hashes in case/frozen_identity.json)."""
    ref = CASE / 'reference'; ident = read(CASE / 'frozen_identity.json')['files']
    for n in ('pcombo_counts.json', 'pcombo_keeps.json', 'pcombo_selectors.json', 'pcombo_fits.json', 'pcombo_science_result.json'):
        assert sha(ref / n) == ident['reference/' + n]['sha256'], ('REFERENCE', n)
    rc = read(ref / 'pcombo_counts.json'); rk = read(ref / 'pcombo_keeps.json'); rs = read(ref / 'pcombo_science_result.json')
    rsel = read(ref / 'pcombo_selectors.json')['models']; rf = read(ref / 'pcombo_fits.json')
    assert rc['files'] == files and rk['files'] == files
    counts = {a: all(np.array_equal(np.asarray(per[s][a], float), np.asarray(rc['per_partition_counts'][s][a], float)) for s in seeds) for a in REPLAY_ARMS}
    keeps = {a: int(sum(kj['keeps'][f][s][a] != rk['keeps'][f][s][a] for f in files for s in seeds)) for a in ('inc_pcand_v1', 'pcombo')}
    taus = int(sum(kj['v1_outer_taus'][f][s] != rk['v1_outer_taus'][f][s] for f in files for s in seeds))
    v1_stats = fits['v1'] == rf['v1']
    sels = selector_reproduction(sel, rsel, 'pcombo')
    scores = {c: {a: r['subsets'][c]['scores'][a] == rs['subsets'][c]['scores'][a] for a in REPLAY_ARMS} for c in rs['subsets']}
    stats = {c: {n: all(r['subsets'][c]['contrasts'][n][f] == rs['subsets'][c]['contrasts'][n][f] for f in STAT_FIELDS) for n in M.REPLAY_CONTRASTS} for c in rs['subsets']}
    partition = {c: r['subsets'][c]['partition_pq']['pcombo'] == rs['subsets'][c]['partition_pq']['pcombo'] for c in rs['subsets']}
    exact = bool(all(counts.values()) and not any(keeps.values()) and taus == 0 and v1_stats and sels['exact'] and all(partition.values())
                 and all(all(v.values()) for v in scores.values()) and all(all(v.values()) for v in stats.values()) and len(rs['subsets']) == 9)
    return dict(exact=exact, arms=REPLAY_ARMS, counts_exact=counts, keeps_differing_image_partitions=keeps, v1_outer_taus_differing=taus, v1_fit_statistics_equal=v1_stats,
                selectors=sels, scores_exact=all(all(v.values()) for v in scores.values()), contrast_statistics_exact=stats, partition_pq_exact=partition,
                all707_pcombo=dict(value=r['subsets']['all707']['scores']['pcombo'], recorded=rs['subsets']['all707']['scores']['pcombo']),
                cohorts=sorted(rs['subsets']))


def find_reference(pins, chain_root=CHAIN):
    """A validated chain CH1 call with the same single-set fold tokens (its arm new = this call's w25), or None."""
    if int(pins['sets']) != 1: return None
    for p in sorted((chain_root / 'results').glob('*.json')):
        try: x = read(p)
        except ValueError: continue
        if x.get('fold_sources') == pins['folds'] and x.get('validated') is True and (chain_root / 'out' / x['label'] / 'case/out/counts.json').is_file(): return x['label']
    return None


def reference_replay(label, ref, r, per, kj, sel, csv, seeds, files, chain_root=CHAIN):
    """Arm w25 against the CH1 reference's arm new: everything must be equal (it is the same computation)."""
    rb = chain_root / 'out' / ref / 'case/out'; rr = read(chain_root / 'results' / f'{ref}.json')
    rc = read(rb / 'counts.json'); rk = read(rb / 'keeps.json'); rs = read(rb / 'science_result.json'); rsel = read(rb / 'selectors.json')['models']
    assert rc['files'] == files
    counts = all(np.array_equal(np.asarray(per[s]['w25'], float), np.asarray(rc['per_partition_counts'][s]['new'], float)) for s in seeds)
    keeps = int(sum(kj['keeps'][f][s]['w25'] != rk['keeps'][f][s]['new'] for f in files for s in seeds))
    taus = int(sum(kj['v1_outer_taus_variants']['w25'][f][s] != rk['v1_outer_taus_variants']['new'][f][s] for f in files for s in seeds))
    sels = selector_reproduction(sel, rsel, 'w25', 'new')
    scores = {c: r['subsets'][c]['scores']['w25'] == rs['subsets'][c]['scores']['new'] for c in rs['subsets']}
    partition = {c: r['subsets'][c]['partition_pq']['w25'] == rs['subsets'][c]['partition_pq']['new'] for c in rs['subsets']}
    stats = {c: all(r['subsets'][c]['contrasts']['w25_minus_pcombo'][f] == rs['subsets'][c]['contrasts']['new_minus_pcombo'][f] for f in STAT_FIELDS) for c in rs['subsets']}
    a = read(H / 'out' / label / 'case/out/ship/w25_selector.json'); b = read(rb / 'ship/new_selector.json')
    ship = {f: a[f] == b[f] for f in SHIP_FIELDS}
    ref_csv = rr['csv']['arms']['new']['sha256']; my_csv = csv['arms'].get('w25', {}).get('sha256')
    exact = bool(counts and keeps == 0 and taus == 0 and sels['exact'] and all(scores.values()) and all(partition.values()) and all(stats.values())
                 and all(ship.values()) and my_csv == ref_csv and len(rs['subsets']) == 9)
    return dict(reference=ref, reference_arm='new', exact=exact, per_partition_counts_equal=counts, keeps_differing_image_partitions=keeps, v1_taus_differing=taus,
                selectors=sels, scores_equal=scores, partition_pq_equal=partition, contrast_vs_pcombo_statistics_equal=stats, ship_selector_fields_equal=ship,
                csv_sha256=my_csv, reference_csv_sha256=ref_csv,
                all707=dict(w25=r['subsets']['all707']['scores']['w25'], reference_new=rs['subsets']['all707']['scores']['new'],
                            w25_partition_pq=r['subsets']['all707']['partition_pq']['w25'], reference_partition_pq=rs['subsets']['all707']['partition_pq']['new']))


def recompute(per, files, cohort, groups, cfg, contrasts):
    seeds = [str(s) for s in cfg['partition_seeds']]; ix = [files.index(f) for f in cohort]; arms = sorted({a for _, x, y in contrasts for a in (x, y)})
    means = {a: np.mean([np.asarray(per[s][a], float)[ix] for s in seeds], axis=0) for a in arms}
    gid = {f: min(g) for g in groups for f in g}; gids = sorted({gid[f] for f in cohort}); idx = [[i for i, f in enumerate(cohort) if gid[f] == g] for g in gids]
    mats = {a: np.stack([means[a][i].sum(0) for i in idx]) for a in arms}
    rng = np.random.default_rng(cfg['bootstrap']['seed']); boots = {a: [] for a in arms}
    for _ in range(cfg['bootstrap']['draws']):
        b = rng.integers(len(gids), size=len(gids))
        for a in arms: boots[a].append(pq(mats[a][b].sum(0)))
    part = {a: [pq(np.asarray(per[s][a], float)[ix].sum(0)) for s in seeds] for a in arms}
    out = dict(scores={a: pq(means[a].sum(0)) for a in arms}, contrasts={})
    for name, a, b in contrasts:
        ds = np.asarray(part[a]) - part[b]; d = np.asarray(boots[a]) - np.asarray(boots[b])
        out['contrasts'][name] = dict(delta=out['scores'][a] - out['scores'][b], mean_paired_partition_delta=float(ds.mean()), positive_of20=int((ds > 0).sum()),
                                      **{n: np.quantile(d, q).tolist() for n, q in R.CI_LEVELS.items()})
    return out


def chain2_replay(label, ref, r, per, kj, sel, csv, seeds, files, arms, chain2_root=CHAIN2, csv_refs=None):
    """chain2r without a rule (or with SHIP_R3 itself): every weight arm against the same arm of the reused chain2 call,
    everything equal (the same computation): per-partition counts, keeps, v1 taus, 100 selectors, scores and partition PQ of
    every cohort, contrast statistics, deployment selector numbers, CSV bytes."""
    rb = chain2_root / 'out' / ref / 'case/out'; rr = read(chain2_root / 'results' / f'{ref}.json')
    rc = read(rb / 'counts.json'); rk = read(rb / 'keeps.json'); rs = read(rb / 'science_result.json'); rsel = read(rb / 'selectors.json')['models']
    assert rc['files'] == files and rr['validated'] is True
    out = dict(reference=ref, arms={}, cohorts=sorted(rs['subsets']))
    for a in arms:
        counts = all(np.array_equal(np.asarray(per[s][a], float), np.asarray(rc['per_partition_counts'][s][a], float)) for s in seeds)
        keeps = int(sum(kj['keeps'][f][s][a] != rk['keeps'][f][s][a] for f in files for s in seeds))
        taus = int(sum(kj['v1_outer_taus_variants'][a][f][s] != rk['v1_outer_taus_variants'][a][f][s] for f in files for s in seeds))
        sels = selector_reproduction(sel, rsel, a)
        scores = all(r['subsets'][c]['scores'][a] == rs['subsets'][c]['scores'][a] for c in rs['subsets'])
        partition = all(r['subsets'][c]['partition_pq'][a] == rs['subsets'][c]['partition_pq'][a] for c in rs['subsets'])
        names = [f'{a}_minus_pcombo'] + ([f'{a}_minus_w25'] if a != 'w25' and 'w25' in arms else [])   # chainF: no w25 contrast without the w25 arm
        stats = all(r['subsets'][c]['contrasts'][n][f] == rs['subsets'][c]['contrasts'][n][f] for c in rs['subsets'] for n in names for f in STAT_FIELDS)
        x = read(H / 'out' / label / f'case/out/ship/{a}_selector.json'); y = read(rb / f'ship/{a}_selector.json')
        ship = all(x[f] == y[f] for f in SHIP_FIELDS)
        my_csv = csv['arms'].get(a, {}).get('sha256'); ref_csv = rr['csv']['arms'][a]['sha256']
        if csv_refs is not None: ref_csv = (csv_refs.get(a) or {}).get('sha256')     # chainF --test-unet: the fold5mean reference (chain3), or none
        csv_ok = (my_csv == ref_csv) if (csv_refs is None or ref_csv is not None) else True
        out['arms'][a] = dict(exact=bool(counts and keeps == 0 and taus == 0 and sels['exact'] and scores and partition and stats and ship and csv_ok),
                              csv_reference=None if csv_refs is None else (csv_refs.get(a) or dict(source='none: no fold5mean CSV of this chain2 arm; CV replay only')),
                              per_partition_counts_equal=counts, keeps_differing_image_partitions=keeps, v1_taus_differing=taus, selectors_exact=sels['exact'],
                              scores_equal=scores, partition_pq_equal=partition, contrast_statistics_equal=stats, ship_selector_fields_equal=ship,
                              csv_sha256=my_csv, reference_csv_sha256=ref_csv, all707=r['subsets']['all707']['scores'][a], reference_all707=rs['subsets']['all707']['scores'][a])
    out['exact'] = bool(out['arms'] and all(v['exact'] for v in out['arms'].values()) and len(rs['subsets']) == 9)
    return out


def vs_reference(ref, per, files, folds, groups, cfg, arms, chain2_root=CHAIN2):
    """arms: names (each vs the same arm of the reference) or (arm, reference arm) pairs (chain2m: a level arm vs its plain
    weight arm in the reference call)."""
    pairs = [(a, a) if isinstance(a, str) else tuple(a) for a in arms]
    return _vs_reference(ref, per, files, folds, groups, cfg, pairs, chain2_root)


def _vs_reference(ref, per, files, folds, groups, cfg, pairs, chain2_root=CHAIN2):
    """chain2r with a rule: each weight arm against the same arm of the reused chain2 call (SHIP_R3), paired across calls by
    image and partition (the partitions, bootstrap draws and pcombo are identical; the replay call checks the identity).
    Same statistics as the in-kernel contrasts (finish.recompute): delta, CI95 (paired 24 h-group bootstrap), positives/20."""
    rc = read(chain2_root / 'out' / ref / 'case/out/counts.json'); assert rc['files'] == files
    seeds = [str(s) for s in cfg['partition_seeds']]
    merged = {s: dict({a: per[s][a] for a, _ in pairs}, **{'ref_' + b: rc['per_partition_counts'][s][b] for _, b in pairs}) for s in seeds}
    con = [(f'{a}_minus_ref_{b}', a, 'ref_' + b) for a, b in pairs]
    cohorts = dict(all707=list(files), without_fold2=[f for f in files if folds[f] != 2], **{f'fold{k}': [f for f in files if folds[f] == k] for k in range(5)})
    out = {n: {} for n, _, _ in con}
    for c, fs in cohorts.items():
        x = recompute(merged, files, fs, groups, cfg, con)
        for n, a, b in con:
            v = x['contrasts'][n]; out[n][c] = dict(v, score=x['scores'][a], reference_score=x['scores'][b])
    return out


F0_LABEL = 'f0rep'                               # chainF: the replay call (decision 2, 2026-11-14)


# Ports added after F0 (decision A, 2026-11-14 21:4x): the only payload files a real call may have different from F0's. Each
# port's flag-absent path must equal F0's code (tools/flagless_vs_f0.py -> logs/flagless_vs_f0.json, synthetic + real images),
# and the sha256 of every differing file is recorded in the call's results.
PORTED_AFTER_F0 = {'combined (ST1, chain2s)': ['case/mt.py', 'case/worker.py', 'csv_worker.py', 'case/tests/test_combined.py', 'case/tests/test_combined_f.py'],
                   'tests added after F0 (main end to end with --no-w25 + --pcand-refit)': ['case/tests/test_main_flags_f.py'],
                   'test-unet fold5mean (orchestrator 22:3x)': ['csv_worker.py']}


def payload_vs_f0(label, root=None):
    """chainF: a real call's kernel payload must equal F0's file for file (sha256), except case/chain_inputs.json and the files of
    the ports listed in PORTED_AFTER_F0 (whose flag-absent equality with F0 is in logs/flagless_vs_f0.json for exactly these
    file versions); F0 must be validated. None for F0 itself."""
    root = root or H
    if label == F0_LABEL: return None
    f0 = root / 'state' / f'{F0_LABEL}_render.json'; mine = root / 'state' / f'{label}_render.json'
    if not f0.is_file() or not mine.is_file(): return dict(exact=False, reason='render record missing', f0_validated=False)
    a = read(mine)['payload_files']; b = read(f0)['payload_files']
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k != 'case/chain_inputs.json')
    res = root / 'results' / f'{F0_LABEL}.json'; ok0 = res.is_file() and read(res).get('validated') is True
    allowed = set().union(*PORTED_AFTER_F0.values()); ported_ok = set(differ) <= allowed
    fl = root / 'logs/flagless_vs_f0.json'; flx = read(fl) if fl.is_file() else {}
    flag_ok = bool(not differ or (flx.get('cv_equal') and flx.get('csv_equal') and all(flx.get('current_sha256', {}).get(k) == a.get(k) for k in differ)))
    return dict(exact=bool(ok0 and ported_ok and flag_ok), differing_files=differ, differing_sha256={k: a.get(k) for k in differ}, f0_sha256={k: b.get(k) for k in differ},
                ports=PORTED_AFTER_F0, only_ported_files=bool(ported_ok), flagless_equal_to_f0=flag_ok, flagless_log_date=flx.get('date'), f0_validated=bool(ok0), files=len(a))


CHAIN3 = H.parent / 'chain3'                     # chain3 (read only): fold5mean-U-Net CSVs of chain2 arms
FOLD5MEAN_FILES = {v: str(H.parent / f'unet_refit/out/fold5mean/{v}.npz') for v in ('full', 'local')}


def fold5mean_csv_refs(ref2, arms, ship_dir, chain3_root=CHAIN3):
    """chainF --test-unet: per arm, a validated chain3 CSV of the same chain2 call and arm, with the fold5mean U-Net maps only and
    the same ship selector -> {arm: dict(sha256, source)}; arms without one are absent (their CSV is then not replayed)."""
    out = {}
    for v in sorted((chain3_root / 'out').glob('*/csv_validation.json')):
        x = read(v)
        if x.get('chain2') != ref2 or x.get('arm') not in arms or x.get('validated') is not True or x['arm'] in out: continue
        if {k: list(w) for k, w in (x.get('unet') or {}).items()} != {k: [f] for k, f in FOLD5MEAN_FILES.items()}: continue
        if x.get('selector_sha256') != sha(Path(ship_dir) / f"{x['arm']}_selector.json"): continue
        out[x['arm']] = dict(sha256=sha(v.parent / 'submission.csv'), source=f'chain3 {x["label"]} (fold5mean U-Net test maps)')
    return out


def unet_swapped(pins):
    """chainF: True when a U-Net view of this call is not stored-equivalent (the weight arms differ from chain2's)."""
    return any(not u['stored_equivalent'] for u in (pins.get('unet') or {}).values())


def same_mf_reference(pins, chain2_root=CHAIN2):
    """chainF: a validated chain2 call with the same fold tokens (the same MF map sets), pulled locally, or None."""
    for p in sorted((chain2_root / 'results').glob('*.json')):
        try: x = read(p)
        except ValueError: continue
        if x.get('fold_sources') == pins['folds'] and x.get('validated') is True and (chain2_root / 'out' / x['label'] / 'case/out/counts.json').is_file(): return x['label']
    return None


PCAND_CONTRASTS = ('refit_minus_{v}_frozen', 'refit_minus_pcombo', 'frozen_minus_pcombo', 'frozen_minus_{v}_raw', 'refit_minus_{v}_raw')


def pcand_reference(ref2, arms, pcand_root=PCAND):
    """chainF: {arm: pcand run} for the arms covered by a validated pcand run on chain2 label ref2."""
    out = {}
    if not ref2: return out
    for st in sorted((pcand_root / 'stage').glob('*/pcand_inputs.json')):
        pin = read(st); run = st.parent.name
        if pin.get('label') != ref2: continue
        for a in arms:
            r = pcand_root / 'results' / f'{run}_{a}.json'
            if a in pin['arms'] and a not in out and r.is_file() and read(r).get('validated') is True and (pcand_root / 'out' / run / 'case/out/counts.json').is_file(): out[a] = run
    return out


def pcand_replay(co, csv, ref2, arms, pcand_root=PCAND, csv_compared=True):
    """chainF: the in-kernel refit of each arm against pcand's validated run on the same chain2 call (everything equal)."""
    refs = pcand_reference(ref2, arms, pcand_root)
    if not refs: return None
    mc = read(co / 'pcand_counts.json')['per_partition_counts']; mt = read(co / 'pcand_taus.json')['refit']; md = read(co / 'deploy_fit.json'); mr = read(co / 'pcand_refit.json')
    out = dict(reference_label=ref2, arms={})
    for a, run in refs.items():
        base = pcand_root / 'out' / run / 'case/out'
        rc = read(base / 'counts.json')['per_partition_counts']; rt = read(base / 'taus.json')['refit'][a]; rd = read(base / 'deploy_fit.json')[a]; rs = read(base / 'science_result.json')
        counts = sorted(mc) == sorted(rc) and all(mc[s][f'{a}_{x}'] == rc[s][f'{a}_{x}'] for s in rc for x in ('raw', 'frozen', 'refit')) and all(mc[s]['pcombo'] == rc[s]['pcombo'] for s in rc)
        names = [f'{a}_' + n.format(v=a) for n in PCAND_CONTRASTS]
        cohorts = sorted(set(mr['subsets']) & set(rs['subsets']))
        stats = bool(cohorts) and all(mr['subsets'][c]['contrasts'][n][f] == rs['subsets'][c]['contrasts'][n][f] for c in cohorts for n in names for f in STAT_FIELDS)
        scores = all(mr['subsets'][c]['scores'][f'{a}_refit'] == rs['subsets'][c]['scores'][f'{a}_refit'] for c in cohorts)
        rv = read(pcand_root / 'out' / run / 'csv_validation.json')['arms'].get(f'{a}_refit', {}).get('sha256')
        mine_csv = csv['arms'].get(f'{a}_refit', {}).get('sha256')
        csv_ok = (mine_csv == rv) if csv_compared else True        # chainF --test-unet: pcand's CSV used the package U-Net maps (not compared)
        out['arms'][a] = dict(run=run, exact=bool(counts and mt[a] == rt and md[a]['regressor_sha256'] == rd['regressor_sha256'] and stats and scores and csv_ok), csv_compared=csv_compared,
                              counts_equal=bool(counts), refit_taus_equal=bool(mt[a] == rt), regressor_sha256=md[a]['regressor_sha256'], reference_regressor_sha256=rd['regressor_sha256'],
                              contrast_statistics_equal=bool(stats), scores_equal=bool(scores), cohorts=cohorts, csv_sha256=mine_csv, reference_csv_sha256=rv,
                              refit_minus_frozen_all707=mr['subsets']['all707']['contrasts'][f'{a}_refit_minus_{a}_frozen']['delta'],
                              reference_refit_minus_frozen_all707=rs['subsets']['all707']['contrasts'][f'{a}_refit_minus_{a}_frozen']['delta'])
    out['exact'] = all(x['exact'] for x in out['arms'].values())
    return out


def pcand_summary(co, arms):
    """chainF: the in-kernel refit report (pcand_refit.json) in the README's form, plus its own integrity flags."""
    if (co / 'pcand_refit_failure.json').is_file(): return dict(ok=False, failure=read(co / 'pcand_refit_failure.json'))
    if not (co / 'pcand_refit.json').is_file(): return dict(ok=False, failure='pcand_refit.json missing')
    r = read(co / 'pcand_refit.json'); d = read(co / 'deploy_fit.json')
    ok = bool(r['status'] == 'complete' and r['duplicate_zero'] and r['tau_path_equal'] and sorted(r['arms_requested']) == sorted(arms)
              and all(d[a]['refit_pickle_identical'] and d[a]['deterministic'] for a in arms))
    return dict(ok=ok, report=r['report'], elapsed_seconds=r['elapsed_seconds'], deploy={a: d[a]['regressor_sha256'] for a in arms})


def count_table(per, files, folds, arms):
    """Partition-mean counts (summed over images and annotation sets) per arm: iou_sum, tp, fp, fn, split, merge, pq;
    cohorts all707 and without fold 2. Counts rows: [iou_sum, tp, fp, fn, split (GT rows overlapping > 1 prediction), merge
    (predictions overlapping > 1 GT)]."""
    seeds = sorted(per); out = {}
    for a in arms:
        out[a] = {}
        for c, fs in (('all707', files), ('without_fold2', [f for f in files if folds[f] != 2])):
            ix = [files.index(f) for f in fs]
            m = np.mean([np.asarray(per[s][a], float)[ix].sum(0) for s in seeds], axis=0)
            out[a][c] = dict(iou_sum=float(m[0]), tp=float(m[1]), fp=float(m[2]), fn=float(m[3]), split=float(m[4]), merge=float(m[5]), pq=pq(m))
    return out


def contrast_table(S, name):
    x = S['all707']['contrasts'][name]
    return dict(delta_combined707=x['delta'], CI95=x['paired_group_CI95'], positive_of20=x['positive_of20'], mean_paired_partition_delta=x['mean_paired_partition_delta'],
                fold_means={f'fold{k}': S[f'fold{k}']['contrasts'][name]['mean_paired_partition_delta'] for k in range(5)},
                combined425=S['combined425']['contrasts'][name]['delta'], combined282=S['combined282']['contrasts'][name]['delta'])


def csv_checks(label, base):
    """Copy the pulled CSVs to out/<label>/submission_<arm>.csv and re-validate them here with the package's validator."""
    for m in [m for m in sys.modules if m == 'filseg' or m.startswith('filseg.')]: del sys.modules[m]   # case/frozen/eval/filseg shadows the package
    sys.path.insert(0, str(PKG_SRC))
    from filseg.reproduction.pipeline import test_keys
    from filseg.reproduction.validate import validate_submission, compare_submissions
    kc = read(base / 'out/csv/csv_check.json'); keys = test_keys(); out = dict(kernel_status=kc['status'], arms={})
    pc = base / 'out/csv/pcombo/submission.csv'
    expected = ['pcombo'] + M.NEW_VARIANTS + [a + '_refit' for a in REFIT_ARMS]     # chainF: the refit CSVs
    for arm in expected:
        src = base / f'out/csv/{arm}/submission.csv'
        if not src.is_file(): out['arms'][arm] = dict(present=False); continue
        dst = H / 'out' / label / f'submission_{arm}.csv'; shutil.copyfile(src, dst)
        v = validate_submission(dst, keys); kernel = kc['pcombo'] if arm == 'pcombo' else kc['arms'].get(arm, {})
        row = dict(present=True, file=str(dst), sha256=sha(dst), kernel_sha256=kernel.get('csv_sha256'), equals_kernel=sha(dst) == kernel.get('csv_sha256'),
                   rows=v['rows'], images=180, files_with_rows=v['files_with_rows'], overlapping_pixel_images=len(v['overlapping_pixel_images']),
                   ids_unique=v['ids_unique'], decoded_rows=v['decoded_rows'], valid=bool(v['passed'] and v['expected_files'] == 180 and not v['overlapping_pixel_images'] and v['ids_unique']))
        if arm != 'pcombo':
            d = compare_submissions(dst, pc, keys); row.update(weight=kernel.get('weight'), cutoff=kernel.get('cutoff'), selector_sha256=kernel.get('selector_sha256'), kept=kernel.get('kept'),
                                                               candidates=kernel.get('candidates'), difference_vs_pcombo=d)
        else:
            row.update(expected_sha256=PCOMBO_CSV_SHA, byte_identical_to_shipped=sha(dst) == PCOMBO_CSV_SHA, deployment_refit_equals_pinned=kc['pcombo']['deployment_refit_equals_pinned'])
        out['arms'][arm] = row
    out['all_valid'] = all(a.get('valid') for a in out['arms'].values()) and len(out['arms']) == len(expected)
    (H / 'out' / label / 'csv_validation.json').write_text(json.dumps(out, indent=1))
    return out


def finish(label):
    import ledger
    base = H / 'out' / label; co = base / 'case/out'
    pins = read(H / 'stage' / label / 'chain_inputs.json'); M.configure(pins['weights'], pins.get('levels'), pins.get('combined'), require_base=not pins.get('no_w25'))
    REFIT_ARMS[:] = list((pins.get('pcand_refit') or {}).get('arms', []))
    r = read(co / 'science_result.json'); c = read(co / 'counts.json'); cfg = read(CASE / 'protocol.json')
    folds = read(CASE / 'frozen/evidence/fold_table.json'); files = sorted(folds); groups = temporal_groups(files, 24)
    assert c['files'] == files and c['arms'] == M.ARMS and r['weights'] == M.WEIGHTS and r['fold_sources'] == pins['folds']
    per = c['per_partition_counts']; seeds = [str(s) for s in cfg['partition_seeds']]
    query = [f for f in files if folds[f] in (0, 1, 2)]; cohorts = R.cohorts(files, folds, query)
    worst = 0.; positive_match = True
    for name, fs in cohorts.items():
        mine = recompute(per, files, fs, groups, cfg, M.CONTRASTS); rep = r['subsets'][name]
        for a in M.ARMS: worst = max(worst, abs(mine['scores'][a] - rep['scores'][a]))
        for cname, _, _ in M.CONTRASTS:
            x, y = mine['contrasts'][cname], rep['contrasts'][cname]
            err = max(abs(x['delta'] - y['delta']), abs(x['mean_paired_partition_delta'] - y['mean_paired_partition_delta']),
                      *[float(np.abs(np.asarray(x[ci]) - y[ci]).max()) for ci in R.CI_LEVELS])
            worst = max(worst, float(err)); positive_match &= x['positive_of20'] == y['positive_of20']
    kj = read(co / 'keeps.json'); sel = read(co / 'selectors.json')['models']; fits = read(co / 'fits.json')
    replay = pcombo_replay(r, per, kj, sel, fits, files, seeds)
    ship = read(co / 'ship/ship_fit.json')['arms']
    csv = csv_checks(label, base)
    swapped = unet_swapped(pins)                        # chainF: no replay against an unswapped reference
    ref = None if swapped or pins.get('no_w25') or pins.get('test_unet') else find_reference(pins)
    refrep = reference_replay(label, ref, r, per, kj, sel, csv, seeds, files) if ref else None
    rule = pins.get('rule'); ship_rule = rule is None or rule['text'] == SHIP_TEXT; ref2 = pins.get('reuses_chain2_stage')
    cmb = M.COMBINED['arm'] if M.COMBINED is not None else None                                                           # chain2s
    plain = [a for a in M.NEW_VARIANTS if not M.multi_level(a) and a != cmb]; multi = [a for a in M.NEW_VARIANTS if M.multi_level(a)]   # chain2m
    crefs = fold5mean_csv_refs(ref2, plain, co / 'ship') if pins.get('test_unet') and ref2 else None    # chainF --test-unet
    rep2 = chain2_replay(label, ref2, r, per, kj, sel, csv, seeds, files, plain, csv_refs=crefs) if ref2 and ship_rule and plain and not swapped else None
    pcr = pcand_summary(co, REFIT_ARMS) if REFIT_ARMS else None
    pf0 = payload_vs_f0(label) if rep2 is None and refrep is None else None    # chainF: a call with its own chain2 / CH1 replay (blocking) does not lean on F0
    pcrep = pcand_replay(co, csv, ref2, REFIT_ARMS, csv_compared=not pins.get('test_unet')) if REFIT_ARMS and pcr['ok'] and not swapped else None
    mfref = same_mf_reference(pins) if swapped else None
    vs_mf = vs_reference(mfref, per, files, folds, groups, cfg, [a for a in M.NEW_VARIANTS if a in read(CHAIN2 / 'results' / f'{mfref}.json')['weight_arms']]) if mfref else None
    vs_arms = [(a, M.BASE_OF[a]) for a in multi] if ship_rule else [(a, a) for a in M.NEW_VARIANTS]
    if cmb is not None and not swapped: vs_arms = vs_arms + [(cmb, M.COMBINED_REF)]           # chain2s (vs the reused call's w55)
    vs2 = vs_reference(ref2, per, files, folds, groups, cfg, vs_arms) if ref2 and vs_arms else None
    counts_tab = None
    if ref2:
        rc2 = read(CHAIN2 / 'out' / ref2 / 'case/out/counts.json'); refs = sorted({b for _, b in vs_arms} | set(plain))
        merged = {s_: dict({a: per[s_][a] for a in M.NEW_VARIANTS}, **{'ref_' + b: rc2['per_partition_counts'][s_][b] for b in refs}) for s_ in per}
        counts_tab = count_table(merged, files, folds, M.NEW_VARIANTS + ['ref_' + b for b in refs])
    kc = read(base / 'out/csv/csv_check.json'); cv_c = r['descriptive']['by_cohort']['all707']['candidates']
    tvc = {a: dict(cv_candidates_per_image=cv_c[a] / 707, test_candidates_per_image=kc['arms'][a]['candidates'] / 180 if kc['arms'].get(a, {}).get('candidates') is not None else None)
           for a in M.NEW_VARIANTS}
    S = r['subsets']; shown = ['pcombo'] + M.NEW_VARIANTS
    res = dict(label=label, created_at=now(), fold_sources=pins['folds'], map_sets=pins['sets'], weights=M.WEIGHTS, weight_arms=M.WEIGHT, exports=pins.get('exports'),
               scores_combined707={a: S['all707']['scores'][a] for a in shown},
               partition_pq_combined707={a: S['all707']['partition_pq'][a] for a in shown},
               contrasts={n: contrast_table(S, n) for n in M.ARM_CONTRASTS},
               gate_rule_descriptive={n: M.gate707(r, n) for n in M.ARM_CONTRASTS},
               w25_reference_replay=refrep,
               levels=M.LEVELS, levelsets=pins.get('levels'), level_descriptives=r['descriptive'].get('levels'), counts_partition_mean=counts_tab,
               csv_level_counts={a: kc['arms'].get(a, {}).get('level_counts') for a in M.NEW_VARIANTS},
               unet={v: dict(sets=u['sets'], stored_equivalent=u['stored_equivalent']) for v, u in (pins.get('unet') or {}).items()} or None,
               vs_chain2_same_mf=dict(reference=mfref, contrasts=vs_mf) if mfref else None,
               pcand_refit=pcr, pcand_reference_replay=pcrep, payload_vs_f0=pf0,
               test_unet=dict(pins=pins.get('test_unet'), kernel=kc.get('test_unet')) if pins.get('test_unet') else None,
               instancing_rule=rule, reuses_chain2_stage=ref2, chain2_reference_replay=rep2, vs_chain2_reference=vs2, candidates_test_vs_cv=tvc,
               candidates={k: v for k, v in r['descriptive']['by_cohort']['all707']['candidates'].items()},
               kept_per_partition_mean=r['descriptive']['by_cohort']['all707']['kept_per_partition_mean'],
               ship={a: {k: v for k, v in x.items() if k != 'feature_distribution'} for a, x in ship.items()},
               csv=csv, pcombo_replay_exact=replay['exact'], pcombo_replay=replay, recomputation_max_abs_difference=worst, positive_counts_match=bool(positive_match),
               kernel_checks=dict(replay_passed=r['replay_passed'], integrity_passed=r['integrity_passed'], inputs_passed=r['inputs_passed'], inputs=r['inputs']),
               kernel_elapsed_seconds=read(base / 'out/artifact_manifest.json')['elapsed_seconds'] if (base / 'out/artifact_manifest.json').is_file() else None,
               worker_elapsed_seconds=r['elapsed_seconds'], csv_elapsed_seconds=read(base / 'out/csv/csv_check.json').get('elapsed_seconds'))
    ok = bool(replay['exact'] and worst < 1e-12 and positive_match and r['replay_passed'] and r['integrity_passed'] and r['inputs_passed'] and csv['all_valid']
              and csv['arms']['pcombo'].get('byte_identical_to_shipped') and csv['arms']['pcombo'].get('deployment_refit_equals_pinned')
              and all(a.get('equals_kernel') for a in csv['arms'].values()) and (refrep is None or refrep['exact']) and (rep2 is None or rep2['exact'])
              and (pcr is None or pcr['ok']) and (pcrep is None or pcrep['exact']) and (pf0 is None or pf0['exact']))
    if pins.get('test_unet'):                              # chainF: the CSVs used exactly the fold5mean files (kernel record = pins = constants)
        t = kc.get('test_unet') or {}; ok = bool(ok and t.get('sha256') == {v: pins['test_unet'][v]['sha256'] for v in ('full', 'local')} == FOLD5MEAN_SHA)
    if cmb is not None:                                    # chain2s: the arm is the one configured (its replay arms are checked above when --from)
        c = pins['combined']; ok = bool(ok and r.get('combined') == M.COMBINED and r['inputs'].get('combined', {}).get('thr', c['thr']) == c['thr'])
        res['combined'] = dict(arm=cmb, name=c.get('name'), thr=c['thr'], dataset=c.get('dataset'), source=c.get('source'), reference_arm=M.COMBINED_REF,
                               kernel_record=r.get('combined'))
    res['validated'] = ok
    (H / 'results').mkdir(exist_ok=True)
    (H / 'results' / f'{label}.json').write_text(json.dumps(res, indent=1))
    ledger.record(label, 'VALIDATED_COMPLETE' if ok else 'VALIDATION_FAILED', results=str(H / 'results' / f'{label}.json'))
    print(json.dumps(dict(label=label, validated=ok, pcombo_replay_exact=replay['exact'], csv_valid=csv['all_valid'],
                          w25_reference=None if refrep is None else dict(ref=refrep['reference'], exact=refrep['exact']),
                          chain2_replay=None if rep2 is None else rep2['exact'], pcand_refit_ok=None if pcr is None else pcr['ok'],
                          pcand_replay=None if pcrep is None else pcrep['exact'],
                          scores=res['scores_combined707'], **{n: dict(delta=v['delta_combined707'], CI95=v['CI95'], pos=v['positive_of20']) for n, v in res['contrasts'].items()})))
    return res


if __name__ == '__main__': finish(sys.argv[1])
