"""Kaggle-CPU-only worker of PC1: refit the P-cand v1 repair per MF-weight arm of one finished chain2 call.

Reuses chain2's case tree unchanged (worker.py is chain2's file byte for byte: inventory, the new-map inputs, the per-image
maps / candidates / 78 features, decode, IoU, pool_map); pt.py holds the pure parts.

Inputs: the stored + new map datasets of the chain2 call (chain_inputs.json, as chain2), and chain2's own kernel output
(kernel source, SHA-256 pinned in pcand_inputs.json): keeps.json (the arm's pass-2 keeps and its frozen v1 taus),
selectors.json (the arm's 100 nested 68-feature selectors), counts.json (per-image counts of the arm and of pcombo),
science_result.json (scores).

Per arm v (pcand_inputs.json 'arms'):
  phase A  per image on arm v's maps: chain2's candidates and 78 features (worker.candidate_rows(f, v)); the keep pattern of
           each saved C5 LR39 selector (s, k) and of the deployed full-707 LR39 ('deploy'); per distinct pattern the 9-tau
           table (uniform whole-kept-set repair at each grid tau: per-candidate v1 scores and six PQ counts); raw IoU.
  replay   (blocking, before any fit) candidate count per image = length of chain2's frozen tau list; chain2's 100 arm-v
           selectors applied to these features give chain2's keeps; pass 2 with chain2's taus on chain2's keeps gives chain2's
           per-image per-partition counts of arm v and its all707 score and partition PQs exactly. Any difference: stop.
  phase B  100 outer v1 regressors refit on arm v's bank (rows = C5 (s, k) keeps of the outer-training images, the v1 rule;
           HistGradientBoostingRegressor(random_state=42)); one global tau per (s, k) from the pooled PQ of the training images'
           C5 (s, k) uniform repairs. Deployment (all 707 images, keep = deployed LR39, pcand_ship's rule): one regressor
           (fitted twice, b4dcd7d1's pickle layout) and one global tau.
  pass 2   chain2's keeps repaired with the refit taus and with the global tau.
Arms and contrasts: pt.py. No test image is read here; the test CSVs are csv_pcand.py.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import json, sys, time, hashlib, pickle, functools, multiprocessing as mp
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import numpy as np
import worker as W
import pt as T
import pcand as C
import presel as P
import mt as M
from rules import assert_split, pq
from make_time_folds import temporal_groups
from context_features import FEATURES as CTX_FEATURES
from pcand import GRID, snap, prepare as prepare_repair, repair_taus

G = W.G
VS = []
REF_FILES = dict(keeps='case/out/keeps.json', counts='case/out/counts.json', selectors='case/out/selectors.json', science_result='case/out/science_result.json')
DEPLOY_SCOPE = 'filament-repro-weights-1111'
RECIPE = 'HistGradientBoostingRegressor(random_state=42), default parameters (pcand_tau v1)'


def remote(): return Path('/kaggle/input').is_dir()
def read(p): return json.loads(Path(p).read_text())


def configure_arms(vs):
    global VS
    VS = list(vs)


# ---------------------------------------------------------------- inputs
def find_output(slug, rel):
    """A file of a kernel source's output (mounted read only under /kaggle/input), by its relative path."""
    name = slug.split('/')[-1]
    ps = [p for p in Path('/kaggle/input').rglob(Path(rel).name) if str(p).endswith('/' + rel) and name in p.parts]
    assert len(ps) == 1, ('CHAIN2_OUTPUT', rel, len(ps)); return ps[0]


def chain2_reference(pins):
    out = {}
    for key, rel in REF_FILES.items():
        p = find_output(pins['chain2_kernel'], rel); assert W.sha(p) == pins['chain2_reference_sha256'][rel], ('CHAIN2_REFERENCE_IDENTITY', rel); out[key] = read(p)
    return out


def deployed_selector(pins):
    """The package's full-707 LR39 (role deployment_full707_original_only, 0.35): the keep rule of b4dcd7d1's training rows."""
    d = pins['deployed_selector']; path = W.unique('quality_model.json', DEPLOY_SCOPE); assert W.sha(path) == d['quality_model_sha256'], 'DEPLOYED_SELECTOR_IDENTITY'
    obj = read(path); sel = [m for m in obj.get('models', [obj]) if m.get('role') == d['role']]; assert len(sel) == 1
    m = sel[0]; assert m['classes'] == [0, 1] and len(m['feature_order']) == 39 and m['threshold'] == d['threshold']
    return m


# ---------------------------------------------------------------- phase A on an arm's bank
def keep_patterns(rows):
    """GT-free: the keep pattern of each saved C5 selector (s, k) (chain2 worker.prediction_side's arithmetic) and of the deployed selector."""
    x = np.asarray([[r[n] for n in G['order39']] for r in rows], float).reshape(len(rows), 39)
    keeps = {f'{s}:{k}': tuple(np.flatnonzero(W.project(m, x) >= m['threshold']).tolist()) for (s, k), m in G['models'].items()}
    dep = G['deployed']; xd = np.asarray([[r[n] for n in dep['feature_order']] for r in rows], float).reshape(len(rows), 39)
    keeps['deploy'] = tuple(np.flatnonzero(T.keep_of(dep, xd)[0]).tolist()) if len(rows) else ()
    return keeps


def image_pass_arm(v, f):
    """Candidates and features first (no annotation), then the 9-tau table of every distinct keep pattern scored against GT."""
    masks, p, rows = W.candidate_rows(f, v)
    keeps = keep_patterns(rows)
    gts = W.gt_sets(f); patterns = sorted(set(keeps.values())); pid = {key: i for i, key in enumerate(patterns)}
    tabs = [T.tau_table([masks[j] for j in key], p, gts) for key in patterns]
    rle = W.encode(masks); qraw = W.ious(gts, rle)
    return f, dict(n=len(masks), rle=rle, hashes=[W.packed(m) for m in masks], features=rows, patterns=[list(k) for k in patterns],
                   pattern_of={key: pid[val] for key, val in keeps.items()}, scores=[t['scores'] for t in tabs], counts=[t['counts'] for t in tabs],
                   tau_path_equal=bool(all(t['tau_path_equal'] for t in tabs)), n_gt_sets=len(gts), raw_iou=[q.tolist() for q in qraw])


def selector_keep_replay(v, AV, sels, keeps, outer, files, names):
    """chain2's 100 nested selectors of arm v, applied to these 68 features of each query image, must give chain2's keeps."""
    compared = 0; diff = 0
    for x in sels:
        if x['arm'] != v: continue
        s, k = x['partition_seed'], x['outer']; compared += 1
        for f in files:
            if outer[s][f] != k: continue
            mine = np.flatnonzero(P.project(x, T.all_rows(AV, f, names)) >= x['threshold']).tolist() if AV[f]['n'] else []
            diff += mine != list(keeps[f][str(s)][v])
    return dict(compared=compared, differing_image_partitions=int(diff), exact=bool(compared == 100 and diff == 0))


# ---------------------------------------------------------------- pass 2 (chain2's pass2_image arithmetic, arms frozen / refit / gtau)
def pass2_arm(f):
    """For each arm v in G['p2_arms'] (default VS) and each variant in G['p2']: repair chain2's keeps of v on v's mixture with the
    variant's taus (frozen: chain2's; refit: the refit regressors'; gtau: the fit's global tau). raw = the keeps unrepaired."""
    gts = W.gt_sets(f); seeds = G['partition_seeds']; outer = G['outer']; rows = {}; info = {s: dict(dup={}, stats={}, raw={}) for s in seeds}
    for v in G.get('p2_arms', VS):
        AA = G['AV'][v]; p = M.mixture(W.branch_maps(f, v), v); masks = W.decode(f, v); prep = {}; cache = {}
        qraw = [P._mat(q, AA[f]['n']) for q in AA[f]['raw_iou']]
        for s in seeds:
            key = tuple(G['keeps_arm'][v][f][s]); ki = np.asarray(key, dtype=int)
            info[s]['raw'][v] = W.total_counts([q[:, ki] for q in qraw]).tolist()
            for x in G['p2']:
                taus = tuple(G['taus'][v][x][f][s][j] for j in key) if x in ('frozen', 'refit') else (G['tglob'][v][s, outer[s][f]],) * len(key)
                ck = (key, taus)
                if ck not in cache:
                    sd = [masks[j] for j in key]
                    if key and key not in prep: prep[key] = prepare_repair(sd, p)
                    out, st = repair_taus(sd, p, list(taus), prep.get(key))
                    rl = W.encode(out); qs = W.ious(gts, rl)
                    cache[ck] = dict(counts=W.total_counts(qs).tolist(), stats=st, dup=P.duplicate_counts(rl, qs))
                a = T.arm(v, x); rows.setdefault(a, {})[s] = cache[ck]['counts']; info[s]['dup'][a] = cache[ck]['dup']; info[s]['stats'][a] = cache[ck]['stats']
    return f, rows, info


# ---------------------------------------------------------------- phase B: refits, global taus, deployment
def make_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=42)


def fit_task_arm(vsk):
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


def deployment_fit(v, AV, files, names):
    """pcand_ship's deployment recipe on arm v's bank: rows = candidates kept by the deployed LR39 (in-sample) on all images,
    targets as pcand_tau; one regressor fitted twice; payload = pickle of {model, features, recipe, training} (b4dcd7d1's
    layout, read by the package's pcand.load_regressor). Also the deployment global tau (pooled PQ of the same keeps)."""
    X, y, excluded = T.training_rows(AV, files, 'deploy', names); assert len(X) > 0 and np.isfinite(X).all()
    m1 = make_regressor().fit(X, y); b1 = pickle.dumps(m1, protocol=5); m2 = make_regressor().fit(X, y); b2 = pickle.dumps(m2, protocol=5)
    deterministic = bool(np.array_equal(m1.predict(X), m2.predict(X))); assert deterministic, 'REGRESSOR_NOT_DETERMINISTIC'
    payload = pickle.dumps(dict(model=m1, features=list(names), recipe=RECIPE,
                                training=f'all {len(files)} training images, chain2 arm {v} bank; keep = deployed full707 LR39 (in-sample); targets as pcand_tau'), protocol=5)
    path = Path(f'out/deploy/{v}_regressor.pkl'); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    pooled = T.pool_counts({f: AV[f]['counts'] for f in files}, {f: AV[f]['pattern_of'] for f in files}, 'deploy', files)
    kept = int(sum(len(AV[f]['patterns'][AV[f]['pattern_of']['deploy']]) for f in files))
    return dict(arm=v, training_images=len(files), candidates=int(sum(AV[f]['n'] for f in files)), kept_candidates=kept, rows=int(len(y)), excluded_unmatched=excluded,
                target_histogram={str(t): int((y == t).sum()) for t in GRID}, target_mean=float(y.mean()), n_iter=int(m1.n_iter_),
                regressor_sha256=hashlib.sha256(payload).hexdigest(), model_pickle_sha256=hashlib.sha256(b1).hexdigest(), refit_pickle_identical=bool(b1 == b2),
                deterministic=deterministic, global_tau=T.choose_global(pooled), pooled_pq=[pq(c) for c in pooled], features=list(names), file=str(path))


# ---------------------------------------------------------------- main
def main():
    global START
    START = time.monotonic(); W.START = START; assert remote(), 'REMOTE_CPU_ONLY'
    cfg = read(H / 'protocol.json'); pins = read(H / 'pcand_inputs.json'); vs = list(pins['arms']); assert vs and len(set(vs)) == len(vs)
    assert tuple(cfg['tau_grid_v1']) == GRID
    spec, annotations, models, outer, folds, images, query = W.inventory(); files = sorted(folds); groups = temporal_groups(files, 24); seeds = cfg['partition_seeds']
    n_exp = cfg['expected_images']; assert len(files) == n_exp['all'] and len(query) == n_exp['query425']
    cohorts = T.cohorts(files, folds, query)
    newspec, input_rows = W.inputs_inventory(folds); W.configure(input_rows['weights']); assert input_rows['label'] == pins['label'] and input_rows['format_checked']
    assert all(v in M.NEW_VARIANTS for v in vs), ('ARMS', vs, M.NEW_VARIANTS)
    configure_arms(vs)
    ref = chain2_reference(pins); assert ref['counts']['files'] == files and ref['keeps']['files'] == files
    deployed = deployed_selector(pins)
    orders = {tuple(m['feature_order']) for m in models.values()}; assert len(orders) == 1; order39 = list(next(iter(orders)))
    assert sorted(deployed['feature_order']) == sorted(order39)
    names_v1 = order39 + CTX_FEATURES + C.WINDOW_FEATURES; assert names_v1 == cfg['features_v1'] and len(names_v1) == 64
    names = M.feature_names(order39, CTX_FEATURES); assert names == cfg['features'] and len(names) == 68
    state = dict(spec=spec, newspec=newspec, annotations=annotations, folds=folds, images=images, models=models, outer=outer, order39=order39, deployed=deployed)
    AV = {}; keeps_arm = {}; replay = {}; P1 = {}
    G.update(files=files, groups=groups, outer=outer, models=models, names_v1=names_v1, partition_seeds=seeds, AV=AV, rles_v={}, hashes_v={},
             keeps_arm=keeps_arm, taus={}, tglob={})
    W.progress(stage='provenance_verified_before_fit', arms=vs, label=pins['label'])
    # ---- per arm: phase A, then the blocking replay of chain2's arm (no fit before it passes)
    for v in vs:
        A = AV[v] = W.pool_map(functools.partial(image_pass_arm, v), files, f'image_pass_{v}', state)
        fz = {f: {s: ref['keeps']['v1_outer_taus_variants'][v][f][str(s)] for s in seeds} for f in files}
        keeps_arm[v] = {f: {s: tuple(ref['keeps']['keeps'][f][str(s)][v]) for s in seeds} for f in files}
        cand = all(len(fz[f][s]) == A[f]['n'] for f in files for s in seeds)
        sk = selector_keep_replay(v, A, ref['selectors']['models'], ref['keeps']['keeps'], outer, files, names)
        G['rles_v'][v] = {f: A[f]['rle'] for f in files}; G['hashes_v'][v] = {f: A[f]['hashes'] for f in files}; G['taus'][v] = dict(frozen=fz)
        G.update(p2=['frozen'], p2_arms=[v])
        E = P1[v] = W.pool_map(pass2_arm, files, f'pass2_frozen_{v}', state)
        per_v = {s: [E[f][0][T.arm(v, 'frozen')][s] for f in files] for s in seeds}
        counts_exact = bool(all(np.array_equal(np.asarray(per_v[s], float), np.asarray(ref['counts']['per_partition_counts'][str(s)][v], float)) for s in seeds))
        sm = T.summarize({s: {v: per_v[s]} for s in seeds}, files, files, groups, cfg, [v], [])
        rs = ref['science_result']['subsets']['all707']
        score_exact = sm['scores'][v] == rs['scores'][v]; part_exact = sm['partition_pq'][v] == rs['partition_pq'][v]
        tau_ok = all(A[f]['tau_path_equal'] for f in files)
        replay[v] = dict(exact=bool(cand and sk['exact'] and counts_exact and score_exact and part_exact and tau_ok), candidate_count_match=bool(cand), selector_keeps=sk,
                         counts_exact=counts_exact, all707_score=sm['scores'][v], chain2_all707_score=rs['scores'][v], score_exact=bool(score_exact),
                         partition_pq_exact=bool(part_exact), uniform_tau_path_equals_pnat=bool(tau_ok), candidates=int(sum(A[f]['n'] for f in files)),
                         elapsed_seconds=time.monotonic() - START, date=W.now())
        W.save('out/replay.json', replay); W.progress(stage=f'replay_{v}', exact=replay[v]['exact'], score=sm['scores'][v])
        if not replay[v]['exact']: raise RuntimeError(f'REPLAY_FAILED {v}: ' + json.dumps({k: x for k, x in replay[v].items() if k != 'selector_keeps'}))
    # ---- phase B: refits and global taus (outer-training images only), deployment fits
    fitres = {}; jobs = [(v, s, k) for v in vs for s in seeds for k in range(5)]
    with mp.get_context('fork').Pool(4) as pool:
        for i, (key, r) in enumerate(pool.imap(fit_task_arm, jobs)):
            fitres[key] = r
            if (i + 1) % 20 == 0: W.progress(stage='v1_refits', done=i + 1, total=len(jobs))
    refit = {v: {f: {s: fitres[v, s, outer[s][f]]['pred'][f] for s in seeds} for f in files} for v in vs}
    tglob = {v: {(s, k): fitres[v, s, k]['stats']['global_tau'] for s in seeds for k in range(5)} for v in vs}
    deploy = {}
    for v in vs: deploy[v] = deployment_fit(v, AV[v], files, names_v1); W.progress(stage=f'deploy_fit_{v}', sha=deploy[v]['regressor_sha256'][:12], global_tau=deploy[v]['global_tau'])
    W.save('out/fits.json', {v: dict(outer=[fitres[v, s, k]['stats'] for s in seeds for k in range(5)], recipe=RECIPE + f'; refit on chain2 arm {v} bank, rows = C5 (s, k) keeps', features=names_v1) for v in vs})
    W.save('out/deploy_fit.json', deploy)
    for v in vs: G['taus'][v]['refit'] = refit[v]
    G.update(tglob=tglob, p2=['refit', 'gtau'], p2_arms=vs)
    E2 = W.pool_map(pass2_arm, files, 'pass2_refit_gtau', state)
    # ---- summaries
    arms = T.arms_of(vs); contrasts = T.contrasts_of(vs)
    per = {s: dict(pcombo=ref['counts']['per_partition_counts'][str(s)]['pcombo']) for s in seeds}
    for s in seeds:
        for v in vs:
            per[s][T.arm(v, 'raw')] = [P1[v][f][1][s]['raw'][v] for f in files]
            per[s][T.arm(v, 'frozen')] = [P1[v][f][0][T.arm(v, 'frozen')][s] for f in files]
            for x in ('refit', 'gtau'): per[s][T.arm(v, x)] = [E2[f][0][T.arm(v, x)][s] for f in files]
    W.save('out/counts.json', dict(files=files, folds={f: folds[f] for f in files}, arms=arms, per_partition_counts=per,
                                   note='pcombo counts are read from chain2 counts.json; v_frozen equals chain2 arm v (replay)'))
    W.save('out/taus.json', dict(refit={v: {f: {str(s): refit[v][f][s] for s in seeds} for f in files} for v in vs},
                                 global_tau={v: {f'{s}:{k}': tglob[v][s, k] for s in seeds for k in range(5)} for v in vs},
                                 deploy_global_tau={v: deploy[v]['global_tau'] for v in vs}))
    reports = {name: T.summarize(per, fs, files, groups, cfg, arms, contrasts) for name, fs in cohorts.items()}
    info = lambda v, x, s, f: (P1[v][f][1] if x == 'frozen' else E2[f][1])[s]
    dup_zero = all(info(v, x, s, f)['dup'][T.arm(v, x)][k] == 0 for v in vs for x in ('frozen', 'refit', 'gtau') for s in seeds for f in files
                   for k in ('overlapping_prediction_pairs', 'gt_matched_by_more_than_one_prediction'))
    finite = bool(all(np.isfinite(T.all_rows(AV[v], f, names_v1)).all() and np.isfinite(T.all_rows(AV[v], f, names)).all() for v in vs for f in files))
    def descr(v, fs):
        kept = lambda f, s: keeps_arm[v][f][s]
        mean_tau = {x: float(np.mean([t for f in fs for s in seeds for t in (
            [G['taus'][v][x][f][s][j] for j in kept(f, s)] if x != 'gtau' else [tglob[v][s, outer[s][f]]] * len(kept(f, s)))] or [0.])) for x in ('frozen', 'refit', 'gtau')}
        prof = {x: {k: float(np.mean([sum(info(v, x, s, f)['stats'][T.arm(v, x)][k] for f in fs) for s in seeds])) for k in ('kept', 'changed', 'added', 'removed', 'fallback')}
                for x in ('frozen', 'refit', 'gtau')}
        return dict(images=len(fs), kept_per_partition_mean=float(np.mean([sum(len(kept(f, s)) for f in fs) for s in seeds])), tau_mean_on_kept=mean_tau, repair_profile_per_partition=prof)
    gh = {v: {str(t): sum(x == t for x in tglob[v].values()) for t in GRID} for v in vs}
    result = dict(status='complete', label=pins['label'], arms_requested=vs, arms=arms, contrasts=[n for n, _, _ in contrasts], weights=M.WEIGHT,
                  replay_passed=bool(all(r['exact'] for r in replay.values())), replay=replay,
                  integrity_passed=bool(dup_zero and finite), duplicate_zero=bool(dup_zero), features_finite=finite,
                  subsets=reports, report={v: {n: T.report_row(reports, n) for n, a, _ in contrasts if a.startswith(v + '_')} for v in vs},
                  descriptive=dict(by_arm={v: {c: descr(v, cohorts[c]) for c in ('all707', 'no_fold2')} for v in vs}, global_tau_histogram=gh,
                                   deploy={v: {k: x for k, x in deploy[v].items() if k != 'features'} for v in vs}),
                  v1_refits=len(fitres), deployment_fits=2 * len(vs), selector_fits=0, segmentation_fits=0, submission_calls=0, test_images_used=0,
                  partition_seeds=seeds, images=len(files), features_v1=names_v1, elapsed_seconds=time.monotonic() - START, created_at=W.now(),
                  limitations=['Development data reused: the 707 images chose the arm weights and every earlier recipe',
                               'Refit rows and global taus use the saved C5 LR39 keeps on the arm candidates (the frozen v1 row rule); pass 2 repairs the arm selector keeps',
                               'pcombo counts are chain2\'s (read); arm v frozen is replayed exactly'])
    W.save('out/science_result.json', result)
    W.progress(stage='complete', **{f'{v}_{x}': result['subsets']['all707']['scores'][T.arm(v, x)] for v in vs for x in ('frozen', 'refit', 'gtau')})


if __name__ == '__main__': main()
