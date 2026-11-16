"""Synthetic tests of worker.py (chain2 / order CH2; written before worker.py was changed; adapted from chain CH1's). Per-image
functions on 2048-pixel synthetic maps: a weight arm changes only the MF branch (the multi-set average) and the blend weight;
w25 with one set equal to the stored map reproduces the stored candidates exactly; the arm's candidates and 78 features are
those of the arm's mixture/MF map and are fixed before any annotation is read; pass 2 repairs each arm on its own candidates
and mixture. End-to-end plumbing of worker.main with stand-ins: 100 v1 fits on the stored bank only, 4 x 100 x (1 + arms)
selector fits that exclude the query fold's 24-hour groups, one bank per selector arm, outputs and verdict wiring, for the
default weights and for a two-weight call.

Locally (no scikit-image) the frozen 39 quality features are replaced by a deterministic stand-in, the v1 regressor by a
simple predictor and the LR selector by a fixed logistic stand-in (no model fitting on the Mac).
"""
import copy, hashlib, importlib.util, json, os, sys, tempfile, types, unittest
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parents[1]
try: HAS_SKIMAGE = importlib.util.find_spec('skimage') is not None
except ValueError: HAS_SKIMAGE = False   # a stand-in installed by another test module
if not HAS_SKIMAGE:
    sk = types.ModuleType('skimage'); sk.measure = types.ModuleType('skimage.measure'); sk.measure.regionprops = None
    sys.modules.setdefault('skimage', sk); sys.modules.setdefault('skimage.measure', sk.measure)
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import worker as W
import pcand as C
import presel as P
import mr as S
import mt as M
import mf_features as MF
import neigh as NB
from contract import mix
from context_features import FEATURES as CTX
from make_time_folds import temporal_groups, assign_groups
from rules import pq

FAKE39 = [f'q{i}' for i in range(39)]


def fake_features(mask, probs, raw):
    ys, xs = np.nonzero(mask); v = probs['blend'][ys, xs]
    base = [np.log1p(len(ys)), float(v.mean()), float(xs.mean()) / 2048, float(ys.mean()) / 2048]
    return {n: base[i % 4] + .001 * i for i, n in enumerate(FAKE39)}


def quality_names():
    if not HAS_SKIMAGE: return FAKE39
    m = np.zeros((2048, 2048), bool); m[10:20, 10:40] = True; z = np.zeros((2048, 2048), np.float32)
    return list(W.features_for_mask(m, dict(full=z, local=z, blend=z), np.zeros((2048, 2048), np.uint8)))


class Images(unittest.TestCase):
    def setUp(self):
        self.saved = {n: getattr(W, n) for n in ('features_for_mask',)}
        if not HAS_SKIMAGE: W.features_for_mask = fake_features
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        n = 2048; full = np.zeros((n, n), np.uint8); local = full.copy(); mf = np.zeros((n, n), np.uint16)
        full[400:412, 300:900] = 200; local[400:412, 300:900] = 210; mf[400:412, 300:900] = 420
        full[412:418, 300:900] = 100; local[412:418, 300:900] = 90; mf[412:418, 300:900] = 180
        full[1200:1225, 1000:1500] = 230; local[1200:1225, 1000:1500] = 230; mf[1200:1225, 1000:1500] = 460
        # new map: the MF band of the second object removed, a third MF-only object added, the weak band of the first raised;
        # mean = (stored + new) / 2
        new = mf.copy(); new[1200:1225, 1000:1500] = 0; new[1600:1612, 200:700] = 500; new[412:418, 300:900] = 400
        new2 = new.copy(); new2[1600:1612, 200:700] = 300; new2[400:412, 300:900] = 300; self.new2 = new2
        full[1600:1612, 200:700] = 150; local[1600:1612, 200:700] = 150
        maps = dict(full=full, local=local, m2f_public=mf); self.mf = mf; self.new = new
        raw = np.full((n, n), 120, np.uint8); raw[400:418, 300:900] = 70
        self.files = ['20110101000000Bh', '20110301000000Ch']; folds = {self.files[0]: 0, self.files[1]: 3}
        import cv2
        for f in self.files: cv2.imwrite(str(t / (f + '.jpeg')), raw)
        poly = lambda y0, y1, x0, x1: dict(segmentation=[[x0, y0, x1, y0, x1, y1, x0, y1]])
        self.ann = [types.SimpleNamespace(image_id='a', annotations=[poly(400, 418, 300, 900), poly(1200, 1225, 1000, 1500)]),
                    types.SimpleNamespace(image_id='b', annotations=[poly(400, 415, 300, 900), poly(1600, 1612, 200, 700)]),
                    types.SimpleNamespace(image_id='c', annotations=[])]
        den = dict(full=255, local=255, m2f_public=510); self.order39 = quality_names()
        models = {(7, k): dict(feature_order=self.order39, scaler_mean=[0.] * 39, scaler_scale=[1.] * 39, coef=[[0.] * 39], intercept=[5.], threshold=.5) for k in range(5)}
        W.G.clear(); W.G.update(folds=folds, images=t, annotations={f: self.ann for f in self.files}, models=models, order39=self.order39,
                               outer={7: {self.files[0]: 0, self.files[1]: 1}}, spec={k: {v: dict(denominator=d) for v, d in den.items()} for k in (0, 3)},
                               z={k: {v: {f: maps[v] for f in self.files} for v in den} for k in (0, 3)},
                               znew={'new': {k: [{f: new for f in self.files}, {f: new2 for f in self.files}] for k in (0, 3)}})   # two map sets

    def tearDown(self):
        for n, v in self.saved.items(): setattr(W, n, v)
        self.tmp.cleanup(); W.G.clear()

    def test_variant_changes_only_the_mf_branch(self):
        f = self.files[1]; a = W.branch_maps(f); avg = M.mean_levels([self.new, self.new2])
        for v in M.NEW_VARIANTS:
            b = W.branch_maps(f, v); self.assertTrue(np.array_equal(a['full'], b['full']) and np.array_equal(a['local'], b['local']))
            self.assertTrue(np.array_equal(b['m2f_public'], avg))                              # every weight arm reads the same average
        self.assertTrue(np.array_equal(avg, (self.new.astype(np.float32) + self.new2.astype(np.float32)) / np.float32(1020)))

    def test_one_set_equal_to_stored_reproduces_the_stored_candidates(self):
        W.G['znew'] = {'new': {k: [{f: self.mf for f in self.files}] for k in (0, 3)}}
        f = self.files[1]; _, r = W.image_pass(f)
        _, x = W.image_pass_variant('w25', f)                                                   # w25 = the shipped mixture
        for k in ('n', 'rle', 'hashes', 'features', 'raw_iou', 'n_gt_sets'): self.assertEqual(r[k], x[k])
        self.assertEqual(x['map_change']['mf_max_abs'], 0.); self.assertEqual(x['map_change']['mixture_mean_abs'], 0.)
        _, y = W.image_pass_variant('w55', f)                                                   # same MF map, another weight
        self.assertEqual(y['map_change']['mf_max_abs'], 0.); self.assertGreater(y['map_change']['mixture_mean_abs'], 0.)

    def test_variant_candidates_and_features_read_no_annotation(self):
        ann = W.G.pop('annotations'); f = self.files[1]
        ms, ps, rs = W.candidate_rows(f, 'stored'); ps_by = {}
        for v in M.NEW_VARIANTS:
            masks, p, rows = W.candidate_rows(f, v); ps_by[v] = p
            self.assertTrue(np.array_equal(p, M.mix_w(W.branch_maps(f, v), M.WEIGHT[v])))
            self.assertEqual(list(rows[0]), self.order39 + CTX + C.WINDOW_FEATURES + NB.FEATURES + MF.FEATURES)
            for r, w in zip(rows, MF.image_features(masks, W.branch_maps(f, v)['m2f_public'])): self.assertEqual({n: r[n] for n in MF.FEATURES}, w)
            for r, w in zip(rows, NB.neighbour_features(masks, p)): self.assertEqual({n: r[n] for n in NB.FEATURES}, w)
            for r, m in zip(rows, masks): self.assertEqual({n: r[n] for n in C.WINDOW_FEATURES}, C.window_features(m, p))
            self.assertNotEqual([W.packed(m) for m in masks], [W.packed(m) for m in ms])      # the new map moves the candidate bank
        self.assertTrue(np.array_equal(ps_by['w25'], mix(W.branch_maps(f, 'w25'), 'incumbent')))
        self.assertFalse(np.array_equal(ps_by['w25'], ps_by['w55']))
        W.G['annotations'] = ann

    def test_map_change_is_recorded(self):
        _, t = W.image_pass_variant('w40', self.files[1])
        self.assertGreater(t['map_change']['mf_mean_abs'], 0.); self.assertEqual(set(t['map_change']), {'mf_mean_abs', 'mf_max_abs', 'mf_flip_at_half', 'mf_ge_half_stored',
                                                                                                       'mf_ge_half_variant', 'mixture_mean_abs', 'mixture_flip_at_half'})

    def test_pass2_each_arm_on_its_own_candidates(self):
        f = self.files[1]
        _, r = W.image_pass(f); V = {v: W.image_pass_variant(v, f)[1] for v in M.NEW_VARIANTS}
        W.G.update(rles={f: r['rle']}, hashes={f: r['hashes']}, rles_v={v: {f: V[v]['rle']} for v in V},
                   hashes_v={v: {f: V[v]['hashes']} for v in V}, A={f: r}, AV={v: {f: V[v]} for v in V})
        allk = lambda x: tuple(range(x['n']))
        W.G.update(keeps2={f: {7: dict(inc_pcand_v1=allk(r), pcombo=allk(r), **{a: allk(V[M.VARIANT[a]]) for a in M.NEW_VARIANTS})}},
                   pred_outer={f: dict(stored={7: [C.TAU] * r['n']}, **{v: {7: [C.TAU] * V[v]['n']} for v in V})})
        _, rows, info, ncache = W.pass2_image(f)
        self.assertEqual(sorted(rows), sorted(M.PASS2_ARMS)); self.assertEqual(ncache, 1 + len(M.NEW_VARIANTS))
        self.assertEqual(rows['pcombo'][7], rows['inc_pcand_v1'][7])
        for v in M.NEW_VARIANTS:                     # each weight arm = P-nat repair of its candidates on its own mixture
            masks = W.decode(f, v); p = M.mix_w(W.branch_maps(f, v), M.WEIGHT[v]); ref, _ = W.pnat_repair(masks, p)
            self.assertEqual(rows[v][7], W.total_counts(W.ious(W.gt_sets(f), W.encode(ref))).tolist())
            self.assertEqual(info[7]['dup'][v]['overlapping_prediction_pairs'], 0)
        self.assertNotEqual(rows['w25'][7], rows['pcombo'][7])
        gts = W.gt_sets(f)
        self.assertEqual(info[7]['raw_counts']['w40'], W.total_counts(W.ious(gts, V['w40']['rle'])).tolist())
        self.assertEqual(info[7]['raw_counts']['pcombo'], W.total_counts(W.ious(gts, r['rle'])).tolist())
        for a in M.PASS2_ARMS:
            b = info[7]['bins'][a]; c = rows[a][7]
            self.assertEqual(sum(b['tp'].values()), c[1]); self.assertEqual(sum(b['fp'].values()), c[2]); self.assertEqual(sum(b['fn'].values()), c[3])
            rb = info[7]['raw_bins'][a]; rc = info[7]['raw_counts'][a]
            self.assertEqual(sum(rb['tp'].values()), rc[1]); self.assertEqual(sum(rb['fp'].values()), rc[2]); self.assertEqual(sum(rb['fn'].values()), rc[3])
        self.assertGreater(info[7]['bins']['pcombo']['tp']['gt1024'], 0)


def files_and_folds():
    files = sorted(f'2011{m:02d}{d:02d}120000{s}h' for m in range(1, 13) for d, s in [(3, 'B'), (17, 'C'), (25, 'M')])[:30]
    return files, {f: i % 5 for i, f in enumerate(files)}


class FakePool:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def imap(self, fn, items): return map(fn, items)


class FakeRegressor:
    fits = []
    def fit(self, X, y): self.m = float(np.mean(y)); self.n_iter_ = 1; FakeRegressor.fits.append(len(y)); return self
    def predict(self, X): return np.full(len(X), self.m) + np.asarray(X)[:, 0] * .2


class FakeSelector:
    def __init__(self, X, y):
        X = np.asarray(X, float); sc = types.SimpleNamespace(mean_=X.mean(0), scale_=X.std(0) + 1.)
        coef = np.zeros((1, X.shape[1])); coef[0, 0] = 2.; coef[0, -1] = -.5
        lr = types.SimpleNamespace(coef_=coef, intercept_=np.array([2. * float(np.mean(y)) - .5]), classes_=np.array([0, 1]), n_iter_=np.array([3]))
        self.parts = [sc, lr]
    def __getitem__(self, i): return self.parts[i]
    def predict_proba(self, X):
        p = P.project(P.selector_state(self), X); return np.stack([1 - p, p], 1)


class Plumbing(unittest.TestCase):
    def setUp(self):
        files, folds = files_and_folds(); self.files, self.folds = files, folds
        self.query = [f for f in files if folds[f] in (0, 1, 2)]; groups = temporal_groups(files, 24)
        self.seeds = list(range(2026110600, 2026110620)); self.outer = {s: assign_groups(groups, 5, s) for s in self.seeds}
        self.names = FAKE39 + CTX + C.WINDOW_FEATURES; self.feat_names = self.names + NB.FEATURES + MF.FEATURES; self.models = {}
        for s in self.seeds:
            for k in range(5):
                train = sorted(f for f in files if self.outer[s][f] != k)
                self.models[s, k] = dict(feature_order=FAKE39, train_identity_sha256=hashlib.sha256(json.dumps(train, separators=(',', ':')).encode()).hexdigest())
        self.A = {f: self.fake_image_pass(f)[1] for f in files}

    def rng(self, f, salt=''): return np.random.default_rng(int(hashlib.sha256((f + salt).encode()).hexdigest()[:8], 16))

    def fake_image_pass(self, f):
        rng = self.rng(f); n = int(rng.integers(1, 5)); seeds, outer = self.seeds, self.outer
        feats = [{c: float(rng.random()) for c in self.feat_names} for _ in range(n)]
        patterns = [list(range(n)), [0]] if n > 1 else [[0], []]
        pattern_of = {f'{s}:{k}': int(rng.integers(2)) for s in seeds for k in range(5)}
        scores = [[[float(v) if rng.random() > .3 else 0. for v in rng.random(9)] for _ in key] for key in patterns]
        raw_iou = [rng.choice([0., .3, .7], size=(int(g), n)).tolist() for g in rng.integers(1, 3, size=2)]
        r = dict(n=n, rle=[None] * n, hashes=['x'] * n, features=feats, patterns=patterns, pattern_of=pattern_of, scores=scores, tau_path_equal=True, n_gt_sets=2, raw_iou=raw_iou)
        qp = {s: pattern_of[f'{s}:{outer[s][f]}'] for s in seeds}
        inc = {s: [1. + .1 * qp[s], 2., 1., 1., 0., 0.] for s in seeds}
        dz = dict(predictions=1, overlapping_prediction_pairs=0, gt_matched_by_more_than_one_prediction=0, predictions_matched_by_more_than_one_gt_of_a_set=0, tp_pairs=2)
        pn = {i: dict(counts=[1.2 + .1 * i, 2., 1., 1., 0., 0.], stats=dict(kept=len(patterns[i]), fallback=0, changed=1, added=3, removed=1, seed_pixels=9)) for i in set(qp.values())}
        r.update(inc_counts=inc, query_pattern=qp, pnat=pn, dup={i: dict(inc=dz, inc_pnat=dz) for i in set(qp.values())})
        return f, r

    def fake_variant(self, v, f):
        rng = self.rng(f, v); n = int(rng.integers(0, 5))
        feats = [{c: float(rng.random()) for c in self.feat_names} for _ in range(n)]
        raw_iou = [rng.choice([0., .3, .7], size=(int(g), n)).tolist() for g in rng.integers(1, 3, size=2)]
        mc = dict(mf_mean_abs=.01, mf_max_abs=.1, mf_flip_at_half=.001, mf_ge_half_stored=.02, mf_ge_half_variant=.02,
                  mixture_mean_abs=.003, mixture_flip_at_half=.0005)
        return f, dict(n=n, rle=[v] * n, hashes=[v] * n, features=feats, n_gt_sets=2, raw_iou=raw_iou, map_change=mc)

    def fake_pass2(self, f):
        assert W.G['rles'][f] == self.A[f]['rle'] and all(W.G['rles_v'][v][f] == [v] * len(W.G['rles_v'][v][f]) for v in M.NEW_VARIANTS), 'PASS2_STATE'
        rows = {a: {} for a in W.PASS2_ARMS}; info = {}
        dz = dict(predictions=1, overlapping_prediction_pairs=0, gt_matched_by_more_than_one_prediction=0, predictions_matched_by_more_than_one_gt_of_a_set=0, tp_pairs=2)
        for s, kp in W.G['keeps2'][f].items():
            for a in W.PASS2_ARMS:
                taus = [W.G['pred_outer'][f][M.VARIANT[a]][s][j] for j in kp[a]]
                rows[a][s] = [1.25 + .05 * len(kp[a]) + .01 * sum(taus) + (.5 if a == 'w40' else 0.), 2., 1., 1., 0., 0.]
            st = dict(kept=1, fallback=0, changed=1, added=2, removed=1, seed_pixels=9)
            bz = {c: {b: 1 for b in M.SIZE_BINS} for c in ('tp', 'fp', 'fn')}
            info[s] = dict(dup={a: dz for a in W.PASS2_ARMS}, stats={a: st for a in W.PASS2_ARMS}, raw_counts={a: [1., 2., 1., 1., 0., 0.] for a in W.PASS2_ARMS},
                           bins={a: bz for a in W.PASS2_ARMS}, raw_bins={a: bz for a in W.PASS2_ARMS})
        return f, rows, info, 1

    def run_main(self, t, weights=M.DEFAULT_WEIGHTS, format_checked=True):
        A, seeds, query, folds = self.A, self.seeds, self.query, self.folds
        replay = dict(full_pipeline_replay_passed=True, exact_historical_per_image_six_counts=True,
                      per_image_counts={f: dict(incumbent={str(s): A[f]['inc_counts'][s] for s in seeds}) for f in query})
        target = lambda fs: pq(np.mean([np.sum([A[f]['inc_counts'][s] for f in fs], 0) for s in seeds], 0))
        cfg = dict(score_started=False, tau_grid_v1=list(C.GRID), window_features_v1=C.WINDOW_FEATURES, context_features=CTX, features_v1=self.names,
                   threshold_grid=P.THRESHOLD_GRID, fixed_arms=M.FIXED_ARMS, gate_contrast=M.GATE_CONTRAST, replay_contrasts=M.REPLAY_CONTRASTS, mf_features=MF.FEATURES, neighbour_features=NB.FEATURES,
                   features=M.feature_names(FAKE39, CTX),
                   partition_seeds=seeds, bootstrap=dict(draws=50, seed=1), expected_images=dict(all=30, query425=18, query282=12),
                   replay_targets=dict(combined425=target(query), fold2=target([f for f in query if folds[f] == 2]), new284=target([f for f in query if folds[f] in (0, 1)])),
                   repair_code_sha256={n: hashlib.sha256((H / n).read_bytes()).hexdigest() for n in ('pnat.py', 'refine_ops.py')},
                   feature_code_sha256={n: hashlib.sha256((H / n).read_bytes()).hexdigest() for n in ('neigh.py', 'mf_features.py')})
        for n in ('pnat.py', 'refine_ops.py', 'neigh.py', 'mf_features.py'): (t / n).write_bytes((H / n).read_bytes())
        (t / 'protocol.json').write_text(json.dumps(cfg)); (t / 'out').mkdir(exist_ok=True); (t / 'out/replay_result.json').write_text(json.dumps(replay))
        saved = {n: getattr(W, n) for n in ('H', 'remote', 'inventory', 'inputs_inventory', 'image_pass', 'image_pass_variant', 'pass2_image', 'mp', 'make_regressor')}
        saved_fit = P.fit_selector; saved_cut = W.PCOMBO_CUTOFF; cwd = os.getcwd(); FakeRegressor.fits = []
        try:
            W.H = t; W.remote = lambda: True; W.mp = types.SimpleNamespace(get_context=lambda name: types.SimpleNamespace(Pool=FakePool))
            W.image_pass = self.fake_image_pass; W.image_pass_variant = self.fake_variant; W.pass2_image = self.fake_pass2
            W.make_regressor = FakeRegressor; P.fit_selector = FakeSelector; W.PCOMBO_CUTOFF = None
            W.inventory = lambda: ({}, {}, self.models, self.outer, folds, t, query)
            W.inputs_inventory = lambda fo: ({}, dict(label='synthetic', folds=[f'a{k}+b{k}' for k in range(5)], sets=2, weights=list(weights),
                                                       archives={f'fold{k}': [dict(sha256='x'), dict(sha256='y')] for k in range(5)}, format_checked=format_checked))
            os.chdir(t); W.main()
            return {n: json.loads((t / 'out' / n).read_text()) for n in ('science_result.json', 'decision.json', 'counts.json', 'fits.json', 'selectors.json', 'keeps.json',
                                                                          'duplicate_audit.json', 'map_change.json', 'ship/ship_fit.json')}
        finally:
            os.chdir(cwd); P.fit_selector = saved_fit; W.PCOMBO_CUTOFF = saved_cut
            for n, v in saved.items(): setattr(W, n, v)
            W.G.clear(); self.layout = dict(ARMS=list(W.ARMS), SELECTOR_ARMS=list(W.SELECTOR_ARMS), PASS2_ARMS=list(W.PASS2_ARMS), NEW_VARIANTS=list(M.NEW_VARIANTS),
                                            CONTRASTS=list(W.CONTRASTS)); W.configure(M.DEFAULT_WEIGHTS)

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            o = self.run_main(Path(tmp)); r = o['science_result.json']
            self.assertTrue(r['replay_passed']); self.assertTrue(r['integrity_passed']); self.assertTrue(r['inputs_passed'])
            self.assertEqual(r['label'], 'synthetic'); self.assertEqual(r['weights'], [.25, .4, .55]); self.assertEqual(r['weight_arms'], dict(w25=.25, w40=.4, w55=.55))
            self.assertEqual(r['map_sets'], 2)
            self.assertEqual(r['kernel_verdict'], 'Improvement could not be confirmed')          # 30 images, never the 707
            self.assertEqual(r['selector_fits'], 1600); self.assertEqual(len(o['selectors.json']['models']), 400); self.assertEqual(r['v1_regressor_fits'], 100)
            self.assertEqual(len(FakeRegressor.fits), 100)                                        # no v1 refit on the variants
            self.assertEqual(o['counts.json']['arms'], M.ARMS)
            for c in ('all707', 'combined282', 'combined425', 'new284', 'fold0', 'fold1', 'fold2', 'fold3', 'fold4'):
                self.assertEqual(set(r['subsets'][c]['contrasts']), {n for n, _, _ in M.CONTRASTS})
            k = o['keeps.json']
            for f in self.files:
                for s in self.seeds:
                    self.assertEqual(sorted(k['keeps'][f][str(s)]), sorted(M.PASS2_ARMS))
                    for a in M.NEW_VARIANTS:
                        v = M.VARIANT[a]
                        n = self.fake_variant(v, f)[1]['n']
                        self.assertEqual(len(k['v1_outer_taus_variants'][v][f][str(s)]), n); self.assertTrue(all(j < n for j in k['keeps'][f][str(s)][a]))
            self.assertEqual(set(o['map_change.json']), set(M.NEW_VARIANTS))
            d = r['descriptive']['by_cohort']['all707']; self.assertEqual(set(d['candidates']), {'stored', *M.NEW_VARIANTS})
            self.assertEqual(set(d['feature_distribution']['w40']), set(MF.FEATURES) | set(NB.FEATURES))
            self.assertEqual(set(d['before_repair']['pq_partition_mean']), set(M.PASS2_ARMS))
            for n in ('w25_minus_pcombo', 'w55_minus_pcombo', 'w40_minus_w25'): self.assertEqual(len(d['before_repair']['paired_partition_deltas'][n]), 20)
            self.assertEqual(d['size_bins']['after_repair']['w55']['tp']['lt256'], 30.)
            self.assertGreater(r['subsets']['all707']['contrasts']['w40_minus_w25']['delta'], 0)       # the stand-in favours w40
            self.assertGreater(r['subsets']['all707']['contrasts']['w40_minus_pcombo']['delta'], r['subsets']['all707']['contrasts']['w25_minus_pcombo']['delta'])
            # ship stage: one deployment selector per selector arm, cutoff = median of the arm's 100 nested cutoffs, package format
            sf = o['ship/ship_fit.json']; self.assertEqual(sorted(sf['arms']), sorted(M.SELECTOR_ARMS))
            import statistics
            for a in M.SELECTOR_ARMS:
                cut = [x['threshold'] for x in o['selectors.json']['models'] if x['arm'] == a]
                rec = json.loads((Path(tmp) / f'out/ship/{a}_selector.json').read_text()); self.assertEqual(rec['threshold'], statistics.median(cut))
                self.assertEqual(sf['arms'][a]['selector_sha256'], hashlib.sha256((Path(tmp) / f'out/ship/{a}_selector.json').read_bytes()).hexdigest())
                self.assertEqual(rec['feature_order'], M.feature_names(FAKE39, CTX)); self.assertEqual(rec['training']['images'], 30)
                self.assertEqual(rec['role'], W.ship_role(a)); self.assertEqual(sf['arms'][a]['candidates'], sum(self.fake_variant(M.VARIANT[a], f)[1]['n'] for f in self.files) if a != 'pcombo' else sum(self.A[f]['n'] for f in self.files))
            self.assertEqual(W.ship_role('pcombo'), 'deployment_refit_pcombo_full707_mf_mixture')
            self.assertEqual(W.ship_role('w40'), 'deployment_refit_chain2_w40_full707_mf_mixture')
            self.assertIn('w = 0.40', json.loads((Path(tmp) / 'out/ship/w40_selector.json').read_text())['training']['bank'])
            o2 = self.run_main(Path(tmp), format_checked=False)['science_result.json']             # an input check failed
            self.assertFalse(o2['inputs_passed']); self.assertTrue(o2['kernel_verdict'].startswith('no verdict')); self.assertFalse(o2['gate'])
            o3 = self.run_main(Path(tmp), weights=[.4, .25]); r3 = o3['science_result.json']        # a two-weight call
            self.assertEqual(r3['weights'], [.25, .4]); self.assertEqual(self.layout['SELECTOR_ARMS'], ['pcombo', 'w25', 'w40'])
            self.assertEqual(r3['selector_fits'], 1200); self.assertEqual(o3['counts.json']['arms'], ['inc', 'inc_pnat', 'inc_pcand_v1', 'pcombo', 'w25', 'w40'])
            self.assertEqual(set(r3['subsets']['all707']['contrasts']), {'w25_minus_pcombo', 'w40_minus_pcombo', 'w40_minus_w25', *M.REPLAY_CONTRASTS})
            self.assertEqual(sorted(o3['ship/ship_fit.json']['arms']), ['pcombo', 'w25', 'w40'])
        sel = o['selectors.json']; groups = temporal_groups(self.files, 24)
        self.assertEqual(sel['features'], M.feature_names(FAKE39, CTX))
        for x in sel['models']:
            train = {f for f in self.files if self.outer[x['partition_seed']][f] != x['outer']}
            self.assertEqual(x['train_images'], len(train)); self.assertEqual(len(x['coef'][0]), 68)
            for g in groups: self.assertFalse(set(g) & train and set(g) - train)
        self.assertEqual(sorted({x['arm'] for x in sel['models']}), sorted(M.SELECTOR_ARMS))

    def test_variant_v1_taus_come_from_the_stored_bank_regressors(self):
        with tempfile.TemporaryDirectory() as tmp:
            o = self.run_main(Path(tmp))
        k = o['keeps.json']; fits = {(x['partition_seed'], x['outer']): x for x in o['fits.json']['v1']}
        for f in self.files[:6]:
            for s in self.seeds[:4]:
                kk = self.outer[s][f]; m = fits[s, kk]['target_mean']
                for v in M.NEW_VARIANTS:
                    rows = self.fake_variant(v, f)[1]['features']
                    self.assertEqual(k['v1_outer_taus_variants'][v][f][str(s)], [C.snap(m + .2 * r[self.names[0]]) for r in rows])

    def test_selector_fits_exclude_query_groups(self):
        seen = []; orig = P.label_rows
        def spy(bank, train): seen.append(set(train)); return orig(bank, train)
        with tempfile.TemporaryDirectory() as tmp:
            P.label_rows = spy
            try: self.run_main(Path(tmp))
            finally: P.label_rows = orig
        na = len(M.SELECTOR_ARMS); self.assertEqual(na, 4)
        self.assertEqual(len(seen), 1600 + na)                                  # + the ship stage: one label_rows per selector arm on all images
        outer_sets = {frozenset(f for f in self.files if self.outer[s][f] != k) for s in self.seeds for k in range(5)}
        groups = temporal_groups(self.files, 24)
        for i in range(0, 1600, 4 * na):
            final = seen[i + 3]; self.assertIn(frozenset(final), outer_sets)
            for a in range(na):
                self.assertEqual(seen[i + 4 * a + 3], final); self.assertEqual(seen[i + 4 * a:i + 4 * a + 3], seen[i:i + 3])
                for tr in seen[i + 4 * a:i + 4 * a + 3]: self.assertTrue(tr < final)
            for g in groups: self.assertFalse(set(g) & final and set(g) - final)


if __name__ == '__main__':
    unittest.main()
