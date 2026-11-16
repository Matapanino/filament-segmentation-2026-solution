"""Synthetic tests of pworker.py (order PC1), written before pworker.py.

Images (2048-pixel synthetic maps, chain2's test scene): the arm's phase A = chain2's image_pass_variant candidates and
features exactly, plus the keep pattern of every saved C5 selector and of the deployed LR39 and the 9-tau table of every
distinct pattern; pass 2 with chain2's taus ('frozen') = chain2's pass2_image counts for that arm; 'refit' and 'gtau' are
repair_taus with the given per-candidate / global taus.
Plumbing (fake banks, stand-in regressor, no fitting): the 100 refits use only outer-training images and the C5 row rule,
global taus are chosen on the outer-training images only, the deployment fit uses the 'deploy' pattern of all images;
main() runs replay -> refits -> pass 2 -> summaries, and stops before any refit when the replay is not exact.
"""
import copy, hashlib, importlib.util, json, os, sys, tempfile, types, unittest
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parents[1]
try: HAS_SKIMAGE = importlib.util.find_spec('skimage') is not None
except ValueError: HAS_SKIMAGE = False
if not HAS_SKIMAGE:
    sk = types.ModuleType('skimage'); sk.measure = types.ModuleType('skimage.measure'); sk.measure.regionprops = None
    sys.modules.setdefault('skimage', sk); sys.modules.setdefault('skimage.measure', sk.measure)
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import worker as W
import pworker as PW
import pt as T
import pcand as C
import presel as P
import mt as M
import mf_features as MF
import neigh as NB
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
    """chain2's synthetic scene (tests/test_worker_synthetic.Images), with one partition seed 7."""
    def setUp(self):
        self.saved = {n: getattr(W, n) for n in ('features_for_mask',)}
        if not HAS_SKIMAGE: W.features_for_mask = fake_features
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        n = 2048; full = np.zeros((n, n), np.uint8); local = full.copy(); mf = np.zeros((n, n), np.uint16)
        full[400:412, 300:900] = 200; local[400:412, 300:900] = 210; mf[400:412, 300:900] = 420
        full[412:418, 300:900] = 100; local[412:418, 300:900] = 90; mf[412:418, 300:900] = 180
        full[1200:1225, 1000:1500] = 230; local[1200:1225, 1000:1500] = 230; mf[1200:1225, 1000:1500] = 460
        new = mf.copy(); new[1200:1225, 1000:1500] = 0; new[1600:1612, 200:700] = 500; new[412:418, 300:900] = 400
        new2 = new.copy(); new2[1600:1612, 200:700] = 300; new2[400:412, 300:900] = 300
        full[1600:1612, 200:700] = 150; local[1600:1612, 200:700] = 150
        maps = dict(full=full, local=local, m2f_public=mf)
        raw = np.full((n, n), 120, np.uint8); raw[400:418, 300:900] = 70
        self.files = ['20110101000000Bh', '20110301000000Ch']; folds = {self.files[0]: 0, self.files[1]: 3}
        import cv2
        for f in self.files: cv2.imwrite(str(t / (f + '.jpeg')), raw)
        poly = lambda y0, y1, x0, x1: dict(segmentation=[[x0, y0, x1, y0, x1, y1, x0, y1]])
        self.ann = [types.SimpleNamespace(image_id='a', annotations=[poly(400, 418, 300, 900), poly(1200, 1225, 1000, 1500)]),
                    types.SimpleNamespace(image_id='b', annotations=[poly(400, 415, 300, 900), poly(1600, 1612, 200, 700)]),
                    types.SimpleNamespace(image_id='c', annotations=[])]
        den = dict(full=255, local=255, m2f_public=510); self.order39 = quality_names()
        keep_all = lambda: dict(feature_order=self.order39, scaler_mean=[0.] * 39, scaler_scale=[1.] * 39, coef=[[0.] * 39], intercept=[5.], threshold=.5)
        models = {(7, k): keep_all() for k in range(5)}
        c1 = [0.] * 39; c1[1] = 40.                                       # C5 (7, 1): keeps candidates whose feature 1 is large
        models[7, 1] = dict(keep_all(), coef=[c1], intercept=[-20.])
        deployed = dict(feature_order=self.order39[::-1], scaler_mean=[0.] * 39, scaler_scale=[1.] * 39, coef=[[0.] * 39], intercept=[-5.], threshold=.35, classes=[0, 1])
        W.configure(M.DEFAULT_WEIGHTS)
        W.G.clear(); W.G.update(folds=folds, images=t, annotations={f: self.ann for f in self.files}, models=models, order39=self.order39, deployed=deployed,
                               outer={7: {self.files[0]: 0, self.files[1]: 1}}, spec={k: {v: dict(denominator=d) for v, d in den.items()} for k in (0, 3)},
                               z={k: {v: {f: maps[v] for f in self.files} for v in den} for k in (0, 3)},
                               znew={'new': {k: [{f: new for f in self.files}, {f: new2 for f in self.files}] for k in (0, 3)}})

    def tearDown(self):
        for n, v in self.saved.items(): setattr(W, n, v)
        self.tmp.cleanup(); W.G.clear()

    def test_phase_a_equals_chain2_variant_pass(self):
        f = self.files[1]
        for v in ('w40', 'w55'):
            _, a = PW.image_pass_arm(v, f); _, b = W.image_pass_variant(v, f)
            for k in ('n', 'rle', 'hashes', 'features', 'raw_iou', 'n_gt_sets'): self.assertEqual(a[k], b[k], (v, k))

    def test_patterns_and_tables(self):
        f = self.files[1]; _, a = PW.image_pass_arm('w55', f)
        self.assertEqual(set(a['pattern_of']), {f'7:{k}' for k in range(5)} | {'deploy'})
        self.assertEqual(a['patterns'][a['pattern_of']['deploy']], [])                 # the stand-in deployed selector keeps nothing
        self.assertEqual(a['patterns'][a['pattern_of']['7:0']], list(range(a['n'])))
        self.assertTrue(a['tau_path_equal']); self.assertEqual(len(a['patterns']), len(a['scores'])); self.assertEqual(len(a['patterns']), len(a['counts']))
        masks = W.decode.__globals__['mu']                                          # pycocotools via worker
        p = M.mix_w(W.branch_maps(f, 'w55'), .55); gts = W.gt_sets(f)
        seeds = [np.ascontiguousarray(masks.decode(r).astype(bool)) for r in a['rle']]
        for i, key in enumerate(a['patterns']):
            ref = T.tau_table([seeds[j] for j in key], p, gts)
            self.assertEqual(a['scores'][i], ref['scores']); self.assertEqual(a['counts'][i], ref['counts'])
            self.assertEqual(len(a['counts'][i]), 9)
        empty = a['counts'][a['pattern_of']['deploy']]
        self.assertTrue(all(c[1] == 0 and c[3] == 4 for c in empty))                  # 2 nonempty sets x 2 GT, all unmatched

    def pass2_state(self, f, taus_of):
        V = {v: PW.image_pass_arm(v, f)[1] for v in ('w25', 'w40', 'w55')}
        allk = lambda x: tuple(range(x['n']))
        PW.configure_arms(['w55', 'w40'])
        W.G.update(AV={v: {f: V[v]} for v in V}, rles_v={v: {f: V[v]['rle']} for v in V}, hashes_v={v: {f: V[v]['hashes']} for v in V},
                   keeps_arm={v: {f: {7: allk(V[v])}} for v in V},
                   taus={v: dict(frozen={f: {7: taus_of(V[v]['n'], 0)}}, refit={f: {7: taus_of(V[v]['n'], 1)}}) for v in V},
                   tglob={v: {(7, 1): .30} for v in V}, partition_seeds=[7])
        return V

    def test_pass2_frozen_equals_chain2_pass2(self):
        f = self.files[1]; tau = lambda n, salt: [C.GRID[(3 * j + salt) % 9] for j in range(n)]
        V = self.pass2_state(f, tau); W.G['p2'] = ['frozen']
        _, rows, info = PW.pass2_arm(f)
        self.assertEqual(sorted(rows), ['w40_frozen', 'w55_frozen'])
        r = W.image_pass(f)[1]
        W.G.update(rles={f: r['rle']}, hashes={f: r['hashes']}, A={f: r},
                   keeps2={f: {7: dict(inc_pcand_v1=tuple(range(r['n'])), pcombo=tuple(range(r['n'])), **{v: tuple(range(V[v]['n'])) for v in V})}},
                   pred_outer={f: dict(stored={7: [C.TAU] * r['n']}, **{v: {7: tau(V[v]['n'], 0)} for v in V})})
        _, ref_rows, ref_info, _ = W.pass2_image(f)                                  # chain2's own pass 2
        for v in ('w40', 'w55'):
            self.assertEqual(rows[f'{v}_frozen'][7], ref_rows[v][7])
            self.assertEqual(info[7]['raw'][v], ref_info[7]['raw_counts'][v])
            self.assertEqual(info[7]['stats'][f'{v}_frozen'], ref_info[7]['stats'][v])
            self.assertEqual(info[7]['dup'][f'{v}_frozen'], ref_info[7]['dup'][v])

    def test_refit_and_gtau_variants(self):
        f = self.files[1]; tau = lambda n, salt: [C.GRID[(2 * j + salt) % 9] for j in range(n)]
        V = self.pass2_state(f, tau); W.G['p2'] = ['refit', 'gtau']
        _, rows, info = PW.pass2_arm(f); gts = W.gt_sets(f)
        self.assertEqual(sorted(rows), ['w40_gtau', 'w40_refit', 'w55_gtau', 'w55_refit'])
        for v in ('w40', 'w55'):
            masks = W.decode(f, v); p = M.mixture(W.branch_maps(f, v), v)
            out, _ = C.repair_taus(masks, p, tau(V[v]['n'], 1)); self.assertEqual(rows[f'{v}_refit'][7], W.total_counts(W.ious(gts, W.encode(out))).tolist())
            out, _ = C.repair_taus(masks, p, [.30] * V[v]['n']); self.assertEqual(rows[f'{v}_gtau'][7], W.total_counts(W.ious(gts, W.encode(out))).tolist())
            self.assertEqual(info[7]['dup'][f'{v}_gtau']['overlapping_prediction_pairs'], 0)

    def test_keep_patterns_read_no_annotation(self):
        f = self.files[1]; ann = W.G.pop('annotations')
        _, _, rows = W.candidate_rows(f, 'w55'); k = PW.keep_patterns(rows)
        self.assertEqual(k['deploy'], ()); self.assertEqual(k['7:0'], tuple(range(len(rows))))
        x = np.asarray([[r[n] for n in self.order39] for r in rows], float)
        self.assertEqual(k['7:1'], tuple(np.flatnonzero(W.project(W.G['models'][7, 1], x) >= .5).tolist()))
        W.G['annotations'] = ann


# ---------------------------------------------------------------- plumbing on fake banks
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
    def __init__(self, *a, **k): pass
    def fit(self, X, y): self.m = float(np.mean(y)); self.n_iter_ = 1; FakeRegressor.fits.append((len(y), round(float(np.sum(X)), 9))); return self
    def predict(self, X): return np.full(len(X), self.m) + np.asarray(X)[:, 0] * .2


class Plumbing(unittest.TestCase):
    VS = ['w55', 'w40']

    def setUp(self):
        files, folds = files_and_folds(); self.files, self.folds = files, folds
        self.query = [f for f in files if folds[f] in (0, 1, 2)]; self.groups = temporal_groups(files, 24)
        self.seeds = list(range(2026110600, 2026110620)); self.outer = {s: assign_groups(self.groups, 5, s) for s in self.seeds}
        self.names_v1 = FAKE39 + CTX + C.WINDOW_FEATURES; self.feat_names = self.names_v1 + NB.FEATURES + MF.FEATURES; self.models = {}
        for s in self.seeds:
            for k in range(5):
                train = sorted(f for f in files if self.outer[s][f] != k)
                self.models[s, k] = dict(feature_order=FAKE39, train_identity_sha256=hashlib.sha256(json.dumps(train, separators=(',', ':')).encode()).hexdigest())
        self.names68 = M.feature_names(FAKE39, CTX)

    def rng(self, f, salt=''): return np.random.default_rng(int(hashlib.sha256((f + salt).encode()).hexdigest()[:8], 16))

    def fake_pass_a(self, v, f):
        rng = self.rng(f, v); n = int(rng.integers(1, 5))
        feats = [{c: float(rng.random()) for c in self.feat_names} for _ in range(n)]
        patterns = [list(range(n)), [0], []]
        pattern_of = {f'{s}:{k}': int(rng.integers(2)) for s in self.seeds for k in range(5)}; pattern_of['deploy'] = int(rng.integers(3))
        scores = [[[float(x) if rng.random() > .3 else 0. for x in rng.random(9)] for _ in key] for key in patterns]
        counts = [[[float(rng.random()) * 2, 2., 1., float(rng.integers(1, 3)), 0., 0.] for _ in range(9)] for _ in patterns]
        raw_iou = [rng.choice([0., .3, .7], size=(int(g), n)).tolist() for g in rng.integers(1, 3, size=2)]
        return f, dict(n=n, rle=[v] * n, hashes=[v] * n, features=feats, patterns=patterns, pattern_of=pattern_of, scores=scores, counts=counts,
                       tau_path_equal=True, n_gt_sets=2, raw_iou=raw_iou)

    def fake_counts(self, f, s, arm, taus):
        return [1.25 + .05 * len(taus) + .01 * sum(taus) + (.3 if arm.endswith('refit') else 0.), 2., 1., 1., 0., 0.]

    def fake_pass2(self, f):
        G = W.G; rows = {}; info = {}
        dz = dict(predictions=1, overlapping_prediction_pairs=0, gt_matched_by_more_than_one_prediction=0, predictions_matched_by_more_than_one_gt_of_a_set=0, tp_pairs=2)
        for s in G['partition_seeds']:
            info[s] = dict(dup={}, stats={}, raw={})
            for v in G.get('p2_arms', PW.VS):
                assert G['rles_v'][v][f] == [v] * G['AV'][v][f]['n'], 'PASS2_STATE'
                key = G['keeps_arm'][v][f][s]; info[s]['raw'][v] = [1., 2., 1., 1., 0., 0.]
                for x in G['p2']:
                    a = T.arm(v, x)
                    taus = [G['taus'][v][x][f][s][j] for j in key] if x in ('frozen', 'refit') else [G['tglob'][v][s, self.outer[s][f]]] * len(key)
                    rows.setdefault(a, {})[s] = self.fake_counts(f, s, a, taus); info[s]['dup'][a] = dz
                    info[s]['stats'][a] = dict(kept=len(key), fallback=0, changed=1, added=2, removed=1, seed_pixels=9)
        return f, rows, info

    def reference(self, perturb=False):
        """chain2's outputs for the fake call: keeps and frozen taus per arm, the 68-feature selectors that reproduce the keeps,
        counts of the frozen arm (from the same fake pass 2), pcombo counts, the frozen arm's scores."""
        A = {v: {f: self.fake_pass_a(v, f)[1] for f in self.files} for v in self.VS}; sels = []; keeps = {f: {str(s): {} for s in self.seeds} for f in self.files}
        taus = {v: {f: {str(s): [C.GRID[(j + s) % 9] for j in range(A[v][f]['n'])] for s in self.seeds} for f in self.files} for v in self.VS}
        for v in self.VS:
            for s in self.seeds:
                for k in range(5):
                    st = dict(scaler_mean=[.5] * 68, scaler_scale=[1.] * 68, coef=[[1.] + [0.] * 67], intercept=[0.], threshold=.55)
                    sels.append(dict(st, arm=v, partition_seed=s, outer=k))
                    for f in self.files:
                        if self.outer[s][f] != k: continue
                        x = T.all_rows(A[v], f, self.names68); keeps[f][str(s)][v] = np.flatnonzero(P.project(st, x) >= .55).tolist()
        per = {str(s): {} for s in self.seeds}
        for v in self.VS:
            for s in self.seeds:
                per[str(s)][v] = [self.fake_counts(f, s, f'{v}_frozen', [taus[v][f][str(s)][j] for j in keeps[f][str(s)][v]]) for f in self.files]
        for s in self.seeds: per[str(s)]['pcombo'] = [[1.3, 2., 1., 1., 0., 0.] for _ in self.files]
        if perturb: per[str(self.seeds[3])]['w55'][4] = [9., 2., 1., 1., 0., 0.]
        cfg = dict(partition_seeds=self.seeds, bootstrap=dict(draws=50, seed=1))
        perint = {s: {a: per[str(s)][a] for a in per[str(s)]} for s in self.seeds}
        sci = dict(subsets={'all707': dict(scores={v: T.summarize(perint, self.files, self.files, self.groups, cfg, [v], [])['scores'][v] for v in self.VS},
                                           partition_pq={v: T.summarize(perint, self.files, self.files, self.groups, cfg, [v], [])['partition_pq'][v] for v in self.VS})})
        return A, dict(keeps=dict(files=self.files, keeps=keeps, v1_outer_taus_variants=taus), counts=dict(files=self.files, per_partition_counts=per),
                       selectors=dict(models=sels), science_result=sci), cfg

    def run_main(self, t, perturb=False):
        A, ref, cfg0 = self.reference(perturb)
        cfg = dict(cfg0, expected_images=dict(all=30, query425=18, query282=12), tau_grid_v1=list(C.GRID), features_v1=self.names_v1, features=self.names68)
        pins = dict(label='synthetic', arms=list(self.VS), chain2_kernel='<kaggle-user>/filament-mainpc-chain2-synthetic')
        (t / 'protocol.json').write_text(json.dumps(cfg)); (t / 'pcand_inputs.json').write_text(json.dumps(pins))
        saved = {n: getattr(PW, n) for n in ('H', 'remote', 'image_pass_arm', 'pass2_arm', 'mp', 'make_regressor', 'chain2_reference', 'deployed_selector')}
        saved_w = {n: getattr(W, n) for n in ('inventory', 'inputs_inventory', 'mp')}
        cwd = os.getcwd(); FakeRegressor.fits = []
        fake_mp = types.SimpleNamespace(get_context=lambda name: types.SimpleNamespace(Pool=FakePool))
        try:
            PW.H = t; PW.remote = lambda: True; PW.mp = fake_mp; W.mp = fake_mp
            PW.image_pass_arm = lambda v, f: (f, A[v][f]); PW.pass2_arm = self.fake_pass2; PW.make_regressor = FakeRegressor
            PW.chain2_reference = lambda pins: ref; PW.deployed_selector = lambda pins: dict(feature_order=FAKE39, threshold=.35)
            W.inventory = lambda: ({}, {}, self.models, self.outer, self.folds, t, self.query)
            W.inputs_inventory = lambda fo: ({}, dict(label='synthetic', folds=[f'a{k}' for k in range(5)], sets=1, weights=[.25, .4, .55], archives={}, format_checked=True))
            os.chdir(t)
            try: PW.main(); raised = None
            except RuntimeError as e: raised = e
            out = {p.name: json.loads(p.read_text()) for p in (t / 'out').glob('*.json')}
            return out, raised, A, ref
        finally:
            os.chdir(cwd)
            for n, v in saved.items(): setattr(PW, n, v)
            for n, v in saved_w.items(): setattr(W, n, v)
            W.G.clear(); W.configure(M.DEFAULT_WEIGHTS)

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            o, raised, A, ref = self.run_main(Path(tmp))
            self.assertIsNone(raised, raised)
            r = o['science_result.json']; rp = o['replay.json']
            for v in self.VS:
                self.assertTrue(rp[v]['exact'], rp[v]); self.assertTrue(rp[v]['selector_keeps']['exact']); self.assertTrue(rp[v]['counts_exact'])
                self.assertEqual(rp[v]['all707_score'], ref['science_result']['subsets']['all707']['scores'][v])
            self.assertTrue(r['replay_passed']); self.assertEqual(r['arms'], T.arms_of(self.VS))
            self.assertEqual(len(FakeRegressor.fits), 2 * 100 + 2 * 2)          # 100 outer refits + the deployment fit twice, per arm
            for c in ('all707', 'no_fold2', 'fold0', 'fold4', 'combined425'):
                self.assertEqual(set(r['subsets'][c]['contrasts']), {n for n, _, _ in T.contrasts_of(self.VS)})
            self.assertEqual(r['subsets']['no_fold2']['images'], 24)
            self.assertGreater(r['subsets']['all707']['contrasts']['w55_refit_minus_w55_frozen']['delta'], 0)   # the stand-in favours refit
            # frozen arm = chain2's arm (same counts), so frozen - pcombo = chain2's own contrast
            per = o['counts.json']['per_partition_counts']
            for s in self.seeds: self.assertEqual(per[str(s)]['w55_frozen'], ref['counts']['per_partition_counts'][str(s)]['w55'])
            fits = o['fits.json']
            for v in self.VS:
                self.assertEqual(len(fits[v]['outer']), 100)
                for x in fits[v]['outer']:
                    s, k = x['partition_seed'], x['outer']; train = sorted(f for f in self.files if self.outer[s][f] != k)
                    X, y, ex = T.training_rows(A[v], train, f'{s}:{k}', self.names_v1)
                    self.assertEqual((x['rows'], x['excluded_unmatched']), (len(y), ex))
                    pooled = T.pool_counts({f: A[v][f]['counts'] for f in train}, {f: A[v][f]['pattern_of'] for f in train}, f'{s}:{k}', train)
                    self.assertEqual(x['global_tau'], T.choose_global(pooled))
                X, y, ex = T.training_rows(A[v], self.files, 'deploy', self.names_v1)
                d = o['deploy_fit.json'][v]; self.assertEqual(d['rows'], len(y)); self.assertEqual(d['training_images'], 30)
                pooled = T.pool_counts({f: A[v][f]['counts'] for f in self.files}, {f: A[v][f]['pattern_of'] for f in self.files}, 'deploy', self.files)
                self.assertEqual(d['global_tau'], T.choose_global(pooled)); self.assertTrue((Path(tmp) / f'out/deploy/{v}_regressor.pkl').is_file())
                self.assertEqual(d['regressor_sha256'], hashlib.sha256((Path(tmp) / f'out/deploy/{v}_regressor.pkl').read_bytes()).hexdigest())
            taus = o['taus.json']
            for v in self.VS:
                for f in self.files[:5]:
                    for s in self.seeds[:3]: self.assertEqual(len(taus['refit'][v][f][str(s)]), A[v][f]['n'])

    def test_replay_failure_stops_before_any_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            o, raised, _, _ = self.run_main(Path(tmp), perturb=True)
            self.assertIsInstance(raised, RuntimeError); self.assertIn('REPLAY', str(raised))
            self.assertFalse(o['replay.json']['w55']['exact']); self.assertFalse(o['replay.json']['w55']['counts_exact'])
            self.assertEqual(FakeRegressor.fits, []); self.assertNotIn('science_result.json', o)

    def test_selector_keep_replay_detects_a_changed_keep(self):
        A, ref, _ = self.reference()
        ok = PW.selector_keep_replay('w55', A['w55'], ref['selectors']['models'], ref['keeps']['keeps'], self.outer, self.files, self.names68)
        self.assertTrue(ok['exact']); self.assertEqual(ok['compared'], 100)
        bad = copy.deepcopy(ref['keeps']['keeps']); f = self.files[2]; s = str(self.seeds[0]); bad[f][s]['w55'] = bad[f][s]['w55'] + [99]
        no = PW.selector_keep_replay('w55', A['w55'], ref['selectors']['models'], bad, self.outer, self.files, self.names68)
        self.assertFalse(no['exact']); self.assertEqual(no['differing_image_partitions'], 1)


if __name__ == '__main__': unittest.main()
