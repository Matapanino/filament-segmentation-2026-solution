"""Synthetic specification tests for the P-cand v2 helpers (wide grid, v1 grid, global tau), written before pcand2.py."""
import hashlib, json, sys, unittest
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H))
import refine_ops as R
import pnat
import pcand2 as C

TAU = 0.4016711682033202
PNAT_LANE = Path('<workspace>/filament/detect_20261111/pnat_crossfold')
WIDE = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, TAU, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
V1 = (0.20, 0.25, 0.30, 0.35, TAU, 0.45, 0.50, 0.55, 0.60)


class Grids(unittest.TestCase):
    def test_wide_grid_is_fixed(self):
        self.assertEqual(C.GRID_W, WIDE); self.assertEqual(len(C.GRID_W), 17)
        self.assertIs(C.GRID_W[8], pnat.TAU)
        self.assertTrue(all(type(t) is float for t in C.GRID_W))
        self.assertEqual(list(C.GRID_W), sorted(C.GRID_W))
        # the 0.05 steps are the decimal literals (same doubles as v1's literals)
        self.assertEqual([t for t in C.GRID_W if t != TAU], [round(0.05 * i, 2) for i in range(1, 17)])

    def test_v1_grid_is_a_column_subset(self):
        self.assertEqual(C.GRID_V1, V1)
        self.assertEqual([C.GRID_W[i] for i in C.V1_COLS], list(V1))
        self.assertEqual(C.V1_COLS, [3, 4, 5, 6, 8, 9, 10, 11, 12])

    def test_snap_wide(self):
        g = C.GRID_W
        self.assertEqual(C.snap(-1., g), 0.05); self.assertEqual(C.snap(0.99, g), 0.80)
        self.assertEqual(C.snap(0.12, g), 0.10); self.assertEqual(C.snap(0.13, g), 0.15)
        self.assertEqual(C.snap(0.4005, g), 0.40); self.assertEqual(C.snap(0.4012, g), TAU)
        self.assertEqual(C.snap(0.7, g), 0.70); self.assertIs(type(C.snap(np.float64(0.52), g)), float)

    def test_snap_wide_ties(self):
        g = C.GRID_W
        self.assertEqual(abs(0.125 - 0.10), abs(0.125 - 0.15)); self.assertEqual(C.snap(0.125, g), 0.15)   # nearer to TAU
        self.assertEqual(abs(0.725 - 0.70), abs(0.725 - 0.75)); self.assertEqual(C.snap(0.725, g), 0.70)
        mid = (0.40 + TAU) / 2
        if abs(mid - 0.40) == abs(TAU - mid): self.assertEqual(C.snap(mid, g), TAU)

    def test_snap_v1_unchanged(self):
        g = C.GRID_V1
        self.assertEqual(C.snap(0.0, g), 0.20); self.assertEqual(C.snap(0.99, g), 0.60)
        self.assertEqual(C.snap(0.31, g), 0.30); self.assertEqual(C.snap(0.326, g), 0.35)
        self.assertEqual(C.snap(0.38, g), TAU); self.assertEqual(C.snap(0.43, g), 0.45)
        self.assertEqual(C.snap(0.225, g), 0.25); self.assertEqual(C.snap(0.525, g), 0.50)
        mid = (TAU + 0.45) / 2; self.assertEqual(C.snap(mid, g), TAU)


class Target(unittest.TestCase):
    def test_candidate_scores_sum_over_sets_strict(self):
        q1 = np.array([[0.6, 0.2, 0.0], [0.1, 0.51, 0.0]]); q2 = np.array([[0.7, 0.5, 0.0]]); q3 = np.zeros((0, 3))
        self.assertTrue(np.allclose(C.candidate_scores([q1, q2, q3], 3), [1.3, 0.51, 0.0]))

    def test_choose_target_wide(self):
        g = C.GRID_W; s = [0.] * 17
        s[0] = .9; self.assertEqual(C.choose_target(s, g), 0.05)
        s = [.7] * 17; self.assertEqual(C.choose_target(s, g), TAU)
        s = [0.] * 17; s[7] = s[9] = .7; self.assertEqual(C.choose_target(s, g), 0.40)    # |0.40-TAU| < |0.45-TAU|
        s = [0.] * 17; s[0] = s[16] = .7; self.assertEqual(C.choose_target(s, g), 0.05)   # |0.05-TAU|=.352 < |0.80-TAU|=.398
        self.assertIsNone(C.choose_target([0.] * 17, g))
        with self.assertRaises(AssertionError): C.choose_target([.1] * 9, g)

    def test_choose_target_v1_unchanged(self):
        g = C.GRID_V1
        self.assertEqual(C.choose_target([0, .1, .2, .3, .5, .6, .4, 0, 0], g), 0.45)
        self.assertEqual(C.choose_target([.7] * 9, g), TAU)
        self.assertEqual(C.choose_target([.7, 0, 0, 0, .1, 0, 0, 0, .7], g), 0.60)
        self.assertEqual(C.choose_target([0, 0, 0, .7, .1, .7, 0, 0, 0], g), 0.45)
        self.assertEqual(C.choose_target([0, .7, 0, 0, .1, 0, 0, .7, 0], g), 0.55)
        self.assertIsNone(C.choose_target([0.] * 9, g))

    def test_v1_target_from_wide_columns(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            row = np.where(rng.random(17) > .5, np.round(rng.random(17), 1), 0.)
            sub = row[C.V1_COLS]
            self.assertEqual(C.choose_target(sub, C.GRID_V1), C.choose_target(list(sub), C.GRID_V1))

    def test_oracle_taus_wide(self):
        m = np.zeros((2, 17)); m[1, 2] = .6
        self.assertEqual(C.oracle_taus(m, C.GRID_W), [TAU, 0.15])


class GlobalTau(unittest.TestCase):
    def test_pq(self):
        self.assertAlmostEqual(C.pq([3., 4., 2., 2., 0, 0]), 3 / 6)
        self.assertEqual(C.pq([0., 0., 0., 0., 0, 0]), 0.)

    def test_choose_global_argmax_and_ties(self):
        g = C.GRID_W
        counts = [[1., 2., 2., 2., 0, 0]] * 17                       # equal PQ everywhere -> TAU
        self.assertEqual(C.choose_global(counts, g), TAU)
        counts = [[1., 2., 2., 2., 0, 0]] * 17; counts[3] = [1.5, 2., 2., 2., 0, 0]
        self.assertEqual(C.choose_global(counts, g), 0.20)
        counts = [[1., 2., 2., 2., 0, 0]] * 17; counts[7] = counts[9] = [1.5, 2., 2., 2., 0, 0]
        self.assertEqual(C.choose_global(counts, g), 0.40)
        with self.assertRaises(AssertionError): C.choose_global(counts[:9], g)

    def test_choose_global_uses_pooled_counts_not_mean_of_pq(self):
        # image A: PQ 1.0 at t0, image B large: pooled ratio decides
        a = np.array([[1., 1., 0., 0., 0, 0], [0., 0., 1., 1., 0, 0]])
        b = np.array([[1., 2., 4., 4., 0, 0], [3., 4., 1., 1., 0, 0]])
        pooled = (a + b).tolist()
        self.assertEqual(C.choose_global(pooled, (0.30, 0.50)), 0.50)


class Repair(unittest.TestCase):
    def setUp(self): self.old = R.N; R.N = 128
    def tearDown(self): R.N = self.old

    def mask(self, y=60, x=40, h=3, w=15):
        m = np.zeros((128, 128), bool); m[y:y + h, x:x + w] = True; return m

    def test_uniform_tau_equals_pnat_exactly(self):
        rng = np.random.default_rng(7)
        seeds = [self.mask(40, 25), self.mask(55, 70), self.mask(90, 30, 4, 30)]
        p = rng.random((128, 128)).astype(np.float32)
        p[np.abs(p - np.float32(TAU)) < 0.02] = np.float32(TAU)
        want, ws = pnat.repair(seeds, p)
        got, gs = C.repair_taus(seeds, p, [TAU] * 3)
        for a, b in zip(got, want): self.assertTrue(np.array_equal(a, b))
        self.assertEqual(gs, ws)
        got64, _ = C.repair_taus(seeds, p, np.full(3, TAU))
        for a, b in zip(got64, want): self.assertTrue(np.array_equal(a, b))

    def test_very_low_tau_is_bounded_by_window_and_separation(self):
        a = self.mask(60, 25); b = self.mask(60, 65); p = np.full((128, 128), .06, np.float32); p[a] = .9; p[b] = .9
        out, st = C.repair_taus([a, b], p, [0.05, 0.05])
        self.assertFalse((out[0] & out[1]).any()); self.assertFalse((out[0] & b).any())
        g = R.geom_from_vector(R.geom_vector(R.geometry(a))); bb = R.window_bbox(g); w = R.window_native(g, bb)
        self.assertTrue((out[0][bb[0]:bb[1], bb[2]:bb[3]] <= w).all())          # never outside its own window
        self.assertEqual(int(out[0][:bb[0]].sum() + out[0][bb[1]:].sum()), 0)

    def test_per_candidate_tau_is_local(self):
        a = self.mask(30, 20); b = self.mask(90, 70)
        p = np.zeros((128, 128), np.float32); p[a] = .9; p[b] = .9; p[33, 20:35] = .1; p[93, 70:85] = .1
        lo, _ = C.repair_taus([a, b], p, [0.05, 0.45])
        self.assertTrue(lo[0][33, 25]); self.assertFalse(lo[1][93, 75])

    def test_high_tau_trims(self):
        a = self.mask(30, 20); p = np.zeros((128, 128), np.float32); p[a] = .9; p[30, 20:25] = .7
        out, st = C.repair_taus([a], p, [0.80])
        self.assertFalse(out[0][30, 21]); self.assertTrue(out[0][31, 21]); self.assertEqual(st['removed'], 5)

    def test_empty(self):
        out, stats = C.repair_taus([], np.zeros((128, 128), np.float32), [])
        self.assertEqual(out, []); self.assertEqual(stats['kept'], 0)

    def test_prepared_context_reused(self):
        seeds = [self.mask(40, 25), self.mask(55, 70)]; p = np.random.default_rng(3).random((128, 128)).astype(np.float32)
        prep = C.prepare(seeds, p)
        for t in C.GRID_W:
            a, _ = C.repair_taus(seeds, p, [t, t], prep); b, _ = C.repair_taus(seeds, p, [t, t])
            for x, y in zip(a, b): self.assertTrue(np.array_equal(x, y))


class WindowFeatures(unittest.TestCase):
    def setUp(self): self.old = R.N; R.N = 128
    def tearDown(self): R.N = self.old

    def test_names_and_values(self):
        m = np.zeros((128, 128), bool); m[60:63, 40:55] = True
        p = np.zeros((128, 128), np.float32); p[m] = .5
        f = C.window_features(m, p)
        self.assertEqual(list(f), C.WINDOW_FEATURES_W); self.assertEqual(len(C.WINDOW_FEATURES_W), 18)
        g = R.geom_from_vector(R.geom_vector(R.geometry(m))); bb = R.window_bbox(g); w = R.window_native(g, bb)
        self.assertAlmostEqual(f['seed_mean_p'], 0.5, places=6)
        self.assertAlmostEqual(f['window_frac_ge_0.05'], m.sum() / w.sum())
        self.assertEqual(f['window_frac_ge_0.80'], 0.0)

    def test_v1_window_features_are_identical_columns(self):
        # v1 computed (v >= t).mean() for t in its grid; the wide grid holds the same doubles
        rng = np.random.default_rng(5); m = np.zeros((128, 128), bool); m[50:54, 30:70] = True
        p = rng.random((128, 128)).astype(np.float32); f = C.window_features(m, p)
        g = R.geom_from_vector(R.geom_vector(R.geometry(m))); bb = R.window_bbox(g); w = R.window_native(g, bb); v = p[bb[0]:bb[1], bb[2]:bb[3]][w]
        self.assertEqual(len(C.WINDOW_FEATURES_V1), 10)
        for i, t in enumerate(C.GRID_V1):
            self.assertEqual(f[C.V1_WINDOW_MAP[f'window_frac_ge_{i}']], float((v >= t).mean()))
        self.assertEqual(C.V1_WINDOW_MAP['seed_mean_p'], 'seed_mean_p')

    def test_signature_has_no_annotation_input(self):
        import inspect
        self.assertEqual(list(inspect.signature(C.window_features).parameters), ['mask', 'p'])


class Identity(unittest.TestCase):
    def test_repair_code_byte_identical_to_pnat_crossfold(self):
        ident = json.loads((H / 'frozen_identity.json').read_text())
        for name in ('pnat.py', 'refine_ops.py'):
            digest = hashlib.sha256((H / name).read_bytes()).hexdigest()
            self.assertEqual(digest, ident['files'][name]['sha256'])
            self.assertEqual(ident['files'][name]['source'], str(PNAT_LANE / name))
            if PNAT_LANE.is_dir():
                self.assertEqual(digest, hashlib.sha256((PNAT_LANE / name).read_bytes()).hexdigest())
        self.assertEqual(hashlib.sha256((H / 'pnat.py').read_bytes()).hexdigest(), 'dcfb2ce0f2c8a15c6c541bc935811a185d7fc4f5fc69df5e15eda36d3c37d038')
        self.assertEqual(hashlib.sha256((H / 'refine_ops.py').read_bytes()).hexdigest(), '2fbd926ac78eb6b5aeebe1eb0be944988ba12af6074e784668a4c2b389bd9337')

    def test_repair_restatement_identical_to_v1(self):
        # pcand2.repair_taus/prepare must be the same source as pcand_tau/pcand.py's (only the grid changed)
        import inspect, pcand_v1_reference as V
        for n in ('prepare', 'repair_taus', 'candidate_scores', '_geom'):
            self.assertEqual(inspect.getsource(getattr(C, n)), inspect.getsource(getattr(V, n)))


class Gate(unittest.TestCase):
    def fixture(self, contrast):
        a = dict(images=425, contrasts={contrast: dict(paired_group_CI95=[.001, .004], partition_deltas=[.003] * 20, mean_paired_partition_delta=.003)})
        return dict(replay_passed=True, integrity_passed=True, subsets={'combined425': a, **{f'fold{k}': json.loads(json.dumps(a)) for k in range(3)}})

    def test_names(self):
        self.assertEqual(C.PRIMARY, 'inc_pcand_v2_minus_inc_pcand_v1')
        self.assertEqual(C.SECONDARY, ['inc_pglobal_minus_inc_pnat', 'inc_pcand_v2_minus_inc_pglobal'])

    def test_confirmed_and_blocks(self):
        for c in [C.PRIMARY] + C.SECONDARY:
            self.assertTrue(C.gate(self.fixture(c), c))
            x = self.fixture(c); x['replay_passed'] = False; self.assertFalse(C.gate(x, c))
            x = self.fixture(c); x['integrity_passed'] = False; self.assertFalse(C.gate(x, c))
            x = self.fixture(c); x['subsets']['combined425']['contrasts'][c]['paired_group_CI95'][0] = 0.; self.assertFalse(C.gate(x, c))
            x = self.fixture(c); x['subsets']['combined425']['contrasts'][c]['partition_deltas'] = [.003] * 15 + [0.] * 5; self.assertFalse(C.gate(x, c))
            x['subsets']['combined425']['contrasts'][c]['partition_deltas'] = [.003] * 16 + [0.] * 4; self.assertTrue(C.gate(x, c))
            x = self.fixture(c); x['subsets']['fold2']['contrasts'][c]['mean_paired_partition_delta'] = 0.; self.assertFalse(C.gate(x, c))
            x = self.fixture(c); x['subsets']['combined425']['images'] = 424; self.assertFalse(C.gate(x, c))
        self.assertFalse(C.gate(self.fixture(C.PRIMARY), C.SECONDARY[0]))
        self.assertFalse(C.gate({}, C.PRIMARY))


if __name__ == '__main__':
    unittest.main()
