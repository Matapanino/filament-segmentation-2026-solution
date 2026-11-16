"""Synthetic specification tests for the P-cand helpers, written before pcand.py."""
import hashlib, json, sys, unittest
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H))
import refine_ops as R
import pnat
import pcand as C

TAU = 0.4016711682033202
PNAT_LANE = Path('<workspace>/filament/detect_20261111/pnat_crossfold')


class Grid(unittest.TestCase):
    def test_grid_is_fixed(self):
        self.assertEqual(C.GRID, (0.20, 0.25, 0.30, 0.35, TAU, 0.45, 0.50, 0.55, 0.60))
        self.assertIs(C.GRID[4], pnat.TAU)
        self.assertTrue(all(type(t) is float for t in C.GRID))

    def test_snap_nearest(self):
        self.assertEqual(C.snap(0.0), 0.20); self.assertEqual(C.snap(0.99), 0.60)
        self.assertEqual(C.snap(0.31), 0.30); self.assertEqual(C.snap(0.326), 0.35)
        self.assertEqual(C.snap(0.38), TAU); self.assertEqual(C.snap(0.43), 0.45)
        self.assertEqual(C.snap(np.float64(0.52)), 0.50); self.assertIs(type(C.snap(np.float64(0.52))), float)

    def test_snap_ties(self):
        # equidistant between two grid values: the one nearer to TAU (same order as the target tie rule)
        self.assertEqual(abs(0.225 - 0.20), abs(0.225 - 0.25)); self.assertEqual(C.snap(0.225), 0.25)
        self.assertEqual(abs(0.525 - 0.50), abs(0.525 - 0.55)); self.assertEqual(C.snap(0.525), 0.50)
        # equidistant with TAU on one side: TAU wins
        mid = (TAU + 0.45) / 2
        self.assertEqual(abs(mid - TAU), abs(0.45 - mid)); self.assertEqual(C.snap(mid), TAU)


class Target(unittest.TestCase):
    def test_candidate_scores_sum_over_sets_strict(self):
        q1 = np.array([[0.6, 0.2, 0.0], [0.1, 0.51, 0.0]])      # set 1: 2 GT x 3 pred
        q2 = np.array([[0.7, 0.5, 0.0]])                        # set 2: 0.5 is not > 0.5
        q3 = np.zeros((0, 3))                                   # empty set contributes nothing
        s = C.candidate_scores([q1, q2, q3], 3)
        self.assertTrue(np.allclose(s, [0.6 + 0.7, 0.51, 0.0]))

    def test_candidate_scores_best_match_per_set(self):
        q = np.array([[0.55, 0.0], [0.8, 0.0]])
        self.assertTrue(np.allclose(C.candidate_scores([q], 2), [0.8, 0.0]))

    def test_choose_target_argmax(self):
        s = [0, .1, .2, .3, .5, .6, .4, 0, 0]
        self.assertEqual(C.choose_target(s), 0.45)

    def test_choose_target_ties_prefer_tau(self):
        self.assertEqual(C.choose_target([.7] * 9), TAU)
        self.assertEqual(C.choose_target([.7, 0, 0, 0, .7, 0, 0, 0, .7]), TAU)

    def test_choose_target_ties_nearer_then_lower(self):
        self.assertEqual(C.choose_target([.7, 0, 0, 0, .1, 0, 0, 0, .7]), 0.60)  # |0.6-TAU| < |0.2-TAU|
        self.assertEqual(C.choose_target([0, 0, 0, .7, .1, .7, 0, 0, 0]), 0.45)  # 0.0483 < 0.0517
        self.assertEqual(C.choose_target([0, .7, 0, 0, .1, 0, 0, .7, 0]), 0.55)

    def test_choose_target_excludes_unmatched(self):
        self.assertIsNone(C.choose_target([0.] * 9))

    def test_choose_target_length(self):
        with self.assertRaises(AssertionError): C.choose_target([.1] * 8)

    def test_oracle_taus_defaults_to_tau(self):
        m = np.array([[0.] * 9, [0, 0, 0, 0, .5, .6, 0, 0, 0]])
        self.assertEqual(C.oracle_taus(m), [TAU, 0.45])


class Repair(unittest.TestCase):
    def setUp(self): self.old = R.N; R.N = 128
    def tearDown(self): R.N = self.old

    def mask(self, y=60, x=40, h=3, w=15):
        m = np.zeros((128, 128), bool); m[y:y + h, x:x + w] = True; return m

    def test_uniform_tau_equals_pnat_exactly(self):
        rng = np.random.default_rng(7)
        seeds = [self.mask(40, 25), self.mask(55, 70), self.mask(90, 30, 4, 30)]
        p = rng.random((128, 128)).astype(np.float32)
        p[np.abs(p - np.float32(TAU)) < 0.02] = np.float32(TAU)       # values exactly at the float32 threshold
        want, ws = pnat.repair(seeds, p)
        got, gs = C.repair_taus(seeds, p, [TAU] * 3)
        for a, b in zip(got, want): self.assertTrue(np.array_equal(a, b))
        self.assertEqual(gs, ws)
        got64, _ = C.repair_taus(seeds, p, np.full(3, TAU))           # numpy float64 taus must not change semantics
        for a, b in zip(got64, want): self.assertTrue(np.array_equal(a, b))

    def test_per_candidate_tau_is_local(self):
        a = self.mask(30, 20); b = self.mask(90, 70)
        p = np.zeros((128, 128), np.float32); p[a] = .9; p[b] = .9
        p[33, 20:35] = .3; p[93, 70:85] = .3                           # one extra row under each seed
        lo, _ = C.repair_taus([a, b], p, [0.25, 0.45])
        self.assertTrue(lo[0][33, 25]); self.assertFalse(lo[1][93, 75])
        hi, _ = C.repair_taus([a, b], p, [0.45, 0.25])
        self.assertFalse(hi[0][33, 25]); self.assertTrue(hi[1][93, 75])

    def test_lower_tau_cannot_cross_other_seed(self):
        a = self.mask(60, 25); b = self.mask(60, 65); p = np.full((128, 128), .3, np.float32)
        out, _ = C.repair_taus([a, b], p, [0.2, 0.6])
        self.assertFalse((out[0] & b).any()); self.assertFalse((out[0] & out[1]).any())
        self.assertFalse(out[0][61, 63])                               # within 3 px of the other seed

    def test_empty(self):
        out, stats = C.repair_taus([], np.zeros((128, 128), np.float32), [])
        self.assertEqual(out, []); self.assertEqual(stats['kept'], 0)

    def test_prepared_context_reused(self):
        seeds = [self.mask(40, 25), self.mask(55, 70)]; p = np.random.default_rng(3).random((128, 128)).astype(np.float32)
        prep = C.prepare(seeds, p)
        for t in C.GRID:
            a, _ = C.repair_taus(seeds, p, [t, t], prep); b, _ = C.repair_taus(seeds, p, [t, t])
            for x, y in zip(a, b): self.assertTrue(np.array_equal(x, y))


class WindowFeatures(unittest.TestCase):
    def setUp(self): self.old = R.N; R.N = 128
    def tearDown(self): R.N = self.old

    def test_names_and_values(self):
        m = np.zeros((128, 128), bool); m[60:63, 40:55] = True
        p = np.zeros((128, 128), np.float32); p[m] = .5
        f = C.window_features(m, p)
        self.assertEqual(list(f), C.WINDOW_FEATURES); self.assertEqual(len(C.WINDOW_FEATURES), 10)
        g = R.geom_from_vector(R.geom_vector(R.geometry(m))); bb = R.window_bbox(g); w = R.window_native(g, bb)
        self.assertAlmostEqual(f['seed_mean_p'], 0.5, places=6)
        self.assertAlmostEqual(f[C.WINDOW_FEATURES[0]], m.sum() / w.sum())    # fraction >= 0.20
        self.assertEqual(f[C.WINDOW_FEATURES[8]], 0.0)                          # fraction >= 0.60

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
            if PNAT_LANE.is_dir():                                              # local check against the source lane
                self.assertEqual(digest, hashlib.sha256((PNAT_LANE / name).read_bytes()).hexdigest())
        self.assertEqual(hashlib.sha256((H / 'pnat.py').read_bytes()).hexdigest(), 'dcfb2ce0f2c8a15c6c541bc935811a185d7fc4f5fc69df5e15eda36d3c37d038')
        self.assertEqual(hashlib.sha256((H / 'refine_ops.py').read_bytes()).hexdigest(), '2fbd926ac78eb6b5aeebe1eb0be944988ba12af6074e784668a4c2b389bd9337')


class Gate(unittest.TestCase):
    def fixture(self):
        a = dict(images=425, contrasts={C.PRIMARY: dict(paired_group_CI95=[.001, .004], partition_deltas=[.003] * 20, mean_paired_partition_delta=.003)})
        return dict(replay_passed=True, integrity_passed=True, subsets={'combined425': a, **{f'fold{k}': json.loads(json.dumps(a)) for k in range(3)}})

    def test_confirmed(self): self.assertTrue(C.gate(self.fixture()))

    def test_blocks(self):
        for path, value in [(('replay_passed',), False), (('integrity_passed',), False)]:
            x = self.fixture(); x[path[0]] = value; self.assertFalse(C.gate(x))
        x = self.fixture(); x['subsets']['combined425']['contrasts'][C.PRIMARY]['paired_group_CI95'][0] = 0.; self.assertFalse(C.gate(x))
        x = self.fixture(); x['subsets']['combined425']['contrasts'][C.PRIMARY]['partition_deltas'] = [.003] * 15 + [0.] * 5; self.assertFalse(C.gate(x))
        x['subsets']['combined425']['contrasts'][C.PRIMARY]['partition_deltas'] = [.003] * 16 + [0.] * 4; self.assertTrue(C.gate(x))
        x = self.fixture(); x['subsets']['fold1']['contrasts'][C.PRIMARY]['mean_paired_partition_delta'] = 0.; self.assertFalse(C.gate(x))
        x = self.fixture(); x['subsets']['combined425']['images'] = 424; self.assertFalse(C.gate(x))
        self.assertFalse(C.gate({}))
        self.assertEqual(C.PRIMARY, 'inc_pcand_minus_inc_pnat')


if __name__ == '__main__':
    unittest.main()
