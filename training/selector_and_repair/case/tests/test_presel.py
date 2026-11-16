"""Synthetic specification tests for the P-resel helpers, written before presel.py (red log: tdd_red.log)."""
import inspect, sys, unittest
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H), str(H / 'support')]
import presel as P

REMOTE = Path('/kaggle/input').is_dir()


def rle(m): return mu.encode(np.asfortranarray(m, dtype=np.uint8))


class Names(unittest.TestCase):
    def test_fixed_names_and_order(self):
        self.assertEqual(P.DELTA_FEATURES, ['repair_area_ratio', 'repair_pixels_added', 'repair_pixels_removed',
                                            'repair_touches_window_border', 'repair_added_mean_p'])
        self.assertEqual(P.ARMS, ['inc', 'inc_pnat', 'inc_pcand_v1', 'presel', 'fresh_ctrl'])
        self.assertEqual(P.PRIMARY, 'presel_minus_inc_pcand_v1')
        self.assertEqual(P.SECONDARY, 'presel_minus_fresh_ctrl')
        names = [c[0] for c in P.CONTRASTS]
        self.assertEqual(names[:2], [P.PRIMARY, P.SECONDARY]); self.assertEqual(len(set(names)), len(names))
        for name, a, b in P.CONTRASTS:
            self.assertEqual(name, f'{a}_minus_{b}'); self.assertIn(a, P.ARMS); self.assertIn(b, P.ARMS)
        self.assertEqual(P.THRESHOLD_GRID, [round(i * .025, 6) for i in range(41)])

    def test_presel_feature_names(self):
        o39 = [f'q{i}' for i in range(39)]; ctx = [f'c{i}' for i in range(15)]
        names = P.presel_feature_names(o39, ctx)
        self.assertEqual(names, o39 + ctx + P.DELTA_FEATURES); self.assertEqual(len(names), 59)
        self.assertEqual(P.control_feature_names(o39, ctx), o39 + ctx)


class Border(unittest.TestCase):
    def test_rectangle_border(self):
        w = np.zeros((10, 12), bool); w[2:8, 3:9] = True
        b = P.window_border(w)
        inner = np.zeros_like(w); inner[3:7, 4:8] = True
        self.assertTrue(np.array_equal(b, w & ~inner))

    def test_bbox_edge_counts_as_border(self):
        w = np.ones((5, 6), bool); b = P.window_border(w)
        self.assertTrue(b[0].all() and b[-1].all() and b[:, 0].all() and b[:, -1].all()); self.assertFalse(b[1:-1, 1:-1].any())

    def test_diagonal_neighbour_outside_makes_border(self):
        w = np.ones((5, 5), bool); w[0, 0] = False
        b = P.window_border(w); self.assertTrue(b[1, 1])


class Delta(unittest.TestCase):
    def setUp(self):
        self.seed = np.zeros((10, 10), bool); self.seed[4:6, 2:8] = True          # 12 pixels
        self.w = np.zeros((10, 10), bool); self.w[1:9, 1:9] = True
        self.p = np.full((10, 10), .3, np.float32)

    def test_definitions(self):
        m = self.seed.copy(); m[4, 2] = False; m[3, 3:6] = True; self.p[3, 3:6] = [.5, .6, .7]
        d = P.delta_features(self.seed, m, self.p, P.window_border(self.w))
        self.assertEqual(list(d), P.DELTA_FEATURES)
        self.assertAlmostEqual(d['repair_area_ratio'], 14 / 12); self.assertEqual(d['repair_pixels_added'], 3.0)
        self.assertEqual(d['repair_pixels_removed'], 1.0); self.assertEqual(d['repair_touches_window_border'], 0.0)
        self.assertAlmostEqual(d['repair_added_mean_p'], float(np.mean(np.float32([.5, .6, .7]).astype(np.float64))))
        self.assertTrue(all(type(v) is float for v in d.values()))

    def test_nothing_added_gives_zero_mean(self):
        d = P.delta_features(self.seed, self.seed.copy(), self.p, P.window_border(self.w))
        self.assertEqual(d['repair_added_mean_p'], 0.0); self.assertEqual(d['repair_pixels_added'], 0.0); self.assertEqual(d['repair_area_ratio'], 1.0)

    def test_touch_border(self):
        m = self.seed.copy(); m[4, 1] = True
        self.assertEqual(P.delta_features(self.seed, m, self.p, P.window_border(self.w))['repair_touches_window_border'], 1.0)

    def test_shape_mismatch_and_empty_rejected(self):
        with self.assertRaises(AssertionError): P.delta_features(self.seed, self.seed[:5], self.p, P.window_border(self.w))
        with self.assertRaises(AssertionError): P.delta_features(np.zeros_like(self.seed), self.seed, self.p, P.window_border(self.w))

    def test_no_annotation_input(self):
        for fn in (P.delta_features, P.window_border, P.presel_row):
            self.assertFalse({'gt', 'gts', 'annotations', 'labels', 'iou'} & set(inspect.signature(fn).parameters))


class Rows(unittest.TestCase):
    def test_presel_row_order(self):
        o39 = [f'q{i}' for i in range(39)]
        q = {n: float(i) for i, n in enumerate(o39)}; q['extra'] = -1.
        ctx = {f'c{i}': 100. + i for i in range(15)}; d = {n: 200. + i for i, n in enumerate(P.DELTA_FEATURES)}
        row = P.presel_row(q, ctx, d, o39, list(ctx))
        self.assertEqual(row, [float(i) for i in range(39)] + [100. + i for i in range(15)] + [200. + i for i in range(5)])

    def test_presel_row_missing_feature_fails(self):
        o39 = ['a', 'b']
        with self.assertRaises(KeyError): P.presel_row({'a': 1.}, {}, {n: 0. for n in P.DELTA_FEATURES}, o39, [])


class Labels(unittest.TestCase):
    def test_per_set_rule(self):
        q1 = np.array([[.6, .2, .0], [.1, .51, .5]]); q2 = np.array([[.5, .0, .9]])
        self.assertEqual(P.set_labels([q1, q2], 3), [[1, 1, 0], [0, 0, 1]])

    def test_strict_threshold_and_empty(self):
        self.assertEqual(P.set_labels([np.array([[.5]])], 1), [[0]]); self.assertEqual(P.set_labels([], 2), [])


class Bank(unittest.TestCase):
    def fixture(self):
        rows = {'a': np.array([[1., 2.], [3., 4.]]), 'b': np.array([[5., 6.]]), 'c': np.zeros((0, 2))}
        ious = {'a': [np.array([[.6, .0]]), np.array([[.2, .7], [.0, .1]])], 'b': [np.array([[.3]])], 'c': [np.zeros((1, 0))]}
        return rows, ious

    def test_bank_layout(self):
        rows, ious = self.fixture(); b = P.build_bank(['a', 'b', 'c'], rows, ious)
        np.testing.assert_array_equal(b['X'], [[1, 2], [3, 4], [5, 6]]); self.assertEqual(b['file'].tolist(), ['a', 'a', 'b'])
        self.assertEqual(b['records']['a']['candidate_ids'], [0, 1]); self.assertEqual(b['records']['b']['candidate_ids'], [2])
        self.assertEqual(b['records']['c']['candidate_ids'], [])
        self.assertEqual(b['records']['a']['annotations'][1], dict(n_gt=2, iou=[[.2, .7], [.0, .1]]))
        # one label row per (candidate, nonempty annotation set), set-major as in the historical labels
        self.assertEqual(b['label_ids'].tolist(), [0, 1, 0, 1, 2]); self.assertEqual(b['label_y'].tolist(), [1, 0, 0, 1, 0])

    def test_fit_rows_respect_split(self):
        rows, ious = self.fixture(); b = P.build_bank(['a', 'b', 'c'], rows, ious)
        ids, y = P.label_rows(b, ['a'])
        self.assertEqual(ids.tolist(), [0, 1, 0, 1]); self.assertEqual(y.tolist(), [1, 0, 0, 1])
        self.assertEqual(P.candidate_ids(b, ['b', 'c']).tolist(), [2])

    def test_threshold_sweep_matches_historical_core(self):
        from core import tune_threshold
        from count_plan import count_coefficients, threshold_sweep
        rows, ious = self.fixture(); b = P.build_bank(['a', 'b', 'c'], rows, ious)
        plan, ngt = count_coefficients(b['records'], len(b['X'])); v = np.array([.2, .7, .4])
        self.assertEqual(threshold_sweep(plan, ngt, b['records'], ['a', 'b', 'c'], v, P.THRESHOLD_GRID), tune_threshold(b['records'], ['a', 'b', 'c'], v))


class Roles(unittest.TestCase):
    def test_pass1_roles(self):
        outer = {7: {'a': 0, 'b': 1, 'c': 1}}; inner = {(7, 0): {'b': 2, 'c': 0}, (7, 1): {'a': 1}}
        r = P.pass1_roles('a', [7], outer, inner, {'a', 'b'}, 2)
        self.assertEqual(r, {(7, 0): ('outer', 7, 0), (7, 1): ('inner', 7, 1, 1)})
        r = P.pass1_roles('c', [7], outer, inner, {'a', 'b'}, 2)          # c is not a query image: no outer role
        self.assertEqual(r, {(7, 0): ('inner', 7, 0, 0)})

    def test_every_image_has_one_role_per_training_fit(self):
        rng = np.random.default_rng(0); files = [f'f{i}' for i in range(30)]; seeds = [1, 2]
        outer = {s: {f: int(rng.integers(5)) for f in files} for s in seeds}
        inner = {(s, k): {f: int(rng.integers(3)) for f in files if outer[s][f] != k} for s in seeds for k in range(5)}
        for f in files:
            r = P.pass1_roles(f, seeds, outer, inner, set(files[:10]), 5)
            self.assertEqual(sum(v[0] == 'inner' for v in r.values()), 2 * 4)
            self.assertEqual(sum(v[0] == 'outer' for v in r.values()), 2 if f in files[:10] else 0)


class Duplicates(unittest.TestCase):
    def test_counts(self):
        a = np.zeros((20, 20), bool); a[0:4, 0:10] = True
        b = np.zeros((20, 20), bool); b[1:5, 0:10] = True          # overlaps a
        c = np.zeros((20, 20), bool); c[10:14, 0:10] = True
        g1 = np.zeros((20, 20), bool); g1[0:5, 0:10] = True         # IoU .8 with a and with b -> matched twice
        g2 = np.zeros((20, 20), bool); g2[10:14, 0:10] = True; g3 = np.zeros((20, 20), bool); g3[10:14, 0:9] = True   # c matched by g2 and g3 of one set
        rl = [rle(m) for m in (a, b, c)]
        qs = [np.asarray(mu.iou([rle(g1), rle(g2), rle(g3)], rl, [0, 0, 0]))]
        d = P.duplicate_counts(rl, qs)
        self.assertEqual(d, dict(predictions=3, overlapping_prediction_pairs=1, gt_matched_by_more_than_one_prediction=1,
                                 predictions_matched_by_more_than_one_gt_of_a_set=1, tp_pairs=4))

    def test_disjoint_and_empty(self):
        a = np.zeros((20, 20), bool); a[0:4, 0:10] = True; c = np.zeros((20, 20), bool); c[10:14, 0:10] = True
        d = P.duplicate_counts([rle(a), rle(c)], [np.asarray(mu.iou([rle(a)], [rle(a), rle(c)], [0, 0]))])
        self.assertEqual(d['overlapping_prediction_pairs'], 0); self.assertEqual(d['gt_matched_by_more_than_one_prediction'], 0)
        self.assertEqual(P.duplicate_counts([], [np.zeros((2, 0))])['tp_pairs'], 0)

    def test_add(self):
        x = dict(predictions=1, overlapping_prediction_pairs=0, gt_matched_by_more_than_one_prediction=2, predictions_matched_by_more_than_one_gt_of_a_set=0, tp_pairs=3)
        self.assertEqual(P.add_counts(x, x, times=3)['tp_pairs'], 12)


class Origin(unittest.TestCase):
    def test_kept_by_origin_and_pairs(self):
        d = P.origin_summary(kept=(0, 2, 3), original=(0, 1), correct_final={0: [1, 0], 2: [1, 1], 3: [0, 0]}, correct_v1={0: [1, 0], 1: [1, 0]})
        self.assertEqual(d, dict(kept_originally_kept=1, kept_originally_rejected=2, dropped_originally_kept=1,
                                 added_correct_pairs=2, added_false_pairs=2, dropped_correct_pairs_under_v1=1, dropped_false_pairs_under_v1=1))


class Gate(unittest.TestCase):
    def result(self, ds, ci, folds=(.001, .001, .001), ok=True):
        c = {P.PRIMARY: dict(partition_deltas=ds, paired_group_CI95=ci)}
        sub = dict(combined425=dict(images=425, contrasts=c), **{f'fold{k}': dict(contrasts={P.PRIMARY: dict(mean_paired_partition_delta=folds[k])}) for k in range(3)})
        return dict(replay_passed=ok, integrity_passed=ok, subsets=sub)

    def test_gate(self):
        self.assertTrue(P.gate(self.result([.001] * 20, [1e-5, .002])))
        self.assertFalse(P.gate(self.result([.001] * 20, [0., .002])))
        self.assertFalse(P.gate(self.result([.001] * 15 + [-.001] * 5, [1e-5, .002])))
        self.assertTrue(P.gate(self.result([.001] * 16 + [-.001] * 4, [1e-5, .002])))
        self.assertFalse(P.gate(self.result([.001] * 20, [1e-5, .002], folds=(.001, 0., .001))))
        self.assertFalse(P.gate(self.result([.001] * 20, [1e-5, .002], ok=False)))


@unittest.skipUnless(REMOTE, 'sklearn fits run only on Kaggle (no model fitting on the Mac)')
class Recipe(unittest.TestCase):
    def test_lr_recipe(self):
        rng = np.random.default_rng(0); X = rng.normal(size=(200, 4)); y = (X[:, 0] + .3 * rng.normal(size=200) > 0).astype(int)
        m = P.fit_selector(X, y); s = P.selector_state(m)
        self.assertEqual(m[1].C, 1.); self.assertEqual(m[1].max_iter, 2000); self.assertEqual(m[1].solver, 'lbfgs'); self.assertEqual(m[1].random_state, 42)
        self.assertIsNone(m[1].class_weight)
        np.testing.assert_allclose(P.project(s, X), m.predict_proba(X)[:, 1], rtol=1e-10, atol=1e-12)


if __name__ == '__main__': unittest.main()
