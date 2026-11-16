"""Synthetic tests of pt.py (order PC1), written before pt.py: the 9-tau table builder (uniform whole-kept-set repair at every
grid tau, scored per candidate and as six PQ counts), the global-tau chooser and its pooling over outer-training images, the
v1 training rows, the arms / contrasts / cohorts (incl. 'no_fold2'), and summarize (same format and the same bootstrap
draws as chain2's worker.summarize)."""
import sys, types, unittest
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import pcand as C
import pnat
import rules
import pt as T

N = 2048


def enc(m): return mu.encode(np.asfortranarray(np.asarray(m, np.uint8)))


def scene():
    """One bar object whose probability decays away from its core: lower thresholds grow the repaired mask toward a wide GT.
    A second, separate object that matches its GT at every threshold."""
    p = np.zeros((N, N), np.float32)
    p[500:510, 300:800] = .9                       # core (the seed)
    p[510:514, 300:800] = .33                      # weak band, kept only at tau <= 0.30
    p[496:500, 300:800] = .33
    p[1200:1220, 1000:1400] = .95
    seed1 = np.zeros((N, N), bool); seed1[500:510, 300:800] = True
    seed2 = np.zeros((N, N), bool); seed2[1200:1220, 1000:1400] = True
    gt1 = np.zeros((N, N), bool); gt1[496:514, 300:800] = True          # the GT includes the weak band
    gt2 = seed2.copy()
    return p, [seed1, seed2], [[enc(gt1), enc(gt2)]]


class TauTable(unittest.TestCase):
    def test_shapes_and_pnat_equivalence(self):
        p, seeds, gts = scene(); t = T.tau_table(seeds, p, gts)
        self.assertEqual(np.asarray(t['scores']).shape, (2, 9)); self.assertEqual(np.asarray(t['counts']).shape, (9, 6))
        self.assertTrue(t['tau_path_equal'])
        ref, _ = pnat.repair(seeds, p)                                     # column TAU = the P-nat repair
        q = [np.asarray(mu.iou(g, [enc(m) for m in ref], [0] * 2)).reshape(len(g), 2) for g in gts]
        self.assertEqual(list(np.asarray(t['scores'])[:, 4]), list(C.candidate_scores(q, 2)))
        self.assertEqual(list(t['counts'][4]), list(sum(rules.counts(x) for x in q)))

    def test_every_column_is_the_uniform_repair_at_that_tau(self):
        p, seeds, gts = scene(); t = T.tau_table(seeds, p, gts)
        for j, tau in enumerate(C.GRID):
            out, _ = C.repair_taus(seeds, p, [tau] * 2)
            q = [np.asarray(mu.iou(g, [enc(m) for m in out], [0] * 2)).reshape(len(g), 2) for g in gts]
            self.assertEqual(list(np.asarray(t['scores'])[:, j]), list(C.candidate_scores(q, 2)))
            self.assertEqual(list(t['counts'][j]), list(sum(rules.counts(x) for x in q)))

    def test_low_tau_grows_toward_the_wide_gt(self):
        p, seeds, gts = scene(); s = np.asarray(T.tau_table(seeds, p, gts)['scores'])
        self.assertEqual(C.choose_target(s[0]), 0.30)                     # the weak band enters at tau <= 0.30; equal scores there -> nearest TAU
        self.assertGreater(s[0][0], s[0][4])
        self.assertTrue(np.all(s[1] > .5))                                # second object matched at every tau

    def test_empty_kept_set(self):
        p, _, gts = scene(); t = T.tau_table([], p, gts)
        self.assertEqual(t['scores'], [])
        for c in t['counts']: self.assertEqual(list(c), [0., 0., 0., 2., 0., 0.])   # both GT unmatched, no prediction
        self.assertTrue(t['tau_path_equal'])

    def test_no_gt(self):
        p, seeds, _ = scene(); t = T.tau_table(seeds, p, [])
        self.assertTrue(np.all(np.asarray(t['scores']) == 0)); self.assertTrue(all(list(c) == [0.] * 6 for c in t['counts']))


class GlobalTau(unittest.TestCase):
    def c(self, iou, tp, fp, fn): return [iou, tp, fp, fn, 0., 0.]

    def test_choose_global_argmax(self):
        pooled = [self.c(1, 2, 2, 2)] * 9; pooled = list(pooled); pooled[6] = self.c(1.6, 2, 2, 2)
        self.assertEqual(T.choose_global(pooled), C.GRID[6])

    def test_choose_global_ties(self):
        self.assertEqual(T.choose_global([self.c(1, 2, 2, 2)] * 9), C.TAU)               # all equal: TAU
        pooled = [self.c(1, 2, 2, 2)] * 9; pooled = list(pooled); pooled[0] = pooled[8] = self.c(2, 2, 2, 2)
        self.assertEqual(T.choose_global(pooled), 0.60)                                  # nearer to TAU wins
        pooled = [self.c(1, 2, 2, 2)] * 9; pooled = list(pooled); pooled[3] = pooled[5] = self.c(2, 2, 2, 2)
        self.assertEqual(T.choose_global(pooled), 0.45)                                  # 0.45 - TAU < TAU - 0.35
        self.assertIs(type(T.choose_global(pooled)), float)
        with self.assertRaises(AssertionError): T.choose_global([self.c(1, 2, 2, 2)] * 8)

    def test_pool_uses_each_training_image_pattern_of_the_fit(self):
        table = {'a': [[self.c(1, 1, 0, 0)] * 9, [self.c(0, 0, 0, 1)] * 9], 'b': [[self.c(2, 2, 0, 0)] * 9], 'q': [[self.c(9, 9, 0, 0)] * 9]}
        pattern_of = {'a': {'7:0': 1, '7:1': 0}, 'b': {'7:0': 0, '7:1': 0}, 'q': {'7:0': 0, '7:1': 0}}
        pooled = T.pool_counts(table, pattern_of, '7:0', ['a', 'b'])
        self.assertEqual(len(pooled), 9); self.assertEqual(list(pooled[0]), [2., 2., 0., 1., 0., 0.])   # 'q' (query) is not pooled
        self.assertEqual(list(T.pool_counts(table, pattern_of, '7:1', ['a'])[3]), [1., 1., 0., 0., 0., 0.])

    def test_chooser_on_a_real_table(self):
        p, seeds, gts = scene(); t = T.tau_table(seeds, p, gts)
        pooled = T.pool_counts({'x': [t['counts']]}, {'x': {'k': 0}}, 'k', ['x'])
        best = T.choose_global(pooled)
        self.assertEqual(best, max(C.GRID, key=lambda tau: (rules.pq(pooled[C.GRID.index(tau)]), -abs(tau - C.TAU))))
        self.assertLessEqual(best, 0.30)


class Rows(unittest.TestCase):
    def test_training_rows_row_rule_and_exclusion(self):
        A = {'a': dict(n=3, features=[{'x': 1.}, {'x': 2.}, {'x': 3.}], patterns=[[0, 2], [1]], pattern_of={'5:1': 0, '5:2': 1},
                       scores=[[[0] * 4 + [.7] + [0] * 4, [0] * 9], [[.8] + [0] * 8]]),
             'b': dict(n=1, features=[{'x': 9.}], patterns=[[]], pattern_of={'5:1': 0, '5:2': 0}, scores=[[]])}
        X, y, ex = T.training_rows(A, ['a', 'b'], '5:1', ['x'])
        self.assertEqual(X.tolist(), [[1.]]); self.assertEqual(y.tolist(), [C.TAU]); self.assertEqual(ex, 1)   # never-matched row 2 excluded
        X, y, ex = T.training_rows(A, ['a'], '5:2', ['x'])
        self.assertEqual(X.tolist(), [[2.]]); self.assertEqual(y.tolist(), [0.20]); self.assertEqual(ex, 0)
        self.assertEqual(T.all_rows(A, 'a', ['x']).tolist(), [[1.], [2.], [3.]]); self.assertEqual(T.all_rows(A, 'b', ['x']).shape, (1, 1))


class Keep(unittest.TestCase):
    def test_keep_of_is_the_logistic_rule(self):
        m = dict(scaler_mean=[1., 0.], scaler_scale=[2., 1.], coef=[[3., -1.]], intercept=[.5], threshold=.6)
        x = np.array([[1., 0.], [3., 0.], [1., 4.]]); keep, prob = T.keep_of(m, x)
        z = (x - [1., 0.]) / [2., 1.]; ref = 1 / (1 + np.exp(-(z @ [3., -1.] + .5)))
        self.assertTrue(np.allclose(prob, ref)); self.assertEqual(keep.tolist(), (ref >= .6).tolist())


class Layout(unittest.TestCase):
    def test_arms_and_contrasts(self):
        self.assertEqual(T.arms_of(['w55', 'w40']), ['pcombo', 'w55_raw', 'w55_frozen', 'w55_refit', 'w55_gtau', 'w40_raw', 'w40_frozen', 'w40_refit', 'w40_gtau'])
        names = [c[0] for c in T.contrasts_of(['w55'])]
        for n in ('w55_refit_minus_w55_frozen', 'w55_gtau_minus_w55_frozen', 'w55_refit_minus_pcombo', 'w55_gtau_minus_pcombo', 'w55_frozen_minus_pcombo'):
            self.assertIn(n, names)
        self.assertEqual(names[:2], ['w55_refit_minus_w55_frozen', 'w55_gtau_minus_w55_frozen'])
        for n, a, b in T.contrasts_of(['w55', 'w40']): self.assertEqual(n, f'{a}_minus_{b}')

    def test_cohorts_add_no_fold2(self):
        files = [f'f{i}' for i in range(10)]; folds = {f: i % 5 for i, f in enumerate(files)}
        q = [f for f in files if folds[f] in (0, 1, 2)]; c = T.cohorts(files, folds, q)
        self.assertEqual(c['no_fold2'], [f for f in files if folds[f] != 2]); self.assertEqual(c['all707'], files)
        for k in range(5): self.assertIn(f'fold{k}', c)


class Summary(unittest.TestCase):
    def setUp(self):
        from make_time_folds import temporal_groups
        self.files = sorted(f'2011{m:02d}{d:02d}120000Bh' for m in range(1, 13) for d in (3, 17)); self.groups = temporal_groups(self.files, 24)
        self.seeds = [11, 12, 13]; self.cfg = dict(partition_seeds=self.seeds, bootstrap=dict(seed=5, draws=200))
        rng = np.random.default_rng(0); self.arms = ['pcombo', 'w55_frozen', 'w55_refit', 'w55_gtau', 'w55_raw']
        self.per = {s: {a: [[float(rng.integers(5, 9)) * .7, float(rng.integers(5, 9)), float(rng.integers(0, 4)), float(rng.integers(0, 4)), 0., 0.] for _ in self.files]
                        for a in self.arms} for s in self.seeds}

    def test_matches_chain2_worker_summarize(self):
        import worker as W
        con = [('w55_minus_pcombo', 'w55', 'pcombo')]
        per2 = {s: dict(pcombo=self.per[s]['pcombo'], w55=self.per[s]['w55_frozen']) for s in self.seeds}
        saved = W.CONTRASTS; W.CONTRASTS = con
        try: ref = W.summarize(per2, self.files, self.files, self.groups, self.cfg, ['pcombo', 'w55'])
        finally: W.CONTRASTS = saved
        mine = T.summarize(self.per, self.files, self.files, self.groups, self.cfg, self.arms, T.contrasts_of(['w55']))
        a, b = mine['contrasts']['w55_frozen_minus_pcombo'], ref['contrasts']['w55_minus_pcombo']
        self.assertEqual(a, b)                                  # same draws although more arms are summarized
        self.assertEqual(mine['scores']['w55_frozen'], ref['scores']['w55'])
        self.assertEqual(mine['partition_pq']['w55_frozen'], ref['partition_pq']['w55'])

    def test_delta_and_positives(self):
        m = T.summarize(self.per, self.files, self.files, self.groups, self.cfg, self.arms, T.contrasts_of(['w55']))
        c = m['contrasts']['w55_refit_minus_w55_frozen']
        self.assertAlmostEqual(c['delta'], m['scores']['w55_refit'] - m['scores']['w55_frozen'], places=15)
        ds = np.asarray(m['partition_pq']['w55_refit']) - m['partition_pq']['w55_frozen']
        self.assertEqual(c['positive_of20'], int((ds > 0).sum())); self.assertEqual(len(c['paired_group_CI95']), 2)

    def test_report_row(self):
        folds = {f: i % 5 for i, f in enumerate(self.files)}; q = [f for f in self.files if folds[f] in (0, 1, 2)]
        S = {n: T.summarize(self.per, fs, self.files, self.groups, self.cfg, self.arms, T.contrasts_of(['w55'])) for n, fs in T.cohorts(self.files, folds, q).items()}
        r = T.report_row(S, 'w55_refit_minus_w55_frozen')
        self.assertEqual(set(r), {'all707', 'no_fold2'})
        self.assertEqual(sorted(r['all707']['fold_means']), [f'fold{k}' for k in range(5)])
        self.assertEqual(sorted(r['no_fold2']['fold_means']), ['fold0', 'fold1', 'fold3', 'fold4'])
        self.assertEqual(r['no_fold2']['delta'], S['no_fold2']['contrasts']['w55_refit_minus_w55_frozen']['delta'])
        self.assertEqual(r['all707']['positive_of20'], S['all707']['contrasts']['w55_refit_minus_w55_frozen']['positive_of20'])


if __name__ == '__main__': unittest.main()
