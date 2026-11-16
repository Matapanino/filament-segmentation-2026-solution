"""Synthetic specification tests for mr.py (the mfrich helpers), written before mr.py (red log: tdd_red_mr.log)."""
import sys, unittest
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H), str(H / 'support')]
import mr as R
import mf_features as MF
from context_features import FEATURES as CTX


def contrast(delta, ci, pos, mean):
    ds = [mean] * 20 if pos == 20 else [abs(mean) + .001] * pos + [-.001] * (20 - pos)
    return dict(delta=delta, paired_group_CI95=list(ci), positive_of20=pos, partition_deltas=ds, mean_paired_partition_delta=mean)


def result(ci=(.001, .005), pos=20, means=(.002, .003, .001, .002, .004), images=707, ok=True, name='mfrich_minus_fresh_ctrl'):
    sub = dict(all707=dict(images=images, contrasts={name: contrast(.003, ci, pos, .003)}))
    for k, m in enumerate(means): sub[f'fold{k}'] = dict(images=141, contrasts={name: contrast(m, [-.01, .01], 20, m)})
    return dict(replay_passed=ok, integrity_passed=True, reproduction707_exact=True, subsets=sub)


class Names(unittest.TestCase):
    def test_arms_and_contrasts(self):
        self.assertEqual(R.ARMS, ['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'mfrich'])
        self.assertEqual(R.PASS2_ARMS, ['inc_pcand_v1', 'fresh_ctrl', 'mfrich'])
        self.assertEqual(R.SELECTOR_ARMS, ['fresh_ctrl', 'mfrich']); self.assertEqual(R.TREATMENT, 'mfrich')
        self.assertEqual(R.GATE_CONTRAST, 'mfrich_minus_fresh_ctrl'); self.assertEqual(R.SECONDARY_CONTRAST, 'mfrich_minus_inc_pcand_v1')
        names = [c[0] for c in R.CONTRASTS]; self.assertEqual(len(set(names)), len(names))
        for name, a, b in R.CONTRASTS:
            self.assertEqual(name, f'{a}_minus_{b}'); self.assertIn(a, R.ARMS); self.assertIn(b, R.ARMS)
        for c in [R.GATE_CONTRAST, R.SECONDARY_CONTRAST] + R.REFIT_REP_CONTRASTS: self.assertIn(c, names)
        self.assertEqual(R.REFIT_REP_CONTRASTS, ['fresh_ctrl_minus_inc_pcand_v1', 'inc_pcand_v1_minus_inc_pnat', 'inc_pcand_v1_minus_inc',
                                                 'inc_pnat_minus_inc', 'fresh_ctrl_minus_inc'])
        self.assertEqual(R.REFIT_REP_ARMS, ['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl'])
        self.assertEqual(R.CI_LEVELS, dict(paired_group_CI95=[.025, .975]))
        self.assertEqual(R.MF_CONTEXT, ['mf_mean', 'full_mf_abs_mean', 'local_mf_abs_mean', 'branches_range_mean', 'branches_all_high_fraction', 'branches_one_high_fraction'])

    def test_feature_names(self):
        o39 = [f'q{i}' for i in range(39)]
        n = R.mfrich_feature_names(o39, CTX)
        self.assertEqual(n, o39 + CTX + MF.FEATURES); self.assertEqual(len(n), 62)
        with self.assertRaises(AssertionError): R.mfrich_feature_names(o39, CTX[:-1])
        with self.assertRaises(AssertionError): R.mfrich_feature_names(o39[:-1], CTX)
        with self.assertRaises(AssertionError): R.mfrich_feature_names(o39[:-1] + ['mfr_q10'], CTX)


class Cohorts(unittest.TestCase):
    def test_cohorts(self):
        files = [f'f{i:02d}' for i in range(20)]; folds = {f: i % 5 for i, f in enumerate(files)}
        q425 = [f for f in files if folds[f] in (0, 1, 2)]
        c = R.cohorts(files, folds, q425)
        self.assertEqual(c['all707'], files); self.assertEqual(c['combined282'], [f for f in files if folds[f] in (3, 4)])
        self.assertEqual(c['combined425'], q425); self.assertEqual(c['new284'], [f for f in q425 if folds[f] in (0, 1)])
        for k in range(5): self.assertEqual(c[f'fold{k}'], [f for f in files if folds[f] == k])
        with self.assertRaises(AssertionError): R.cohorts(files, folds, q425[:-1])


class Gate(unittest.TestCase):
    def test_pass(self): self.assertTrue(R.gate707(result()))

    def test_each_condition_blocks(self):
        self.assertFalse(R.gate707(result(ci=(0., .005))))                      # CI95 lower must be strictly > 0
        self.assertFalse(R.gate707(result(ci=(-.0001, .005))))
        self.assertFalse(R.gate707(result(pos=15)))                             # >= 16/20
        self.assertTrue(R.gate707(result(pos=16)))
        self.assertFalse(R.gate707(result(means=(.002, .003, 0., .002, .004))))  # every base fold mean > 0
        self.assertFalse(R.gate707(result(means=(.002, .003, .001, .002, -.001))))
        self.assertFalse(R.gate707(result(images=706)))                         # the gate cohort is the 707
        self.assertFalse(R.gate707(result(ok=False)))
        r = result(); r['reproduction707_exact'] = False; self.assertFalse(R.gate707(r))
        r = result(); r['integrity_passed'] = False; self.assertFalse(R.gate707(r))
        r = result(); del r['subsets']['fold3']; self.assertFalse(R.gate707(r))
        r = result(); r['subsets']['all707']['contrasts']['mfrich_minus_fresh_ctrl']['paired_group_CI95'] = [float('nan'), 1.]; self.assertFalse(R.gate707(r))

    def test_statistics_met(self):
        r = result(name='mfrich_minus_inc_pcand_v1')
        self.assertTrue(R.statistics_met(r, 'all707', 'mfrich_minus_inc_pcand_v1'))
        self.assertFalse(R.statistics_met(result(ci=(-.001, .004), name='x'), 'all707', 'x')); self.assertFalse(R.statistics_met(r, 'all707', 'missing'))


class Helpers(unittest.TestCase):
    def test_intervals(self):
        d = np.linspace(-1, 1, 2001); i = R.intervals(d)
        self.assertEqual(set(i), {'paired_group_CI95'}); np.testing.assert_allclose(i['paired_group_CI95'], [-.95, .95], atol=1e-12)

    def test_keep_agreement_and_compare(self):
        self.assertEqual(R.keep_agreement((0, 1, 3), (1, 2), 5), dict(both=1, first_only=2, second_only=1, neither=1))
        files = ['a', 'b']; per = {1: {'fresh_ctrl': [[1, 2, 3, 4, 5, 6], [0] * 6]}}
        ref = dict(files=files, per_partition_counts={'1': {'fresh_ctrl': [[1, 2, 3, 4, 5, 6], [0] * 6]}})
        self.assertTrue(R.compare_counts(per, files, 'fresh_ctrl', ref, [1])['exact'])
        ref['per_partition_counts']['1']['fresh_ctrl'][1][0] = 1e-12; self.assertFalse(R.compare_counts(per, files, 'fresh_ctrl', ref, [1])['exact'])
        keeps = {'a': {1: dict(fresh_ctrl=(0, 2))}, 'b': {1: dict(fresh_ctrl=())}}
        self.assertTrue(R.compare_keeps(keeps, {'a': {'1': dict(fresh_ctrl=[0, 2])}, 'b': {'1': dict(fresh_ctrl=[])}}, files, [1], 'fresh_ctrl')['exact'])
        self.assertFalse(R.compare_keeps(keeps, {'a': {'1': dict(fresh_ctrl=[0])}, 'b': {'1': dict(fresh_ctrl=[])}}, files, [1], 'fresh_ctrl')['exact'])

    def test_quantiles_and_distribution(self):
        q = R.quantiles(np.arange(101.))
        self.assertEqual(list(q), ['min', 'q05', 'q25', 'median', 'q75', 'q95', 'max', 'mean', 'n'])
        self.assertEqual((q['q05'], q['median'], q['q95'], q['n']), (5., 50., 95., 101))
        rows = [{n: float(i) for n in MF.FEATURES} for i in range(11)]
        d = R.feature_distribution(rows, MF.FEATURES); self.assertEqual(list(d), MF.FEATURES); self.assertEqual(d['mfr_q50']['median'], 5.)
        z = R.feature_distribution([dict(mfr_blob_count=0.), dict(mfr_blob_count=2.)], ['mfr_blob_count'])
        self.assertEqual(z['mfr_blob_count']['fraction_zero'], .5)

    def test_collinearity(self):
        rng = np.random.default_rng(0); old = rng.random((200, 3)); new = np.stack([2 * old[:, 0] + 1, rng.random(200)], 1)
        c = R.collinearity(new, old, ['n0', 'n1'], ['o0', 'o1', 'o2'], corr_with=['o0', 'o2'])
        self.assertAlmostEqual(c['n0']['r2_on_existing'], 1., places=10); self.assertLess(c['n1']['r2_on_existing'], .1)
        self.assertAlmostEqual(c['n0']['corr']['o0'], 1., places=10); self.assertEqual(set(c['n0']['corr']), {'o0', 'o2'})
        self.assertEqual(c['n0']['max_abs_corr_feature'], 'o0')
        const = np.stack([np.ones(200), rng.random(200)], 1)                    # a constant column has no correlation (reported as 0)
        c = R.collinearity(const, old, ['k', 'r'], ['o0', 'o1', 'o2'], corr_with=['o0'])
        self.assertEqual(c['k']['corr']['o0'], 0.)

    def test_shift_table(self):
        cv = {n: dict(q05=0., median=1., q95=2.) for n in MF.FEATURES}; test = {n: dict(q05=0., median=1.5, q95=2.) for n in MF.FEATURES}
        coef = {n: .5 for n in MF.FEATURES}; scale = {n: 2. for n in MF.FEATURES}
        t = R.shift_table(cv, test, coef, scale)
        self.assertEqual(list(t), MF.FEATURES)
        self.assertEqual(t['mfr_q10']['median_shift'], .5); self.assertEqual(t['mfr_q10']['implied_logit_shift_at_median'], .5 * .5 / 2.)
        self.assertEqual(t['mfr_q10']['cv'], dict(q05=0., median=1., q95=2.)); self.assertEqual(t['mfr_q10']['test'], dict(q05=0., median=1.5, q95=2.))


if __name__ == '__main__':
    unittest.main()
