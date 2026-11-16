"""Synthetic specification tests for pc.py (the P-combo helpers), written before pc.py (red log: tdd_red_pc.log)."""
import sys, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen/eval'), str(H), str(H / 'support')]
import pc as PC
import mr as R
import neigh as NB
import mf_features as MF
from context_features import FEATURES as CTX


def contrast(mean, ci=(-.01, .01), pos=20):
    ds = [mean] * 20 if pos == 20 else [abs(mean) + .001] * pos + [-.001] * (20 - pos)
    return dict(delta=mean, paired_group_CI95=list(ci), positive_of20=pos, partition_deltas=ds, mean_paired_partition_delta=mean)


def result(name='pcombo_minus_mfrich', ci=(.001, .005), pos=20, means=(.002, .003, .001, .002, .004)):
    sub = dict(all707=dict(images=707, contrasts={name: contrast(.003, ci, pos)}))
    for k, m in enumerate(means): sub[f'fold{k}'] = dict(images=141, contrasts={name: contrast(m)})
    return dict(replay_passed=True, integrity_passed=True, reproduction707_exact=True, subsets=sub)


class Names(unittest.TestCase):
    def test_arms(self):
        self.assertEqual(PC.ARMS, ['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'pneigh', 'mfrich', 'pcombo'])
        self.assertEqual(PC.PASS2_ARMS, ['inc_pcand_v1', 'fresh_ctrl', 'pneigh', 'mfrich', 'pcombo'])
        self.assertEqual(PC.SELECTOR_ARMS, ['fresh_ctrl', 'pneigh', 'mfrich', 'pcombo'])
        self.assertEqual(PC.TREATMENT, 'pcombo'); self.assertEqual(PC.GATE_CONTRAST, 'pcombo_minus_mfrich')
        self.assertEqual(PC.SECONDARY_CONTRASTS, ['pcombo_minus_pneigh', 'pcombo_minus_inc_pcand_v1'])
        self.assertEqual(PC.ADDED_FEATURES, NB.FEATURES + MF.FEATURES); self.assertEqual(len(PC.ADDED_FEATURES), 14)

    def test_contrasts(self):
        names = [c[0] for c in PC.CONTRASTS]; self.assertEqual(len(set(names)), len(names))
        self.assertEqual(names[0], PC.GATE_CONTRAST)
        for name, a, b in PC.CONTRASTS:
            self.assertEqual(name, f'{a}_minus_{b}'); self.assertIn(a, PC.ARMS); self.assertIn(b, PC.ARMS)
        # every contrast of the three base lanes is recomputed here (exact-replay checks), plus the descriptive ones
        for c in R.REFIT_REP_CONTRASTS + PC.LANE_CONTRASTS['mfrich'] + PC.LANE_CONTRASTS['pneigh'] + PC.SECONDARY_CONTRASTS + PC.DESCRIPTIVE_CONTRASTS:
            self.assertIn(c, names)
        self.assertEqual(PC.LANE_CONTRASTS['mfrich'], ['mfrich_minus_fresh_ctrl', 'mfrich_minus_inc_pcand_v1'] + R.REFIT_REP_CONTRASTS)
        self.assertEqual(PC.LANE_CONTRASTS['pneigh'], ['pneigh_minus_fresh_ctrl', 'pneigh_minus_inc_pcand_v1'] + R.REFIT_REP_CONTRASTS + ['pneigh_minus_inc'])
        self.assertEqual(PC.LANE_ARMS, dict(mfrich=['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'mfrich'], pneigh=['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'pneigh']))
        self.assertEqual(PC.DESCRIPTIVE_CONTRASTS, ['pcombo_minus_fresh_ctrl', 'mfrich_minus_pneigh'])

    def test_feature_names(self):
        o39 = [f'q{i}' for i in range(39)]; base = o39 + CTX
        self.assertEqual(PC.feature_names(o39, CTX, 'fresh_ctrl'), base)
        self.assertEqual(PC.feature_names(o39, CTX, 'pneigh'), base + NB.FEATURES)
        self.assertEqual(PC.feature_names(o39, CTX, 'mfrich'), base + MF.FEATURES)
        self.assertEqual(PC.feature_names(o39, CTX, 'pcombo'), base + NB.FEATURES + MF.FEATURES)
        self.assertEqual(len(PC.feature_names(o39, CTX, 'pcombo')), 68)
        self.assertEqual(PC.feature_names(o39, CTX, 'mfrich'), R.mfrich_feature_names(o39, CTX))
        with self.assertRaises(AssertionError): PC.feature_names(o39, CTX[:-1], 'pcombo')
        with self.assertRaises(AssertionError): PC.feature_names(o39[:-1] + ['neigh_count_64'], CTX, 'pcombo')
        with self.assertRaises(KeyError): PC.feature_names(o39, CTX, 'other')


class Gate(unittest.TestCase):
    def test_gate_reads_the_pcombo_minus_mfrich_contrast(self):
        self.assertTrue(PC.gate707(result()))
        self.assertFalse(PC.gate707(result(name='pcombo_minus_pneigh')))           # another contrast does not count
        self.assertFalse(PC.gate707(result(ci=(0., .005))))
        self.assertFalse(PC.gate707(result(pos=15))); self.assertTrue(PC.gate707(result(pos=16)))
        self.assertFalse(PC.gate707(result(means=(-.001, .003, .001, .002, .004))))
        r = result(); r['reproduction707_exact'] = False; self.assertFalse(PC.gate707(r))

    def test_final_verdict_needs_the_local_replays(self):
        self.assertEqual(PC.final_verdict(True, True), 'confirmed')
        self.assertEqual(PC.final_verdict(False, True), 'Improvement could not be confirmed')
        self.assertTrue(PC.final_verdict(True, False).startswith('no verdict'))
        self.assertTrue(PC.final_verdict(False, False).startswith('no verdict'))


class Origin(unittest.TestCase):
    def test_pairs(self):
        self.assertEqual(PC.ORIGIN_PAIRS, [('fresh_ctrl', 'inc_pcand_v1'), ('pneigh', 'inc_pcand_v1'), ('mfrich', 'inc_pcand_v1'), ('pcombo', 'inc_pcand_v1'),
                                           ('pcombo', 'mfrich'), ('pcombo', 'pneigh')])


if __name__ == '__main__':
    unittest.main()
