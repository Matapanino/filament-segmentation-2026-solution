"""chain2r (order PP1): the instancing-rule override in worker.py, on the synthetic images of test_worker_synthetic.
No rule: every arm uses SHIP_R3 and the bank text is chain2's. A rule: only the weight arms' candidates use it (pcombo /
stored keep SHIP_R3), and the arm's bank text names it."""
import sys, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / 'tests'))
import numpy as np
import test_worker_synthetic as T
W = T.W; M = T.M
from filseg.instancing import iter_instances, SHIP_R3
import rulespec as RS


class RuleOverride(unittest.TestCase):
    setUp = T.Images.setUp

    def tearDown(self):
        W.set_rule(None); T.Images.tearDown(self)

    def test_default_is_ship(self):
        W.set_rule(None); self.assertIs(W.RULE, SHIP_R3)
        self.assertEqual(W.bank_text('w55'), W.bank_text_chain2('w55'))

    def test_spec_round_trip(self):
        spec = RS.rule_spec(RS.parse_rule('thr=0.4,link_gap=90'))
        self.assertEqual(spec, dict(text='thr=0.4,min_area=256,close_px=21,link_gap=90,link_angle=45', fields=dict(thr=0.4, min_area=256, close_px=21, link_gap=90, link_angle=45.0)))
        W.set_rule(spec); self.assertEqual(W.RULE, SHIP_R3.with_(thr=0.4, link_gap=90))
        with self.assertRaises(AssertionError): W.set_rule(dict(text='thr=0.45', fields=dict(thr=0.4)))

    def test_ship_spec_is_ship(self):
        W.set_rule(RS.rule_spec(SHIP_R3)); self.assertEqual(W.RULE, SHIP_R3)

    def test_weight_arm_uses_rule_stored_keeps_ship(self):
        f = self.files[1]; rule = RS.parse_rule('thr=0.3,min_area=64,close_px=0,link_gap=0')
        W.set_rule(None); m0, p0, _ = W.candidate_rows(f, 'w55'); s0, _, _ = W.candidate_rows(f, 'stored')
        W.set_rule(RS.rule_spec(rule)); m1, p1, _ = W.candidate_rows(f, 'w55'); s1, _, _ = W.candidate_rows(f, 'stored')
        self.assertTrue(np.array_equal(p0, p1))
        self.assertEqual([W.packed(m) for m in m1], [W.packed(np.asarray(m, bool)) for m in iter_instances(p1, rule)])
        self.assertNotEqual([W.packed(m) for m in m0], [W.packed(m) for m in m1])
        self.assertEqual([W.packed(m) for m in s0], [W.packed(m) for m in s1])
        self.assertIn('thr=0.3,min_area=64,close_px=0,link_gap=0,link_angle=45', W.bank_text('w55'))
        self.assertNotIn('thr=', W.bank_text('pcombo'))


if __name__ == '__main__':
    unittest.main()
