"""Synthetic tests for the instancing-rule override (order PP1): parser, canonical text and the no-flag pass-through."""
import sys, unittest
from pathlib import Path
CASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CASE / 'frozen'), str(CASE / 'frozen/eval'), str(CASE)]
import numpy as np
from filseg.instancing import SHIP_R3, iter_instances, Rule
import rulespec as RS


def synthetic_map(seed=0):
    """Soft 2048 px map: a few broken, faint line segments plus noise (exercises close, link and min_area)."""
    rng = np.random.default_rng(seed); p = rng.random((2048, 2048), dtype=np.float32) * 0.3
    for y0, x0, n, v in ((300, 200, 600, 0.62), (900, 400, 500, 0.47), (1500, 900, 300, 0.42), (1200, 1500, 40, 0.9)):
        for i in range(n):
            if 250 < i < 290: continue                     # a 40 px gap: link_gap 60 bridges it, 0 does not
            p[y0 + i // 4: y0 + i // 4 + 4, x0 + i: x0 + i + 1] = v
    return p


class TestParse(unittest.TestCase):
    def test_none_and_empty_are_ship_r3(self):
        for t in (None, '', '  '):
            self.assertIs(RS.parse_rule(t), SHIP_R3)

    def test_full_ship_text_equals_ship_r3(self):
        r = RS.parse_rule('thr=0.5,min_area=256,close_px=21,link_gap=60,link_angle=45')
        self.assertEqual(r, SHIP_R3); self.assertTrue(RS.is_ship(r))

    def test_partial_spec_keeps_other_ship_values(self):
        r = RS.parse_rule('thr=0.45')
        self.assertEqual(r, SHIP_R3.with_(thr=0.45)); self.assertFalse(RS.is_ship(r))
        self.assertEqual((r.min_area, r.close_px, r.link_gap, r.link_angle), (256, 21, 60, 45.0))

    def test_types(self):
        r = RS.parse_rule('min_area=128,close_px=31,link_gap=0')
        self.assertIsInstance(r.min_area, int); self.assertIsInstance(r.close_px, int); self.assertIsInstance(r.link_gap, int)
        self.assertIsInstance(RS.parse_rule('thr=0.4').thr, float)

    def test_rejects(self):
        for bad in ('thr', 'thr=', 'foo=1', 'thr=0.4,thr=0.5', 'min_area=12.5', 'thr=1.5', 'thr=0', 'close_px=-1',
                    'link_gap=-5', 'min_area=0', 'thr=abc', 'hys_hi=0.6', 'link_angle=0', 'thr=nan'):
            with self.assertRaises(ValueError, msg=bad): RS.parse_rule(bad)

    def test_canonical_text_round_trip(self):
        for t in ('thr=0.45', 'min_area=384,link_gap=90', 'thr=0.4,close_px=11,link_gap=0', ''):
            r = RS.parse_rule(t); self.assertEqual(RS.parse_rule(RS.rule_text(r)), r)
        self.assertEqual(RS.rule_text(SHIP_R3), 'thr=0.5,min_area=256,close_px=21,link_gap=60,link_angle=45')
        self.assertEqual(RS.rule_text(RS.parse_rule('link_gap=90,thr=0.40')), 'thr=0.4,min_area=256,close_px=21,link_gap=90,link_angle=45')

    def test_short_tag(self):
        self.assertEqual(RS.rule_tag(SHIP_R3), 'r3')
        self.assertEqual(RS.rule_tag(RS.parse_rule('thr=0.45')), 't45')
        self.assertEqual(RS.rule_tag(RS.parse_rule('thr=0.4,min_area=128,close_px=31,link_gap=90')), 't40a128c31g90')


class TestPassThrough(unittest.TestCase):
    def test_no_flag_masks_identical(self):
        p = synthetic_map()
        a = [m.copy() for m in iter_instances(p, SHIP_R3)]; b = [m.copy() for m in iter_instances(p, RS.parse_rule(None))]
        self.assertEqual(len(a), len(b)); self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertGreater(len(a), 0)

    def test_override_changes_masks(self):
        p = synthetic_map()
        a = [m for m in iter_instances(p, SHIP_R3)]; b = [m for m in iter_instances(p, RS.parse_rule('thr=0.40'))]
        self.assertNotEqual(sum(int(m.sum()) for m in a), sum(int(m.sum()) for m in b))

    def test_link_gap_zero_splits_the_gapped_segment(self):
        p = synthetic_map()
        self.assertGreater(len(list(iter_instances(p, RS.parse_rule('link_gap=0')))), len(list(iter_instances(p, SHIP_R3))))


if __name__ == '__main__':
    unittest.main()
