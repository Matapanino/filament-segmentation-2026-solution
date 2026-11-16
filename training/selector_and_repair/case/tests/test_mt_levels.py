"""chain2m (order MT1): the arm layout with level sets (mt.configure(weights, levelsets)).

No level sets: chain2's layout exactly. With level sets: w25 stays the plain SHIP_R3 arm (the anchor, no levels); every other
weight gets one arm per level set, named w<NN><tag> (tag '' for [0.5]); contrasts <arm>_minus_pcombo, <arm>_minus_w25, and
<arm>_minus_w<NN> when the plain arm of that weight is in the call.
"""
import sys, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import mt as M


class Layout(unittest.TestCase):
    def tearDown(self): M.configure(M.DEFAULT_WEIGHTS)

    def test_no_levels_is_chain2(self):
        a = M.layout([.25, .55]); b = M.layout([.25, .55], None)
        self.assertEqual(a, b); self.assertEqual(a['NEW_VARIANTS'], ['w25', 'w55']); self.assertEqual(a['LEVELS'], dict(w25=None, w55=None))
        self.assertEqual([c[0] for c in a['CONTRASTS']][:3], ['w25_minus_pcombo', 'w55_minus_pcombo', 'w55_minus_w25'])

    def test_replay_and_levels(self):
        x = M.layout([.25, .55], [[.5], [.5, .4]])
        self.assertEqual(x['NEW_VARIANTS'], ['w25', 'w55', 'w55l50l40'])
        self.assertEqual(x['LEVELS'], dict(w25=None, w55=[.5], w55l50l40=[.5, .4]))
        self.assertEqual(x['WEIGHT'], dict(w25=.25, w55=.55, w55l50l40=.55)); self.assertEqual(x['BASE_OF']['w55l50l40'], 'w55')
        self.assertEqual(x['ARM_CONTRASTS'], ['w25_minus_pcombo', 'w55_minus_pcombo', 'w55l50l40_minus_pcombo', 'w55_minus_w25', 'w55l50l40_minus_w25',
                                              'w55l50l40_minus_w55'])
        self.assertEqual(x['SELECTOR_ARMS'], ['pcombo', 'w25', 'w55', 'w55l50l40']); self.assertEqual(x['VARIANT']['w55l50l40'], 'w55l50l40')
        self.assertTrue(all('_minus_' not in a for a in x['NEW_VARIANTS']))

    def test_second_call(self):
        x = M.layout([.25, .55], [[.5, .4, .3]])
        self.assertEqual(x['NEW_VARIANTS'], ['w25', 'w55l50l40l30']); self.assertNotIn('w55l50l40l30_minus_w55', x['ARM_CONTRASTS'])

    def test_configure_sets_globals(self):
        M.configure([.25, .55], [[.5, .4]]); self.assertEqual(M.LEVELS['w55l50l40'], [.5, .4]); self.assertEqual(M.WEIGHT['w55l50l40'], .55)
        M.configure([.25, .55]); self.assertEqual(M.LEVELS, dict(w25=None, w55=None))

    def test_feature_names_with_level(self):
        self.assertEqual(M.arm_feature_names(['a', 'b'], 'w55'), ['a', 'b'])
        M.configure([.25, .55], [[.5], [.5, .4]])
        self.assertEqual(M.arm_feature_names(['a', 'b'], 'w55'), ['a', 'b']); self.assertEqual(M.arm_feature_names(['a', 'b'], 'w55l50l40'), ['a', 'b', 'level'])
        self.assertEqual(M.arm_feature_names(['a', 'b'], 'pcombo'), ['a', 'b']); self.assertEqual(M.arm_feature_names(['a', 'b'], 'w25'), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
