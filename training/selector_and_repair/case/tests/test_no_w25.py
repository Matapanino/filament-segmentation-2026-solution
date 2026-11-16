"""chainF (order INT1) tests, written before the code: the opt-in no-w25 layout (chain_inputs.json 'no_w25': true). Without it
every call needs weight 0.25 (chain2's anchor arm) exactly as before; with it the weights need not include 0.25, there is no
w25 arm and no <arm>_minus_w25 contrast, and the worker's descriptive gate contrast is not required."""
import sys, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import mt as M
import worker as W


class Layout(unittest.TestCase):
    def tearDown(self): W.configure(M.DEFAULT_WEIGHTS)

    def test_default_unchanged(self):
        with self.assertRaises(ValueError): M.parse_weights('0.55,0.70')
        lay = M.layout([.25, .55, .7])
        self.assertEqual(lay['NEW_VARIANTS'], ['w25', 'w55', 'w70'])
        self.assertEqual(lay['ARM_CONTRASTS'], ['w25_minus_pcombo', 'w55_minus_pcombo', 'w70_minus_pcombo', 'w55_minus_w25', 'w70_minus_w25'])

    def test_no_w25(self):
        self.assertEqual(M.parse_weights('0.70,0.55', require_base=False), [.55, .7])
        lay = M.layout([.55, .7], require_base=False)
        self.assertEqual(lay['NEW_VARIANTS'], ['w55', 'w70']); self.assertEqual(lay['ARM_CONTRASTS'], ['w55_minus_pcombo', 'w70_minus_pcombo'])
        self.assertEqual(lay['SELECTOR_ARMS'], ['pcombo', 'w55', 'w70'])
        lv = M.layout([.55], [[.5], [.5, .4]], require_base=False)
        self.assertEqual(lv['NEW_VARIANTS'], ['w55', 'w55l50l40']); self.assertIn('w55l50l40_minus_w55', lv['ARM_CONTRASTS'])
        self.assertEqual(M.layout([.25, .55], require_base=False)['NEW_VARIANTS'], ['w25', 'w55'])        # 0.25 still allowed

    def test_worker_configure(self):
        W.configure([.55, .7], None, require_base=False); self.assertEqual(W.ARMS, M.FIXED_ARMS + ['w55', 'w70'])
        self.assertNotIn(M.GATE_CONTRAST, [c[0] for c in W.CONTRASTS])
        W.configure([.25, .55]); self.assertIn(M.GATE_CONTRAST, [c[0] for c in W.CONTRASTS])
        with self.assertRaises(ValueError): W.configure([.55, .7])


if __name__ == '__main__': unittest.main()
