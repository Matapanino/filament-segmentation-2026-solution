"""chainF (order INT1) tests of the combined arm (chain2s, ported after F0) with chainF's other flags: --no-w25 keeps cmb
(it needs w55, not w25); --pcand-refit may name cmb (single level); U-Net swapped views reach cmb's maps like every
weight arm (branch_maps), while its mixture stays the MF map."""
import sys, unittest
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / 'tests'))
import test_worker_synthetic as T
W = T.W; M = T.M


class WithChainF(unittest.TestCase):
    def tearDown(self): W.set_pcand([]); W.configure(M.DEFAULT_WEIGHTS)

    def test_no_w25_and_refit(self):
        x = M.layout([.55], None, dict(thr=.4), require_base=False)
        self.assertEqual(x['NEW_VARIANTS'], ['w55', 'cmb']); self.assertEqual(x['ARM_CONTRASTS'], ['w55_minus_pcombo', 'cmb_minus_pcombo'])
        with self.assertRaises(ValueError): M.layout([.7], None, dict(thr=.4), require_base=False)        # w55 is the combined arm's reference
        W.configure([.55, .7], None, dict(thr=.4), require_base=False); W.set_pcand(['w55', 'cmb']); self.assertEqual(W.PCAND_ARMS, ['w55', 'cmb'])
        W.configure([.25, .55]); self.assertIsNone(M.COMBINED)
        with self.assertRaises(AssertionError): W.set_pcand(['cmb'])                                      # no cmb arm without the option


class UnetReachesCmb(T.Images):
    def tearDown(self): W.configure(M.DEFAULT_WEIGHTS); super().tearDown()

    def test_unet(self):
        f = self.files[1]; W.configure([.25, .55], None, dict(thr=.4))
        cmb = np.zeros((2048, 2048), np.uint16); cmb[400:412, 300:900] = 430
        full2 = np.zeros((2048, 2048), np.uint8); full2[1600:1612, 200:700] = 220
        W.G['znew']['cmb'] = {k: [{x: cmb for x in self.files}] for k in (0, 3)}
        W.G['znew']['unet_full'] = {k: [{x: full2 for x in self.files}] for k in (0, 3)}
        b = W.branch_maps(f, 'cmb')
        self.assertTrue(np.array_equal(b['full'], full2.astype(np.float32) / np.float32(255)))
        self.assertTrue(np.array_equal(M.mixture(b, 'cmb'), cmb.astype(np.float32) / np.float32(510)))


if __name__ == '__main__': unittest.main()
