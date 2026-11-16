"""chain2u (order CH3) tests, written before the code: the U-Net map swap in the CV tree.

mt: the set-spec parser ('a' or 'a+b'), the U-Net level average (uint8 levels / (255 n) in float32; n = 1 is the stored
rule z / 255 bit for bit). worker: with U-Net sets in the call, the weight arms read the swapped full/local maps of the
image's base fold (fold-k archive for a fold-k image), stored arms never do; a swapped set equal to the stored maps gives
the stored candidates and features exactly; two sets average per image; map_change records the U-Net change; without U-Net
sets nothing differs from chain2 (same keys, same maps)."""
import importlib.util, sys, tempfile, types, unittest
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parents[1]
try: HAS_SKIMAGE = importlib.util.find_spec('skimage') is not None
except ValueError: HAS_SKIMAGE = False
if not HAS_SKIMAGE:
    sk = types.ModuleType('skimage'); sk.measure = types.ModuleType('skimage.measure'); sk.measure.regionprops = None
    sys.modules.setdefault('skimage', sk); sys.modules.setdefault('skimage.measure', sk.measure)
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H), str(H / 'tests')]
import worker as W
import mt as M
from test_worker_synthetic import fake_features, quality_names


class Parser(unittest.TestCase):
    def test_set_spec(self):
        self.assertEqual(M.parse_set_spec('d4'), ['d4'])
        self.assertEqual(M.parse_set_spec('d4+s43'), ['d4', 's43'])
        for bad in ('', 'a++b', 'a+a', '../x', 'A B', '+a', 'a/b'):
            with self.assertRaises(ValueError, msg=bad): M.parse_set_spec(bad)


class Average(unittest.TestCase):
    def test_one_set_is_the_stored_rule(self):
        r = np.random.default_rng(0); z = r.integers(0, 256, size=(64, 64)).astype(np.uint8)
        got = M.mean_unet([z]); self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.array_equal(got, z.astype(np.float32) / 255))                     # worker.branch_maps' stored rule
        self.assertEqual(got.tobytes(), (z.astype(np.float32) / 255).tobytes())

    def test_two_sets(self):
        r = np.random.default_rng(1); a, b = (r.integers(0, 256, size=(64, 64)).astype(np.uint8) for _ in range(2))
        got = M.mean_unet([a, b])
        self.assertTrue(np.array_equal(got, (a.astype(np.uint32) + b).astype(np.float32) / np.float32(510)))   # no requantisation
        self.assertTrue(np.allclose(got, (a / 255. + b / 255.) / 2, rtol=0, atol=1e-6))
        self.assertEqual(float(M.mean_unet([np.full((2, 2), 255, np.uint8)] * 3).max()), 1.0)   # no overflow
        with self.assertRaises(AssertionError): M.mean_unet([a.astype(np.uint16)])


class Swap(unittest.TestCase):
    def setUp(self):
        self.saved = W.features_for_mask
        if not HAS_SKIMAGE: W.features_for_mask = fake_features
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        n = 2048; full = np.zeros((n, n), np.uint8); local = full.copy(); mf = np.zeros((n, n), np.uint16)
        full[400:412, 300:900] = 200; local[400:412, 300:900] = 210; mf[400:412, 300:900] = 420
        full[1200:1225, 1000:1500] = 230; local[1200:1225, 1000:1500] = 230; mf[1200:1225, 1000:1500] = 460
        self.full, self.local, self.mf = full, local, mf
        # swapped U-Net maps: a new object in both views, the first object weakened in the full view
        self.full2 = full.copy(); self.full2[1600:1612, 200:700] = 220; self.full2[400:412, 300:900] = 120
        self.local2 = local.copy(); self.local2[1600:1612, 200:700] = 230
        import cv2
        self.files = ['20110101000000Bh', '20110301000000Ch']; self.folds = {self.files[0]: 0, self.files[1]: 3}
        raw = np.full((n, n), 120, np.uint8); raw[400:418, 300:900] = 70
        for f in self.files: cv2.imwrite(str(t / (f + '.jpeg')), raw)
        poly = lambda y0, y1, x0, x1: dict(segmentation=[[x0, y0, x1, y0, x1, y1, x0, y1]])
        ann = [types.SimpleNamespace(image_id='a', annotations=[poly(400, 412, 300, 900), poly(1600, 1612, 200, 700)])]
        den = dict(full=255, local=255, m2f_public=510); maps = dict(full=full, local=local, m2f_public=mf)
        W.G.clear(); W.G.update(folds=self.folds, images=t, annotations={f: ann for f in self.files}, order39=quality_names(),
                               models={(7, k): dict(feature_order=quality_names(), scaler_mean=[0.] * 39, scaler_scale=[1.] * 39, coef=[[0.] * 39], intercept=[5.], threshold=.5) for k in range(5)},
                               outer={7: {self.files[0]: 0, self.files[1]: 1}},
                               spec={k: {v: dict(denominator=d) for v, d in den.items()} for k in (0, 3)},
                               z={k: {v: {f: maps[v] for f in self.files} for v in den} for k in (0, 3)},
                               znew={'new': {k: [{f: mf for f in self.files}] for k in (0, 3)}})

    def tearDown(self):
        W.features_for_mask = self.saved; self.tmp.cleanup(); W.G.clear()

    def swap(self, full_sets=None, local_sets=None):
        if full_sets is not None: W.G['znew']['unet_full'] = full_sets
        if local_sets is not None: W.G['znew']['unet_local'] = local_sets

    def test_without_sets_nothing_changes(self):
        f = self.files[1]; a = W.branch_maps(f); b = W.branch_maps(f, 'w25')
        for v in ('full', 'local'): self.assertTrue(np.array_equal(a[v], b[v]))
        _, x = W.image_pass_variant('w25', f)
        self.assertEqual(set(x['map_change']), {'mf_mean_abs', 'mf_max_abs', 'mf_flip_at_half', 'mf_ge_half_stored', 'mf_ge_half_variant', 'mixture_mean_abs', 'mixture_flip_at_half'})

    def test_arms_read_the_fold_archive_of_the_image(self):
        # fold 0 archive = stored maps, fold 3 archive = the changed maps: image 0 (fold 0) is unchanged, image 1 (fold 3) changes
        self.swap({0: [{f: self.full for f in self.files}], 3: [{f: self.full2 for f in self.files}]},
                  {0: [{f: self.local for f in self.files}], 3: [{f: self.local2 for f in self.files}]})
        a0, b0 = W.branch_maps(self.files[0]), W.branch_maps(self.files[0], 'w55')
        a1, b1 = W.branch_maps(self.files[1]), W.branch_maps(self.files[1], 'w55')
        for v in ('full', 'local'): self.assertTrue(np.array_equal(a0[v], b0[v]))
        self.assertTrue(np.array_equal(b1['full'], self.full2.astype(np.float32) / 255)); self.assertTrue(np.array_equal(b1['local'], self.local2.astype(np.float32) / 255))
        self.assertTrue(np.array_equal(a1['full'], self.full.astype(np.float32) / 255))          # the stored arms never read a swapped set
        _, x = W.image_pass_variant('w55', self.files[1])
        self.assertGreater(x['map_change']['full_mean_abs'], 0.); self.assertGreater(x['map_change']['local_mean_abs'], 0.)
        _, x0 = W.image_pass_variant('w55', self.files[0]); self.assertEqual(x0['map_change']['full_mean_abs'], 0.)
        ms, p, rows = W.candidate_rows(self.files[1], 'w25')
        self.assertTrue(np.array_equal(p, M.mix_w(b1, .25)))                                     # the blend reads the swapped maps
        _, ps, _ = W.candidate_rows(self.files[1], 'stored'); self.assertFalse(np.array_equal(p, ps))

    def test_only_one_view_swapped(self):
        self.swap(local_sets={k: [{f: self.local2 for f in self.files}] for k in (0, 3)})
        a, b = W.branch_maps(self.files[1]), W.branch_maps(self.files[1], 'w25')
        self.assertTrue(np.array_equal(a['full'], b['full'])); self.assertFalse(np.array_equal(a['local'], b['local']))
        _, x = W.image_pass_variant('w25', self.files[1])
        self.assertNotIn('full_mean_abs', x['map_change']); self.assertGreater(x['map_change']['local_mean_abs'], 0.)   # keys only for a swapped view

    def test_stored_equal_set_reproduces_the_stored_candidates(self):
        self.swap({k: [{f: self.full for f in self.files}] for k in (0, 3)}, {k: [{f: self.local for f in self.files}] for k in (0, 3)})
        f = self.files[1]; _, r = W.image_pass(f); _, x = W.image_pass_variant('w25', f)
        for k in ('n', 'rle', 'hashes', 'features', 'raw_iou', 'n_gt_sets'): self.assertEqual(r[k], x[k])
        self.assertEqual(x['map_change']['full_mean_abs'], 0.); self.assertEqual(x['map_change']['mixture_mean_abs'], 0.)

    def test_two_sets_average(self):
        self.swap({k: [{f: self.full for f in self.files}, {f: self.full2 for f in self.files}] for k in (0, 3)})
        b = W.branch_maps(self.files[1], 'w25')
        self.assertTrue(np.array_equal(b['full'], M.mean_unet([self.full, self.full2])))

    def test_bank_text_names_the_sets(self):
        self.assertNotIn('U-Net', W.bank_text('w25'))                                             # chain2's text when nothing is swapped
        try:
            W.UNET_SETS.update(full=['d4'], local=['d4'])
            self.assertIn('U-Net full = d4', W.bank_text('w25')); self.assertEqual(W.bank_text('pcombo'), W.PCOMBO_BANK)
        finally: W.UNET_SETS.clear()


if __name__ == '__main__': unittest.main()
