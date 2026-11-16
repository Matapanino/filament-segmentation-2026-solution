import unittest
import numpy as np
from context_features import FEATURES, context_features

class FeaturesTest(unittest.TestCase):
    def fixture(self):
        m=np.zeros((33,35),bool);m[15:18,16:19]=True
        return m,np.full(m.shape,128,np.uint8),{k:np.full(m.shape,.7) for k in ('full','local','m2f_public')}
    def test_fixed_finite_order_and_constant_context(self):
        m,r,p=self.fixture();f=context_features(m,p,r)
        self.assertEqual(list(f),FEATURES);self.assertEqual(len(f),15)
        self.assertTrue(np.isfinite(list(f.values())).all())
        for radius in (16,64,128):
            self.assertAlmostEqual(f[f'context_{radius}_mean'],128/255)
            self.assertAlmostEqual(f[f'context_{radius}_std'],0)
            self.assertEqual(f[f'context_{radius}_dark_fraction'],0)
        self.assertEqual(f['branches_range_mean'],0)
        self.assertEqual(f['branches_all_high_fraction'],1)
    def test_dark_ring_and_candidate_exclusion(self):
        m,r,p=self.fixture();r[:]=0;r[m]=255;f=context_features(m,p,r)
        for radius in (16,64,128):
            self.assertEqual(f[f'context_{radius}_mean'],0)
            self.assertEqual(f[f'context_{radius}_dark_fraction'],1)
    def test_pairwise_and_votes(self):
        m,r,p=self.fixture();p['full'][:]=.8;p['local'][:]=.2;p['m2f_public'][:]=.4
        f=context_features(m,p,r)
        self.assertAlmostEqual(f['full_mf_abs_mean'],.4)
        self.assertAlmostEqual(f['local_mf_abs_mean'],.2)
        self.assertAlmostEqual(f['branches_range_mean'],.6)
        self.assertEqual(f['branches_all_high_fraction'],0)
        self.assertEqual(f['branches_one_high_fraction'],1)
    def test_empty_ring_fallback(self):
        m,r,p=self.fixture();m[:]=True;f=context_features(m,p,r)
        self.assertAlmostEqual(f['context_16_mean'],128/255)
        self.assertEqual(f['context_16_dark_fraction'],0)
    def test_empty_mask_rejected(self):
        m,r,p=self.fixture();m[:]=False
        with self.assertRaises(AssertionError):context_features(m,p,r)
    def test_clipped_edge_no_wrap(self):
        m,r,p=self.fixture();m[:]=False;m[0,0]=True;r[:]=255;r[-1,-1]=0
        self.assertEqual(context_features(m,p,r)['context_16_mean'],1)
    def test_bad_map_rejected(self):
        m,r,p=self.fixture();p['full'][m]=np.nan
        with self.assertRaises(AssertionError):context_features(m,p,r)
    def test_gt_free_signature(self):
        import inspect
        self.assertEqual(list(inspect.signature(context_features).parameters),['mask','maps','raw'])
