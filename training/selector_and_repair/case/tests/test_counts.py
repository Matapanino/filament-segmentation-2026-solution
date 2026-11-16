import sys,unittest
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(H/'support'))
from core import image_counts,tune_threshold
from count_plan import count_coefficients,evaluate,threshold_sweep
class CountParityTest(unittest.TestCase):
    def fixture(self):
        return {'a':dict(candidate_ids=[0,1],annotations=[dict(n_gt=2,iou=[[.6,.2],[.1,.7]])]),
                'b':dict(candidate_ids=[2],annotations=[dict(n_gt=1,iou=[[.5]]),dict(n_gt=1,iou=[[.8]])])}
    def test_every_subset_exact(self):
        r=self.fixture();p,n=count_coefficients(r,3)
        for k in range(8):
            keep=np.array([(k>>i)&1 for i in range(3)],bool)
            np.testing.assert_array_equal(evaluate(p,n,r,['a','b'],keep),np.asarray(list(image_counts(r,['a','b'],keep).values())))
    def test_grid_tie_and_pooled_parity(self):
        r=self.fixture();p,n=count_coefficients(r,3);v=np.array([.2,.7,.4]);grid=[round(i*.025,6) for i in range(41)]
        a,sa=threshold_sweep(p,n,r,['a','b'],v,grid);b,sb=tune_threshold(r,['a','b'],v)
        self.assertEqual(a,b);self.assertEqual(sa,sb)
    def test_threshold_cannot_read_query_gt(self):
        r=self.fixture();p,n=count_coefficients(r,3);v=np.array([.2,.7,np.nan])
        a,_=threshold_sweep(p,n,r,['a'],v,[0,.3,.8])
        r['b']['annotations'][0]['iou']=[[1.0]];p,n=count_coefficients(r,3)
        b,_=threshold_sweep(p,n,r,['a'],v,[0,.3,.8]);self.assertEqual(a,b)
