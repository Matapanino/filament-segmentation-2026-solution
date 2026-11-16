"""Synthetic specification tests, written before the new adapter."""
import sys, unittest, copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import refine_ops as R
from pnat import repair, classify

class Rules(unittest.TestCase):
    def setUp(self):
        self.old=R.N;R.N=128
    def tearDown(self):R.N=self.old
    def mask(self,y=60,x=40):
        m=np.zeros((128,128),bool);m[y:y+3,x:x+15]=True;return m
    def test_empty(self):
        out,stats=repair([],np.zeros((128,128),np.float32));self.assertEqual(out,[]);self.assertEqual(stats['fallback'],0)
    def test_threshold_inclusive_and_trim(self):
        s=self.mask();p=np.zeros(s.shape,np.float64);p[s]=.7;p[60,40]=0;p[61,55]=.4016711682033202
        out,stats=repair([s],p)
        self.assertFalse(out[0][60,40]);self.assertTrue(out[0][61,55]);self.assertEqual(stats['removed'],1)
    def test_empty_fallback(self):
        s=self.mask();out,stats=repair([s],np.zeros(s.shape,np.float32));self.assertTrue(np.array_equal(s,out[0]));self.assertEqual(stats['fallback'],1)
    def test_detached_island_removed(self):
        s=self.mask();p=s.astype(float);p[75:77,50:52]=1
        out,_=repair([s],p);self.assertTrue(np.array_equal(s,out[0]))
    def test_window_limits_growth(self):
        s=self.mask();out,_=repair([s],np.ones(s.shape))
        g=R.geometry(s);bb=R.window_bbox(g);want=R.full(R.window_native(g,bb),bb)
        self.assertTrue(np.array_equal(out[0],want));self.assertFalse(out[0][0,0])
    def test_other_kept_seed_separation_and_contested(self):
        a=self.mask(x=25);b=self.mask(x=65);p=np.ones(a.shape)
        out,_=repair([a,b],p)
        self.assertFalse((out[0]&out[1]).any());self.assertFalse(out[0][61,52]);self.assertFalse(out[1][61,52])
        self.assertFalse(out[0][61,64]);self.assertFalse(out[1][61,40])
    def test_rejected_candidate_does_not_block(self):
        a=self.mask(x=25);b=self.mask(x=65);p=np.ones(a.shape)
        only,_=repair([a],p);both,_=repair([a,b],p)
        self.assertTrue(only[0][61,65]);self.assertFalse(both[0][61,65])
    def test_clipped_border_and_single_pixel(self):
        s=np.zeros((128,128),bool);s[0,0]=1;out,_=repair([s],np.ones(s.shape));self.assertTrue(out[0][0,0]);self.assertFalse(out[0][-1,-1])
    def test_no_input_mutation(self):
        s=self.mask();p=s.astype(float);before=(s.copy(),p.copy());repair([s],p);self.assertTrue(np.array_equal(s,before[0]));self.assertTrue(np.array_equal(p,before[1]))
    def test_exact_historical_sequence(self):
        rng=np.random.default_rng(42);seeds=[self.mask(40,25),self.mask(55,70)];p=rng.random((128,128))
        gs=[R.geom_from_vector(R.geom_vector(R.geometry(s))) for s in seeds];bbs=[R.window_bbox(g) for g in gs];ws=[R.window_native(g,b) for g,b in zip(gs,bbs)];ctx=R.image_context(seeds,bbs)
        ms,_=R.paste_back(ctx,[(p[b[0]:b[1],b[2]:b[3]]>=.4016711682033202)&w for b,w in zip(bbs,ws)])
        got,_=repair(seeds,p)
        for a,b,bb in zip(got,ms,bbs):self.assertTrue(np.array_equal(a,R.full(b,bb)))

class Decision(unittest.TestCase):
    def fixture(self):
        a=dict(images=425,paired_group_CI95=[.001,.004],partition_deltas=[.003]*20,mean_paired_partition_delta=.003)
        return dict(full_pipeline_replay_passed=True,integrity_passed=True,subsets={'combined425':a,**{f'fold{k}':dict(a,images=n) for k,n in enumerate([142,142,141])}})
    def test_confirmed(self):self.assertEqual(classify(self.fixture())['classification'],'confirmed')
    def test_replay_blocks(self):
        x=self.fixture();x['full_pipeline_replay_passed']=False;self.assertEqual(classify(x)['classification'],'invalid')
    def test_lower_boundary(self):
        x=self.fixture();x['subsets']['combined425']['paired_group_CI95'][0]=0;self.assertEqual(classify(x)['classification'],'inconclusive')
    def test_fold_nonpositive(self):
        x=self.fixture();x['subsets']['fold2']['mean_paired_partition_delta']=0;self.assertEqual(classify(x)['classification'],'inconclusive')
    def test_partition_boundary(self):
        x=self.fixture();x['subsets']['combined425']['partition_deltas']=[.003]*15+[0]*5;self.assertEqual(classify(x)['classification'],'inconclusive')
        x['subsets']['combined425']['partition_deltas']=[.003]*16+[0]*4;self.assertEqual(classify(x)['classification'],'confirmed')
    def test_no_effect_boundary(self):
        x=self.fixture();x['subsets']['combined425']['paired_group_CI95']=[-.001,.001999];self.assertEqual(classify(x)['classification'],'no_effect')
        x['subsets']['combined425']['paired_group_CI95'][1]=.002;self.assertEqual(classify(x)['classification'],'inconclusive')
    def test_missing_or_nonfinite(self):
        x=self.fixture();x['subsets']['fold0']['mean_paired_partition_delta']=float('nan');self.assertEqual(classify(x)['classification'],'invalid')
        self.assertEqual(classify({})['classification'],'invalid')
if __name__=='__main__':unittest.main()
