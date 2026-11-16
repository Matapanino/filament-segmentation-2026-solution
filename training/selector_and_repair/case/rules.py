"""Small pure rules for matched comparisons; no model fit or data transport."""
import numpy as np

def counts(q):
    q=np.asarray(q,float);assert q.ndim==2 and np.isfinite(q).all()
    hit=q>.5;overlap=q>0
    return np.array([q[hit].sum(),hit.sum(),(hit.sum(0)==0).sum(),(hit.sum(1)==0).sum(),(overlap.sum(1)>1).sum(),(overlap.sum(0)>1).sum()],float)

def pq(c):
    denominator=c[1]+.5*(c[2]+c[3]);return float(c[0]/denominator) if denominator else 0.

def assert_split(train,query,groups):
    a,b=set(train),set(query);assert not a&b and a and b
    assert a|b<=set(f for g in groups for f in g)
    for g in groups:assert not(set(g)&a and set(g)&b),'TEMPORAL_GROUP_LEAKAGE'

def choose_threshold(sweep):return max(sweep,key=lambda r:(r['pq'],-r['threshold']))['threshold']

def adoption_gate(result):
    try:
        if result['replay_passed'] is not True:return False
        for contrast in ('context_minus_refit39','context_minus_incumbent'):
            r=result['subsets']['combined425']['contrasts'][contrast];ds=r['partition_deltas'];ci=r['paired_group_CI95']
            if len(ds)!=20 or not np.isfinite(ds+ci).all() or ci[0]<=0 or sum(d>0 for d in ds)<16:return False
            for k in range(3):
                d=result['subsets'][f'fold{k}']['contrasts'][contrast]['mean_paired_partition_delta']
                if not np.isfinite(d) or d<=0:return False
        return True
    except (KeyError,TypeError,ValueError):return False
