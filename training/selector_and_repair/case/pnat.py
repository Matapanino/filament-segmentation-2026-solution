"""One fixed prediction-only adapter around byte-identical historical P-nat."""
import numpy as np
import refine_ops as R
TAU=0.4016711682033202

def repair(seeds,p):
    stats=dict(kept=len(seeds),fallback=0,changed=0,added=0,removed=0,seed_pixels=0)
    if not seeds:return [],stats
    assert p.shape==(R.N,R.N) and np.isfinite(p).all()
    gs=[R.geom_from_vector(R.geom_vector(R.geometry(s))) for s in seeds]
    bbs=[R.window_bbox(g) for g in gs];ws=[R.window_native(g,b) for g,b in zip(gs,bbs)]
    ctx=R.image_context(seeds,bbs)
    q=[p[b[0]:b[1],b[2]:b[3]] for b in bbs]
    ms,fb=R.paste_back(ctx,[(v>=TAU)&w for v,w in zip(q,ws)],grow=False)
    checks=R.check_member(ctx,ms,q,TAU,ws,strict=True)
    assert all(checks.values()),('PNAT_INVARIANT',checks)
    out=[R.full(m,b) for m,b in zip(ms,bbs)]
    stats.update(fallback=fb,changed=sum(not np.array_equal(s,m) for s,m in zip(seeds,out)),
        added=sum(int((m&~s).sum()) for s,m in zip(seeds,out)),removed=sum(int((s&~m).sum()) for s,m in zip(seeds,out)),seed_pixels=sum(int(s.sum()) for s in seeds))
    return out,stats

def classify(r):
    try:
        assert r['full_pipeline_replay_passed'] is True and r['integrity_passed'] is True
        for name,n in [('combined425',425),('fold0',142),('fold1',142),('fold2',141)]:
            a=r['subsets'][name]
            assert a['images']==n and len(a['partition_deltas'])==20
            assert np.isfinite([*a['paired_group_CI95'],*a['partition_deltas'],a['mean_paired_partition_delta']]).all()
        a=r['subsets']['combined425']
        checks=dict(ci95_lower_positive=a['paired_group_CI95'][0]>0,positive_partitions_at_least16=sum(d>0 for d in a['partition_deltas'])>=16,
                    all_fold_means_positive=all(r['subsets'][f'fold{k}']['mean_paired_partition_delta']>0 for k in (0,1,2)))
        confirmed=all(checks.values());label='confirmed' if confirmed else ('no_effect' if a['paired_group_CI95'][1]<.002 else 'inconclusive')
        return dict(classification=label,confirmed=confirmed,checks=checks,improvement_confirmed=confirmed,adopt=False,test_proposal_required=confirmed,submission_authorized=False)
    except (KeyError,AssertionError,TypeError,ValueError):
        return dict(classification='invalid',confirmed=False,adopt=False,reason='Replay, integrity or completeness failed',submission_authorized=False)
