"""P-nat: fixed historical within-window re-threshold of the kept candidates.

Copied without change from the cross-validated adapter (detect_20261111/pnat_crossfold/pnat.py,
SHA256 dcfb2ce0f2c8a15c6c541bc935811a185d7fc4f5fc69df5e15eda36d3c37d038) and its byte-identical
w6_refine geometry module `refine_ops.py`. Order (pnat_crossfold/protocol.json): SHIP_R3 ->
original-mask LR39 features -> saved keep -> P-nat on kept seeds only; never rescore.
The probability map is the same flat float32 blend that generated the candidates.
"""
import numpy as np
from . import refine_ops as R
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
