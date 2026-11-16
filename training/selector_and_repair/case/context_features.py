"""Fixed raw and predicted-map statistics. No annotations or fitted parameters."""
import cv2
import numpy as np
RADII=(16,64,128)
FEATURES=[f'context_{r}_{s}' for r in RADII for s in ('mean','std','dark_fraction')]+[
    'mf_mean','full_mf_abs_mean','local_mf_abs_mean','branches_range_mean',
    'branches_all_high_fraction','branches_one_high_fraction']

def context_features(mask,maps,raw):
    mask=np.asarray(mask,bool)
    assert mask.ndim==2 and mask.any() and raw.shape==mask.shape
    assert raw.dtype==np.uint8
    ys,xs=np.where(mask);h,w=mask.shape
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    inside=raw[mask].astype(np.float64)/255.;median=float(np.median(inside));out={}
    for radius in RADII:
        a,b,c,d=max(0,y0-radius),min(h,y1+radius),max(0,x0-radius),min(w,x1+radius)
        crop=mask[a:b,c:d]
        grown=cv2.dilate(crop.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        ring=grown&~crop
        values=raw[a:b,c:d][ring].astype(np.float64)/255.
        if not len(values):values=inside
        out.update({f'context_{radius}_mean':float(values.mean()),f'context_{radius}_std':float(values.std()),
                    f'context_{radius}_dark_fraction':float((values<median).mean())})
    p=np.stack([np.asarray(maps[k])[mask] for k in ('full','local','m2f_public')]).astype(np.float64)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    votes=(p>=.5).sum(axis=0)
    out.update(mf_mean=float(p[2].mean()),full_mf_abs_mean=float(np.abs(p[0]-p[2]).mean()),
        local_mf_abs_mean=float(np.abs(p[1]-p[2]).mean()),branches_range_mean=float(np.ptp(p,axis=0).mean()),
        branches_all_high_fraction=float((votes==3).mean()),branches_one_high_fraction=float((votes==1).mean()))
    assert list(out)==FEATURES and np.isfinite(list(out.values())).all()
    return out
