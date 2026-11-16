"""Exact submitted 39-feature function; no fitting code."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

def features_for_mask(mask, probs, raw):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    region = regionprops(crop)[0]
    distance = distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1][crop > 0]
    area = len(ys)
    perimeter = region.perimeter
    result = dict(log_area=np.log1p(area), log_perimeter=np.log1p(perimeter),
                  compactness=4*np.pi*area/max(perimeter**2, 1),
                  extent=region.extent, solidity=region.solidity, eccentricity=region.eccentricity,
                  log_major_axis=np.log1p(region.axis_major_length),
                  log_minor_axis=np.log1p(region.axis_minor_length),
                  bbox_width=(x1-x0)/2048, bbox_height=(y1-y0)/2048,
                  width_mean=float(distance.mean()), width_max=float(distance.max()),
                  centroid_radius=np.hypot(xs.mean()-1023.5, ys.mean()-1023.5)/1024,
                  radial_extent=float(np.hypot(xs-1023.5, ys-1023.5).max()/1024),
                  edge_fraction=float(((xs == 0) | (xs == 2047) | (ys == 0) | (ys == 2047)).mean()))
    for name, prob in probs.items():
        p = prob[ys, xs]
        result.update({f"{name}_{k}": float(v) for k, v in dict(
            mean=p.mean(), std=p.std(), q10=np.quantile(p, .1), median=np.median(p),
            q90=np.quantile(p, .9), high_fraction=(p >= .8).mean()).items()})
    diff = np.abs(probs["full"][ys, xs] - probs["local"][ys, xs])
    result.update(disagreement_mean=float(diff.mean()), disagreement_q90=float(np.quantile(diff, .9)))
    a, b, c, d = max(0, y0-8), min(2048, y1+8), max(0, x0-8), min(2048, x1+8)
    mask_crop = mask[a:b, c:d].astype(np.uint8)
    ring = (cv2.dilate(mask_crop, np.ones((15, 15), np.uint8)) > 0) & (mask_crop == 0)
    inside = raw[ys, xs].astype(float)/255
    outside = raw[a:b, c:d][ring].astype(float)/255
    outside_mean = float(outside.mean()) if len(outside) else float(inside.mean())
    result.update(image_mean=float(inside.mean()), image_std=float(inside.std()),
                  ring_mean=outside_mean, ring_contrast=outside_mean-float(inside.mean()))
    assert all(np.isfinite(v) for v in result.values())
    return result
