"""mfrich: eight MF-derived features of one candidate, from the candidate mask and the per-image MF probability map only.

The MF map is the one the six existing MF context features read (context_features.py: maps['m2f_public']); in CV the single
OOF MF map of the image's base fold (uint16 / 510), at test the three-fold MF mean ((MF0 + MF1 + MF2) / 1530). No annotation,
fitted parameter or other map enters.

  mfr_q10, mfr_q50, mfr_q90  10th / 50th / 90th percentile of MF over the candidate's pixels (numpy linear interpolation)
  mfr_frac_ge50              fraction of candidate pixels with MF >= 0.5
  mfr_blob_iou               IoU of the candidate and the union of the MF >= 0.5 connected components (8-connectivity, whole
                             image) that overlap it; 0 if none overlaps
  mfr_ring_area_ratio        MF >= 0.5 pixels in the 16 px ring / candidate area; ring = square (33 x 33) dilation of the
                             candidate minus the candidate, clipped at the image border (the ring of context_features.py)
  mfr_ring_max               max MF in that ring; 0 if the ring is empty
  mfr_blob_count             number of distinct MF >= 0.5 components that overlap the candidate
"""
import cv2
import numpy as np

THRESHOLD = .5
RING = 16
CONNECTIVITY = 8
FEATURES = ['mfr_q10', 'mfr_q50', 'mfr_q90', 'mfr_frac_ge50', 'mfr_blob_iou', 'mfr_ring_area_ratio', 'mfr_ring_max', 'mfr_blob_count']


def components(mf):
    """Connected components of MF >= 0.5 over the whole image: (label image, area per label; label 0 = background)."""
    high = (np.asarray(mf) >= THRESHOLD).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(high, connectivity=CONNECTIVITY, ltype=cv2.CV_32S)
    return labels, stats[:, cv2.CC_STAT_AREA].astype(np.int64)


def candidate_features(mask, mf, labels, areas):
    mask = np.asarray(mask, bool); mf = np.asarray(mf)
    assert mask.ndim == 2 and mask.shape == mf.shape == labels.shape and mask.any()
    inside = mf[mask]; v = inside.astype(np.float64)
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
    q10, q50, q90 = np.percentile(v, [10, 50, 90])
    high = inside >= THRESHOLD; inter = int(high.sum()); area = int(mask.sum())
    lab = labels[mask]; assert int((lab > 0).sum()) == inter
    ids = np.unique(lab[lab > 0])
    iou = inter / (area + int(areas[ids].sum()) - inter) if len(ids) else 0.
    ys, xs = np.nonzero(mask); h, w = mask.shape
    a, b, c, d = max(0, ys.min() - RING), min(h, ys.max() + 1 + RING), max(0, xs.min() - RING), min(w, xs.max() + 1 + RING)
    crop = mask[a:b, c:d]
    grown = cv2.dilate(crop.astype(np.uint8), np.ones((2 * RING + 1, 2 * RING + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    ring = mf[a:b, c:d][grown & ~crop]
    out = dict(mfr_q10=float(q10), mfr_q50=float(q50), mfr_q90=float(q90), mfr_frac_ge50=inter / area, mfr_blob_iou=float(iou),
               mfr_ring_area_ratio=int((ring >= THRESHOLD).sum()) / area, mfr_ring_max=float(ring.max()) if ring.size else 0., mfr_blob_count=float(len(ids)))
    assert list(out) == FEATURES and np.isfinite(list(out.values())).all()
    return out


def image_features(masks, mf):
    """The 8 features of every candidate of one image (components computed once per image)."""
    if not len(masks): return []
    labels, areas = components(mf)
    return [candidate_features(m, mf, labels, areas) for m in masks]
