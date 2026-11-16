"""mfvar: five view-disagreement features of one candidate from the 8 D4 per-view MF maps of one MF model.

Input: views = (8, H, W) per-view MF maps (Stage A: each inverse-transformed 256-px view upsampled to 2048 and quantised
like the stored maps, uint16 / 510; floats in [0, 1] are used as they are). The candidate is the one built from the
MF-TTA mixture (the 8-view mean); no annotation, fitted parameter or other map enters.

  mfv_pixel_std_mean  mean over the candidate's pixels of the per-pixel population standard deviation across the 8 views
  mfv_view_vote       fraction of the 8 views whose MF >= 0.5 blob overlaps the candidate with IoU > 0.5; the blob of a
                      view = the union of that view's MF >= 0.5 connected components (8-connectivity, whole image) that
                      overlap the candidate, i.e. mf_features' mfr_blob_iou computed on the view (0 if none overlaps)
  mfv_view_mean_min   min over the 8 views of the candidate's mean MF
  mfv_view_mean_max   max over the 8 views of the candidate's mean MF
  mfv_blob_iou_std    population standard deviation over the 8 views of that blob IoU

Test-time rule (preregistered): the five features are computed per MF model (MF0, MF1, MF2, each from its own 8 views) on
the test candidate and averaged over the three models (test_image_features).
"""
import numpy as np
from . import mfrich_features as MF

FEATURES = ['mfv_pixel_std_mean', 'mfv_view_vote', 'mfv_view_mean_min', 'mfv_view_mean_max', 'mfv_blob_iou_std']
N_VIEWS = 8
VOTE_IOU = .5
DENOMINATOR = 510


def as_prob(views):
    v = np.asarray(views)
    if np.issubdtype(v.dtype, np.integer):
        assert int(v.max(initial=0)) <= DENOMINATOR and int(v.min(initial=0)) >= 0; v = v.astype(np.float32) / np.float32(DENOMINATOR)
    assert v.ndim == 3 and v.shape[0] == N_VIEWS, ('VIEWS_SHAPE', v.shape)
    return v


def blob_iou(mask, high, labels, areas):
    """mf_features.candidate_features' blob IoU (same formula) of one candidate on one view."""
    inside = high[mask]; inter = int(inside.sum()); area = int(mask.sum())
    lab = labels[mask]; ids = np.unique(lab[lab > 0])
    return inter / (area + int(areas[ids].sum()) - inter) if len(ids) else 0.


def image_features(masks, views):
    """The five features of every candidate of one image (8 component labelings computed once per image)."""
    if not len(masks): return []
    v = as_prob(views); comps = [MF.components(v[i]) for i in range(N_VIEWS)]; highs = [v[i] >= MF.THRESHOLD for i in range(N_VIEWS)]
    out = []
    for m in masks:
        m = np.asarray(m, bool); assert m.shape == v.shape[1:] and m.any()
        px = v[:, m].astype(np.float64); assert np.isfinite(px).all() and ((px >= 0) & (px <= 1)).all()
        means = px.mean(1)
        ious = np.asarray([blob_iou(m, highs[i], *comps[i]) for i in range(N_VIEWS)], float)
        r = dict(mfv_pixel_std_mean=float(px.std(0).mean()), mfv_view_vote=float((ious > VOTE_IOU).mean()), mfv_view_mean_min=float(means.min()),
                 mfv_view_mean_max=float(means.max()), mfv_blob_iou_std=float(ious.std()))
        assert list(r) == FEATURES and np.isfinite(list(r.values())).all(); out.append(r)
    return out


def test_image_features(masks, views_by_model):
    """Test-time rule: the features of each of the 3 MF models' views, averaged over the models."""
    assert len(views_by_model) == 3
    per = [image_features(masks, v) for v in views_by_model]
    return [{n: float(np.mean([p[j][n] for p in per])) for n in FEATURES} for j in range(len(masks))]
