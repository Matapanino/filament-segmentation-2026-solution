"""Six prediction-only neighbour-context features of every unrepaired SHIP_R3 candidate of one image (the P-neigh selector).

A copy of the cross-validated code (detect_20261112/pneigh/neigh.py) and of the pmerge geometry it calls, with the same
arithmetic. Input: the image's candidates (pixel-disjoint full-size masks, candidate order) and the mixture map p (the
candidate blend). No annotation.

  distance(a, b)  minimum Euclidean distance between pixel centres (boundary pixels, cKDTree, exact integer d2, no cap)
  closest points  the minimising pixel pair, ties to the lexicographically smallest (y_a, x_a, y_b, x_b), pair in candidate order
  axis            the R3 linker's principal axis of the whole candidate (postproc._axis_of over all its pixels)
  bridge          the 1-px cv2.line (LINE_8, endpoints included) between the closest points, in the two bboxes' crop (pad 4)
  nearest         the other candidate with the smallest distance (ties: lowest candidate index)

  neigh_count_64               number of other candidates with distance <= 64
  neigh_count_128              number of other candidates with distance <= 128
  neigh_gap_nearest            min(distance to the nearest other candidate, 512); 512 when there is none
  neigh_axis_abs_cos_nearest   |cos| between the candidate's axis and its nearest neighbour's; 0.0 when there is none
  neigh_area_share_aligned_128 area / (area + total area of the candidates with distance <= 128 and |cos| >= 0.7)
  neigh_bridge_p_min_nearest   min of p along the bridge to the nearest neighbour; 1.0 when there is none
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .postproc import _axis_of, EPS

FEATURES = ['neigh_count_64', 'neigh_count_128', 'neigh_gap_nearest', 'neigh_axis_abs_cos_nearest',
            'neigh_area_share_aligned_128', 'neigh_bridge_p_min_nearest']
R_NEAR, R_FAR, GAP_CAP, ALIGN_COS = 64, 128, 512, .7
NO_NEIGHBOUR_COS, NO_NEIGHBOUR_BRIDGE = 0.0, 1.0
NO_CAP = 2900                         # > the image diagonal (2047 * sqrt(2) = 2894.9): every pair has a distance
SIZE = 2048
CROP_PAD = 4


def _tight(full):
    M = np.asarray(full, bool); ys = np.flatnonzero(M.any(1)); xs = np.flatnonzero(M.any(0))
    if not len(ys):raise ValueError('Empty candidate')
    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
    return np.ascontiguousarray(M[y0:y1, x0:x1]), (y0, y1, x0, x1)


def _boundary(M, bbox):
    P = np.pad(M, 1); inner = P[1:-1, 1:-1] & P[:-2, 1:-1] & P[2:, 1:-1] & P[1:-1, :-2] & P[1:-1, 2:]
    ys, xs = np.nonzero(M & ~inner)
    return np.stack([ys + bbox[0], xs + bbox[2]], 1).astype(np.int64)


def _closest_pair(Ba, Bb, max_gap=NO_CAP):
    _, idx = cKDTree(Ba.astype(np.float64)).query(Bb.astype(np.float64), k=1, distance_upper_bound=max_gap + 0.5)
    ok = idx < len(Ba)
    if not ok.any():return None
    nb = Bb[ok]; na = Ba[idx[ok]]; d2 = ((nb - na) ** 2).sum(1)
    m = int(d2.min())
    if m > max_gap * max_gap:return None
    best = None
    for yb, xb in nb[d2 == m]:
        da = (Ba[:, 0] - yb) ** 2 + (Ba[:, 1] - xb) ** 2
        row = min((int(Ba[i, 0]), int(Ba[i, 1]), int(yb), int(xb)) for i in np.flatnonzero(da == m))
        best = row if best is None else min(best, row)
    return (m, (best[0], best[1]), (best[2], best[3]))


def _crop_of(bboxes):
    y0 = max(0, min(b[0] for b in bboxes) - CROP_PAD); y1 = min(SIZE, max(b[1] for b in bboxes) + CROP_PAD)
    x0 = max(0, min(b[2] for b in bboxes) - CROP_PAD); x1 = min(SIZE, max(b[3] for b in bboxes) + CROP_PAD)
    return (y0, y1, x0, x1)


def _centre_line(pa, pb, crop):
    y0, y1, x0, x1 = crop; c = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.line(c, (pa[1] - x0, pa[0] - y0), (pb[1] - x0, pb[0] - y0), 1, 1)
    return c.astype(bool)


def _axis(M, bbox):
    ys, xs = np.nonzero(M); xs = xs + bbox[2]; ys = ys + bbox[0]
    return _axis_of(np.stack([xs, ys]).astype(np.float64))[0]


def abs_cos(u, v):
    u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
    return float(min(1.0, abs(float(u @ v)) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), EPS)))


def neighbour_features(masks, p):
    """One dict of the 6 FEATURES per candidate (candidate order). masks: full-size bool masks; p: the mixture map."""
    n = len(masks)
    if not n:return []
    objs = [_tight(m) for m in masks]
    bd = [_boundary(M, bb) for M, bb in objs]
    axes = [_axis(M, bb) for M, bb in objs]
    area = [float(np.asarray(M, bool).sum()) for M, _ in objs]
    pair = {}
    for a in range(n):
        for b in range(a + 1, n):
            r = _closest_pair(bd[a], bd[b])
            if r is None:raise ValueError('Candidate pair without a distance')
            pair[a, b] = r
    d2 = lambda i, j: pair[min(i, j), max(i, j)][0]
    out = []
    for a in range(n):
        others = [b for b in range(n) if b != a]
        f = dict(neigh_count_64=float(sum(d2(a, b) <= R_NEAR * R_NEAR for b in others)),
                 neigh_count_128=float(sum(d2(a, b) <= R_FAR * R_FAR for b in others)))
        if others:
            nb = min(others, key=lambda b: (d2(a, b), b)); lo, hi = min(a, nb), max(a, nb); m2, pa, pb = pair[lo, hi]
            crop = _crop_of([objs[lo][1], objs[hi][1]]); L = _centre_line(pa, pb, crop); y0, y1, x0, x1 = crop
            f.update(neigh_gap_nearest=float(min(np.sqrt(m2), GAP_CAP)), neigh_axis_abs_cos_nearest=abs_cos(axes[a], axes[nb]))
            aligned = sum(area[b] for b in others if d2(a, b) <= R_FAR * R_FAR and abs_cos(axes[a], axes[b]) >= ALIGN_COS)
            f.update(neigh_area_share_aligned_128=float(area[a] / (area[a] + aligned)),
                     neigh_bridge_p_min_nearest=float(np.asarray(p)[y0:y1, x0:x1][L].astype(np.float64).min()))
        else:
            f.update(neigh_gap_nearest=float(GAP_CAP), neigh_axis_abs_cos_nearest=NO_NEIGHBOUR_COS, neigh_area_share_aligned_128=1.0,
                     neigh_bridge_p_min_nearest=NO_NEIGHBOUR_BRIDGE)
        f = {k: f[k] for k in FEATURES}
        if not np.isfinite(list(f.values())).all():raise ValueError('Nonfinite neighbour feature')
        out.append(f)
    return out
