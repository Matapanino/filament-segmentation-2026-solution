"""P-neigh: 6 fixed, prediction-only neighbour-context features of every unrepaired SHIP_R3 candidate of one image.

Input: the image's candidates (pixel-disjoint full-size masks, candidate order) and the mixture map p. No annotation.
Geometry is pmerge's, reused unchanged:
  distance(a, b)  = minimum Euclidean distance between pixel centres (pmerge.boundary + pmerge.closest_pair, no cap);
  closest points  = pmerge.closest_pair's, for the pair taken in candidate order (lower index first);
  principal axis  = the R3 linker's axis of the whole candidate (pmerge.linker_piece -> postproc._axis_of);
  bridge          = pmerge.centre_line (1 px, cv2.line LINE_8, endpoints included) between the closest points, in
                    pmerge.crop_of of the two bboxes, the same values as pmerge.pair_features' bridge_p_min.
Nearest neighbour = the other candidate with the smallest distance (ties: lowest candidate index).

  neigh_count_64               number of other candidates with distance <= 64
  neigh_count_128              number of other candidates with distance <= 128
  neigh_gap_nearest            min(distance to the nearest other candidate, 512); 512 when there is none
  neigh_axis_abs_cos_nearest   |cos| between the candidate's axis and its nearest neighbour's; 0.0 when there is none
  neigh_area_share_aligned_128 area / (area + total area of the candidates with distance <= 128 and |cos| >= 0.7)
  neigh_bridge_p_min_nearest   min of p along the bridge to the nearest neighbour; 1.0 when there is none
"""
import numpy as np
import pmerge as PM

FEATURES = ['neigh_count_64', 'neigh_count_128', 'neigh_gap_nearest', 'neigh_axis_abs_cos_nearest',
            'neigh_area_share_aligned_128', 'neigh_bridge_p_min_nearest']
R_NEAR, R_FAR, GAP_CAP, ALIGN_COS = 64, 128, 512, .7
NO_NEIGHBOUR_COS, NO_NEIGHBOUR_BRIDGE = 0.0, 1.0
NO_CAP = 2900                         # > the image diagonal (2047 * sqrt(2) = 2894.9): every pair has a distance


def abs_cos(u, v):
    u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
    return float(min(1.0, abs(float(u @ v)) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), PM.EPS)))


def neighbour_features(masks, p):
    """One dict of the 6 FEATURES per candidate (candidate order). masks: full-size bool masks; p: the mixture map."""
    n = len(masks)
    if not n: return []
    objs = [PM.obj_from_full(m) for m in masks]
    bd = [PM.boundary(M, bb) for M, bb in objs]
    axes = [PM.linker_piece(M, bb)['axis'] for M, bb in objs]
    area = [float(np.asarray(M, bool).sum()) for M, _ in objs]
    pair = {}
    for a in range(n):
        for b in range(a + 1, n):
            r = PM.closest_pair(bd[a], bd[b], NO_CAP); assert r is not None, 'PAIR_WITHOUT_DISTANCE'
            pair[a, b] = r
    d2 = lambda i, j: pair[min(i, j), max(i, j)][0]
    out = []
    for a in range(n):
        others = [b for b in range(n) if b != a]
        f = dict(neigh_count_64=float(sum(d2(a, b) <= R_NEAR * R_NEAR for b in others)),
                 neigh_count_128=float(sum(d2(a, b) <= R_FAR * R_FAR for b in others)))
        if others:
            nb = min(others, key=lambda b: (d2(a, b), b)); lo, hi = min(a, nb), max(a, nb); m2, pa, pb = pair[lo, hi]
            crop = PM.crop_of([objs[lo][1], objs[hi][1]]); L = PM.centre_line(pa, pb, crop); y0, y1, x0, x1 = crop
            f.update(neigh_gap_nearest=float(min(np.sqrt(m2), GAP_CAP)), neigh_axis_abs_cos_nearest=abs_cos(axes[a], axes[nb]))
            aligned = sum(area[b] for b in others if d2(a, b) <= R_FAR * R_FAR and abs_cos(axes[a], axes[b]) >= ALIGN_COS)
            f.update(neigh_area_share_aligned_128=float(area[a] / (area[a] + aligned)),
                     neigh_bridge_p_min_nearest=float(np.asarray(p)[y0:y1, x0:x1][L].astype(np.float64).min()))
        else:
            f.update(neigh_gap_nearest=float(GAP_CAP), neigh_axis_abs_cos_nearest=NO_NEIGHBOUR_COS, neigh_area_share_aligned_128=1.0,
                     neigh_bridge_p_min_nearest=NO_NEIGHBOUR_BRIDGE)
        f = {k: f[k] for k in FEATURES}; assert np.isfinite(list(f.values())).all(); out.append(f)
    return out
