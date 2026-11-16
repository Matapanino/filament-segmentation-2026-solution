"""P-merge: a learned per-pair decision to join neighbouring final objects of INC + P-cand v1.

Objects are the final INC + P-cand v1 masks of an image (kept candidates, repaired with their v1 taus). Pure helpers
(numpy + cv2 + scipy + the byte-identical refine_ops/pcand2/spatial modules and the frozen R3 linker source).

- Candidate pairs: two objects of the same image whose minimum Euclidean distance between pixel centres is <= 48 px.
  Closest points: the minimising pixel pair; ties go to the lexicographically smallest (y_a, x_a, y_b, x_b).
- Merge of (a, b): Ma | Mb | bridge, bridge = band & (p >= min(tau_a, tau_b)) & ~(every other current object), band =
  cv2.line(closest a, closest b, thickness 5) (the R3 linker's stroke). If the bridge does not connect them, the merged
  object is still the union, as one instance. Greedy in descending predicted probability, each object at most once, one round.
- Label of a pair (training images only): the linearised pooled-PQ gain of merging that pair alone,
  dN - PQ_ref * (dTP + 0.5 dFP + 0.5 dFN), summed over the image's annotation sets; label = 1 iff gain > 0.

No function here reads an annotation except iou_cols / pair_delta, which are used only for labels and scoring.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree
import refine_ops as R
import pcand2 as C
import spatial as S
from rules import counts as count6
from filseg.postproc import _axis_of, EPS

MAX_GAP = 48
BRIDGE_THICKNESS = 5
MERGE_THRESHOLD = 0.5
CROP_PAD = 4
LINKER_GAP, LINKER_ANGLE, LINKER_WINDOW, LINKER_MIN_AREA, LINKER_MIN_ELONG = 60, 45.0, 30, 32, 1.0
PRIMARY = 'inc_pmerge_minus_inc_pcand_v1'
SECONDARY = []
FEATURES = ['gap_px', 'axis_angle_deg', 'gap_axis_angle_a_deg', 'gap_axis_angle_b_deg', 'bridge_p_min', 'bridge_p_mean',
            'area_a', 'area_b', 'elong_a', 'elong_b', 'tau_a', 'tau_b', 'selector_a', 'selector_b', 'linker_join',
            'bridge_full_mean', 'bridge_local_mean', 'bridge_mf_mean']


# ------------------------------------------------------------------ objects as tight bbox-local masks
def tight(M, bbox):
    """Crop a bbox-local mask to the tight bbox of its pixels; returns (mask, (y0, y1, x0, x1)) in image coordinates."""
    M = np.asarray(M, bool); ys = np.flatnonzero(M.any(1)); xs = np.flatnonzero(M.any(0)); assert len(ys), 'EMPTY_OBJECT'
    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
    return np.ascontiguousarray(M[y0:y1, x0:x1]), (bbox[0] + y0, bbox[0] + y1, bbox[2] + x0, bbox[2] + x1)


def obj_from_full(full): return tight(full, (0, R.N, 0, R.N))


def boundary(M, bbox):
    """(n, 2) int64 (y, x) image coordinates of the pixels with a 4-neighbour outside the object. Every minimising pixel
    of the distance to any outside point is such a pixel, so the closest pairs of two objects are unchanged."""
    P = np.pad(M, 1); inner = P[1:-1, 1:-1] & P[:-2, 1:-1] & P[2:, 1:-1] & P[1:-1, :-2] & P[1:-1, 2:]
    ys, xs = np.nonzero(M & ~inner)
    return np.stack([ys + bbox[0], xs + bbox[2]], 1).astype(np.int64)


def bbox_gap2(b1, b2):
    dy = max(0, b2[0] - (b1[1] - 1), b1[0] - (b2[1] - 1)); dx = max(0, b2[2] - (b1[3] - 1), b1[2] - (b2[3] - 1))
    return dy * dy + dx * dx


def closest_pair(Ba, Bb, max_gap=MAX_GAP):
    """None if the minimum distance exceeds max_gap; else (d2, (ya, xa), (yb, xb)) with lexicographic tie order."""
    _, idx = cKDTree(Ba.astype(np.float64)).query(Bb.astype(np.float64), k=1, distance_upper_bound=max_gap + 0.5)
    ok = idx < len(Ba)
    if not ok.any(): return None
    nb = Bb[ok]; na = Ba[idx[ok]]; d2 = ((nb - na) ** 2).sum(1)          # exact integer nearest distances of each b point
    m = int(d2.min())
    if m > max_gap * max_gap: return None
    best = None
    for yb, xb in nb[d2 == m]:
        da = (Ba[:, 0] - yb) ** 2 + (Ba[:, 1] - xb) ** 2
        row = min((int(Ba[i, 0]), int(Ba[i, 1]), int(yb), int(xb)) for i in np.flatnonzero(da == m))
        best = row if best is None else min(best, row)
    return (m, (best[0], best[1]), (best[2], best[3]))


def candidate_pairs(objs, max_gap=MAX_GAP):
    """Pairs (a < b) of objects (mask, bbox, ...) with minimum pixel distance <= max_gap."""
    out = []; bd = {}
    for a in range(len(objs)):
        for b in range(a + 1, len(objs)):
            if bbox_gap2(objs[a][1], objs[b][1]) > max_gap * max_gap: continue
            for i in (a, b):
                if i not in bd: bd[i] = boundary(objs[i][0], objs[i][1])
            r = closest_pair(bd[a], bd[b], max_gap)
            if r is not None: out.append(dict(a=a, b=b, d2=r[0], pa=r[1], pb=r[2]))
    return out


# ------------------------------------------------------------------ the bridge and the merge
def crop_of(bboxes):
    y0 = max(0, min(b[0] for b in bboxes) - CROP_PAD); y1 = min(R.N, max(b[1] for b in bboxes) + CROP_PAD)
    x0 = max(0, min(b[2] for b in bboxes) - CROP_PAD); x1 = min(R.N, max(b[3] for b in bboxes) + CROP_PAD)
    return (y0, y1, x0, x1)


def _draw(pa, pb, crop, thickness):
    y0, y1, x0, x1 = crop; c = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.line(c, (pa[1] - x0, pa[0] - y0), (pb[1] - x0, pb[0] - y0), 1, thickness)
    return c.astype(bool)


def band(pa, pb, crop):
    """The linker's 5-px stroke between the closest points, in crop coordinates; it must not touch an inner crop edge."""
    B = _draw(pa, pb, crop, BRIDGE_THICKNESS); y0, y1, x0, x1 = crop
    assert not ((B[0].any() and y0 > 0) or (B[-1].any() and y1 < R.N) or (B[:, 0].any() and x0 > 0) or (B[:, -1].any() and x1 < R.N)), 'BAND_CLIPPED'
    return B


def centre_line(pa, pb, crop): return _draw(pa, pb, crop, 1)


def place(M, bbox, crop):
    """Bbox-local mask -> crop-local mask (the part inside the crop)."""
    out = np.zeros((crop[1] - crop[0], crop[3] - crop[2]), bool)
    y0, y1, x0, x1 = max(bbox[0], crop[0]), min(bbox[1], crop[1]), max(bbox[2], crop[2]), min(bbox[3], crop[3])
    if y0 < y1 and x0 < x1: out[y0 - crop[0]:y1 - crop[0], x0 - crop[2]:x1 - crop[2]] = M[y0 - bbox[0]:y1 - bbox[0], x0 - bbox[2]:x1 - bbox[2]]
    return out


def merge_core(crop, Ma, Mb, others, p_crop, pa, pb, tau):
    """Crop-local merge: (merged mask, info). tau = min(tau_a, tau_b) as a Python float (the repair's comparison)."""
    Bd = band(pa, pb, crop) & (p_crop >= float(tau)) & ~others
    merged = Ma | Mb | Bd; new = Bd & ~Ma & ~Mb
    _, lab = cv2.connectedComponents(merged.astype(np.uint8), connectivity=8)
    la = set(np.unique(lab[Ma]).tolist()); lb = set(np.unique(lab[Mb]).tolist())
    return merged, dict(bridge_pixels=int(new.sum()), connected=bool(la & lb))


def merge_objs(objs, a, b, pa, pb, tau, p):
    """Local path: merge objects a and b of a list of (mask, bbox, ...) objects; every other object is excluded."""
    crop = crop_of([objs[a][1], objs[b][1]]); others = np.zeros((crop[1] - crop[0], crop[3] - crop[2]), bool)
    for i, o in enumerate(objs):
        if i not in (a, b): others |= place(o[0], o[1], crop)
    merged, info = merge_core(crop, place(objs[a][0], objs[a][1], crop), place(objs[b][0], objs[b][1], crop), others,
                              p[crop[0]:crop[1], crop[2]:crop[3]], pa, pb, tau)
    return tight(merged, crop), info


def apply_merges(full_masks, merges, p):
    """Full path: apply merges (a, b, pa, pb, tau) in order to full-size masks. The merged object takes a's slot and b's
    slot is dropped; a later bridge excludes every current object other than its own two (earlier bridges included)."""
    cur = list(full_masks); infos = []; dropped = set()
    for a, b, pa, pb, tau in merges:
        assert a not in dropped and b not in dropped and a != b, 'OBJECT_MERGED_TWICE'
        ba, bb = obj_from_full(cur[a])[1], obj_from_full(cur[b])[1]; crop = crop_of([ba, bb]); y0, y1, x0, x1 = crop
        others = np.zeros((y1 - y0, x1 - x0), bool)
        for i, m in enumerate(cur):
            if i not in (a, b) and i not in dropped: others |= m[y0:y1, x0:x1]
        merged, info = merge_core(crop, cur[a][y0:y1, x0:x1], cur[b][y0:y1, x0:x1], others, p[y0:y1, x0:x1], pa, pb, tau)
        F = cur[a] | cur[b]; F[y0:y1, x0:x1] |= merged; cur[a] = F; dropped.add(b); infos.append(info)
    return [m for i, m in enumerate(cur) if i not in dropped], infos


def overlap_pixels(full_masks):
    if not full_masks: return 0
    acc = np.zeros(full_masks[0].shape, np.uint8)
    for m in full_masks: acc += np.asarray(m, bool)
    return int((acc >= 2).sum())


# ------------------------------------------------------------------ the R3 linker's rule on a pair of whole objects
def linker_piece(M, bbox):
    """postproc._pieces applied to the whole object as one piece (row-major pixel order, the same tip and axis rules)."""
    ys, xs = np.nonzero(M); xs = xs + bbox[2]; ys = ys + bbox[0]
    pts = np.stack([xs, ys]).astype(np.float64)
    axis, l1, l2 = _axis_of(pts); t = pts.T @ np.array(axis)
    tips, tip_axes = [], []
    for k in (int(np.argmin(t)), int(np.argmax(t))):
        tx, ty = int(xs[k]), int(ys[k]); tips.append((tx, ty))
        near = (xs - tx) ** 2 + (ys - ty) ** 2 <= LINKER_WINDOW ** 2
        local, _, _ = _axis_of(pts[:, near]) if near.sum() >= 3 else (axis, 0, 0)
        tip_axes.append(local)
    return dict(area=int(len(xs)), axis=axis, tips=(tips[0], tips[1]), tip_axes=(tip_axes[0], tip_axes[1]), elong=(l1 / max(l2, EPS)) ** .5)


def linker_joins(pa, pb):
    """postproc.link_collinear's test for one pair of pieces (gap 60, 45 deg, pieces under 32 px never link)."""
    if pa['area'] < LINKER_MIN_AREA or pb['area'] < LINKER_MIN_AREA or pa['elong'] < LINKER_MIN_ELONG or pb['elong'] < LINKER_MIN_ELONG: return False
    ka, kb = min(((i, j) for i in (0, 1) for j in (0, 1)),
                 key=lambda ij: (pa['tips'][ij[0]][0] - pb['tips'][ij[1]][0]) ** 2 + (pa['tips'][ij[0]][1] - pb['tips'][ij[1]][1]) ** 2)
    (ax, ay), (bx, by) = pa['tips'][ka], pb['tips'][kb]; dx, dy = bx - ax, by - ay; gap = (dx * dx + dy * dy) ** .5
    if gap < 1.0 or gap > LINKER_GAP: return False
    ux, uy = dx / gap, dy / gap; va, vb = pa['tip_axes'][ka], pb['tip_axes'][kb]; cos_min = float(np.cos(np.deg2rad(LINKER_ANGLE)))
    return abs(ux * va[0] + uy * va[1]) >= cos_min and abs(ux * vb[0] + uy * vb[1]) >= cos_min


# ------------------------------------------------------------------ prediction-only pair features
def angle_deg(u, v):
    u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
    c = abs(float(u @ v)) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), EPS)
    return float(np.degrees(np.arccos(min(1.0, c))))


def pair_features(objs, pair, taus, selector, p, maps, pieces=None):
    """The 18 prediction-only features of one candidate pair. objs: (mask, bbox, ...) objects; taus / selector: per object.
    maps: the native full / local / m2f_public branch maps. Reads no annotation."""
    a, b = pair['a'], pair['b']; pa, pb = pair['pa'], pair['pb']
    A = pieces[a] if pieces is not None else linker_piece(objs[a][0], objs[a][1])
    B = pieces[b] if pieces is not None else linker_piece(objs[b][0], objs[b][1])
    crop = crop_of([objs[a][1], objs[b][1]]); L = centre_line(pa, pb, crop); y0, y1, x0, x1 = crop
    vals = lambda m: np.asarray(m)[y0:y1, x0:x1][L].astype(np.float64)
    pv = vals(p); gap = np.array([pb[1] - pa[1], pb[0] - pa[0]], np.float64)    # (x, y)
    f = dict(gap_px=float(np.sqrt(pair['d2'])), axis_angle_deg=angle_deg(A['axis'], B['axis']),
             gap_axis_angle_a_deg=angle_deg(gap, A['axis']), gap_axis_angle_b_deg=angle_deg(gap, B['axis']),
             bridge_p_min=float(pv.min()), bridge_p_mean=float(pv.mean()),
             area_a=float(A['area']), area_b=float(B['area']), elong_a=float(A['elong']), elong_b=float(B['elong']),
             tau_a=float(taus[a]), tau_b=float(taus[b]), selector_a=float(selector[a]), selector_b=float(selector[b]),
             linker_join=float(linker_joins(A, B)),
             bridge_full_mean=float(vals(maps['full']).mean()), bridge_local_mean=float(vals(maps['local']).mean()),
             bridge_mf_mean=float(vals(maps['m2f_public']).mean()))
    assert list(f) == FEATURES and np.isfinite(list(f.values())).all()
    return f


# ------------------------------------------------------------------ labels (annotations enter only here)
def iou_cols(M, bbox, gts): return S.iou_column(M, bbox, gts)


def pair_delta(cols_a, cols_b, cols_m):
    """[dN, dTP, dFP, dFN] of replacing objects a and b by their merge, summed over the annotation sets, with the matching
    computed for the involved objects only (exact for the whole image because objects are disjoint and a GT can be hit by
    at most one of them)."""
    d = np.zeros(4)
    for ca, cb, cm in zip(cols_a, cols_b, cols_m):
        before = count6(np.column_stack([ca, cb]).reshape(len(ca), 2)); after = count6(np.asarray(cm, float).reshape(len(cm), 1))
        d += (after - before)[:4]
    return d


def gain(delta, pq_ref):
    d = np.asarray(delta, float); return float(d[0] - pq_ref * (d[1] + .5 * d[2] + .5 * d[3]))


def label(g): return int(g > 0)


# ------------------------------------------------------------------ merge selection
def select_merges(pairs, scores, threshold=MERGE_THRESHOLD):
    """Greedy: indices of the pairs to merge, in descending score (ties by (a, b)), score >= threshold, each object once."""
    order = sorted(range(len(pairs)), key=lambda i: (-float(scores[i]), pairs[i][0], pairs[i][1])); used = set(); out = []
    for i in order:
        a, b = pairs[i]
        if float(scores[i]) < threshold or a in used or b in used: continue
        used |= {a, b}; out.append(i)
    return out


def oracle_merges(pairs, gains):
    """ORACLE (query GT): the pairs with a positive gain, greedy in descending gain (ties by (a, b)), each object once."""
    order = sorted(range(len(pairs)), key=lambda i: (-float(gains[i]), pairs[i][0], pairs[i][1])); used = set(); out = []
    for i in order:
        a, b = pairs[i]
        if not float(gains[i]) > 0 or a in used or b in used: continue
        used |= {a, b}; out.append(i)
    return out


# ------------------------------------------------------------------ base objects: the v1 repair per interaction cluster
class BaseSet:
    """The P-cand v1 repair of one kept set (pcand2.repair_taus) evaluated per interaction cluster with a cache.
    masks(taus) -> per kept candidate (tight mask, tight bbox, digest), equal to repair_taus's full masks."""

    def __init__(self, seeds, p):
        self.n = len(seeds); self.cache = {}; self.computed = 0
        self.prep = C.prepare(seeds, p) if seeds else None
        self.cl = S.clusters(self.prep) if seeds else []

    def masks(self, taus):
        assert len(taus) == self.n; out = [None] * self.n; pr = self.prep
        for c, mem in enumerate(self.cl):
            key = (c, tuple(float(taus[i]) for i in mem))
            if key not in self.cache:
                ctx = [pr['ctx'][i] for i in mem]
                Rl = [(pr['q'][i] >= float(taus[i])) & pr['ws'][i] for i in mem]
                ms, _ = R.paste_back(ctx, Rl, grow=False); self.computed += 1
                self.cache[key] = {i: tight(M, pr['bbs'][i]) for i, M in zip(mem, ms)}
                self.cache[key] = {i: (M, bb, S.digest_local(M, bb)) for i, (M, bb) in self.cache[key].items()}
            for i, v in self.cache[key].items(): out[i] = v
        return out


# ------------------------------------------------------------------ classifier wrapper
class Constant:
    def __init__(self, v): self.v = float(v); self.n_iter = 0
    def prob(self, X): return np.full(len(X), self.v)


class Fitted:
    def __init__(self, m): self.m = m; self.n_iter = int(getattr(m, 'n_iter_', 0)); self.col = list(m.classes_).index(1)
    def prob(self, X): return np.asarray(self.m.predict_proba(np.asarray(X, float)))[:, self.col] if len(X) else np.zeros(0)


def fit_classifier(X, y, make):
    """Fixed recipe: make() fitted on (X, y). Fewer than two classes: no fit, a constant probability equal to the only
    label (0 when there are no rows)."""
    y = np.asarray(y, int)
    if len(set(y.tolist())) < 2: return Constant(float(y[0]) if len(y) else 0.0)
    return Fitted(make().fit(np.asarray(X, float), y))
