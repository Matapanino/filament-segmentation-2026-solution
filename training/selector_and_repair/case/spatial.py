"""P-spatial: P-cand v1's per-candidate tau, modulated pixel by pixel along each kept candidate by its seed's brightness.

For a kept candidate with snapped v1 threshold tau_v1, inside its P-nat window (pnat_crossfold refine_ops geometry):
  s(x)    = (x - m) . e1, the coordinate along the window's principal axis (m = window centre, e1 = principal axis);
  bin(x)  = floor((s(x) + Lc / 2) / 16), clipped to 0 .. nb - 1 with nb = ceil(Lc / 16) (bins start at the window's edge);
  peak(b) = max of the native mixture p over the candidate's seed pixels in bin b; a bin without seed pixels takes the
            value of the nearest bin that has them (ties to the lower bin index);
  m       = median of peak(b) over the bins with seed pixels;
  tau(x)  = clip(tau_v1 * (peak(bin(x)) / m) ** gamma, 0.05, 0.80), cast to the dtype of p before the comparison
            (the same comparison semantics as a Python-float scalar threshold against the float32 map).
Everything else is pcand2.repair_taus line by line (windows, separation, contested pixels, connectivity to the seed,
seed fallback, paste-back); only the threshold comparison changes. gamma = 0 gives exactly the scalar repair.

The fast path (FastImage) evaluates uniform-gamma configurations exactly but cheaply: a candidate's repaired mask depends
only on its own threshold map and on candidates whose potential new pixels (window & ~seed & ~separation) overlap its own,
so paste-back runs per interaction cluster and the IoU columns are computed inside the window bbox. It is tested bitwise
against the full repair + pycocotools IoU, and checked again in the kernel on every query image and partition.

No function here reads an annotation except FastImage's IoU columns, which are only used for scoring.
"""
import hashlib
import numpy as np
import refine_ops as R
from pcand2 import prepare, _geom
from rules import counts as count6

GAMMAS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
BIN_PX = 16
TAU_MIN, TAU_MAX = 0.05, 0.80
PRIMARY = 'inc_pspatial_minus_inc_pcand_v1'
SECONDARY = ['inc_pspatial_minus_inc_pnat']


def fill_nearest(peak, has):
    """Bins without seed pixels take the value of the nearest bin with seed pixels; ties go to the lower bin index."""
    idx = np.flatnonzero(has); assert len(idx)
    near = idx[np.argmin(np.abs(np.arange(len(peak))[:, None] - idx[None, :]), axis=1)]   # argmin: first = lower index
    return np.asarray(peak, np.float64)[near]


def profile(seed, p):
    """Prediction-only along-axis brightness profile of one kept candidate inside its P-nat window bbox."""
    seed = np.asarray(seed, bool); g = _geom(seed); bb = R.window_bbox(g); y0, y1, x0, x1 = bb
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    u = (xx - g['m'][0]) * g['e1'][0] + (yy - g['m'][1]) * g['e1'][1]      # the same expression as refine_ops.window_native
    nb = max(1, int(np.ceil(g['Lc'] / BIN_PX)))
    bins = np.clip(np.floor((u + g['Lc'] / 2) / BIN_PX).astype(np.int64), 0, nb - 1)
    sd = seed[y0:y1, x0:x1]; assert int(sd.sum()) == int(seed.sum()), 'SEED_OUTSIDE_WINDOW_BBOX'
    pv = np.asarray(p)[y0:y1, x0:x1][sd].astype(np.float64); bv = bins[sd]
    raw = np.full(nb, -np.inf); np.maximum.at(raw, bv, pv)
    has = np.zeros(nb, bool); has[bv] = True
    peak = fill_nearest(raw, has); m = float(np.median(peak[has]))
    ratio = peak / m if m > 0 else np.ones(nb)
    idx = np.flatnonzero(has); tail = int(idx[0] + (nb - 1 - idx[-1]))
    return dict(bbox=bb, bins=bins, nb=nb, peak=peak, has=has, m=m, ratio=ratio, copied=int((~has).sum()), copied_tail=tail,
                copied_interior=int((~has).sum()) - tail)


def tau_map(prof, tau, gamma, dtype):
    """Per-pixel threshold over the window bbox: clip(tau * ratio ** gamma, 0.05, 0.80), cast to dtype."""
    with np.errstate(divide='ignore', over='ignore'):
        t = np.clip(float(tau) * np.asarray(prof['ratio'], np.float64) ** float(gamma), TAU_MIN, TAU_MAX)
    return t.astype(dtype)[prof['bins']]


def repair_maps(seeds, p, taus, gammas, profs, prep=None):
    """pcand2.repair_taus with the scalar comparison replaced by the threshold map; returns (full masks, stats)."""
    stats = dict(kept=len(seeds), fallback=0, changed=0, added=0, removed=0, seed_pixels=0)
    if not seeds: return [], stats
    assert len(taus) == len(seeds) == len(gammas) == len(profs)
    pr = prepare(seeds, p) if prep is None else prep
    bbs, ws, ctx, q = pr['bbs'], pr['ws'], pr['ctx'], pr['q']
    assert all(tuple(f['bbox']) == tuple(b) for f, b in zip(profs, bbs)), 'PROFILE_BBOX'
    above = [v >= tau_map(f, t, g, v.dtype) for v, f, t, g in zip(q, profs, taus, gammas)]
    ms, fb = R.paste_back(ctx, [a & w for a, w in zip(above, ws)], grow=False)
    checks = R.check_member(ctx, ms, [a.astype(np.float64) for a in above], 1.0, ws, strict=True)
    assert all(checks.values()), ('PSPATIAL_INVARIANT', checks)
    out = [R.full(m, b) for m, b in zip(ms, bbs)]
    stats.update(fallback=fb, changed=sum(not np.array_equal(s, m) for s, m in zip(seeds, out)),
                 added=sum(int((m & ~s).sum()) for s, m in zip(seeds, out)), removed=sum(int((s & ~m).sum()) for s, m in zip(seeds, out)),
                 seed_pixels=sum(int(s.sum()) for s in seeds))
    return out, stats


def digest(full_mask):
    """Identity of a full-size mask: sha1 of its flat nonzero indices."""
    return hashlib.sha1(np.flatnonzero(np.asarray(full_mask, bool)).astype(np.int64).tobytes()).hexdigest()


def digest_local(M, bbox):
    y0, y1, x0, x1 = bbox; ys, xs = np.nonzero(M)
    return hashlib.sha1(((ys + y0).astype(np.int64) * R.N + (xs + x0)).tobytes()).hexdigest()


def _overlap(b1, b2):
    y0, y1, x0, x1 = max(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), min(b1[3], b2[3])
    return (y0, y1, x0, x1) if y0 < y1 and x0 < x1 else None


def clusters(prep):
    """Interaction clusters: candidates whose potential new pixels (window & ~seed & ~separation) overlap, transitively."""
    bbs = prep['bbs']; pot = [w & ~c['seed'] & ~c['B'] for w, c in zip(prep['ws'], prep['ctx'])]
    n = len(bbs); parent = list(range(n))

    def root(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            o = _overlap(bbs[i], bbs[j])
            if o is None: continue
            y0, y1, x0, x1 = o; a, b = bbs[i], bbs[j]
            if (pot[i][y0 - a[0]:y1 - a[0], x0 - a[2]:x1 - a[2]] & pot[j][y0 - b[0]:y1 - b[0], x0 - b[2]:x1 - b[2]]).any():
                parent[root(i)] = root(j)
    out = {}
    for i in range(n): out.setdefault(root(i), []).append(i)
    return [sorted(v) for v in sorted(out.values())]


def decode_gts(gt_sets):
    """Nonempty annotation sets of RLEs -> per set a list of (bbox, cropped bool mask, area). Used only for scoring."""
    from pycocotools import mask as mu
    out = []
    for rles in gt_sets:
        row = []
        for r in rles:
            m = np.asarray(mu.decode(r), bool); ys, xs = np.nonzero(m)
            if len(ys) == 0: row.append(((0, 0, 0, 0), np.zeros((0, 0), bool), 0)); continue
            b = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            row.append((b, m[b[0]:b[1], b[2]:b[3]].copy(), int(m.sum())))
        out.append(row)
    return out


def iou_column(M, bbox, gts):
    """IoU of the bbox-local mask M with every GT of every set, as pycocotools computes it ((double) i / (double) u)."""
    area = int(M.sum()); cols = []
    for row in gts:
        c = np.zeros(len(row))
        for gi, (gb, gm, ga) in enumerate(row):
            o = _overlap(bbox, gb) if ga else None
            if o is None: continue
            y0, y1, x0, x1 = o
            i = int((M[y0 - bbox[0]:y1 - bbox[0], x0 - bbox[2]:x1 - bbox[2]] & gm[y0 - gb[0]:y1 - gb[0], x0 - gb[2]:x1 - gb[2]]).sum())
            if i: c[gi] = i / (area + ga - i)
        cols.append(c)
    return cols


class FastImage:
    """Exact uniform-gamma evaluation of one kept set on one image with a per-cluster cache."""

    def __init__(self, seeds, p, gts, profs=None):
        self.seeds = list(seeds); self.p = p; self.gts = gts; self.computed = 0; self.cache = {}
        self.prep = prepare(self.seeds, p) if self.seeds else None
        self.profs = profs if profs is not None else [profile(s, p) for s in self.seeds]
        self.cl = clusters(self.prep) if self.seeds else []
        self.of = {i: c for c, mem in enumerate(self.cl) for i in mem}

    def _cluster(self, c, taus, gammas):
        mem = self.cl[c]; key = (c, tuple(float(taus[i]) for i in mem), tuple(float(gammas[i]) for i in mem))
        if key not in self.cache:
            pr = self.prep; ctx = [pr['ctx'][i] for i in mem]
            Rl = [(pr['q'][i] >= tau_map(self.profs[i], taus[i], gammas[i], pr['q'][i].dtype)) & pr['ws'][i] for i in mem]
            ms, _ = R.paste_back(ctx, Rl, grow=False); self.computed += 1
            self.cache[key] = {i: (iou_column(M, pr['bbs'][i], self.gts), digest_local(M, pr['bbs'][i])) for i, M in zip(mem, ms)}
        return self.cache[key]

    def columns(self, taus, gammas):
        assert len(taus) == len(self.seeds) == len(gammas)
        res = {}
        for c in range(len(self.cl)): res.update(self._cluster(c, taus, gammas))
        return [res[i] for i in range(len(self.seeds))]

    def iou_matrices(self, taus, gamma):
        cols = self.columns(taus, [gamma] * len(taus))
        return [np.column_stack([cols[i][0][a] for i in range(len(cols))]) if cols else np.zeros((len(row), 0)) for a, row in enumerate(self.gts)]

    def evaluate(self, taus, gamma, gammas=None):
        """(six counts summed over the nonempty annotation sets, per-candidate scores, per-candidate mask digests)."""
        gs = [gamma] * len(taus) if gammas is None else list(gammas)
        cols = self.columns(taus, gs); n = len(cols)
        qs = [np.column_stack([cols[i][0][a] for i in range(n)]) if n else np.zeros((len(row), 0)) for a, row in enumerate(self.gts)]
        total = sum((count6(q) for q in qs), start=np.zeros(6))
        sc = np.zeros(n)
        for q in qs:
            if q.shape[0]: best = q.max(0); sc += np.where(best > .5, best, 0.)
        return total, sc, [cols[i][1] for i in range(n)]


def choose_gamma(pq_by_gamma):
    """Training-side choice: the gamma with the best pooled PQ; ties to 0, then the smaller |gamma|, then the lower gamma."""
    v = [float(x) for x in pq_by_gamma]; assert len(v) == len(GAMMAS)
    top = max(v)
    return min((g for g, x in zip(GAMMAS, v) if x == top), key=lambda g: (g != 0, abs(g), g))


def oracle_gammas(score_matrix):
    """ORACLE (query GT): per candidate the gamma with the best score, same tie order; 0 if never matched. Descriptive only."""
    out = []
    for row in np.asarray(score_matrix, float).reshape(-1, len(GAMMAS)):
        top = row.max()
        out.append(0.0 if top <= 0 else min((g for g, x in zip(GAMMAS, row) if x == top), key=lambda g: (g != 0, abs(g), g)))
    return out


def inner_assignment(groups, train, s, k):
    """The selector machinery's inner 3-fold split of fit (s, k): assign_groups over the temporal groups of the
    outer-training images with seed s + 10000 + k (instrument_source.py). Groups must not straddle the training set."""
    from make_time_folds import assign_groups
    tr = set(train)
    for g in groups: assert len({f in tr for f in g}) == 1, 'GROUP_STRADDLES_TRAINING_SET'
    return assign_groups([g for g in groups if g[0] in tr], k=3, seed=s + 10000 + k)
