"""P-cand v2: the P-cand v1 recipe on a fixed wide tau grid, plus a training-chosen global tau (P-global).

Pure helpers (numpy + the byte-identical refine_ops/pnat modules). No fitting, no annotations in the
prediction-side functions. prepare/repair_taus/candidate_scores/_geom are the same source as
pcand_tau/pcand.py (tested against the verbatim copy pcand_v1_reference.py); only the grid is a parameter.
"""
import numpy as np
import refine_ops as R
import pnat

TAU = pnat.TAU
GRID_V1 = (0.20, 0.25, 0.30, 0.35, TAU, 0.45, 0.50, 0.55, 0.60)
GRID_W = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, TAU, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
V1_COLS = [GRID_W.index(t) for t in GRID_V1]


def wname(t): return 'window_frac_ge_tau' if t == TAU else f'window_frac_ge_{t:.2f}'


WINDOW_FEATURES_W = [wname(t) for t in GRID_W] + ['seed_mean_p']
WINDOW_FEATURES_V1 = [f'window_frac_ge_{i}' for i in range(len(GRID_V1))] + ['seed_mean_p']
V1_WINDOW_MAP = {**{f'window_frac_ge_{i}': wname(t) for i, t in enumerate(GRID_V1)}, 'seed_mean_p': 'seed_mean_p'}
PRIMARY = 'inc_pcand_v2_minus_inc_pcand_v1'
SECONDARY = ['inc_pglobal_minus_inc_pnat', 'inc_pcand_v2_minus_inc_pglobal']


def _order(t):
    """Tie order among equally good grid values: TAU first, then nearer to TAU, then lower."""
    return (t != TAU, abs(t - TAU), t)


def snap(x, grid):
    x = float(x); best = min(abs(x - t) for t in grid)
    return min((t for t in grid if abs(x - t) == best), key=_order)


def candidate_scores(q_sets, n_pred):
    """q_sets: per nonempty annotation set an (n_gt, n_pred) IoU matrix. Score of a prediction = sum over sets of
    the IoU with its best-matching GT of that set, counted only when that IoU is > 0.5."""
    s = np.zeros(n_pred)
    for q in q_sets:
        q = np.asarray(q, float).reshape(-1, n_pred)
        if not q.shape[0]: continue
        best = q.max(0); s += np.where(best > .5, best, 0.)
    return s


def choose_target(scores, grid):
    """Grid tau with the highest score; ties toward TAU, then the nearer value, then the lower. None if never matched."""
    scores = [float(v) for v in scores]; assert len(scores) == len(grid)
    top = max(scores)
    if top <= 0: return None
    return min((t for t, v in zip(grid, scores) if v == top), key=_order)


def oracle_taus(score_matrix, grid):
    """ORACLE (query GT): per-candidate best grid tau, TAU where a candidate never matches. Descriptive only."""
    out = []
    for row in np.asarray(score_matrix, float).reshape(-1, len(grid)):
        t = choose_target(row, grid); out.append(TAU if t is None else t)
    return out


def pq(c):
    """Pooled PQ of six counts (iou_sum, tp, fp, fn, ...), the same formula as rules.pq."""
    d = c[1] + .5 * (c[2] + c[3]); return float(c[0] / d) if d else 0.


def choose_global(pooled_counts, grid):
    """P-global: the grid tau with the highest pooled PQ (counts already summed over the training images);
    ties toward TAU, then the nearer value, then the lower."""
    assert len(pooled_counts) == len(grid)
    v = [pq(c) for c in pooled_counts]; top = max(v)
    return min((t for t, x in zip(grid, v) if x == top), key=_order)


def _geom(mask):
    return R.geom_from_vector(R.geom_vector(R.geometry(mask)))


def window_features(mask, p):
    """Prediction-only statistics of the native mixture inside the candidate's own P-nat window (wide grid)."""
    g = _geom(mask); b = R.window_bbox(g); w = R.window_native(g, b); v = p[b[0]:b[1], b[2]:b[3]][w]
    out = {wname(t): float((v >= t).mean()) for t in GRID_W}
    out['seed_mean_p'] = float(np.asarray(p)[np.asarray(mask, bool)].astype(np.float64).mean())
    assert list(out) == WINDOW_FEATURES_W and np.isfinite(list(out.values())).all()
    return out


def prepare(seeds, p):
    """Window geometry and separation context of a kept set; independent of the threshold(s)."""
    assert p.shape == (R.N, R.N) and np.isfinite(p).all()
    gs = [_geom(s) for s in seeds]
    bbs = [R.window_bbox(g) for g in gs]; ws = [R.window_native(g, b) for g, b in zip(gs, bbs)]
    ctx = R.image_context(seeds, bbs)
    q = [p[b[0]:b[1], b[2]:b[3]] for b in bbs]
    return dict(bbs=bbs, ws=ws, ctx=ctx, q=q)


def repair_taus(seeds, p, taus, prep=None):
    stats = dict(kept=len(seeds), fallback=0, changed=0, added=0, removed=0, seed_pixels=0)
    if not seeds: return [], stats
    taus = [float(t) for t in taus]; assert len(taus) == len(seeds)   # Python floats: same comparison semantics as pnat.TAU
    pr = prepare(seeds, p) if prep is None else prep
    bbs, ws, ctx, q = pr['bbs'], pr['ws'], pr['ctx'], pr['q']
    above = [v >= t for v, t in zip(q, taus)]
    ms, fb = R.paste_back(ctx, [a & w for a, w in zip(above, ws)], grow=False)
    # Same invariants as pnat.repair; the new-pixel rule is checked against each candidate's own threshold.
    checks = R.check_member(ctx, ms, [a.astype(np.float64) for a in above], 1.0, ws, strict=True)
    assert all(checks.values()), ('PCAND_INVARIANT', checks)
    out = [R.full(m, b) for m, b in zip(ms, bbs)]
    stats.update(fallback=fb, changed=sum(not np.array_equal(s, m) for s, m in zip(seeds, out)),
                 added=sum(int((m & ~s).sum()) for s, m in zip(seeds, out)), removed=sum(int((s & ~m).sum()) for s, m in zip(seeds, out)),
                 seed_pixels=sum(int(s.sum()) for s in seeds))
    return out, stats


def gate(result, contrast):
    """Shared gate: replay and integrity passed, combined425 CI95 lower > 0, >= 16/20 positive, fold0/1/2 means > 0."""
    try:
        if result['replay_passed'] is not True or result['integrity_passed'] is not True: return False
        c = result['subsets']['combined425']['contrasts'][contrast]; ds, ci = c['partition_deltas'], c['paired_group_CI95']
        if result['subsets']['combined425']['images'] != 425 or len(ds) != 20: return False
        if not np.isfinite(list(ds) + list(ci)).all() or ci[0] <= 0 or sum(d > 0 for d in ds) < 16: return False
        for k in range(3):
            d = result['subsets'][f'fold{k}']['contrasts'][contrast]['mean_paired_partition_delta']
            if not np.isfinite(d) or d <= 0: return False
        return True
    except (KeyError, TypeError, ValueError):
        return False
