"""P-cand: the P-nat repair with one threshold per kept candidate, predicted by the deployment regressor.

The helpers below are copied verbatim from the cross-validated implementation (detect_20261111/pcand_tau/pcand.py,
SHA256 87e96105ccc78adc0cf690547cc3f6f3e0b5e2d61bf899a6f118e7731540e201): the tau grid, the snap and tie rules,
the target score and choice used to build the training rows, the window features and the per-candidate repair.
With every threshold equal to pnat.TAU the repair returns exactly pnat.repair's masks (tested).
Order on test: SHIP_R3 -> original-mask LR39 features -> saved keep -> 64 features of each kept candidate ->
deployment regressor -> snap to the grid -> per-candidate repair of the kept seeds; never rescore.
"""
import hashlib
import importlib.metadata
import pickle
import warnings
from pathlib import Path

import numpy as np

from . import refine_ops as R
from . import pnat

TAU = pnat.TAU
# The deployment regressor ships in the weights dataset next to reproduction_weights.json; its bytes are pinned here.
REGRESSOR_FILE = 'regressor.pkl'
REGRESSOR_SHA256 = 'b4dcd7d1ec79fc6c42135733721c53d107ea53f1cefe2934c7fd2c95acba6798'
SKLEARN_VERSION = '1.6.1'  # the version that fitted and pickled it (also pinned in requirements-reproduce-cpu.txt)
GRID = (0.20, 0.25, 0.30, 0.35, TAU, 0.45, 0.50, 0.55, 0.60)
WINDOW_FEATURES = [f'window_frac_ge_{i}' for i in range(len(GRID))] + ['seed_mean_p']
# The preregistered 64 prediction-only features, in regressor column order (pcand_tau/protocol.json).
FEATURES = ["log_area", "log_perimeter", "compactness", "extent", "solidity", "eccentricity", "log_major_axis", "log_minor_axis", "bbox_width", "bbox_height", "width_mean", "width_max", "centroid_radius", "radial_extent", "edge_fraction", "full_mean", "full_std", "full_q10", "full_median", "full_q90", "full_high_fraction", "local_mean", "local_std", "local_q10", "local_median", "local_q90", "local_high_fraction", "blend_mean", "blend_std", "blend_q10", "blend_median", "blend_q90", "blend_high_fraction", "disagreement_mean", "disagreement_q90", "image_mean", "image_std", "ring_mean", "ring_contrast", "context_16_mean", "context_16_std", "context_16_dark_fraction", "context_64_mean", "context_64_std", "context_64_dark_fraction", "context_128_mean", "context_128_std", "context_128_dark_fraction", "mf_mean", "full_mf_abs_mean", "local_mf_abs_mean", "branches_range_mean", "branches_all_high_fraction", "branches_one_high_fraction", "window_frac_ge_0", "window_frac_ge_1", "window_frac_ge_2", "window_frac_ge_3", "window_frac_ge_4", "window_frac_ge_5", "window_frac_ge_6", "window_frac_ge_7", "window_frac_ge_8", "seed_mean_p"]


def _order(t):
    """Tie order among equally good grid values: TAU first, then nearer to TAU, then lower."""
    return (t != TAU, abs(t - TAU), t)


def snap(x):
    x = float(x); best = min(abs(x - t) for t in GRID)
    return min((t for t in GRID if abs(x - t) == best), key=_order)


def candidate_scores(q_sets, n_pred):
    """q_sets: per nonempty annotation set an (n_gt, n_pred) IoU matrix. Score of a prediction = sum over sets of
    the IoU with its best-matching GT of that set, counted only when that IoU is > 0.5."""
    s = np.zeros(n_pred)
    for q in q_sets:
        q = np.asarray(q, float).reshape(-1, n_pred)
        if not q.shape[0]: continue
        best = q.max(0); s += np.where(best > .5, best, 0.)
    return s


def choose_target(scores):
    """Grid tau with the highest score; ties toward TAU, then the nearer value, then the lower. None if never matched."""
    scores = [float(v) for v in scores]; assert len(scores) == len(GRID)
    top = max(scores)
    if top <= 0: return None
    return min((t for t, v in zip(GRID, scores) if v == top), key=_order)


def _geom(mask):
    return R.geom_from_vector(R.geom_vector(R.geometry(mask)))


def window_features(mask, p):
    """Prediction-only statistics of the native mixture inside the candidate's own P-nat window."""
    g = _geom(mask); b = R.window_bbox(g); w = R.window_native(g, b); v = p[b[0]:b[1], b[2]:b[3]][w]
    out = {name: float((v >= t).mean()) for name, t in zip(WINDOW_FEATURES, GRID)}
    out['seed_mean_p'] = float(np.asarray(p)[np.asarray(mask, bool)].astype(np.float64).mean())
    assert list(out) == WINDOW_FEATURES and np.isfinite(list(out.values())).all()
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


def uniform(tau):
    """Predictor that gives every kept candidate the same threshold (the regression check uses pnat.TAU)."""
    tau = float(tau)

    def predictor(x):
        return [tau] * len(x)
    predictor.raw = []
    return predictor


def deployed(weights_root):
    """(path, sha256) of the deployment regressor in the weights dataset, for load_regressor."""
    path = Path(weights_root) / REGRESSOR_FILE
    if not path.is_file():
        raise FileNotFoundError(f'{REGRESSOR_FILE} is missing from the weights dataset at {weights_root}; attach a version that contains it')
    return str(path), REGRESSOR_SHA256


def _sklearn_version():
    return importlib.metadata.version('scikit-learn')


def mean_raw(models, x):
    """Plain mean of the models' predictions of x, each as float64 (the 4-regressor repair: np.mean over the stacked
    predictions in the given order). One model gives its own predictions unchanged."""
    return np.mean(np.stack([np.asarray(m.predict(x), float).ravel() for m in models]), axis=0)


def load_regressor(path, sha256, members=()):
    """Hash-checked deployment regressor; returns a predictor mapping (n, 64) features to snapped thresholds.
    members: further fitted regressors of the same 64 features (chain.repair_trees); the threshold is then the snap of
    the plain mean of this regressor's and the members' predictions (mean_raw), with the same grid and tie rule."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError('Regressor file hash differs from the pinned value')
    if _sklearn_version() != SKLEARN_VERSION:
        raise RuntimeError(f'The regressor was pickled with scikit-learn {SKLEARN_VERSION}; found {_sklearn_version()}. Install requirements-reproduce-cpu.txt')
    obj = pickle.loads(data)
    if list(obj['features']) != FEATURES:
        raise ValueError('Regressor feature order differs from the preregistered 64 features')
    model = obj['model']

    def predictor(x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError('Expected finite (n, 64) features')
        if members:
            with warnings.catch_warnings():   # LightGBM's wrapper was fitted on a named frame; the columns are FEATURES
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                raw = [float(v) for v in mean_raw([model, *members], x)]
        else:raw = [float(v) for v in model.predict(x)]
        predictor.raw.append(raw)
        return [snap(v) for v in raw]
    predictor.raw = []
    return predictor
