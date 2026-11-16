"""P-resel: pure helpers (repair-delta features, bank layout, labels, roles, duplicate counts, origin summary, gate).

Prediction-side helpers (window_border, delta_features, presel_row) take no annotation. set_labels / build_bank only
turn ordinary training annotations (IoU records) into the historical per-annotation-set targets and count records.
The selector recipe (fit_selector) is the fresh LogisticRegression recipe of mf_aware_selector, unchanged.
"""
import numpy as np
import cv2
from pycocotools import mask as mu

DELTA_FEATURES = ['repair_area_ratio', 'repair_pixels_added', 'repair_pixels_removed', 'repair_touches_window_border', 'repair_added_mean_p']
ARMS = ['inc', 'inc_pnat', 'inc_pcand_v1', 'presel', 'fresh_ctrl']
PRIMARY = 'presel_minus_inc_pcand_v1'
SECONDARY = 'presel_minus_fresh_ctrl'
CONTRASTS = [(PRIMARY, 'presel', 'inc_pcand_v1'),
             (SECONDARY, 'presel', 'fresh_ctrl'),
             ('fresh_ctrl_minus_inc_pcand_v1', 'fresh_ctrl', 'inc_pcand_v1'),
             ('presel_minus_inc_pnat', 'presel', 'inc_pnat'),
             ('inc_pcand_v1_minus_inc_pnat', 'inc_pcand_v1', 'inc_pnat'),
             ('inc_pnat_minus_inc', 'inc_pnat', 'inc')]
THRESHOLD_GRID = [round(i * .025, 6) for i in range(41)]
DUP_KEYS = ['predictions', 'overlapping_prediction_pairs', 'gt_matched_by_more_than_one_prediction',
            'predictions_matched_by_more_than_one_gt_of_a_set', 'tp_pairs']


def presel_feature_names(order39, ctx): return list(order39) + list(ctx) + DELTA_FEATURES


def control_feature_names(order39, ctx): return list(order39) + list(ctx)


# ---------------------------------------------------------------- prediction side (no annotations)
def window_border(w):
    """Window pixels with an 8-neighbour outside the window; pixels outside the bbox count as outside."""
    w = np.asarray(w, bool)
    pad = np.pad(w, 1, constant_values=False).astype(np.uint8)
    inner = cv2.erode(pad, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)[1:-1, 1:-1] > 0
    return w & ~inner


def delta_features(seed, mask, p, border):
    """Five repair-delta statistics of one candidate, on bbox-local crops (seed, repaired mask, mixture p, window border)."""
    seed = np.asarray(seed, bool); mask = np.asarray(mask, bool)
    assert seed.shape == mask.shape == np.shape(p) == np.shape(border)
    s = int(seed.sum()); a = int(mask.sum()); assert s > 0 and a > 0
    add = mask & ~seed; rem = seed & ~mask
    out = dict(repair_area_ratio=float(a / s), repair_pixels_added=float(add.sum()), repair_pixels_removed=float(rem.sum()),
               repair_touches_window_border=float(bool((mask & np.asarray(border, bool)).any())),
               repair_added_mean_p=float(np.asarray(p)[add].astype(np.float64).mean()) if add.any() else 0.0)
    assert list(out) == DELTA_FEATURES and np.isfinite(list(out.values())).all()
    return out


def presel_row(q39, ctx, delta, order39, ctx_names):
    """Fixed 59-value row: 39 quality features of the repaired mask, 15 context features, 5 repair deltas."""
    return [float(q39[n]) for n in order39] + [float(ctx[n]) for n in ctx_names] + [float(delta[n]) for n in DELTA_FEATURES]


# ---------------------------------------------------------------- labels, bank, rows
def _mat(q, n):
    q = np.asarray(q, float)
    if q.ndim != 2: q = q.reshape(-1, n)
    assert q.shape[1] == n; return q


def set_labels(q_sets, n):
    """Historical per-annotation-set target: per nonempty set, 1 if IoU > 0.5 with any GT of that set."""
    return [(_mat(q, n) > .5).any(0).astype(int).tolist() for q in q_sets]


def build_bank(files, rows, ious):
    """files: ordered images; rows[f]: (n_f, d) features; ious[f]: per nonempty annotation set an (n_gt, n_f) IoU matrix.
    Candidate ids are consecutive in file order. Label rows are set-major per image (historical layout)."""
    X, fid, recs, lid, ly = [], [], {}, [], []
    start = 0
    for f in files:
        r = np.asarray(rows[f], float); n = len(r); ids = list(range(start, start + n))
        if n: X.append(r.reshape(n, -1))
        fid += [f] * n
        ann = []
        for q in ious[f]:
            q = _mat(q, n); ann.append(dict(n_gt=int(q.shape[0]), iou=q.tolist()))
        for y in set_labels(ious[f], n):
            lid += ids; ly += y
        recs[f] = dict(candidate_ids=ids, annotations=ann); start += n
    d = X[0].shape[1] if X else 0
    return dict(X=np.vstack(X) if X else np.zeros((0, d)), file=np.asarray(fid, dtype=object), records=recs,
                label_ids=np.asarray(lid, int), label_y=np.asarray(ly, int))


def label_rows(bank, train):
    """Expanded training rows (candidate id per label row) of the images in train."""
    t = set(train); sel = np.asarray([bank['file'][i] in t for i in bank['label_ids']], bool)
    return bank['label_ids'][sel], bank['label_y'][sel]


def candidate_ids(bank, files):
    t = set(files); return np.asarray([i for i, f in enumerate(bank['file']) if f in t], int)


# ---------------------------------------------------------------- pass-1 roles
def pass1_roles(f, seeds, outer, inner, query, n_outer=5):
    """For each outer fit (s, k) that needs image f: ('outer', s, k) when f is a query image of fold k (outer v1 taus),
    ('inner', s, k, j) when f is an outer-training image of (s, k) in inner fold j (inner cross-fit v1 taus)."""
    out = {}
    for s in seeds:
        for k in range(n_outer):
            if outer[s][f] == k:
                if f in query: out[(s, k)] = ('outer', s, k)
            else:
                out[(s, k)] = ('inner', s, k, inner[(s, k)][f])
    return out


# ---------------------------------------------------------------- duplicate-match audit
def duplicate_counts(rles, q_sets):
    """Pairs of predictions sharing a pixel; GT rows matched (IoU > 0.5) by more than one prediction; predictions matched by
    more than one GT of the same annotation set; all qualifying pairs (the evaluator's TP)."""
    n = len(rles); pairs = 0
    if n > 1:
        o = np.asarray(mu.iou(rles, rles, [0] * n)).reshape(n, n); pairs = int(np.triu(o > 0, 1).sum())
    hits = [_mat(q, n) > .5 for q in q_sets]
    return dict(predictions=n, overlapping_prediction_pairs=pairs, gt_matched_by_more_than_one_prediction=int(sum((h.sum(1) > 1).sum() for h in hits)),
                predictions_matched_by_more_than_one_gt_of_a_set=int(sum((h.sum(0) > 1).sum() for h in hits)), tp_pairs=int(sum(h.sum() for h in hits)))


def add_counts(total, d, times=1):
    return {k: total.get(k, 0) + times * d[k] for k in DUP_KEYS}


# ---------------------------------------------------------------- descriptive (final masks; never chooses anything)
def origin_summary(kept, original, correct_final, correct_v1):
    """kept: the selector's kept candidates; original: the saved selector's kept candidates. correct_final[j]: per nonempty
    set 1/0 for candidate j's final mask; correct_v1[j]: the same for the INC+P-cand-v1 mask of an originally kept j."""
    kept, original = set(kept), set(original)
    added = sorted(kept - original); dropped = sorted(original - kept)
    return dict(kept_originally_kept=len(kept & original), kept_originally_rejected=len(added), dropped_originally_kept=len(dropped),
                added_correct_pairs=int(sum(sum(correct_final[j]) for j in added)), added_false_pairs=int(sum(len(correct_final[j]) - sum(correct_final[j]) for j in added)),
                dropped_correct_pairs_under_v1=int(sum(sum(correct_v1[j]) for j in dropped)),
                dropped_false_pairs_under_v1=int(sum(len(correct_v1[j]) - sum(correct_v1[j]) for j in dropped)))


# ---------------------------------------------------------------- selector recipe (Kaggle only) and gate
def fit_selector(X, y):
    """mf_aware_selector's fresh recipe: StandardScaler + LogisticRegression(C=1, max_iter=2000, random_state=42, lbfgs),
    no class weights; a ConvergenceWarning is an error."""
    import warnings
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    assert set(np.asarray(y).tolist()) == {0, 1}
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=42, solver='lbfgs'))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning); model.fit(np.asarray(X, float), np.asarray(y, int))
    return model


def selector_state(model):
    return dict(scaler_mean=model[0].mean_.tolist(), scaler_scale=model[0].scale_.tolist(), coef=model[1].coef_.tolist(),
                intercept=model[1].intercept_.tolist(), classes=model[1].classes_.tolist(), iterations=model[1].n_iter_.tolist())


def project(m, x):
    from scipy.special import expit
    return expit(((np.asarray(x, float) - np.asarray(m['scaler_mean'])) / np.asarray(m['scaler_scale'])) @ np.asarray(m['coef'])[0] + m['intercept'][0])


def gate(result, contrast=PRIMARY):
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
