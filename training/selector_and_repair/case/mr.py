"""mfrich: pure helpers (arms, contrasts, feature names, cohorts, the 707-image gate, intervals, keep agreement, exact
comparisons, GT-free feature distributions and the test-shift table). No model fit, no data transport.

The fresh_ctrl recipe lives in presel.py (verbatim) and worker.py. mfrich = fresh_ctrl with the 8 MF-derived features of
mf_features.py appended (62 features); everything else is fresh_ctrl's.
"""
import numpy as np
import mf_features as MF

ARMS = ['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'mfrich']
PASS2_ARMS = ['inc_pcand_v1', 'fresh_ctrl', 'mfrich']
SELECTOR_ARMS = ['fresh_ctrl', 'mfrich']
TREATMENT = 'mfrich'
GATE_CONTRAST = 'mfrich_minus_fresh_ctrl'
SECONDARY_CONTRAST = 'mfrich_minus_inc_pcand_v1'
REFIT_REP_ARMS = ['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl']
REFIT_REP_CONTRASTS = ['fresh_ctrl_minus_inc_pcand_v1', 'inc_pcand_v1_minus_inc_pnat', 'inc_pcand_v1_minus_inc', 'inc_pnat_minus_inc', 'fresh_ctrl_minus_inc']
CONTRASTS = [(GATE_CONTRAST, 'mfrich', 'fresh_ctrl'), (SECONDARY_CONTRAST, 'mfrich', 'inc_pcand_v1')] + [(n, *n.split('_minus_')) for n in REFIT_REP_CONTRASTS]
CI_LEVELS = dict(paired_group_CI95=[.025, .975])
# The six context features of context_features.py that read the MF map (selvar, derived mechanically there).
MF_CONTEXT = ['mf_mean', 'full_mf_abs_mean', 'local_mf_abs_mean', 'branches_range_mean', 'branches_all_high_fraction', 'branches_one_high_fraction']
_CTX = [f'context_{r}_{s}' for r in (16, 64, 128) for s in ('mean', 'std', 'dark_fraction')] + MF_CONTEXT


def mfrich_feature_names(order39, ctx):
    """fresh_ctrl's 54 names (presel.control_feature_names: order39 + the 15 context features), then the 8 MF features."""
    assert list(ctx) == _CTX and len(order39) == 39 and not set(order39) & (set(_CTX) | set(MF.FEATURES))
    return list(order39) + list(ctx) + list(MF.FEATURES)


def cohorts(files, folds, query425):
    """Ordered cohorts: all707 (the gate cohort, 'combined707'), the 425 and 282 query sets and the five base folds."""
    q425 = list(query425); assert q425 == [f for f in files if folds[f] in (0, 1, 2)], 'QUERY425_ORDER'
    q282 = [f for f in files if folds[f] in (3, 4)]
    return dict(all707=list(files), combined425=q425, combined282=q282, new284=[f for f in q425 if folds[f] in (0, 1)],
                **{f'fold{k}': [f for f in files if folds[f] == k] for k in range(5)})


def intervals(diff):
    d = np.asarray(diff, float); return {n: np.quantile(d, q).tolist() for n, q in CI_LEVELS.items()}


def _contrast_ok(c):
    ds, ci = c['partition_deltas'], c['paired_group_CI95']
    if len(ds) != 20 or not np.isfinite(list(ds) + list(ci)).all(): return False
    return ci[0] > 0 and sum(d > 0 for d in ds) >= 16


def gate707(result, contrast=GATE_CONTRAST):
    """Gate: blocking checks passed; all707 group CI95 lower > 0; >= 16/20 positive; each of the five base-fold mean paired deltas > 0."""
    try:
        if result['replay_passed'] is not True or result['integrity_passed'] is not True or result['reproduction707_exact'] is not True: return False
        S = result['subsets']
        if S['all707']['images'] != 707 or not _contrast_ok(S['all707']['contrasts'][contrast]): return False
        for k in range(5):
            d = S[f'fold{k}']['contrasts'][contrast]['mean_paired_partition_delta']
            if not np.isfinite(d) or d <= 0: return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def statistics_met(result, cohort, contrast):
    """Secondary (decides nothing): CI95 lower > 0 and >= 16/20 positive on the cohort."""
    try:
        return bool(_contrast_ok(result['subsets'][cohort]['contrasts'][contrast]))
    except (KeyError, TypeError, ValueError):
        return False


def keep_agreement(first, second, n):
    a, b = set(first), set(second)
    return dict(both=len(a & b), first_only=len(a - b), second_only=len(b - a), neither=n - len(a | b))


def compare_counts(per, files, arm, ref, seeds):
    """per[s][arm]: per-image six counts in `files` order; ref: refit_rep counts.json (same file order), exact equality."""
    assert list(ref['files']) == list(files), 'FILE_ORDER'
    worst = 0.; exact = True
    for s in seeds:
        a = np.asarray(per[s][arm], float); b = np.asarray(ref['per_partition_counts'][str(s)][arm], float); assert a.shape == b.shape
        exact &= bool(np.array_equal(a, b)); worst = max(worst, float(np.abs(a - b).max()) if a.size else 0.)
    return dict(images=len(files), exact=bool(exact), max_abs_difference=worst)


def compare_keeps(keeps, ref, files, seeds, arm):
    diff = sum(tuple(keeps[f][s][arm]) != tuple(ref[f][str(s)][arm]) for f in files for s in seeds)
    return dict(images=len(files), differing_image_partitions=int(diff), exact=diff == 0)


FRACTION_LEVELS = (.1, .25, .5, .75)


def map_stats(native, den):
    """GT-free statistics of one native-integer probability map (value / den). refit_rep's rep.map_stats, verbatim."""
    z = np.asarray(native); assert np.issubdtype(z.dtype, np.integer) and int(z.max()) <= den and int(z.min()) >= 0
    hist = np.bincount(z.ravel().astype(np.int64), minlength=den + 1); n = int(hist.sum()); v = np.arange(den + 1) / den
    frac = {str(t): float(hist[v >= t].sum() / n) for t in FRACTION_LEVELS}
    return dict(mean=float((hist * v).sum() / n), fraction_ge=frac, hist=hist.tolist())


def quantiles(v):
    v = np.asarray(v, float)
    if not len(v): return {}
    q = np.quantile(v, [0, .05, .25, .5, .75, .95, 1]).tolist()
    return dict(zip(['min', 'q05', 'q25', 'median', 'q75', 'q95', 'max'], q), mean=float(v.mean()), n=int(len(v)))


def feature_distribution(rows, names):
    """Per feature: quantiles over candidates, plus the fraction of candidates at exactly 0."""
    out = {}
    for n in names:
        v = np.asarray([r[n] for r in rows], float); out[n] = dict(quantiles(v), fraction_zero=float((v == 0).mean()) if len(v) else 0.)
    return out


def collinearity(new, old, new_names, old_names, corr_with=MF_CONTEXT):
    """Descriptive: per new feature, Pearson r with selected existing features, the largest |r| over all existing features,
    and R^2 of an ordinary least-squares fit on all existing features (with intercept). A constant column has r = 0."""
    new = np.asarray(new, float); old = np.asarray(old, float); assert new.shape[0] == old.shape[0]
    def r(a, b):
        sa, sb = a.std(), b.std()
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else 0.
    Z = np.column_stack([np.ones(len(old)), old]); out = {}
    for i, n in enumerate(new_names):
        y = new[:, i]; rs = [r(y, old[:, j]) for j in range(old.shape[1])]; j = int(np.argmax(np.abs(rs)))
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]; res = y - Z @ beta; tot = ((y - y.mean()) ** 2).sum()
        out[n] = dict(corr={m: rs[old_names.index(m)] for m in corr_with}, max_abs_corr=float(abs(rs[j])), max_abs_corr_feature=old_names[j],
                      r2_on_existing=float(1 - (res ** 2).sum() / tot) if tot > 0 else 0.)
    return out


def shift_table(cv, test, coef, scale):
    """Per MF feature: CV and test q05 / median / q95, the median shift (test - CV) and the logit shift that median shift
    would cause through the mean mfrich selector (mean standardized coefficient x shift / mean scaler scale). Descriptive."""
    out = {}
    for n in MF.FEATURES:
        sh = test[n]['median'] - cv[n]['median']
        out[n] = dict(cv={k: cv[n][k] for k in ('q05', 'median', 'q95')}, test={k: test[n][k] for k in ('q05', 'median', 'q95')},
                      median_shift=sh, implied_logit_shift_at_median=coef[n] * sh / scale[n])
    return out
