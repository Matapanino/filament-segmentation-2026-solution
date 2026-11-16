"""P-combo: pure helpers (arms, contrasts, feature names, the gate, the final verdict rule). No model fit, no data transport.

pcombo = fresh_ctrl's recipe on the 54 features + the 6 neighbour features (neigh.py, pneigh lane) + the 8 MF features
(mf_features.py, mfrich lane), 68 features. The three base arms (fresh_ctrl, pneigh, mfrich) are exact replays of
refit_rep, pneigh and mfrich. The generic helpers (cohorts, intervals, gate statistics, exact comparisons) are mfrich's mr.py.
"""
import mr as R
import neigh as NB
import mf_features as MF

ARMS = ['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'pneigh', 'mfrich', 'pcombo']
PASS2_ARMS = ['inc_pcand_v1', 'fresh_ctrl', 'pneigh', 'mfrich', 'pcombo']
SELECTOR_ARMS = ['fresh_ctrl', 'pneigh', 'mfrich', 'pcombo']
TREATMENT = 'pcombo'
GATE_CONTRAST = 'pcombo_minus_mfrich'
SECONDARY_CONTRASTS = ['pcombo_minus_pneigh', 'pcombo_minus_inc_pcand_v1']
DESCRIPTIVE_CONTRASTS = ['pcombo_minus_fresh_ctrl', 'mfrich_minus_pneigh']
ADDED_FEATURES = list(NB.FEATURES) + list(MF.FEATURES)
EXTRA = dict(fresh_ctrl=[], pneigh=list(NB.FEATURES), mfrich=list(MF.FEATURES), pcombo=ADDED_FEATURES)
# What each base lane computed (its arms and contrasts), for the exact-replay checks.
LANE_ARMS = dict(mfrich=['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'mfrich'], pneigh=['inc', 'inc_pnat', 'inc_pcand_v1', 'fresh_ctrl', 'pneigh'])
LANE_CONTRASTS = dict(mfrich=['mfrich_minus_fresh_ctrl', 'mfrich_minus_inc_pcand_v1'] + R.REFIT_REP_CONTRASTS,
                      pneigh=['pneigh_minus_fresh_ctrl', 'pneigh_minus_inc_pcand_v1'] + R.REFIT_REP_CONTRASTS + ['pneigh_minus_inc'])
_names = [GATE_CONTRAST] + SECONDARY_CONTRASTS + DESCRIPTIVE_CONTRASTS
for _lane in ('mfrich', 'pneigh'):
    _names += [n for n in LANE_CONTRASTS[_lane] if n not in _names]
CONTRASTS = [(n, *n.split('_minus_')) for n in _names]
# Kept-by-origin summaries in pass 2: (arm, the arm treated as "original").
ORIGIN_PAIRS = [('fresh_ctrl', 'inc_pcand_v1'), ('pneigh', 'inc_pcand_v1'), ('mfrich', 'inc_pcand_v1'), ('pcombo', 'inc_pcand_v1'),
                ('pcombo', 'mfrich'), ('pcombo', 'pneigh')]


def feature_names(order39, ctx, arm):
    """fresh_ctrl's 54 names (mr.mfrich_feature_names' checks), then the arm's added features: pneigh +6, mfrich +8, pcombo +6 +8."""
    base = R.mfrich_feature_names(order39, ctx)[:54]
    extra = EXTRA[arm]; assert not set(base) & set(extra)
    return base + extra


def gate707(result):
    """The kernel-side gate on pcombo minus mfrich (mr.gate707: blocking checks, all707 CI95 lower > 0, >= 16/20, five fold means > 0)."""
    return R.gate707(result, GATE_CONTRAST)


def final_verdict(gate, all_checks_passed):
    """The reported verdict: a verdict only if every blocking check passed, in the kernel and in the local audit
    (the exact pneigh and mfrich replays and every selector state are checked locally)."""
    if not all_checks_passed: return 'no verdict: replay, reproduction, integrity or local audit failure'
    return 'confirmed' if gate else 'Improvement could not be confirmed'
