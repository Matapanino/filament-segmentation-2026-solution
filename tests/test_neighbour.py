"""Neighbour-context selector stage contracts (the cross-validated P-neigh arm), written before the package module.

The P-neigh selector reads the 39 quality + 15 context features of refit_ship's fresh_ctrl stage plus 6 prediction-only
neighbour features of every unrepaired candidate (neighbour.py, a copy of detect_20261112/pneigh/neigh.py and the
pmerge geometry it uses). Switching the selector back to the shipped LR39 or to the 54-feature fresh_ctrl selector gives
the old paths. Synthetic data only; no fitting.
"""
import hashlib
import inspect
import json

import numpy as np
import pytest

from test_pnat import keep_all_model, synthetic_maps
from test_refit import selector54

N = 2048


def bar(y0, y1, x0, x1):
    m = np.zeros((N, N), bool); m[y0:y1, x0:x1] = True; return m


def pmap(masks, inside=.9, outside=.2):
    p = np.full((N, N), outside, np.float32)
    for m in masks: p[m] = inside
    return p


def selector60(coef=None, intercept=0., threshold=.375):
    from filseg.reproduction import refit
    n = len(refit.PNEIGH_FEATURES)
    return dict(role=refit.PNEIGH_ROLE, feature_order=list(refit.PNEIGH_FEATURES), classes=[0, 1], scaler_mean=[0.] * n,
                scaler_scale=[1.] * n, coef=[list(coef) if coef is not None else [0.] * n], intercept=[float(intercept)], threshold=threshold)


# ---------------------------------------------------------------- the 6 features (values from pneigh's tests/test_neigh.py)
def test_names_constants_and_signature():
    from filseg.reproduction import neighbour as NB
    assert NB.FEATURES == ['neigh_count_64', 'neigh_count_128', 'neigh_gap_nearest', 'neigh_axis_abs_cos_nearest',
                           'neigh_area_share_aligned_128', 'neigh_bridge_p_min_nearest']
    assert (NB.R_NEAR, NB.R_FAR, NB.GAP_CAP, NB.ALIGN_COS) == (64, 128, 512, .7)
    assert (NB.NO_NEIGHBOUR_COS, NB.NO_NEIGHBOUR_BRIDGE) == (0.0, 1.0)
    assert NB.NO_CAP >= int(np.ceil(np.hypot(N - 1, N - 1)))
    assert list(inspect.signature(NB.neighbour_features).parameters) == ['masks', 'p']


def test_collinear_pair():
    from filseg.reproduction import neighbour as NB
    a = bar(500, 506, 100, 200); b = bar(500, 506, 230, 330); p = pmap([a, b]); p[502, 215] = .05
    fa, fb = NB.neighbour_features([a, b], p)
    assert list(fa) == NB.FEATURES and list(fb) == NB.FEATURES
    assert (fa['neigh_count_64'], fa['neigh_count_128']) == (1., 1.)
    assert fa['neigh_gap_nearest'] == 31. and fb['neigh_gap_nearest'] == 31.
    assert abs(fa['neigh_axis_abs_cos_nearest'] - 1.) < 1e-12
    assert abs(fa['neigh_area_share_aligned_128'] - .5) < 1e-12 and abs(fb['neigh_area_share_aligned_128'] - .5) < 1e-12
    assert abs(fa['neigh_bridge_p_min_nearest'] - .2) < 1e-6 and fa['neigh_bridge_p_min_nearest'] == fb['neigh_bridge_p_min_nearest']
    p[500, 215] = .05
    assert abs(NB.neighbour_features([a, b], p)[0]['neigh_bridge_p_min_nearest'] - .05) < 1e-6


def test_distance_thresholds_inclusive():
    from filseg.reproduction import neighbour as NB
    a = bar(1000, 1004, 100, 200)
    for gap, c64, c128 in ((64, 1, 1), (65, 0, 1), (128, 0, 1), (129, 0, 0)):
        b = bar(1000, 1004, 199 + gap, 260 + gap)
        f = NB.neighbour_features([a, b], pmap([a, b]))[0]
        assert f['neigh_gap_nearest'] == float(gap) and (f['neigh_count_64'], f['neigh_count_128']) == (c64, c128)


def test_counts_aligned_area_share_and_nearest_tie():
    from filseg.reproduction import neighbour as NB
    a = bar(800, 806, 800, 1000); b = bar(800, 806, 1040, 1140); c = bar(700, 760, 900, 905)
    d = bar(800, 806, 1200, 1300); e = bar(1500, 1506, 100, 200)
    masks = [a, b, c, d, e]; f = NB.neighbour_features(masks, pmap(masks))
    assert (f[0]['neigh_count_64'], f[0]['neigh_count_128']) == (2., 2.)
    assert abs(f[0]['neigh_area_share_aligned_128'] - 1200 / 1800) < 1e-12
    assert abs(f[0]['neigh_axis_abs_cos_nearest'] - 1.) < 1e-12       # b and c tie at 41: the lower index b


def test_far_gap_is_capped_and_lone_candidate_defaults():
    from filseg.reproduction import neighbour as NB
    a = bar(100, 104, 100, 200); b = bar(1900, 1904, 1800, 1900)
    assert NB.neighbour_features([a, b], pmap([a, b]))[0]['neigh_gap_nearest'] == 512.
    (f,) = NB.neighbour_features([a], pmap([a]))
    assert f == dict(neigh_count_64=0., neigh_count_128=0., neigh_gap_nearest=512., neigh_axis_abs_cos_nearest=0.,
                     neigh_area_share_aligned_128=1., neigh_bridge_p_min_nearest=1.)
    assert NB.neighbour_features([], pmap([])) == []


# ---------------------------------------------------------------- the 60-feature selector role
def test_pneigh_feature_order_and_cutoff():
    from filseg.reproduction import refit, neighbour
    assert refit.PNEIGH_FEATURES == list(refit.FEATURES) + neighbour.FEATURES and len(refit.PNEIGH_FEATURES) == 60
    assert refit.THRESHOLD == .375 and refit.PNEIGH_ROLE != refit.ROLE


def test_pneigh_selector_file_is_hash_and_recipe_checked(tmp_path):
    from filseg.reproduction import refit
    good = tmp_path / 'sel.json'; good.write_text(json.dumps(selector60()))
    assert refit.load_selector(good, hashlib.sha256(good.read_bytes()).hexdigest())['role'] == refit.PNEIGH_ROLE
    for change in (dict(threshold=.35), dict(feature_order=list(refit.FEATURES)), dict(feature_order=list(reversed(refit.PNEIGH_FEATURES))),
                   dict(coef=[[0.] * 54]), dict(role=refit.ROLE)):
        bad = tmp_path / 'bad.json'; bad.write_text(json.dumps(dict(selector60(), **change)))
        with pytest.raises(ValueError):
            refit.load_selector(bad, hashlib.sha256(bad.read_bytes()).hexdigest())


def test_neighbour_rows_are_required_by_the_pneigh_selector_only():
    from filseg.reproduction import refit
    rows = [{n: 0. for n in refit.FEATURES[:39]}]; ctx = [{n: 0. for n in refit.FEATURES[39:]}]
    with pytest.raises(ValueError):
        refit.apply_selector(rows, ctx, selector60())
    assert not refit.needs_neighbour(selector54()) and refit.needs_neighbour(selector60())


def test_pneigh_selector_reads_the_neighbour_features_of_the_candidates_on_the_mixture_map():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.arithmetic import candidate_blend
    from filseg.reproduction import refit, neighbour
    from scipy.special import expit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    fresh = image_stages(raw, maps, model, selector=selector54())
    from test_pcand import decode
    masks = [decode(r) for r in fresh['rles']]
    blend = candidate_blend(*(maps[n] for n in ('full', 'local', 'MF0', 'MF1', 'MF2')))
    expected = neighbour.neighbour_features(masks, blend)
    assert len(masks) == 3 and expected[1]['neigh_count_64'] == 1.       # the (800, 800) and (812, 960) bars are 60 px apart
    j = refit.PNEIGH_FEATURES.index('neigh_gap_nearest'); coef = [0.] * 60; coef[j] = -1.
    out = image_stages(raw, maps, model, selector=selector60(coef=coef, intercept=100.))
    assert out['selector']['neighbour'] == expected
    assert out['selector']['context'] == fresh['selector']['context']
    z = 100. - np.asarray([e['neigh_gap_nearest'] for e in expected])
    assert np.array_equal(out['keep'], expit(z) >= .375) and out['keep'].tolist() == [False, True, True]
    assert out['rles'] == fresh['rles']


def test_switching_back_to_fresh_ctrl_or_lr39_gives_the_old_paths():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    from filseg.reproduction import refit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    a = image_stages(raw, maps, model, pcand=uniform(.4016711682033202))
    b = image_stages(raw, maps, model, pcand=uniform(.4016711682033202), selector=refit.lr39(model))
    assert a['rles'] == b['rles'] and 'neighbour' not in b['selector']
    c = image_stages(raw, maps, model, selector=selector54(intercept=-10.))
    assert 'neighbour' not in c['selector'] and set(c['stages']) == set(b['stages'])
