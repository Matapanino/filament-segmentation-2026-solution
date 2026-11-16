"""Refit selector stage contracts, written before the package module.

The stage replaces the shipped LR39 keep with a selector on the 39 quality + 15 context features of the unrepaired
candidates (the cross-validated fresh_ctrl arm). Loaded with the shipped LR39 it must give the old path exactly.
Synthetic data only; no fitting.
"""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from test_pnat import keep_all_model, synthetic_maps
from test_pcand import decode

TAU = 0.4016711682033202


def ctx_maps(maps):
    from filseg.reproduction.arithmetic import mf_mean
    return dict(full=maps['full'].astype(np.float32) / 255, local=maps['local'].astype(np.float32) / 255,
                m2f_public=mf_mean(maps['MF0'], maps['MF1'], maps['MF2']))


def selector54(coef=None, intercept=0., threshold=.375, scale=None):
    from filseg.reproduction import refit
    n = len(refit.FEATURES)
    return dict(role=refit.ROLE, feature_order=list(refit.FEATURES), classes=[0, 1], scaler_mean=[0.] * n,
                scaler_scale=list(scale) if scale is not None else [1.] * n, coef=[list(coef) if coef is not None else [0.] * n],
                intercept=[float(intercept)], threshold=threshold)


# ---------------------------------------------------------------- off by default, recipe constants
def test_selector_stage_is_off_by_default():
    from filseg.reproduction import pipeline
    assert inspect.signature(pipeline.image_stages).parameters['selector'].default is None
    assert inspect.signature(pipeline.run_csv).parameters['selector'].default is None


def test_feature_order_is_the_39_quality_then_the_15_context_features():
    from filseg.reproduction import refit, pcand
    from filseg.reproduction.context_features import FEATURES as CTX
    assert list(refit.FEATURES) == list(pcand.FEATURES[:54]) and list(refit.FEATURES[39:]) == CTX
    assert refit.THRESHOLD == 0.375


# ---------------------------------------------------------------- switching back gives the old path
def test_lr39_through_the_selector_stage_equals_the_default_path():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction import refit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    for intercept in (0., -10.):
        m = dict(model, intercept=[intercept])
        a = image_stages(raw, maps, m)
        b = image_stages(raw, maps, m, selector=refit.lr39(m))
        assert a['rles'] == b['rles'] and np.array_equal(a['keep'], b['keep']) and np.array_equal(a['scores'], b['scores'])
        assert {k: v for k, v in b['stages'].items() if k in a['stages']} == a['stages']
        assert b['selector']['lr39_keep'] == a['keep'].tolist() and b['selector']['context'] is None


def test_lr39_through_the_selector_stage_equals_the_default_pcand_path():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    from filseg.reproduction import refit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    a = image_stages(raw, maps, model, pcand=uniform(TAU))
    b = image_stages(raw, maps, model, pcand=uniform(TAU), selector=refit.lr39(model))
    assert a['rles'] == b['rles'] and np.array_equal(a['keep'], b['keep']) and a['pcand'] == b['pcand'] and a['pcand']['kept'] >= 2


def test_lr39_arithmetic_is_bitwise_the_shipped_apply_quality():
    from filseg.reproduction.quality import apply_quality
    from filseg.reproduction import refit
    rng = np.random.default_rng(1112)
    maps = synthetic_maps()
    raw, model = keep_all_model()
    model = dict(model, scaler_mean=rng.normal(size=39).tolist(), scaler_scale=(rng.random(39) + .5).tolist(),
                 coef=[rng.normal(size=39).tolist()], intercept=[float(rng.normal())])
    from filseg.reproduction.instancing import iter_instances, SHIP_R3
    from filseg.reproduction.arithmetic import candidate_blend
    masks = [np.asarray(m, bool) for m in iter_instances(candidate_blend(*(maps[n] for n in ('full', 'local', 'MF0', 'MF1', 'MF2'))), SHIP_R3)]
    rows, x, prob, keep = apply_quality(masks, maps['full'], maps['local'], raw, model)
    x2, prob2, keep2 = refit.apply_selector(rows, None, refit.lr39(model))
    assert np.array_equal(x, x2) and np.array_equal(prob, prob2) and np.array_equal(keep, keep2) and len(masks) >= 2


# ---------------------------------------------------------------- the 54-feature selector
def test_selector_reads_the_39_quality_and_15_context_features_of_the_unrepaired_candidates():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.context_features import context_features
    from filseg.reproduction.features import features_for_mask
    from filseg.reproduction import refit
    from scipy.special import expit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    base = image_stages(raw, maps, model)
    masks = [decode(r) for r in base['rles']]
    probs = dict(full=maps['full'].astype(np.float32) / 255, local=maps['local'].astype(np.float32) / 255)
    probs['blend'] = (probs['full'] + probs['local']) * .5
    expected = []
    for m in masks:
        q = features_for_mask(m, probs, raw)
        row = {**{n: float(q[n]) for n in refit.FEATURES[:39]}, **context_features(m, ctx_maps(maps), raw)}
        expected.append([row[n] for n in refit.FEATURES])
    expected = np.asarray(expected, float)
    # a selector that keeps exactly the candidates whose mf_mean context feature is above the median
    j = refit.FEATURES.index('mf_mean'); cut = float(np.median(expected[:, j]))
    coef = [0.] * 54; coef[j] = 1000.
    sel = selector54(coef=coef, intercept=-1000. * cut, threshold=.375)
    out = image_stages(raw, maps, model, selector=sel)
    z = expected @ np.asarray(coef) - 1000. * cut
    assert np.array_equal(out['keep'], expit(z) >= .375)
    assert np.array_equal(np.asarray(out['selector']['probability']), expit(z))
    ctx = out['selector']['context']
    assert len(ctx) == len(masks) and all(list(c) == refit.FEATURES[39:] for c in ctx)
    assert np.array_equal(np.asarray([[c[n] for n in refit.FEATURES[39:]] for c in ctx]), expected[:, 39:])
    assert out['selector']['lr39_keep'] == base['keep'].tolist()
    assert out['rles'] == base['rles']  # no repair stage: unrepaired candidates


def test_pcand_after_the_selector_repairs_only_the_new_kept_set():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    from filseg.reproduction import refit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    seen = []

    def predictor(x):
        seen.append(np.asarray(x))
        return [TAU] * len(x)

    drop_all = selector54(intercept=-10.)
    out = image_stages(raw, maps, model, pcand=predictor, selector=drop_all)
    assert not np.any(out['keep']) and seen == [] and out['pcand']['kept'] == 0
    keep_all = selector54(intercept=10.)
    a = image_stages(raw, maps, model, pcand=uniform(TAU))
    b = image_stages(raw, maps, model, pcand=uniform(TAU), selector=keep_all)
    assert a['rles'] == b['rles'] and np.array_equal(a['keep'], b['keep'])  # keep_all_model also keeps everything


def test_selector_and_pnat_compose_like_pcand():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction import refit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    a = image_stages(raw, maps, model, pnat=True)
    b = image_stages(raw, maps, model, pnat=True, selector=refit.lr39(model))
    assert a['rles'] == b['rles'] and a['pnat'] == b['pnat']


# ---------------------------------------------------------------- loader
def test_selector_file_is_hash_checked_and_recipe_checked(tmp_path):
    from filseg.reproduction import refit
    good = tmp_path / 'sel.json'
    good.write_text(json.dumps(selector54()))
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    assert refit.load_selector(good, digest)['threshold'] == .375
    with pytest.raises(ValueError):
        refit.load_selector(good, '0' * 64)
    for change in (dict(threshold=.35), dict(feature_order=list(reversed(refit.FEATURES))), dict(classes=[1, 0]), dict(role='other'),
                   dict(coef=[[0.] * 53])):
        bad = tmp_path / 'bad.json'
        bad.write_text(json.dumps(dict(selector54(), **change)))
        with pytest.raises(ValueError):
            refit.load_selector(bad, hashlib.sha256(bad.read_bytes()).hexdigest())


def test_nonfinite_selector_probability_is_an_error():
    from filseg.reproduction import refit
    rows = [{n: 0. for n in refit.FEATURES[:39]}]
    ctx = [{n: 0. for n in refit.FEATURES[39:]}]
    sel = selector54(scale=[0.] * 54)
    with pytest.raises(ValueError):
        refit.apply_selector(rows, ctx, sel)


# ---------------------------------------------------------------- the deployed refit selector in the package
SELECTOR_SHA256 = '05759a2561e702f1fa4c044c36e32a1e867be15a1f6122451f090edabd2fd2fe'


def test_deployed_selector_file_is_pinned_and_loads():
    from filseg.reproduction import refit
    path, digest = refit.deployed()
    assert digest == SELECTOR_SHA256 == refit.SELECTOR_SHA256
    assert hashlib.sha256(open(path, 'rb').read()).hexdigest() == SELECTOR_SHA256
    sel = refit.load_selector(path, digest)
    assert sel['role'] == refit.ROLE and sel['threshold'] == .375 and sel['feature_order'] == refit.FEATURES
    assert sel['training']['images'] == 707


def test_run_csv_selector_true_resolves_to_the_pinned_file(monkeypatch):
    from filseg.reproduction import pipeline, refit
    seen = []

    class Stop(Exception):
        pass

    def loader(path, digest):
        seen.append((str(path), digest))
        raise Stop

    monkeypatch.setattr(refit, 'load_selector', loader)
    monkeypatch.setattr(pipeline, 'resolve_assets', lambda w: (Path('root'), {}))
    monkeypatch.setattr('filseg.reproduction.quality.load_quality', lambda p: {})
    with pytest.raises(Stop):
        pipeline.run_csv('I', None, 'M', 'O', selector=True)
    assert seen == [(str(refit.deployed()[0]), SELECTOR_SHA256)]
