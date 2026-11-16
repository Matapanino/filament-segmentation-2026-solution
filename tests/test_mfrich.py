"""MF-rich selector stage contracts (the cross-validated mfrich arm), written before the package module.

The MF-rich selector reads the 39 quality + 15 context features of refit_ship's fresh_ctrl stage plus 8 prediction-only
features of every unrepaired candidate on the three-fold MF mean (mfrich_features.py, a byte-identical copy of
detect_20261112/mfrich/mf_features.py). Switching the selector back to the shipped LR39, the 54-feature fresh_ctrl selector
or the 60-feature P-neigh selector gives the old paths. The P-neigh selector is pinned as package data. Synthetic data only;
no fitting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_pnat import keep_all_model, synthetic_maps
from test_refit import selector54
from test_neighbour import selector60

CV_MF_FEATURES_SHA256 = '4ec5e961c27d696606ba4c1a0239b1dd6e2534edda293e536e5fb8d93ae35a85'
PNEIGH_SELECTOR_SHA256 = '12237dd42263df43ae81c894ef585af9d8f82b6910d99dd228349759afe7408c'


def selector62(coef=None, intercept=0., threshold=.375):
    from filseg.reproduction import refit
    n = len(refit.MFRICH_FEATURES)
    return dict(role=refit.MFRICH_ROLE, feature_order=list(refit.MFRICH_FEATURES), classes=[0, 1], scaler_mean=[0.] * n,
                scaler_scale=[1.] * n, coef=[list(coef) if coef is not None else [0.] * n], intercept=[float(intercept)], threshold=threshold)


def blank(n=96):
    return np.zeros((n, n), np.float32)


# ---------------------------------------------------------------- the 8 features (values from mfrich's tests/test_mf_features.py)
def test_module_is_the_cv_source_byte_for_byte():
    from filseg.reproduction import mfrich_features as MF
    assert hashlib.sha256(Path(MF.__file__).read_bytes()).hexdigest() == CV_MF_FEATURES_SHA256


def test_names_and_constants():
    from filseg.reproduction import mfrich_features as MF
    assert MF.FEATURES == ['mfr_q10', 'mfr_q50', 'mfr_q90', 'mfr_frac_ge50', 'mfr_blob_iou', 'mfr_ring_area_ratio', 'mfr_ring_max', 'mfr_blob_count']
    assert (MF.THRESHOLD, MF.RING, MF.CONNECTIVITY) == (.5, 16, 8)


def test_blob_iou_union_of_overlapping_components_and_inclusive_threshold():
    from filseg.reproduction import mfrich_features as MF
    mf = blank(); mask = np.zeros(mf.shape, bool); mask[20:30, 20:30] = True
    mf[20:25, 20:40] = .8; mf[70:80, 70:80] = .9; mf[29:32, 20:30] = .6
    r = MF.image_features([mask], mf)[0]
    inter = 50 + 10; union = 100 + 100 + 30 - inter
    assert abs(r['mfr_blob_iou'] - inter / union) < 1e-15 and r['mfr_blob_count'] == 2. and r['mfr_frac_ge50'] == .6
    mf = blank(); mf[20:30, 20:30] = .5
    assert MF.image_features([mask], mf)[0]['mfr_frac_ge50'] == 1.
    mf[20:30, 20:30] = np.nextafter(np.float32(.5), np.float32(0))
    assert MF.image_features([mask], mf)[0]['mfr_blob_count'] == 0.


def test_ring_is_the_square_ring_clipped_at_the_border():
    from filseg.reproduction import mfrich_features as MF
    mf = blank(128); mask = np.zeros(mf.shape, bool); mask[50:60, 50:60] = True
    mf[50:60, 61:66] = .7; mf[50:60, 80:90] = .99; mf[40, 40] = .3
    r = MF.image_features([mask], mf)[0]
    assert abs(r['mfr_ring_area_ratio'] - .5) < 1e-15 and abs(r['mfr_ring_max'] - float(np.float32(.7))) < 1e-7
    mf = blank(64); mask = np.zeros(mf.shape, bool); mask[0:5, 0:5] = True; mf[0:5, 0:5] = 1.
    assert MF.image_features([mask], mf)[0]['mfr_ring_max'] == 0.
    assert MF.image_features([], mf) == []


# ---------------------------------------------------------------- the 62-feature selector role
def test_mfrich_feature_order_and_cutoff():
    from filseg.reproduction import refit, mfrich_features
    assert refit.MFRICH_FEATURES == list(refit.FEATURES) + mfrich_features.FEATURES and len(refit.MFRICH_FEATURES) == 62
    assert refit.THRESHOLD == .375 and refit.MFRICH_ROLE not in (refit.ROLE, refit.PNEIGH_ROLE, refit.LR39_ROLE)


def test_mfrich_selector_file_is_hash_and_recipe_checked(tmp_path):
    from filseg.reproduction import refit
    good = tmp_path / 'sel.json'; good.write_text(json.dumps(selector62()))
    assert refit.load_selector(good, hashlib.sha256(good.read_bytes()).hexdigest())['role'] == refit.MFRICH_ROLE
    for change in (dict(threshold=.35), dict(feature_order=list(refit.FEATURES)), dict(feature_order=list(reversed(refit.MFRICH_FEATURES))),
                   dict(feature_order=list(refit.PNEIGH_FEATURES)), dict(coef=[[0.] * 60]), dict(role=refit.PNEIGH_ROLE)):
        bad = tmp_path / 'bad.json'; bad.write_text(json.dumps(dict(selector62(), **change)))
        with pytest.raises(ValueError):
            refit.load_selector(bad, hashlib.sha256(bad.read_bytes()).hexdigest())


def test_mfrich_rows_are_required_by_the_mfrich_selector_only():
    from filseg.reproduction import refit
    rows = [{n: 0. for n in refit.FEATURES[:39]}]; ctx = [{n: 0. for n in refit.FEATURES[39:]}]
    with pytest.raises(ValueError):
        refit.apply_selector(rows, ctx, selector62())
    with pytest.raises(ValueError):
        refit.apply_selector(rows, ctx, selector62(), mfrich_rows=[])
    assert refit.needs_mfrich(selector62()) and not refit.needs_mfrich(selector54()) and not refit.needs_mfrich(selector60())
    _, p, keep = refit.apply_selector(rows, ctx, selector62(intercept=1.), mfrich_rows=[{n: 0. for n in refit.MFRICH_FEATURES[54:]}])
    assert keep.tolist() == [True]


def test_mfrich_selector_reads_the_8_features_on_the_three_fold_mf_mean():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.arithmetic import mf_mean
    from filseg.reproduction import refit, mfrich_features
    from test_pcand import decode
    from scipy.special import expit
    maps = synthetic_maps()
    maps['MF1'][800:820, 800:900] = 100                 # the fold maps disagree on one bar: the mean is not any single fold
    raw, model = keep_all_model()
    fresh = image_stages(raw, maps, model, selector=selector54())
    masks = [decode(r) for r in fresh['rles']]
    mean = mf_mean(maps['MF0'], maps['MF1'], maps['MF2'])
    expected = mfrich_features.image_features(masks, mean)
    single = mfrich_features.image_features(masks, maps['MF0'].astype(np.float32) / np.float32(510))
    assert len(masks) == 3 and expected != single
    j = refit.MFRICH_FEATURES.index('mfr_q50'); coef = [0.] * 62; coef[j] = 100.
    cut = sorted(e['mfr_q50'] for e in expected)[1]
    out = image_stages(raw, maps, model, selector=selector62(coef=coef, intercept=-100. * cut))
    assert out['selector']['mfrich'] == expected
    assert out['selector']['context'] == fresh['selector']['context'] and 'neighbour' not in out['selector']
    z = 100. * (np.asarray([e['mfr_q50'] for e in expected]) - cut)
    assert np.array_equal(out['keep'], expit(z) >= .375) and 0 < int(out['keep'].sum()) < 3
    assert np.array_equal(np.asarray(out['selector']['probability']), expit(z))
    assert out['rles'] == fresh['rles']


def test_switching_back_to_lr39_fresh_ctrl_or_pneigh_gives_the_old_paths():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    from filseg.reproduction import refit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    a = image_stages(raw, maps, model, pcand=uniform(.4016711682033202))
    b = image_stages(raw, maps, model, pcand=uniform(.4016711682033202), selector=refit.lr39(model))
    assert a['rles'] == b['rles'] and 'mfrich' not in b['selector']
    for sel in (selector54(intercept=-10.), selector60(intercept=10.)):
        c = image_stages(raw, maps, model, selector=sel)
        assert 'mfrich' not in c['selector'] and set(c['stages']) == set(b['stages'])


# ---------------------------------------------------------------- pinned selectors addressed by name
def test_pinned_pneigh_selector_is_package_data():
    from filseg.reproduction import refit
    path, digest = refit.deployed('pneigh')
    assert digest == PNEIGH_SELECTOR_SHA256 and hashlib.sha256(Path(path).read_bytes()).hexdigest() == PNEIGH_SELECTOR_SHA256
    assert Path(path).parent == Path(refit.__file__).parent / 'data'
    sel = refit.load_selector(path, digest)
    assert sel['role'] == refit.PNEIGH_ROLE and sel['threshold'] == .375 and sel['feature_order'] == refit.PNEIGH_FEATURES
    assert sel['training']['images'] == 707
    assert refit.deployed() == refit.deployed('fresh_ctrl') == (refit.SELECTOR_FILE, refit.SELECTOR_SHA256)
    with pytest.raises(ValueError):
        refit.deployed('unknown')


def test_run_csv_named_selectors_resolve_to_the_pinned_files(monkeypatch):
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
    for selector, name in ((True, 'fresh_ctrl'), ('pneigh', 'pneigh')):
        with pytest.raises(Stop):
            pipeline.run_csv('I', None, 'M', 'O', selector=selector)
        assert seen[-1] == (str(refit.deployed(name)[0]), refit.deployed(name)[1])


# ---------------------------------------------------------------- the deployed MF-rich selector in the package
MFRICH_SELECTOR_SHA256 = 'bf499652dafb7bd2533c63fa2449761454ccd5ee137548d2945b9d8501d78556'


def test_pinned_mfrich_selector_is_package_data(monkeypatch):
    from filseg.reproduction import pipeline, refit
    path, digest = refit.deployed('mfrich')
    assert digest == MFRICH_SELECTOR_SHA256 and hashlib.sha256(Path(path).read_bytes()).hexdigest() == MFRICH_SELECTOR_SHA256
    assert Path(path).parent == Path(refit.__file__).parent / 'data'
    sel = refit.load_selector(path, digest)
    assert sel['role'] == refit.MFRICH_ROLE and sel['threshold'] == .375 and sel['feature_order'] == refit.MFRICH_FEATURES
    assert sel['training']['images'] == 707 and sel['training']['candidates'] == 5467
    seen = []

    class Stop(Exception):
        pass

    def loader(p, d):
        seen.append((str(p), d))
        raise Stop

    monkeypatch.setattr(refit, 'load_selector', loader)
    monkeypatch.setattr(pipeline, 'resolve_assets', lambda w: (Path('root'), {}))
    monkeypatch.setattr('filseg.reproduction.quality.load_quality', lambda p: {})
    with pytest.raises(Stop):
        pipeline.run_csv('I', None, 'M', 'O', selector='mfrich')
    assert seen == [(str(path), MFRICH_SELECTOR_SHA256)]
