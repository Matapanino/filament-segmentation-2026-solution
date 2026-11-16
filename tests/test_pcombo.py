"""P-combo selector stage contracts (the cross-validated pcombo arm), written before the package change.

The P-combo selector reads the 54 features of refit_ship's fresh_ctrl stage, then the 6 neighbour features of the P-neigh
stage (neighbour.py, on the candidate blend), then the 8 MF-rich features of the MF-rich stage (mfrich_features.py, on the
three-fold MF mean): 68 columns in the cross-validated order (detect_20261112/pcombo protocol.json features.pcombo). Its
cutoff is 0.35, the median of pcombo's 100 nested cutoffs. Switching the selector back to LR39, fresh_ctrl, P-neigh or
MF-rich gives the old paths. Synthetic data only; no fitting.
"""
import hashlib
import json

import numpy as np
import pytest

from test_pnat import keep_all_model, synthetic_maps
from test_refit import selector54
from test_neighbour import selector60
from test_mfrich import selector62


def selector68(coef=None, intercept=0., threshold=.35):
    from filseg.reproduction import refit
    n = len(refit.PCOMBO_FEATURES)
    return dict(role=refit.PCOMBO_ROLE, feature_order=list(refit.PCOMBO_FEATURES), classes=[0, 1], scaler_mean=[0.] * n,
                scaler_scale=[1.] * n, coef=[list(coef) if coef is not None else [0.] * n], intercept=[float(intercept)], threshold=threshold)


def test_pcombo_feature_order_and_cutoff():
    from filseg.reproduction import refit, neighbour, mfrich_features
    assert refit.PCOMBO_FEATURES == list(refit.FEATURES) + list(neighbour.FEATURES) + list(mfrich_features.FEATURES)
    assert len(refit.PCOMBO_FEATURES) == 68 and len(set(refit.PCOMBO_FEATURES)) == 68
    assert refit.PCOMBO_FEATURES[:60] == refit.PNEIGH_FEATURES and refit.PCOMBO_FEATURES[60:] == refit.MFRICH_FEATURES[54:]
    assert refit.PCOMBO_THRESHOLD == .35 and refit.THRESHOLD == .375
    assert refit.PCOMBO_ROLE not in (refit.ROLE, refit.PNEIGH_ROLE, refit.MFRICH_ROLE, refit.LR39_ROLE)


def test_pcombo_selector_file_is_hash_and_recipe_checked(tmp_path):
    from filseg.reproduction import refit
    good = tmp_path / 'sel.json'; good.write_text(json.dumps(selector68()))
    assert refit.load_selector(good, hashlib.sha256(good.read_bytes()).hexdigest())['role'] == refit.PCOMBO_ROLE
    for change in (dict(threshold=.375), dict(feature_order=list(refit.MFRICH_FEATURES)), dict(feature_order=list(reversed(refit.PCOMBO_FEATURES))),
                   dict(feature_order=list(refit.FEATURES) + list(refit.MFRICH_FEATURES[54:]) + list(refit.PNEIGH_FEATURES[54:])),
                   dict(coef=[[0.] * 62]), dict(role=refit.MFRICH_ROLE)):
        bad = tmp_path / 'bad.json'; bad.write_text(json.dumps(dict(selector68(), **change)))
        with pytest.raises(ValueError):
            refit.load_selector(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
    for old in (selector54(threshold=.35), selector60(threshold=.35), selector62(threshold=.35)):    # the 0.35 cutoff is P-combo's only
        bad = tmp_path / 'old.json'; bad.write_text(json.dumps(old))
        with pytest.raises(ValueError):
            refit.load_selector(bad, hashlib.sha256(bad.read_bytes()).hexdigest())


def test_pcombo_selector_needs_both_added_blocks():
    from filseg.reproduction import refit
    rows = [{n: 0. for n in refit.FEATURES[:39]}]; ctx = [{n: 0. for n in refit.FEATURES[39:]}]
    nb = [{n: 0. for n in refit.PNEIGH_FEATURES[54:]}]; mf = [{n: 0. for n in refit.MFRICH_FEATURES[54:]}]
    assert refit.needs_neighbour(selector68()) and refit.needs_mfrich(selector68()) and refit.needs_context(selector68())
    for args in ((None, None), (nb, None), (None, mf)):
        with pytest.raises(ValueError):
            refit.apply_selector(rows, ctx, selector68(), *args)
    x, p, keep = refit.apply_selector(rows, ctx, selector68(intercept=float(np.log(.35 / .65))), nb, mf)
    assert x.shape == (1, 68) and keep.tolist() == [bool(p[0] >= .35)]


def test_pcombo_columns_follow_the_feature_order_not_the_dict_order():
    from filseg.reproduction import refit
    from scipy.special import expit
    rng = np.random.default_rng(11); X = rng.normal(size=(7, 68)); coef = rng.normal(size=68)
    names = refit.PCOMBO_FEATURES
    rows = [{n: float(v) for n, v in zip(names[:39], x[:39])} for x in X]
    ctx = [{n: float(v) for n, v in zip(reversed(names[39:54]), reversed(x[39:54]))} for x in X]
    nb = [{n: float(v) for n, v in zip(reversed(names[54:60]), reversed(x[54:60]))} for x in X]
    mf = [{n: float(v) for n, v in zip(reversed(names[60:]), reversed(x[60:]))} for x in X]
    x, p, keep = refit.apply_selector(rows, ctx, selector68(coef=coef, intercept=.3), nb, mf)
    assert np.array_equal(x, X)
    assert np.max(np.abs(p - expit(X @ coef + .3))) < 1e-12 and np.array_equal(keep, p >= .35)


def test_pcombo_selector_reads_the_pneigh_and_mfrich_rows_of_the_same_candidates():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction import refit
    from scipy.special import expit
    maps = synthetic_maps()
    maps['MF1'][800:820, 800:900] = 100                 # the fold maps disagree on one bar: the mean is not any single fold
    raw, model = keep_all_model()
    fresh = image_stages(raw, maps, model, selector=selector54())
    pn = image_stages(raw, maps, model, selector=selector60())
    mr = image_stages(raw, maps, model, selector=selector62())
    coef = [0.] * 68
    coef[refit.PCOMBO_FEATURES.index('neigh_gap_nearest')] = -1.
    coef[refit.PCOMBO_FEATURES.index('mfr_q50')] = 50.
    out = image_stages(raw, maps, model, selector=selector68(coef=coef, intercept=60.))
    assert out['selector']['neighbour'] == pn['selector']['neighbour'] and out['selector']['mfrich'] == mr['selector']['mfrich']
    assert out['selector']['context'] == fresh['selector']['context'] and out['rles'] == fresh['rles']
    z = 60. - np.asarray([e['neigh_gap_nearest'] for e in pn['selector']['neighbour']]) + 50. * np.asarray([e['mfr_q50'] for e in mr['selector']['mfrich']])
    assert np.array_equal(np.asarray(out['selector']['probability']), expit(z)) and np.array_equal(out['keep'], expit(z) >= .35)
    assert out['selector']['lr39_keep'] == fresh['selector']['lr39_keep']


def test_switching_back_to_the_older_selectors_gives_their_paths():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    from filseg.reproduction import refit
    maps = synthetic_maps()
    raw, model = keep_all_model()
    a = image_stages(raw, maps, model, pcand=uniform(.4016711682033202))
    b = image_stages(raw, maps, model, pcand=uniform(.4016711682033202), selector=refit.lr39(model))
    assert a['rles'] == b['rles'] and 'neighbour' not in b['selector'] and 'mfrich' not in b['selector']
    for sel, extra in ((selector54(intercept=-10.), set()), (selector60(intercept=10.), {'neighbour'}), (selector62(intercept=10.), {'mfrich'})):
        c = image_stages(raw, maps, model, selector=sel)
        assert {'neighbour', 'mfrich'} & set(c['selector']) == extra and set(c['stages']) == set(b['stages'])
    assert refit.deployed() == refit.deployed('fresh_ctrl') and set(refit.PINNED) >= {'fresh_ctrl', 'pneigh', 'mfrich'}


# ---------------------------------------------------------------- the deployed P-combo selector in the package
PCOMBO_SELECTOR_SHA256 = 'fd25d491d828aaac1bdd8c07d120edbe32ac24d8f5980599d00190f7573b389c'


def test_pinned_pcombo_selector_is_package_data(monkeypatch):
    from pathlib import Path
    from filseg.reproduction import pipeline, refit
    path, digest = refit.deployed('pcombo')
    assert digest == PCOMBO_SELECTOR_SHA256 and hashlib.sha256(Path(path).read_bytes()).hexdigest() == PCOMBO_SELECTOR_SHA256
    assert Path(path).parent == Path(refit.__file__).parent / 'data'
    sel = refit.load_selector(path, digest)
    assert sel['role'] == refit.PCOMBO_ROLE and sel['threshold'] == .35 and sel['feature_order'] == refit.PCOMBO_FEATURES
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
        pipeline.run_csv('I', None, 'M', 'O', selector='pcombo')
    assert seen == [(str(path), PCOMBO_SELECTOR_SHA256)]
