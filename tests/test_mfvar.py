"""MF-var: the MF-TTA path plus five view-disagreement features of the 8 D4 views of each MF fold model, written before the
package change.

The CV arm (detect_20261113/mfvar) appends to the P-combo 68 features five features of each candidate computed from the 8
per-view MF maps of the image's out-of-fold MF model (view_features.py). At test the preregistered rule computes them for
each of MF0, MF1 and MF2 from that model's 8 views and averages over the three. The candidates and the MF map are the MF-TTA
ones (the three-fold mean of the 8-view means). The GPU stage keeps the 8 inverse-transformed views of every image (MF{k}_views,
(8, 2048, 2048) uint16, value / 510) in the same pass that writes the TTA archives, so the TTA archives are unchanged.
Synthetic data only; no model, no fitting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pnat import keep_all_model, synthetic_maps
from test_mfmaps import pool4, raw_image, variant_maps, selector68

VIEW_FEATURES_SHA256 = '35089c94885d2386f4038668977a9ceb4da08e8eab1632bb3c0ba4ea851e48be'   # detect_20261113/mfvar/view_features.py
VIEWS_IO_SHA256 = '377fac01a4fdaf4f5582d6bdcba3a587553c98d7c41b8f45b1e877adb1e249e3'        # detect_20261113/mfvar/views_io.py
CV_IMPORT = 'import mf_features as MF\n'
PACKAGE_IMPORT = 'from . import mfrich_features as MF\n'
NAMES = ['mfv_pixel_std_mean', 'mfv_view_vote', 'mfv_view_mean_min', 'mfv_view_mean_max', 'mfv_blob_iou_std']


# ---------------------------------------------------------------- the CV sources
def test_view_features_is_the_cv_module_with_the_package_import():
    from filseg.reproduction import view_features
    text = Path(view_features.__file__).read_text()
    assert text.count(PACKAGE_IMPORT) == 1 and CV_IMPORT not in text
    assert hashlib.sha256(text.replace(PACKAGE_IMPORT, CV_IMPORT).encode()).hexdigest() == VIEW_FEATURES_SHA256


def test_views_writer_is_the_cv_module_byte_for_byte():
    from filseg.reproduction import mf_views_io
    assert hashlib.sha256(Path(mf_views_io.__file__).read_bytes()).hexdigest() == VIEWS_IO_SHA256


def test_source_manifest_records_the_view_modules():
    import filseg.reproduction as package
    manifest = json.loads((Path(package.__file__).parent / 'source_manifest.json').read_text())
    assert manifest['mf_views_io.py'] == dict(source='detect_20261113/mfvar/views_io.py', sha256=VIEWS_IO_SHA256, byte_identical=True)
    assert manifest['view_features.py'] == dict(source='detect_20261113/mfvar/view_features.py', sha256=VIEW_FEATURES_SHA256, byte_identical=False,
                                                change='import: ' + CV_IMPORT.strip() + ' -> ' + PACKAGE_IMPORT.strip())


# ---------------------------------------------------------------- the five features (hand-computed cases)
def bar(y0=10, x0=10, h=4, w=20, shape=(64, 64)):
    m = np.zeros(shape, bool); m[y0:y0 + h, x0:x0 + w] = True
    return m


def test_five_features_of_a_hand_computed_case():
    from filseg.reproduction import view_features as VF
    assert VF.FEATURES == NAMES
    m = bar()
    views = np.zeros((8, 64, 64), np.uint16)
    for i in range(8):
        views[i][m] = 510 if i < 6 else 102          # six views see the bar at 1.0, two at 0.2 (below 0.5: no blob)
    (r,) = VF.image_features([m], views)
    assert list(r) == NAMES
    vals = np.asarray([1.] * 6 + [np.float32(102) / np.float32(510)] * 2, np.float64)
    assert r['mfv_pixel_std_mean'] == pytest.approx(float(vals.std()), abs=1e-7)
    assert r['mfv_view_vote'] == 6 / 8
    assert r['mfv_view_mean_min'] == pytest.approx(0.2, abs=1e-7) and r['mfv_view_mean_max'] == 1.
    assert r['mfv_blob_iou_std'] == pytest.approx(float(np.std([1.] * 6 + [0.] * 2)), abs=1e-12)


def test_blob_iou_counts_the_whole_overlapping_component():
    from filseg.reproduction import view_features as VF
    m = bar(h=4, w=20)
    views = np.zeros((8, 64, 64), np.uint16)
    for i in range(8):
        views[i][10:14, 10:50] = 510                  # every view: one blob twice the candidate's length
    (r,) = VF.image_features([m], views)
    assert r['mfv_view_vote'] == 0. and r['mfv_blob_iou_std'] == 0.   # IoU 0.5 is not > 0.5
    assert r['mfv_pixel_std_mean'] == 0. and r['mfv_view_mean_min'] == r['mfv_view_mean_max'] == 1.


def test_no_candidate_gives_no_rows_and_wrong_view_count_is_refused():
    from filseg.reproduction import view_features as VF
    assert VF.image_features([], np.zeros((8, 8, 8), np.uint16)) == []
    with pytest.raises(AssertionError):
        VF.image_features([bar(shape=(8, 8), y0=1, x0=1, h=2, w=2)], np.zeros((7, 8, 8), np.uint16))


def test_test_time_rule_averages_the_three_fold_models():
    from filseg.reproduction import view_features as VF
    rng = np.random.default_rng(3)
    masks = [bar(), bar(y0=30, x0=5, h=3, w=30)]
    per = [rng.integers(0, 511, (8, 64, 64)).astype(np.uint16) for _ in range(3)]
    got = VF.test_image_features(masks, per)
    each = [VF.image_features(masks, v) for v in per]
    for j in range(2):
        for n in NAMES:
            assert got[j][n] == float(np.mean([e[j][n] for e in each]))
    with pytest.raises(AssertionError):
        VF.test_image_features(masks, per[:2])


# ---------------------------------------------------------------- GPU stage: the 8 views in the TTA pass
def test_image_views_with_views_keeps_the_tta_and_identity_maps_unchanged():
    from filseg.reproduction import mfmaps, mf_d4 as D4
    raw = raw_image()
    old = mfmaps.image_views(pool4, raw, device='cpu', views=('tta',))
    new = mfmaps.image_views(pool4, raw, device='cpu', views=('tta', 'views'))
    assert set(new) == {'identity', 'tta', 'views'}
    assert np.array_equal(new['identity'], old['identity']) and np.array_equal(new['tta'], old['tta'])
    v = new['views']
    assert v.shape == (8, 2048, 2048) and v.dtype == np.uint16 and int(v.max()) <= 510
    assert np.array_equal(v[0], new['identity'])
    _, _, views = D4.tta(pool4, mfmaps.model_input(raw, device='cpu'), keep_views=True)
    assert all(np.array_equal(v[i], D4.quantise(D4.upsample(views[i]))) for i in range(8))


def test_views_kind_is_opt_in_and_needs_tta():
    from filseg.reproduction import mfmaps
    assert mfmaps.VIEWS == ('MF0_views', 'MF1_views', 'MF2_views')
    with pytest.raises(ValueError):
        mfmaps.run_mf_view_maps('I', None, 'O', views=('views',))
    with pytest.raises(ValueError):
        mfmaps.run_mf_view_maps('I', None, 'O', views=('tta', 'nope'))


def test_views_archive_is_deterministic_sorted_and_loads_like_save_npz(tmp_path):
    from filseg.reproduction.mf_views_io import ViewsWriter
    arrays = {k: np.random.default_rng(i).integers(0, 511, (8, 16, 16)).astype(np.uint16) for i, k in enumerate(['a', 'b', 'c'])}
    digests = []
    for name in ('one.npz', 'two.npz'):
        w = ViewsWriter(tmp_path / name)
        for k in sorted(arrays): w.write(k, arrays[k])
        w.close(); digests.append(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest())
    assert digests[0] == digests[1]
    with np.load(tmp_path / 'one.npz', allow_pickle=False) as z:
        assert sorted(z.files) == ['a', 'b', 'c'] and all(np.array_equal(z[k], arrays[k]) for k in arrays)
    w = ViewsWriter(tmp_path / 'three.npz'); w.write('b', arrays['b'])
    with pytest.raises(AssertionError):
        w.write('a', arrays['a'])


def test_cli_infer_mfviews_accepts_the_views_kind(monkeypatch):
    import sys
    from filseg.reproduction import __main__ as entry, launch
    calls = []
    monkeypatch.setattr(launch, 'run_gpu_mf_views', lambda *a, **k: calls.append((a, k)) or dict(status='complete'))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'infer-mfviews', '--images', 'I', '--output', 'O', '--views', 'tta', 'views'])
    entry.main()
    assert calls[-1] == (('I', None, 'O', 'bf16', 6600), dict(deterministic=True, views=('tta', 'views')))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'infer-mfviews', '--images', 'I', '--output', 'O', '--views', 'views'])
    with pytest.raises(SystemExit):
        entry.main()


# ---------------------------------------------------------------- CPU: view archives, selector role, mode
def write_views(root, keys, value=None):
    from filseg.reproduction import mfmaps
    from filseg.reproduction.mf_views_io import ViewsWriter
    from filseg.reproduction.audit import identity
    networks = {}
    for k in range(3):
        name = f'MF{k}_views'; p = Path(root) / (name + '.npz'); w = ViewsWriter(p)
        for key in sorted(keys): w.write(key, np.full((8, 4, 4), k if value is None else value, np.uint16))
        w.close(); networks[name] = dict(file=p.name, **identity(p))
    path = Path(root) / mfmaps.RESULT
    old = json.loads(path.read_text()) if path.exists() else dict(status='complete', networks={})
    old['networks'].update(networks); path.write_text(json.dumps(old))
    return networks


def test_view_archives_are_loaded_hash_checked(tmp_path):
    from filseg.reproduction import mfmaps
    keys = ['a', 'b']; write_views(tmp_path, keys)
    got = mfmaps.load_view_archives(tmp_path, keys)
    assert [n for n, _ in got] == list(mfmaps.VIEWS) and got[2][1]['b'][0, 0, 0] == 2
    with pytest.raises(ValueError):
        mfmaps.load_view_archives(tmp_path, keys + ['c'])
    (tmp_path / 'MF1_views.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):
        mfmaps.load_view_archives(tmp_path, keys)


def selector73(threshold=.3, coef=None, intercept=0.):
    from filseg.reproduction import refit
    n = len(refit.MFVAR_FEATURES)
    return dict(role=refit.MFVAR_ROLE, feature_order=list(refit.MFVAR_FEATURES), classes=[0, 1], scaler_mean=[0.] * n, scaler_scale=[1.] * n,
                coef=[list(coef) if coef is not None else [0.] * n], intercept=[float(intercept)], threshold=threshold)


def test_mfvar_role_feature_order_and_cutoff():
    from filseg.reproduction import refit, view_features
    assert refit.MFVAR_FEATURES == list(refit.PCOMBO_FEATURES) + list(view_features.FEATURES) and len(refit.MFVAR_FEATURES) == 73
    assert refit.MFVAR_THRESHOLD == 0.3            # the median of mfvar's 100 nested cutoffs (sorted values 50 and 51: 0.3)
    assert refit.MFVAR_ROLE == 'deployment_refit_mfvar_full707_mf_mixture'
    assert refit.check(selector73())['role'] == refit.MFVAR_ROLE
    for bad in (selector73(threshold=.35), dict(selector73(), feature_order=list(refit.PCOMBO_FEATURES)),
                dict(selector73(), feature_order=list(refit.PCOMBO_FEATURES) + list(reversed(view_features.FEATURES))), dict(selector73(), coef=[[0.] * 68])):
        with pytest.raises(ValueError):
            refit.check(bad)
    assert refit.needs_views(selector73()) and not refit.needs_views(selector68(refit.PCOMBO_ROLE, .35))


def test_mfvar_selector_needs_the_view_rows_and_reads_them_in_order():
    from filseg.reproduction import refit
    from scipy.special import expit
    names = refit.MFVAR_FEATURES
    rng = np.random.default_rng(5); X = rng.normal(size=(4, 73)); coef = rng.normal(size=73)
    rows = [{n: float(v) for n, v in zip(names[:39], x[:39])} for x in X]
    ctx = [{n: float(v) for n, v in zip(names[39:54], x[39:54])} for x in X]
    nb = [{n: float(v) for n, v in zip(names[54:60], x[54:60])} for x in X]
    mf = [{n: float(v) for n, v in zip(names[60:68], x[60:68])} for x in X]
    vw = [{n: float(v) for n, v in zip(reversed(names[68:]), reversed(x[68:]))} for x in X]
    with pytest.raises(ValueError):
        refit.apply_selector(rows, ctx, selector73(coef=coef), nb, mf)
    x, p, keep = refit.apply_selector(rows, ctx, selector73(coef=coef, intercept=.2), nb, mf, vw)
    assert np.array_equal(x, X) and np.max(np.abs(p - expit(X @ coef + .2))) < 1e-12 and np.array_equal(keep, p >= .3)
    sel = selector68(refit.PCOMBO_ROLE, .35)          # an older selector given view rows ignores them
    a = refit.apply_selector(rows, ctx, sel, nb, mf); b = refit.apply_selector(rows, ctx, sel, nb, mf, vw)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def synthetic_views(maps, spread=0):
    """Three fold models' 8 views: each view = the fold's MF map shifted by a view-dependent amount (clipped)."""
    out = []
    for k in range(3):
        m = maps[f'MF{k}_tta'].astype(np.int32)
        out.append(np.stack([np.clip(m + spread * (i - 4), 0, 510) for i in range(8)]).astype(np.uint16))
    return out


def test_image_stages_mfvar_is_the_mftta_path_plus_the_averaged_view_features():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction import refit, view_features as VF
    from scipy.special import expit
    maps = variant_maps(tta_shift=-20); raw, model = keep_all_model(); views = synthetic_views(maps, spread=30)
    coef68 = [0.] * 68; coef68[refit.PCOMBO_FEATURES.index('mfr_q50')] = 10.
    base = image_stages(raw, maps, model, selector=dict(selector68('deployment_refit_mftta_full707_mf_mixture', .35), coef=[coef68], intercept=[-4.]),
                        mf_variant='mftta')
    coef = coef68 + [0.] * 5; coef[refit.MFVAR_FEATURES.index('mfv_pixel_std_mean')] = -20.
    out = image_stages(raw, maps, model, selector=selector73(coef=coef, intercept=-4.), mf_variant='mftta', views=views)
    assert out['stages']['MF_mean'] == base['stages']['MF_mean'] and out['stages']['candidate_blend'] == base['stages']['candidate_blend']
    assert out['candidates'] == base['candidates'] and out['selector']['context'] == base['selector']['context']
    assert out['selector']['mfrich'] == base['selector']['mfrich'] and out['selector']['neighbour'] == base['selector']['neighbour']
    from pycocotools import mask as mu
    masks = [mu.decode(dict(size=[2048, 2048], counts=r.encode())).astype(bool) for r in out['rles']]
    expected = VF.test_image_features(masks, views)
    assert out['selector']['views'] == expected and len(expected) == out['candidates'] > 0
    z = -4. + 10. * np.asarray([r['mfr_q50'] for r in base['selector']['mfrich']]) - 20. * np.asarray([r['mfv_pixel_std_mean'] for r in expected])
    assert np.array_equal(np.asarray(out['selector']['probability']), expit(z)) and np.array_equal(out['keep'], expit(z) >= .3)
    with pytest.raises(ValueError):
        image_stages(raw, maps, model, selector=selector73(), mf_variant='mftta')      # the view maps are required


def test_image_stages_ignores_views_for_the_older_selectors():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    maps = variant_maps(tta_shift=-20); raw, model = keep_all_model(); views = synthetic_views(maps, spread=30)
    sel = selector68('deployment_refit_mftta_full707_mf_mixture', .35)
    a = image_stages(raw, maps, model, selector=sel, mf_variant='mftta', pcand=uniform(.4016711682033202))
    b = image_stages(raw, maps, model, selector=sel, mf_variant='mftta', pcand=uniform(.4016711682033202), views=views)
    assert a['stages'] == b['stages'] and a['rles'] == b['rles'] and 'views' not in b['selector']


def test_run_csv_has_the_views_switch_off_by_default():
    import inspect
    from filseg.reproduction import pipeline
    assert inspect.signature(pipeline.image_stages).parameters['views'].default is None
    assert inspect.signature(pipeline.run_csv).parameters['mf_views'].default is False


def test_mfvar_mode_is_listed_but_not_runnable_until_its_csv_is_pinned():
    from filseg.reproduction import modes, mfmaps
    row = modes.MFMAP_MODES['mfvar']
    assert row['kwargs'] == dict(pnat=False, pcand=True, selector='mfvar', mf_variant='mftta', mf_views=True)
    assert row['map_set'] == mfmaps.EXPECTED_MAP_SETS['mftta'] and row['label']
    assert list(modes.MFMAP_MODES) == ['mftta', 'mfhires', 'mfcomb', 'mfvar']
    if row['sha256'] is None:
        assert 'mfvar' not in modes.runnable()
        with pytest.raises(ValueError):
            modes.resolve('mfvar')


def test_cli_mfvar_mode_reaches_run_csv_with_the_views(monkeypatch):
    from test_modes import cli
    from filseg.reproduction.modes import MFMAP_MODES
    args, kwargs = cli(monkeypatch, '--mode', 'mfvar', '--variant-maps', 'V')
    assert args == ('I', None, 'M', 'O') and kwargs == dict(MFMAP_MODES['mfvar']['kwargs'], variant_maps='V')


# ---------------------------------------------------------------- the deployed MF-var selector in the package
# Fitted on Kaggle CPU (mfvar_ship Stage 1a, kernel fit2): two fits gave these exact bytes (as did the earlier fit kernel); the
# 100 saved 'mfvar' and the 100 saved 'mftta' CV selectors of the mfvar lane were refitted exactly from the same bank.
MFVAR_SELECTOR_SHA256 = '53f4db37e99351451047d2c1c7f9be1377fe1d18332edc864048770aed06cc27'


def test_pinned_mfvar_selector_is_package_data():
    from filseg.reproduction import refit
    path, digest = refit.deployed('mfvar')
    assert digest == MFVAR_SELECTOR_SHA256 and hashlib.sha256(Path(path).read_bytes()).hexdigest() == MFVAR_SELECTOR_SHA256
    assert Path(path) == Path(refit.__file__).parent / 'data/mfvar_selector.json'
    sel = refit.load_selector(path, digest)
    assert sel['role'] == refit.MFVAR_ROLE and sel['threshold'] == .3 and sel['feature_order'] == refit.MFVAR_FEATURES
    assert sel['training']['images'] == 707 and sel['training']['candidates'] == 5482 and sel['training']['expanded_rows'] == 8928
