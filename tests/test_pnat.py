"""P-nat stage contracts, written before the package module.

The cross-validated implementation is pinned as a verbatim fixture (tests/data/pnat_cv_reference.py,
the frozen pnat_crossfold adapter) and compared candidate by candidate. Synthetic data only.
"""
import ast
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / 'data'
CV_ADAPTER_SHA256 = 'dcfb2ce0f2c8a15c6c541bc935811a185d7fc4f5fc69df5e15eda36d3c37d038'
CV_REFINE_OPS_SHA256 = '2fbd926ac78eb6b5aeebe1eb0be944988ba12af6074e784668a4c2b389bd9337'
TAU = 0.4016711682033202


def cv_reference():
    """Import the verbatim CV adapter; its `import refine_ops` resolves to the byte-identical package module."""
    from filseg.reproduction import refine_ops
    path = DATA / 'pnat_cv_reference.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CV_ADAPTER_SHA256
    sys.modules['refine_ops'] = refine_ops
    spec = importlib.util.spec_from_file_location('pnat_cv_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def small_image():
    from filseg.reproduction import refine_ops as R
    old = R.N
    R.N = 128
    yield R
    R.N = old


def mask(y=60, x=40, h=3, w=15, n=128):
    m = np.zeros((n, n), bool)
    m[y:y + h, x:x + w] = True
    return m


def test_refine_ops_is_byte_identical_to_the_cv_source():
    from filseg.reproduction import refine_ops
    assert hashlib.sha256(Path(refine_ops.__file__).read_bytes()).hexdigest() == CV_REFINE_OPS_SHA256


def test_tau_and_repair_definition_are_copied_verbatim():
    from filseg.reproduction import pnat
    assert pnat.TAU == TAU
    tree = ast.parse((DATA / 'pnat_cv_reference.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'repair')
    cv_source = ast.get_source_segment((DATA / 'pnat_cv_reference.py').read_text(), node)
    assert inspect.getsource(pnat.repair).rstrip() == cv_source.rstrip()


def test_threshold_is_inclusive_and_trims_seed_pixels(small_image):
    from filseg.reproduction.pnat import repair
    s = mask()
    p = np.zeros(s.shape)
    p[s] = .7
    p[60, 40] = 0
    p[61, 55] = TAU
    out, stats = repair([s], p)
    assert not out[0][60, 40] and out[0][61, 55] and stats['removed'] == 1


def test_empty_threshold_falls_back_to_the_original_seed(small_image):
    from filseg.reproduction.pnat import repair
    s = mask()
    out, stats = repair([s], np.zeros(s.shape, np.float32))
    assert np.array_equal(out[0], s) and stats['fallback'] == 1
    assert repair([], np.zeros(s.shape, np.float32))[0] == []


def test_detached_island_and_window_limit(small_image):
    from filseg.reproduction.pnat import repair
    R = small_image
    s = mask()
    p = s.astype(float)
    p[75:77, 50:52] = 1
    assert np.array_equal(repair([s], p)[0][0], s)
    out, _ = repair([s], np.ones(s.shape))
    g = R.geometry(s)
    bb = R.window_bbox(g)
    assert np.array_equal(out[0], R.full(R.window_native(g, bb), bb))


def test_neighbouring_seeds_never_overlap_after_growth(small_image):
    from filseg.reproduction.pnat import repair
    a, b = mask(x=25), mask(x=65)
    out, _ = repair([a, b], np.ones(a.shape))
    assert not (out[0] & out[1]).any()
    assert not out[0][61, 52] and not out[1][61, 52]


def test_package_matches_cv_implementation_candidate_by_candidate(small_image):
    from filseg.reproduction.pnat import repair
    cv = cv_reference()
    rng = np.random.default_rng(20261111)
    compared = 0
    for trial in range(25):
        seeds = []
        occupied = np.zeros((128, 128), bool)
        for _ in range(int(rng.integers(1, 5))):
            y, x = int(rng.integers(5, 110)), int(rng.integers(5, 100))
            m = mask(y, x, int(rng.integers(2, 6)), int(rng.integers(5, 25)))
            m &= ~occupied
            if m.any():
                seeds.append(m)
                occupied |= m
        p = rng.random((128, 128)).astype(np.float32)
        p[occupied] = np.maximum(p[occupied], .5)
        ours, our_stats = repair(seeds, p)
        theirs, cv_stats = cv.repair(seeds, p)
        assert our_stats == cv_stats and len(ours) == len(theirs) == len(seeds)
        for a, b in zip(ours, theirs):
            assert np.array_equal(a, b)
            compared += 1
    assert compared > 25


def synthetic_maps():
    full = np.zeros((2048, 2048), np.uint8)
    local = np.zeros((2048, 2048), np.uint8)
    mf = np.zeros((2048, 2048), np.uint16)
    for y, x in ((200, 200), (800, 800), (812, 960)):
        full[y:y + 20, x:x + 100] = 210
        local[y:y + 20, x:x + 100] = 200
        mf[y:y + 20, x:x + 100] = 400
        full[y + 20:y + 24, x:x + 100] = 110  # just above tau after blending: P-nat can grow here
        local[y + 20:y + 24, x:x + 100] = 110
        mf[y + 20:y + 24, x:x + 100] = 210
    return dict(full=full, local=local, MF0=mf, MF1=mf.copy(), MF2=mf.copy())


def keep_all_model():
    from filseg.reproduction.features import features_for_mask
    raw = np.full((2048, 2048), 128, np.uint8)
    m = np.zeros(raw.shape, bool)
    m[0:2, 0:2] = True
    probs = dict(full=np.zeros(raw.shape, np.float32), local=np.zeros(raw.shape, np.float32))
    probs['blend'] = probs['full']
    order = list(features_for_mask(m, probs, raw))
    return raw, dict(feature_order=order, scaler_mean=[0.] * 39, scaler_scale=[1.] * 39, coef=[[0.] * 39], intercept=[0.], threshold=.35)


def test_pnat_stage_is_off_by_default_everywhere():
    from filseg.reproduction import pipeline
    assert inspect.signature(pipeline.image_stages).parameters['pnat'].default is False
    assert inspect.signature(pipeline.run_csv).parameters['pnat'].default is False


def test_pnat_runs_after_original_mask_quality_and_never_rescored():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.arithmetic import candidate_blend
    from filseg.reproduction.pnat import repair
    from pycocotools import mask as mu
    maps = synthetic_maps()
    raw, model = keep_all_model()
    base = image_stages(raw, maps, model)
    treated = image_stages(raw, maps, model, pnat=True)
    assert base['candidates'] == treated['candidates'] >= 2
    assert np.array_equal(base['features'], treated['features'])
    assert np.array_equal(base['scores'], treated['scores']) and np.array_equal(base['keep'], treated['keep'])
    assert 'pnat' not in base and treated['pnat']['kept'] == int(np.sum(base['keep']))
    kept = [r for r, k in zip(base['rles'], base['keep']) if k]
    seeds = [mu.decode(dict(size=[2048, 2048], counts=r.encode())).astype(bool) for r in kept]
    expected, stats = repair(seeds, candidate_blend(*(maps[n] for n in ('full', 'local', 'MF0', 'MF1', 'MF2'))))
    out_rles = [r for r, k in zip(treated['rles'], treated['keep']) if k]
    assert len(out_rles) == len(expected) and treated['pnat']['changed'] == stats['changed'] > 0
    occupied = np.zeros((2048, 2048), bool)
    for r, e in zip(out_rles, expected):
        m = mu.decode(dict(size=[2048, 2048], counts=r.encode())).astype(bool)
        assert np.array_equal(m, e) and m.any() and not (occupied & m).any()
        occupied |= m


def test_pnat_with_no_kept_candidate_yields_no_rows():
    from filseg.reproduction.pipeline import image_stages
    maps = synthetic_maps()
    raw, model = keep_all_model()
    model = dict(model, intercept=[-10.])
    treated = image_stages(raw, maps, model, pnat=True)
    assert not np.any(treated['keep']) and treated['pnat']['kept'] == 0
