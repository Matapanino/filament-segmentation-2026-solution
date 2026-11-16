"""P-cand stage contracts, written before the package module.

The cross-validated implementation (detect_20261111/pcand_tau/pcand.py) is pinned as a verbatim fixture
(tests/data/pcand_cv_reference.py) and compared function by function and candidate by candidate.
Synthetic data only; no fitting.
"""
import ast
import hashlib
import importlib.util
import inspect
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from test_pnat import cv_reference as pnat_cv_reference, keep_all_model, mask, synthetic_maps

DATA = Path(__file__).parent / 'data'
CV_PCAND_SHA256 = '87e96105ccc78adc0cf690547cc3f6f3e0b5e2d61bf899a6f118e7731540e201'
CV_CONTEXT_SHA256 = '313e87b790d62e00316dbaff73f8521bc07c0af6c8bb0349ea3f74eb2f0a191e'
TAU = 0.4016711682033202
GRID = (0.20, 0.25, 0.30, 0.35, TAU, 0.45, 0.50, 0.55, 0.60)
VERBATIM = ('_order', 'snap', 'candidate_scores', 'choose_target', '_geom', 'window_features', 'prepare', 'repair_taus')


def cv_pcand():
    """Import the verbatim CV module; `import refine_ops` -> byte-identical package module, `import pnat` -> CV adapter."""
    path = DATA / 'pcand_cv_reference.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CV_PCAND_SHA256
    sys.modules['pnat'] = pnat_cv_reference()
    spec = importlib.util.spec_from_file_location('pcand_cv_reference', path)
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


def random_seeds(rng, n=128):
    seeds, occupied = [], np.zeros((n, n), bool)
    for _ in range(int(rng.integers(1, 6))):
        y, x = int(rng.integers(5, 110)), int(rng.integers(5, 100))
        m = mask(y, x, int(rng.integers(2, 6)), int(rng.integers(5, 25)), n)
        m &= ~occupied
        if m.any():
            seeds.append(m)
            occupied |= m
    return seeds, occupied


# ---------------------------------------------------------------- source identity
def test_helpers_are_copied_verbatim_from_the_cv_lane():
    from filseg.reproduction import pcand
    text = (DATA / 'pcand_cv_reference.py').read_text()
    tree = ast.parse(text)
    cv = {n.name: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in VERBATIM:
        assert inspect.getsource(getattr(pcand, name)).rstrip() == cv[name].rstrip(), name
    assert pcand.GRID == GRID and pcand.TAU == TAU and pcand.GRID[4] is pcand.TAU
    assert all(type(t) is float for t in pcand.GRID)


def test_context_features_module_is_byte_identical():
    from filseg.reproduction import context_features
    assert hashlib.sha256(Path(context_features.__file__).read_bytes()).hexdigest() == CV_CONTEXT_SHA256


def test_feature_names_are_the_preregistered_64():
    from filseg.reproduction import pcand
    from filseg.reproduction.context_features import FEATURES as CTX
    assert len(pcand.FEATURES) == 64 and len(set(pcand.FEATURES)) == 64
    assert pcand.FEATURES[39:54] == CTX and pcand.FEATURES[54:] == pcand.WINDOW_FEATURES
    assert pcand.FEATURES[:3] == ['log_area', 'log_perimeter', 'compactness'] and pcand.FEATURES[38] == 'ring_contrast'


# ---------------------------------------------------------------- numerical equality with the CV module
def test_snap_and_targets_equal_cv():
    from filseg.reproduction import pcand
    cv = cv_pcand()
    rng = np.random.default_rng(5)
    for x in list(rng.uniform(0, .8, 2000)) + [(a + b) / 2 for a, b in zip(GRID, GRID[1:])] + list(GRID):
        assert pcand.snap(x) == cv.snap(x) and type(pcand.snap(x)) is float
    for _ in range(300):
        s = list(rng.choice([0., .6, .7, .8], size=9))
        assert pcand.choose_target(s) == cv.choose_target(s)


def test_repair_taus_equals_cv_candidate_by_candidate(small_image):
    from filseg.reproduction import pcand
    cv = cv_pcand()
    rng = np.random.default_rng(20261111)
    compared = 0
    for _ in range(40):
        seeds, occupied = random_seeds(rng)
        p = rng.random((128, 128)).astype(np.float32)
        p[occupied] = np.maximum(p[occupied], .5)
        taus = [GRID[int(i)] for i in rng.integers(0, 9, len(seeds))]
        ours, a = pcand.repair_taus(seeds, p, taus)
        theirs, b = cv.repair_taus(seeds, p, taus)
        assert a == b and len(ours) == len(theirs) == len(seeds)
        for x, y in zip(ours, theirs):
            assert np.array_equal(x, y)
            compared += 1
        for m in seeds:
            assert pcand.window_features(m, p) == cv.window_features(m, p)
    assert compared > 40


def test_uniform_tau_equals_pnat_repair(small_image):
    from filseg.reproduction import pcand, pnat
    rng = np.random.default_rng(7)
    for _ in range(25):
        seeds, occupied = random_seeds(rng)
        p = rng.random((128, 128)).astype(np.float32)
        p[np.abs(p - np.float32(TAU)) < .02] = np.float32(TAU)
        want, ws = pnat.repair(seeds, p)
        got, gs = pcand.repair_taus(seeds, p, [TAU] * len(seeds))
        assert gs == ws and all(np.array_equal(x, y) for x, y in zip(got, want))


# ---------------------------------------------------------------- pipeline stage
def decode(r):
    from pycocotools import mask as mu
    return mu.decode(dict(size=[2048, 2048], counts=r.encode())).astype(bool)


def test_pcand_stage_is_off_by_default_and_exclusive_with_pnat():
    from filseg.reproduction import pipeline
    assert inspect.signature(pipeline.image_stages).parameters['pcand'].default is None
    assert inspect.signature(pipeline.run_csv).parameters['pcand'].default is None
    raw, model = keep_all_model()
    with pytest.raises(ValueError):
        pipeline.image_stages(raw, synthetic_maps(), model, pnat=True, pcand=lambda x: [TAU] * len(x))


def test_uniform_tau_path_equals_pnat_path():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    maps = synthetic_maps()
    raw, model = keep_all_model()
    a = image_stages(raw, maps, model, pnat=True)
    b = image_stages(raw, maps, model, pcand=uniform(TAU))
    assert a['rles'] == b['rles'] and a['stages'] == b['stages']
    assert np.array_equal(a['keep'], b['keep']) and np.array_equal(a['scores'], b['scores'])
    assert b['pcand']['taus'] == [TAU] * b['pcand']['kept'] and b['pcand']['kept'] >= 2
    assert {k: b['pcand'][k] for k in a['pnat']} == a['pnat']


def test_pcand_stage_calls_the_predictor_on_the_64_features_and_repairs_like_cv():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.arithmetic import candidate_blend, mf_mean
    from filseg.reproduction.features import features_for_mask
    from filseg.reproduction.context_features import context_features
    from filseg.reproduction import pcand
    cv = cv_pcand()
    maps = synthetic_maps()
    raw, model = keep_all_model()
    seen = []

    def predictor(x):
        seen.append(np.array(x))
        return [GRID[i % 9] for i in range(len(x))]  # different taus per candidate

    base = image_stages(raw, maps, model)
    out = image_stages(raw, maps, model, pcand=predictor)
    assert np.array_equal(base['keep'], out['keep']) and np.array_equal(base['scores'], out['scores'])
    seeds = [decode(r) for r, k in zip(base['rles'], base['keep']) if k]
    blend = candidate_blend(*(maps[n] for n in ('full', 'local', 'MF0', 'MF1', 'MF2')))
    ctx_maps = dict(full=maps['full'].astype(np.float32) / 255, local=maps['local'].astype(np.float32) / 255,
                    m2f_public=mf_mean(maps['MF0'], maps['MF1'], maps['MF2']))
    probs = dict(full=ctx_maps['full'], local=ctx_maps['local'])
    probs['blend'] = (probs['full'] + probs['local']) * .5
    expected_x = []
    for m in seeds:
        q = features_for_mask(m, probs, raw)
        row = {**{n: float(q[n]) for n in pcand.FEATURES[:39]}, **context_features(m, ctx_maps, raw), **cv.window_features(m, blend)}
        expected_x.append([row[n] for n in pcand.FEATURES])
    assert len(seen) == 1 and seen[0].shape == (len(seeds), 64)
    assert np.array_equal(seen[0], np.asarray(expected_x, float))
    taus = [GRID[i % 9] for i in range(len(seeds))]
    expected, stats = cv.repair_taus(seeds, blend, taus)
    got = [decode(r) for r, k in zip(out['rles'], out['keep']) if k]
    assert all(np.array_equal(a, b) for a, b in zip(got, expected)) and len(got) == len(expected)
    assert out['pcand']['taus'] == taus and {k: out['pcand'][k] for k in stats} == stats


def test_pcand_stage_with_no_kept_candidate_never_calls_the_predictor():
    from filseg.reproduction.pipeline import image_stages
    raw, model = keep_all_model()
    model = dict(model, intercept=[-10.])

    def predictor(x):
        raise AssertionError('must not be called')

    out = image_stages(raw, synthetic_maps(), model, pcand=predictor)
    assert not np.any(out['keep']) and out['pcand']['kept'] == 0 and out['pcand']['taus'] == []


# ---------------------------------------------------------------- deployment regressor wrapper
class Halves:
    """Stand-in regressor (no fitting): predicts half the first feature."""
    n_features_in_ = 64

    def predict(self, x):
        return np.asarray(x, float)[:, 0] / 2


def test_regressor_file_is_hash_checked_and_snapped(tmp_path):
    from filseg.reproduction import pcand
    path = tmp_path / 'regressor.pkl'
    path.write_bytes(pickle.dumps(dict(model=Halves(), features=pcand.FEATURES, recipe='stand-in')))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    predictor = pcand.load_regressor(path, digest)
    x = np.zeros((3, 64))
    x[:, 0] = [0.4, 0.8, 2.0]
    assert predictor(x) == [0.20, TAU, 0.60]
    assert predictor.raw == [[0.2, 0.4, 1.0]]
    with pytest.raises(ValueError):
        pcand.load_regressor(path, '0' * 64)
    bad = tmp_path / 'bad.pkl'
    bad.write_bytes(pickle.dumps(dict(model=Halves(), features=list(reversed(pcand.FEATURES)), recipe='stand-in')))
    with pytest.raises(ValueError):
        pcand.load_regressor(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
