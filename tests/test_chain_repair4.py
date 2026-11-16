"""Chain mode ms125m4r4: ms125m4 with only the per-mask repair threshold changed. The threshold of each kept mask is the
snap (pcand.snap: the same 9-value grid and tie rule) of the plain mean of four regressors' predictions of its 64 repair
features: ms125's own gradient-boosted regressor (regressor_ms125.pkl) and a LightGBM, an XGBoost and a CatBoost regressor
fitted on the same rows. No fitted weights.

The reference arithmetic is the one that built the expected CSV (the PR1 CSV kernel's csv_m4r4.py, r4_factory): np.mean
over np.stack of the four members' predictions as float64, in the order hgb, lgbm, xgb, cat, then pcand.snap. Synthetic
rows here; the recipe's own files are checked once pinned.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / 'src/filseg/reproduction/data/chain'
VERSIONS = dict(lightgbm='4.6.0', xgboost='2.1.4', catboost='1.2.8')   # the PR1 kernel's versions


class Const:
    """A regressor that predicts one value for every row."""
    def __init__(self, v): self.v = v
    def predict(self, x): return np.full(len(x), self.v)


def hgb():
    o = pickle.loads((CHAIN / 'regressor_ms125.pkl').read_bytes()); return o['model']


def ms125_regressor():
    r = json.loads((CHAIN / 'ms125.json').read_text())['regressor']; return CHAIN / r['file'], r['sha256']


@pytest.fixture(scope='module')
def rows():
    """Synthetic 64-feature rows on which ms125's regressor gives several different thresholds."""
    r = np.random.default_rng(11); x = r.random((400, 64)); x[:, :10] *= 8
    return x


@pytest.fixture(scope='module')
def fitted(tmp_path_factory, rows):
    """Three small regressors of the 64 features, pickled as the PR1 kernel pickled its members."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    y = 0.2 + 0.4 * rows[:, 0] / 8
    out = tmp_path_factory.mktemp('repair4'); files = {}
    for a, m in (('lgbm', LGBMRegressor(n_estimators=20, num_leaves=7, random_state=42, n_jobs=1, verbose=-1)),
                 ('xgb', XGBRegressor(n_estimators=20, max_depth=3, tree_method='hist', random_state=42, n_jobs=1)),
                 ('cat', CatBoostRegressor(iterations=20, depth=3, random_seed=42, thread_count=1, verbose=0, allow_writing_files=False))):
        b = pickle.dumps(m.fit(rows, y)); (out / f'{a}.pkl').write_bytes(b); files[a] = out / f'{a}.pkl'
    return files


def recipe_dir(tmp_path, fitted, versions=VERSIONS):
    """A copy of data/chain with ms125m4r4's repair regressors pinned to the synthetic files."""
    for p in CHAIN.iterdir():
        if p.is_file(): (tmp_path / p.name).write_bytes(p.read_bytes())
    r = json.loads((CHAIN / 'ms125m4r4.json').read_text())
    spec = dict(versions=dict(versions))
    for a in ('lgbm', 'xgb', 'cat'):
        b = fitted[a].read_bytes(); name = r['repair_trees'][a]['file']; (tmp_path / name).write_bytes(b)
        spec[a] = dict(file=name, sha256=hashlib.sha256(b).hexdigest())
    r.update(repair_trees=spec)
    (tmp_path / 'ms125m4r4.json').write_text(json.dumps(r))
    return r


def reference_taus(members, x):
    """csv_m4r4.py's r4 predictor: np.mean(np.stack([member predictions as float64]), axis=0), then pcand.snap."""
    from filseg.reproduction import pcand as P
    raw = np.mean(np.stack([np.asarray(m.predict(x), float).ravel() for m in members]), axis=0)
    return [float(v) for v in raw], [P.snap(v) for v in raw]


def test_the_repair_with_only_the_gradient_boosted_member_is_the_current_repair_byte_for_byte(rows):
    from filseg.reproduction import pcand as P
    current = P.load_regressor(*ms125_regressor()); want = current(rows)
    raw = P.mean_raw([hgb()], rows)
    assert raw.tobytes() == np.asarray(hgb().predict(rows), float).tobytes()
    assert [float(v) for v in raw] == current.raw[-1]
    assert [P.snap(v) for v in raw] == want
    assert len(set(want)) > 2


def test_the_four_regressor_threshold_is_the_snap_of_the_plain_mean():
    from filseg.reproduction import pcand as P
    x = np.zeros((3, 64))
    assert P.mean_raw([Const(.2), Const(.3), Const(.5), Const(.6)], x).tolist() == [0.4] * 3
    assert P.snap(0.4) == P.TAU                      # 0.4 is nearest to the grid's TAU (0.4017)
    # A tie between two grid values goes to the one nearer TAU (pcand's tie rule, unchanged).
    m = P.mean_raw([Const(.5), Const(.55), Const(.5), Const(.55)], x)[0]
    assert abs(m - .5) == abs(.55 - m) and P.snap(m) == .5


def test_load_regressor_with_members_gives_the_snapped_mean_of_the_four_bit_for_bit(rows):
    from filseg.reproduction import pcand as P
    members = [Const(.25), Const(.55), Const(.6)]
    alone = P.load_regressor(*ms125_regressor())(rows)
    f = P.load_regressor(*ms125_regressor(), members=members); taus = f(rows)
    raw, want = reference_taus([hgb()] + members, rows)
    assert f.raw[-1] == raw and taus == want and all(t in P.GRID for t in taus)
    assert taus != alone   # the members change the thresholds


def test_ms125m4r4_repair_is_the_pr1_arithmetic_on_its_hash_checked_files(tmp_path, fitted, rows, monkeypatch):
    from filseg.reproduction import chain
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); recipe_dir(tmp_path, fitted)
    r = chain.load_recipe('ms125m4r4'); f = chain.repair(r, chain.regressor(r, None)); taus = f(rows)
    members = [hgb()] + [pickle.loads(fitted[a].read_bytes()) for a in ('lgbm', 'xgb', 'cat')]
    raw, want = reference_taus(members, rows)
    assert f.raw[-1] == raw and taus == want
    assert taus != chain.repair(chain.load_recipe('ms125m4'), chain.regressor(chain.load_recipe('ms125m4'), None))(rows)


def test_ms125m4r4_refuses_a_changed_repair_file(tmp_path, fitted, monkeypatch):
    from filseg.reproduction import chain
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); r = recipe_dir(tmp_path, fitted)
    p = tmp_path / r['repair_trees']['cat']['file']; p.write_bytes(p.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash'):
        chain.repair(chain.load_recipe('ms125m4r4'), chain.regressor(r, None))


def test_ms125m4r4_refuses_other_library_versions(tmp_path, fitted, monkeypatch):
    from filseg.reproduction import chain
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); r = recipe_dir(tmp_path, fitted, versions=dict(VERSIONS, xgboost='2.1.3'))
    with pytest.raises(ValueError, match='xgboost'):
        chain.repair(chain.load_recipe('ms125m4r4'), chain.regressor(r, None))


def test_recipes_without_repair_trees_keep_the_regressor_alone(rows):
    from filseg.reproduction import chain, pcand as P
    r = chain.load_recipe('ms125m4')
    assert chain.repair(r, chain.regressor(r, None))(rows) == P.load_regressor(*ms125_regressor())(rows)


def test_ms125m4r4_is_ms125m4_with_only_the_repair_changed():
    from filseg.reproduction import chain
    a, b = chain.load_recipe('ms125m4'), chain.load_recipe('ms125m4r4')
    assert set(a) | {'repair_trees'} == set(b) and 'repair_trees' not in a
    for k in ('weight', 'cutoff', 'unet', 'sets', 'archives', 'regressor', 'selector', 'trees'):
        assert a[k] == b[k], k
    assert set(b['repair_trees']) == {'versions', 'lgbm', 'xgb', 'cat'} and b['repair_trees']['versions'] == VERSIONS
    if b['csv']['sha256'] is not None:
        for t in ('lgbm', 'xgb', 'cat'):
            assert hashlib.sha256((CHAIN / b['repair_trees'][t]['file']).read_bytes()).hexdigest() == b['repair_trees'][t]['sha256']


def test_ms125m4r4_is_a_mode_only_once_its_csv_is_pinned():
    from filseg.reproduction import chain, modes
    r = chain.load_recipe('ms125m4r4')
    assert ('ms125m4r4' in modes.CHAIN_MODES) == (r['csv']['sha256'] is not None)


def test_pinned_repair_files_load_after_torch_in_the_same_process(tmp_path, fitted):
    """As chain.trees: on macOS LightGBM's OpenMP runtime crashed after torch's unless one thread is set first."""
    import os, subprocess, sys
    pytest.importorskip('torch')
    recipe_dir(tmp_path, fitted)
    code = ('import torch\nfrom pathlib import Path\nfrom filseg.reproduction import chain\n'
            f'chain.CHAIN = Path({str(tmp_path)!r})\nimport numpy as np\n'
            'r = chain.load_recipe("ms125m4r4")\nprint(chain.repair(r, chain.regressor(r, None))(np.zeros((2, 64))))\n')
    env = {k: v for k, v in os.environ.items() if k != 'OMP_NUM_THREADS'}; env['PYTHONPATH'] = str(REPO / 'src')
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])


def test_every_chain_rebuild_reads_the_recipe_repair_through_chain_repair():
    """The figure cell rebuilds two images of the CSV run and compares them stage by stage: it must use the same repair as
    the CSV (chain.repair, with the recipe's repair_trees), not the regressor alone (Kaggle T4 check 2026-11-16: RLE differed)."""
    import inspect
    from filseg.reproduction import chain, figures
    assert 'chain.repair(r, reg)' in inspect.getsource(figures._illustrate_chain)
    for mod in (chain, figures):
        assert 'load_regressor(*reg)' not in inspect.getsource(mod), mod.__name__
