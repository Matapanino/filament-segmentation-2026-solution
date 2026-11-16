"""Chain mode ms125m4: the final submission's recipe (ms125) with only the keep/drop selector replaced by mean4, the plain
mean of four probabilities of the same 68 P-combo feature rows: ms125's own logistic regression (selector_ms125.json, the
package arithmetic) and a LightGBM, an XGBoost and a CatBoost model fitted on the same candidates. keep = mean >= the
recipe's cutoff. Everything else (maps, blend weight, candidates, repair regressor) is ms125's.

The reference arithmetic is the one that built the expected CSV (the SL1 CSV kernel's csv_m4.py): the package's LR
probability, LightGBM Booster.predict, XGBClassifier / CatBoostClassifier predict_proba[:, 1] of the reloaded native model
files, np.mean over [lr, lgbm, xgb, cat]. Synthetic candidates here; the recipe's own files are checked once pinned.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / 'src/filseg/reproduction/data/chain'
VERSIONS = dict(lightgbm='4.6.0', xgboost='2.1.4', catboost='1.2.8')   # the SL1 kernel's versions


def split_rows(x):
    """68-column rows -> the four dict lists apply_selector reads (quality 39, context 15, neighbour 6, MF-rich 8)."""
    from filseg.reproduction import refit
    from filseg.reproduction.context_features import FEATURES as CONTEXT
    from filseg.reproduction.mfrich_features import FEATURES as MFRICH
    from filseg.reproduction.neighbour import FEATURES as NEIGHBOUR
    order = refit.PCOMBO_FEATURES; idx = {n: i for i, n in enumerate(order)}
    quality = [n for n in order[:54] if n not in CONTEXT]
    part = lambda names: [{n: float(r[idx[n]]) for n in names} for r in x]
    return part(quality), part(list(CONTEXT)), part(list(NEIGHBOUR)), part(list(MFRICH))


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    """ms125's LR, synthetic candidates on its feature scale, and three small tree models saved in their native formats."""
    import lightgbm, xgboost, catboost
    from filseg.reproduction import refit
    lr = json.loads((CHAIN / 'selector_ms125.json').read_text())
    r = np.random.default_rng(7); n = 300
    x = np.asarray(lr['scaler_mean']) + np.asarray(lr['scaler_scale']) * r.standard_normal((n, 68))
    _, p, _ = refit.apply_selector(*split_rows(x)[:1], split_rows(x)[1], lr, *split_rows(x)[2:])
    y = (r.random(n) < p).astype(int); y[:2] = [0, 1]
    out = tmp_path_factory.mktemp('mean4')
    lightgbm.LGBMClassifier(n_estimators=20, num_leaves=7, min_child_samples=10, random_state=42, n_jobs=1, verbose=-1).fit(x, y).booster_.save_model(str(out / 'lgbm.txt'))
    xgboost.XGBClassifier(n_estimators=20, max_depth=3, tree_method='hist', random_state=42, n_jobs=1).fit(x, y).save_model(str(out / 'xgb.json'))
    catboost.CatBoostClassifier(iterations=20, depth=3, random_seed=42, thread_count=1, verbose=0, allow_writing_files=False).fit(x, y).save_model(str(out / 'cat.cbm'))
    return dict(x=x, lr=lr, dir=out)


def recipe_dir(tmp_path, fitted, cutoff=0.4, versions=VERSIONS):
    """A copy of data/chain with ms125m4 pinned to the synthetic tree files."""
    for p in CHAIN.iterdir():
        if p.is_file(): (tmp_path / p.name).write_bytes(p.read_bytes())
    r = json.loads((CHAIN / 'ms125m4.json').read_text())
    trees = dict(versions=dict(versions))
    for a, f in (('lgbm', 'lgbm.txt'), ('xgb', 'xgb.json'), ('cat', 'cat.cbm')):
        b = (fitted['dir'] / f).read_bytes(); name = r['trees'][a]['file']; (tmp_path / name).write_bytes(b)
        trees[a] = dict(file=name, sha256=hashlib.sha256(b).hexdigest())
    r.update(cutoff=cutoff, trees=trees)
    (tmp_path / 'ms125m4.json').write_text(json.dumps(r))
    return r


def reference(fitted, x, cutoff):
    """csv_m4.py's arithmetic: the package LR probability, then the reloaded native tree models, np.mean of the four."""
    import lightgbm, xgboost, catboost
    from filseg.reproduction import refit
    d = fitted['dir']; q, c, nb, mr = split_rows(x)
    rows, p_lr, _ = refit.apply_selector(q, c, fitted['lr'], nb, mr)
    if not len(rows): return p_lr, p_lr >= cutoff
    lg = lightgbm.Booster(model_file=str(d / 'lgbm.txt'))
    xg = xgboost.XGBClassifier(); xg.load_model(str(d / 'xgb.json'))
    ca = catboost.CatBoostClassifier(); ca.load_model(str(d / 'cat.cbm'))
    p = np.mean([p_lr] + [np.asarray(f(rows), float) for f in (lg.predict, lambda z: xg.predict_proba(z)[:, 1], lambda z: ca.predict_proba(z)[:, 1])], axis=0)
    return p, p >= cutoff


def test_mean4_probability_is_the_plain_mean_of_the_lr_and_the_three_trees_bit_for_bit(tmp_path, fitted, monkeypatch):
    from filseg.reproduction import chain, refit
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); r = recipe_dir(tmp_path, fitted)
    sel = chain.selector(chain.load_recipe('ms125m4'))
    assert sel['threshold'] == r['cutoff'] and sel['feature_order'] == refit.PCOMBO_FEATURES
    x = fitted['x']; q, c, nb, mr = split_rows(x)
    rows, p, keep = refit.apply_selector(q, c, sel, nb, mr)
    want_p, want_keep = reference(fitted, x, r['cutoff'])
    assert np.array_equal(p, want_p) and np.array_equal(keep, want_keep)
    assert 0 < keep.sum() < len(keep)
    _, p_lr, keep_lr = refit.apply_selector(q, c, fitted['lr'], nb, mr)   # the LR alone is a different selector
    assert not np.array_equal(p, p_lr)


def test_mean4_with_no_candidates_keeps_nothing(tmp_path, fitted, monkeypatch):
    from filseg.reproduction import chain, refit
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); recipe_dir(tmp_path, fitted)
    _, p, keep = refit.apply_selector([], [], chain.selector(chain.load_recipe('ms125m4')), [], [])
    assert p.shape == (0,) and keep.shape == (0,)


def test_mean4_selector_refuses_a_changed_tree_file(tmp_path, fitted, monkeypatch):
    from filseg.reproduction import chain
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); r = recipe_dir(tmp_path, fitted)
    p = tmp_path / r['trees']['xgb']['file']; p.write_bytes(p.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash'):
        chain.selector(chain.load_recipe('ms125m4'))


def test_mean4_selector_refuses_other_library_versions(tmp_path, fitted, monkeypatch):
    from filseg.reproduction import chain
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); recipe_dir(tmp_path, fitted, versions=dict(VERSIONS, catboost='1.2.7'))
    with pytest.raises(ValueError, match='catboost'):
        chain.selector(chain.load_recipe('ms125m4'))


def test_mean4_cutoff_must_be_on_the_chain_grid(tmp_path, fitted, monkeypatch):
    from filseg.reproduction import chain
    monkeypatch.setattr(chain, 'CHAIN', tmp_path); recipe_dir(tmp_path, fitted, cutoff=0.41)
    with pytest.raises(ValueError):
        chain.selector(chain.load_recipe('ms125m4'))


def test_ms125m4_is_ms125_with_only_the_selector_changed():
    from filseg.reproduction import chain
    a, b = chain.load_recipe('ms125'), chain.load_recipe('ms125m4')
    for k in ('weight', 'unet', 'sets', 'archives', 'regressor', 'selector'):
        assert a[k] == b[k], k
    assert set(b['trees']) == {'versions', 'lgbm', 'xgb', 'cat'} and b['trees']['versions'] == VERSIONS
    pinned = b['csv']['sha256'] is not None
    if pinned:
        assert b['cutoff'] in __import__('filseg.reproduction.refit', fromlist=['x']).CHAIN_CUTOFF_GRID
        for t in ('lgbm', 'xgb', 'cat'):
            assert hashlib.sha256((CHAIN / b['trees'][t]['file']).read_bytes()).hexdigest() == b['trees'][t]['sha256']


def test_ms125m4_is_a_mode_only_once_its_csv_is_pinned():
    from filseg.reproduction import chain, modes
    r = chain.load_recipe('ms125m4')
    assert ('ms125m4' in modes.CHAIN_MODES) == (r['csv']['sha256'] is not None)


def test_cpu_requirements_pin_the_tree_libraries_at_the_fitted_versions():
    for f in ('requirements.txt', 'requirements-reproduce-cpu.txt'):
        text = (REPO / f).read_text()
        for lib, v in VERSIONS.items():
            assert re.search(rf'^{lib}=={re.escape(v)}$', text, re.M), (f, lib)


def test_the_installed_tree_libraries_are_the_pinned_versions():
    import lightgbm, xgboost, catboost
    assert dict(lightgbm=lightgbm.__version__, xgboost=xgboost.__version__, catboost=catboost.__version__) == VERSIONS


def test_tree_models_load_after_torch_in_the_same_process(tmp_path, fitted):
    """On macOS two OpenMP runtimes (torch's and LightGBM's) crashed the LightGBM load; chain.trees runs the libraries on one
    OpenMP thread (as the CSV kernel that pinned the CSV did), which loads them safely after torch."""
    import subprocess, sys
    pytest.importorskip('torch')
    recipe_dir(tmp_path, fitted)
    code = ('import torch\nfrom pathlib import Path\nfrom filseg.reproduction import chain\n'
            f'chain.CHAIN = Path({str(tmp_path)!r})\n'
            'import numpy as np\nf = chain.selector(chain.load_recipe("ms125m4"))["mean4"]\n'
            'print(sum(float(f[a](np.zeros((2, 68)))[0]) for a in ("lgbm", "xgb", "cat")))\n')
    env = {k: v for k, v in __import__('os').environ.items() if k != 'OMP_NUM_THREADS'}
    env['PYTHONPATH'] = str(REPO / 'src')
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
