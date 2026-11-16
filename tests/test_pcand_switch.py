"""The user-facing P-cand switch: CLI flag, notebook mode and the pinned deployment regressor.

Synthetic only; no fitting and no real model. run_csv, resolve_assets and the regressor loader are replaced
by recorders so that each test checks which code path a switch reaches, not the CSV itself.
"""
import ast
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PIN = 'b4dcd7d1ec79fc6c42135733721c53d107ea53f1cefe2934c7fd2c95acba6798'


class Reached(Exception):
    pass


class Halves:
    n_features_in_ = 64

    def predict(self, x):
        return np.asarray(x, float)[:, 0] / 2


def cli(monkeypatch, *flags):
    """Run `python -m filseg.reproduction csv ...` in-process and return the run_csv call it made."""
    from filseg.reproduction import __main__ as entry, pipeline
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(status='complete', csv=dict(sha256='0' * 64))

    monkeypatch.setattr(pipeline, 'run_csv', recorder)
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'csv', '--images', 'I', '--maps', 'M', '--output', 'O', *flags])
    entry.main()
    assert len(calls) == 1
    return calls[0]


# ---------------------------------------------------------------- CLI switch
def test_cli_pcand_flag_calls_run_csv_with_the_deployed_regressor(monkeypatch):
    args, kwargs = cli(monkeypatch, '--pcand')
    assert args == ('I', None, 'M', 'O') and kwargs == dict(pnat=False, pcand=True)


def test_cli_default_and_pnat_calls_are_unchanged(monkeypatch):
    assert cli(monkeypatch) == (('I', None, 'M', 'O'), dict(pnat=False, pcand=None))
    assert cli(monkeypatch, '--pnat') == (('I', None, 'M', 'O'), dict(pnat=True, pcand=None))


def test_cli_rejects_pnat_together_with_pcand(monkeypatch):
    with pytest.raises(SystemExit):
        cli(monkeypatch, '--pnat', '--pcand')


# ---------------------------------------------------------------- notebook switch
# The notebooks now select a named mode from filseg.reproduction.modes (tests/test_modes.py, tests/test_notebooks_final.py);
# MODE = "pcand" there still reaches run_csv(pnat=False, pcand=True), the same call as --pcand above.


# ---------------------------------------------------------------- pinned regressor from the weights root
def test_pinned_regressor_identity_and_scikit_learn_pin():
    from filseg.reproduction import pcand
    assert pcand.REGRESSOR_FILE == 'regressor.pkl' and pcand.REGRESSOR_SHA256 == PIN
    assert pcand.SKLEARN_VERSION == '1.6.1'
    assert 'scikit-learn==1.6.1' in (REPO / 'requirements-reproduce-cpu.txt').read_text().split()


def run_csv_until_loader(monkeypatch, tmp_path, payload, loader=None):
    """Call run_csv(pcand=True) with a fake weights root holding `payload` as regressor.pkl."""
    from filseg.reproduction import pipeline, pcand
    root = tmp_path / 'weights'
    root.mkdir()
    if payload is not None:
        (root / 'regressor.pkl').write_bytes(payload)
    monkeypatch.setattr(pipeline, 'resolve_assets', lambda weights: (root, {}))

    def no_images(*a, **k):
        raise AssertionError('image work must not start before the regressor is accepted')

    monkeypatch.setattr(pipeline, 'prepare_images', no_images)
    if loader is not None:
        monkeypatch.setattr(pcand, 'load_regressor', loader)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    return root, lambda: pipeline.run_csv(tmp_path / 'images', None, tmp_path / 'maps', tmp_path / 'out', pcand=True)


def test_run_csv_pcand_true_is_the_tuple_path_with_the_pinned_file(monkeypatch, tmp_path):
    seen = []

    def loader(path, sha256):
        seen.append((path, sha256))
        raise Reached

    root, call = run_csv_until_loader(monkeypatch, tmp_path, b'stand-in', loader)
    with pytest.raises(Reached):
        call()
    assert seen == [(str(root / 'regressor.pkl'), PIN)]


def test_wrong_regressor_hash_fails_loudly(monkeypatch, tmp_path):
    from filseg.reproduction import pcand
    payload = pickle.dumps(dict(model=Halves(), features=pcand.FEATURES, recipe='stand-in'))
    assert hashlib.sha256(payload).hexdigest() != PIN
    _, call = run_csv_until_loader(monkeypatch, tmp_path, payload)
    with pytest.raises(ValueError, match='hash'):
        call()


def test_missing_regressor_fails_loudly(monkeypatch, tmp_path):
    _, call = run_csv_until_loader(monkeypatch, tmp_path, None)
    with pytest.raises(FileNotFoundError, match='regressor.pkl'):
        call()


def test_scikit_learn_version_other_than_the_pin_fails_loudly(monkeypatch, tmp_path):
    from filseg.reproduction import pcand
    path = tmp_path / 'regressor.pkl'
    path.write_bytes(pickle.dumps(dict(model=Halves(), features=pcand.FEATURES, recipe='stand-in')))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(pcand, '_sklearn_version', lambda: '1.7.0')
    with pytest.raises(RuntimeError, match='scikit-learn'):
        pcand.load_regressor(path, digest)
    monkeypatch.setattr(pcand, '_sklearn_version', lambda: pcand.SKLEARN_VERSION)
    assert pcand.load_regressor(path, digest)(np.zeros((1, 64))) == [0.20]
