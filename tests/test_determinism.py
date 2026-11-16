"""Deterministic-mode contracts. Synthetic only: no model weights, no competition images."""
import inspect
import json
import os
import time

import numpy as np
import pytest


@pytest.fixture
def restore_torch():
    torch = pytest.importorskip('torch')
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
              torch.backends.cudnn.benchmark, os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    yield torch
    torch.use_deterministic_algorithms(before[0])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[1], before[2]
    if before[3] is None:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = before[3]


def test_enabled_mode_sets_every_backend_switch(restore_torch):
    torch = restore_torch
    from filseg.reproduction.determinism import configure, SEED
    settings = configure(True)
    assert settings['enabled'] is True and settings['seed'] == SEED
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert settings['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert settings['use_deterministic_algorithms'] is True and settings['warn_only'] is False
    assert settings['dataloader'] == 'none: sequential in-process loop over sorted test keys'
    json.dumps(settings)
    a = torch.rand(4)
    configure(True)
    assert torch.equal(a, torch.rand(4))


def test_disabled_mode_is_recorded_and_does_not_force_algorithms(restore_torch):
    torch = restore_torch
    from filseg.reproduction.determinism import configure
    settings = configure(False)
    assert settings['enabled'] is False and not torch.are_deterministic_algorithms_enabled()
    assert settings['use_deterministic_algorithms'] is False


def test_child_environment_sets_cublas_before_cuda_initialisation():
    from filseg.reproduction.determinism import child_environment
    env = child_environment({'PATH': '/bin'}, True)
    assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and env['PYTHONHASHSEED'] == '0' and env['PATH'] == '/bin'
    off = child_environment({'PATH': '/bin', 'CUBLAS_WORKSPACE_CONFIG': ':16:8'}, False)
    assert off['CUBLAS_WORKSPACE_CONFIG'] == ':16:8'


def test_deterministic_mode_is_the_default_of_every_gpu_entry_point():
    from filseg.reproduction import launch, pipeline
    assert inspect.signature(launch.run_gpu_pipeline).parameters['deterministic'].default is True
    assert inspect.signature(pipeline.run_maps).parameters['deterministic'].default is True


def test_map_archive_bytes_do_not_depend_on_write_time(tmp_path):
    from filseg.reproduction.determinism import save_npz, archive_content_sha
    rng = np.random.default_rng(3)
    maps = {'b': rng.integers(0, 511, (16, 16), dtype=np.uint16), 'a': rng.integers(0, 256, (8, 8), dtype=np.uint8)}
    save_npz(tmp_path / 'one.npz', maps)
    time.sleep(2.1)
    save_npz(tmp_path / 'two.npz', dict(reversed(list(maps.items()))))
    assert (tmp_path / 'one.npz').read_bytes() == (tmp_path / 'two.npz').read_bytes()
    with np.load(tmp_path / 'one.npz', allow_pickle=False) as z:
        assert sorted(z.files) == ['a', 'b']
        for k, v in maps.items():
            assert z[k].dtype == v.dtype and np.array_equal(z[k], v)
    assert archive_content_sha(tmp_path / 'one.npz') == archive_content_sha(tmp_path / 'two.npz')
    changed = dict(maps)
    changed['a'] = changed['a'].copy()
    changed['a'][0, 0] ^= 1
    save_npz(tmp_path / 'three.npz', changed)
    assert archive_content_sha(tmp_path / 'three.npz') != archive_content_sha(tmp_path / 'one.npz')


def test_module_trace_orders_leaf_outputs_and_detects_one_value(restore_torch):
    torch = restore_torch
    from filseg.reproduction.determinism import module_trace, first_trace_divergence
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    x = torch.ones(1, 4)
    with module_trace(net) as trace:
        net(x)
    assert [row['module'] for row in trace] == ['0', '1', '2']
    assert all(len(row['sha256']) == 64 for row in trace)
    with module_trace(net) as again:
        net(x)
    assert first_trace_divergence(trace, again) is None
    with torch.no_grad():
        net[2].bias[0] += 1
    with module_trace(net) as changed:
        net(x)
    assert first_trace_divergence(trace, changed)['module'] == '2'
