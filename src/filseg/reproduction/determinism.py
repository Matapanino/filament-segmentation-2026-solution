"""Deterministic execution mode for neural map inference (enabled by default).

`configure(True)` fixes every seed, forces cuDNN deterministic kernels without autotuning, makes PyTorch
raise on any nondeterministic CUDA operation (never a silent fallback) and pins the cuBLAS workspace.
`CUBLAS_WORKSPACE_CONFIG` must exist before the CUDA context starts, so GPU child processes receive it
through `child_environment`. Inference has no DataLoader: images are processed sequentially in sorted key
order inside one process, which fixes the order and leaves no worker seeding to control.

Map archives are written with fixed ZIP metadata so that equal arrays always give equal file bytes
(numpy's savez stamps the current time into every member). `module_trace` records a hash of every leaf
module output so that a run-to-run difference can be localised to the first diverging operation.
"""
import contextlib
import hashlib
import io
import json
import os
import random
import subprocess
import zipfile

import numpy as np

SEED = 0
CUBLAS = ':4096:8'
DATALOADER = 'none: sequential in-process loop over sorted test keys'


def child_environment(base, enabled=True):
    env = dict(base)
    if enabled:
        env['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS
        env['PYTHONHASHSEED'] = str(SEED)
    return env


def configure(enabled=True, seed=SEED):
    import torch
    if enabled:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # Disabled mode is the previous package behaviour: deterministic cuDNN selection without autotuning,
    # but no seeding, no global enforcement and no cuBLAS workspace pin.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(enabled), warn_only=False)
    return dict(enabled=bool(enabled), seed=seed if enabled else None,
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                dataloader=DATALOADER)


def environment():
    """Runtime identity recorded with every run."""
    import platform
    import torch
    row = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), numpy=np.__version__)
    if torch.cuda.is_available():
        row.update(GPU=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                   hardware_native_bf16=torch.cuda.get_device_capability(0)[0] >= 8)
        try:
            row['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'], text=True).strip()
        except (OSError, subprocess.CalledProcessError) as e:
            row['driver'] = 'unavailable: ' + type(e).__name__
    return row


def save_npz(path, arrays):
    """np.savez_compressed equivalent with sorted members and fixed timestamps: equal arrays, equal bytes."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, buffer.getvalue())


def archive_content_sha(path):
    """Hash of sorted keys, dtypes, shapes and array bytes; independent of the container."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for key in sorted(z.files):
            a = np.ascontiguousarray(z[key])
            h.update(json.dumps(dict(key=key, dtype=a.dtype.str, shape=a.shape)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def _tensors(value):
    if hasattr(value, 'detach'):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _tensors(v)
    elif isinstance(value, dict):
        for k in sorted(value):
            yield from _tensors(value[k])
    elif hasattr(value, 'to_tuple'):
        yield from _tensors(value.to_tuple())


@contextlib.contextmanager
def module_trace(model):
    """Record, in call order, a SHA256 of each leaf module's output tensors. Values are not modified."""
    rows = []
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())):
            continue

        def hook(_module, _inputs, output, name=name):
            import torch
            h = hashlib.sha256()
            shapes = []
            for t in _tensors(output):
                x = t.detach().contiguous().cpu()
                shapes.append([str(x.dtype), list(x.shape)])
                h.update(str(x.dtype).encode())
                h.update(str(tuple(x.shape)).encode())
                if x.dtype == torch.bfloat16:
                    x = x.view(torch.int16)
                h.update(x.numpy().tobytes())
            rows.append(dict(index=len(rows), module=name, type=type(_module).__name__, outputs=shapes, sha256=h.hexdigest()))
        handles.append(module.register_forward_hook(hook))
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def first_trace_divergence(a, b):
    for x, y in zip(a, b):
        if (x['module'], x['sha256']) != (y['module'], y['sha256']):
            return dict(index=x['index'], module=x['module'], type=x['type'], other_module=y['module'])
    if len(a) != len(b):
        return dict(index=min(len(a), len(b)), module=None, type='trace_length', lengths=[len(a), len(b)])
    return None
