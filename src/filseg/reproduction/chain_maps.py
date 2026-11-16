"""Chain maps re-inferred from the release model files (GPU, Kaggle T4), and their difference from the stored maps (CPU).

The chain modes (chain.py) read stored test maps. This module rebuilds those maps from the model files listed in the
same release (data/chain/release_files.json) and writes them under the SAME file names, so that the chain CSV stage
(`csv --mode <recipe> --release <these files> --maps-from-models`) reads them in place of the stored ones.

Mask2Former (one archive per recipe entry: model file, input size, views), as the as-run exporters computed it
(a1native/export_tta.py at 2048; ms1/export_scale.py and ms1/export_scale_v.py for 2560 and 3072):
  the native view raw / 255 (float32) is resized to the input size (torch bilinear, align_corners=False; untouched at
  2048); for each view in the recipe's order the view is transformed, the network runs under BF16 autocast,
  fg = max_q(sigmoid(mask_q) * score_q) is taken at the network output (size / 4 px) and inverse-transformed; the views
  are summed in that order and divided by their number (no division for one view); then bilinear to 2048
  (align_corners=False) and rint(fg * 510) -> uint16, once. One archive member per test image (MapWriter format).
U-Net (recipes with the 5-fold-mean U-Net maps): each fold checkpoint unetmodel_f{k}_{view}.pt gives
  unet_f{k}_{view}_test.npz through the package's own inference.unet_map (as unet_refit/test_maps.py `infer`), and
  unet_fold5mean_{view}.npz is uint8(rint(sum_k q_k / 5)) over the five (unet_refit/fold5mean.py), package save_npz.

Execution follows the package GPU stage (launch.py, pipeline.run_maps): Kaggle T4 only, the pinned U-Net and MF
environments installed in turn, a fresh interpreter per GPU, TF32 off, determinism.configure (seeded, deterministic
algorithms enforced). The archives of one environment are split over the GPUs (one process per GPU, CUDA_VISIBLE_DEVICES);
an archive whose output file exists is skipped (resume); files are written as <name>.partial and renamed when complete.
Receipts: chain_maps_result.json (environment of every process; per archive seconds, seconds per view, peak GPU memory,
sha256, equal_to_release). compare_maps (CPU) gives, per archive, equal images, changed pixels and the level differences.
The stored maps were exported on an RTX 3090 (torch 2.5.1, native BF16); the T4 has no native BF16, so the re-inferred
maps are expected to differ slightly: the comparison measures by how much.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from . import chain

DATA = Path(__file__).parent / 'data'
RESULT = 'chain_maps_result.json'
COMPARE = 'chain_maps_compare.json'
RECEIPTS = 'chain_maps_receipts'
LOCK = '.chain_maps.lock'
UNET_VIEWS = ('full', 'local')
UNET_COST = dict(full=1, local=9)        # tiles per image (inference.unet_map: full 1 tile, local 9 tiles)
MF_CONFIG = 'mf_model_config.json'
SIZE = 2048
DEN = 510
# a1native/export_tta.TRANSFORMS (= pkg mf_d4.TRANSFORMS): y = rot90(flip_W(x) if flip else x, k); 'flipH' is
# ms1/export_scale_v.VIEW_SETS[4][3] (flip_W + rot180 = flip_H), named so in the recipes.
TRANSFORMS = {t[0]: t for t in [('id', False, 0), ('rot90', False, 1), ('rot180', False, 2), ('rot270', False, 3),
                                ('flipW', True, 0), ('flipW_rot90', True, 1), ('flipW_rot180', True, 2), ('flipW_rot270', True, 3),
                                ('flipH', True, 2)]}


def transforms(views):
    unknown = [v for v in views if v not in TRANSFORMS]
    if unknown:raise ValueError(f'Unknown view {unknown}: allowed {list(TRANSFORMS)}')
    return [TRANSFORMS[v] for v in views]


# ---------------------------------------------------------------- the exporters' arithmetic
def forward(x, t):
    """a1native/export_tta.py forward (= ms1/export_scale_v.py forward, pkg mf_d4.forward)."""
    import torch
    _, flip, k = t
    y = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(y, k, dims=(-2, -1))


def inverse(y, t):
    """a1native/export_tta.py inverse (= ms1/export_scale_v.py inverse, pkg mf_d4.inverse)."""
    import torch
    _, flip, k = t
    x = torch.rot90(y, -k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def scaled_input(x, size):
    """ms1/export_scale_v.py scaled_input: x [1, 1, 2048, 2048] float32 -> [1, 1, size, size] (bilinear,
    align_corners=False); unchanged (the same tensor) at size 2048."""
    import torch
    if size % 32:
        raise ValueError(f'size {size} is not a multiple of 32')
    if size == x.shape[-1]:
        return x
    return torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)


def query_scores(class_queries_logits):
    """a1native/decode.py query_scores: [Q, 2] logits -> [Q] probability of class 0 ("filament")."""
    return class_queries_logits.float().softmax(-1)[:, 0]


def fg_lowres(model, x, size, views, timer=None):
    """ms1/export_scale_v.py fg_lowres with the views named (a1native/export_tta.py fg_lowres at size 2048: the same
    operations in the same order). x [1, 1, 2048, 2048] on the model's device -> mean over the views of the
    inverse-transformed fg, float32 [size/4, size/4]; one view: that view's fg (no division). timer: {view: seconds}
    accumulated (the device is synchronised after each view)."""
    import torch
    xs = scaled_input(x, size)
    total = None
    for t in transforms(views):
        tick = time.perf_counter()
        xv = xs if t[0] == 'id' else forward(xs, t).contiguous()
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16):
            o = model(xv)
        scores = query_scores(o.class_queries_logits[0])
        fg = inverse((o.masks_queries_logits[0].float().sigmoid() * scores[:, None, None]).amax(0), t)
        total = fg if total is None else total + fg
        del o, xv
        if timer is not None:
            if x.is_cuda:torch.cuda.synchronize(x.device)
            timer[t[0]] = timer.get(t[0], 0.0) + time.perf_counter() - tick
    return total if len(views) == 1 else total / len(views)


def level_map(fg):
    """a1native/export_tta.py level_map: float [h, w] fg -> int64 [2048, 2048] levels rint(bilinear_2048(fg) * 510)."""
    import torch
    return np.rint(torch.nn.functional.interpolate(fg[None, None], size=(2048, 2048), mode='bilinear',
                                                   align_corners=False)[0, 0].cpu().numpy() * 510).astype(np.int64)


def model_input(raw, device='cuda'):
    """a1native/export_tta.py main maps() with a1native/model.py view_native: raw / 255 float32, [1, 1, H, W] on device."""
    import torch
    return torch.from_numpy((raw.astype(np.float32) / 255)[None, None].copy()).to(device)


def mf_archive_map(model, raw, size, views, device='cuda', timer=None):
    """One image's member of a Mask2Former archive: uint16 rint(bilinear_2048(view mean of fg) * 510) (the archive writer
    and maps_for_image both call this)."""
    return level_map(fg_lowres(model, model_input(raw, device), size, views, timer)).astype(np.uint16)


class MapWriter:
    """a1native/eval_ckpt.py MapWriter: image-by-image npz writer, byte-compatible with pkg determinism.save_npz when keys
    arrive sorted."""

    def __init__(self, path):
        self.z = zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True)
        self.last = None

    def write(self, key, arr):
        assert arr.dtype == np.uint16 and arr.shape == (2048, 2048)
        assert self.last is None or key > self.last
        self.last = key
        buf = io.BytesIO()
        np.lib.format.write_array(buf, arr, allow_pickle=False)
        info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        self.z.writestr(info, buf.getvalue())

    def close(self):
        self.z.close()


def unet_level_mean(arrays):
    """unet_refit/test_maps.py level_mean: uint8(rint(sum / n)) of uint8 (2048, 2048) maps; n = 1 returns the map."""
    for q in arrays:
        if q.dtype != np.uint8 or q.shape != (SIZE, SIZE): raise ValueError('U-Net maps must be uint8 2048x2048')
    if len(arrays) == 1: return arrays[0]
    s = np.zeros((SIZE, SIZE), np.uint32)
    for q in arrays: s += q
    return np.rint(s / np.float64(len(arrays))).astype(np.uint8)


# ---------------------------------------------------------------- models
def config_model(mf_config_path):
    """a1native/model.py config_model with its Net / Dense wrapper (the as-run state-dict key layout; the forward of
    pkg models.load_network): grey input repeated to 3 channels, ImageNet mean/std."""
    import torch
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    class Net(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x = x.repeat(1, 3, 1, 1)
            mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
            std = x.new_tensor([.229, .224, .225])[None, :, None, None]
            return self.model(pixel_values=(x - mean) / std)

    class Dense(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.net = Net(model)

        def forward(self, x):
            return self.net(x)

    config = Mask2FormerConfig.from_dict(json.loads(Path(mf_config_path).read_text()))
    config.num_labels = 1; config.id2label = {0: 'filament'}; config.label2id = {'filament': 0}
    return Dense(Mask2FormerForUniversalSegmentation(config))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):h.update(b)
    return h.hexdigest()


def verified(root, name):
    """The release file `name` under root, refused unless its bytes and sha256 are the release record's."""
    pin = chain.release_files()[name]; p = chain.find(name, [root])
    if p.stat().st_size != pin['bytes'] or _sha(p) != pin['sha256']:raise ValueError(f'{name}: size or hash differs from the release record')
    return p


def load_mf(root, unit, device):
    """The archive's Mask2Former (strict load, as a1native/export_tta.py main): config + EMA weights from the release."""
    from safetensors.torch import load_file
    model = config_model(verified(root, MF_CONFIG))
    model.load_state_dict(load_file(str(verified(root, unit['model']))), strict=True)
    return model.to(device).eval()


def load_unet(root, unit, device):
    """A U-Net fold checkpoint (torch dict {model, config, epoch}; train/unet/oof_maps.load_model, smp.Unet as pkg
    models.load_network) and the package view spec it must carry."""
    import torch
    import segmentation_models_pytorch as smp
    ck = torch.load(verified(root, unit['model']), map_location='cpu', weights_only=True)
    spec = json.loads((DATA / f"{unit['view']}_config.json").read_text())['view']; cfg = ck['config']
    if cfg['view'] != spec or int(cfg['fold']) != unit['fold'] or cfg['encoder'] != 'efficientnet-b0':
        raise ValueError(f"{unit['model']}: checkpoint config differs from the package {unit['view']} view or fold {unit['fold']}")
    model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval(), spec


def unet_test_map(model, raw, spec):
    from .inference import unet_map
    return unet_map(model, raw, spec)[0]


# ---------------------------------------------------------------- what a recipe needs
def _recipes(recipe):
    return [recipe] if isinstance(recipe, str) else list(recipe)


def plan(recipe, names=None):
    """The archives to infer for one or several recipes, in recipe order (MF archives, then U-Net fold models); names: a
    subset of their file names."""
    pins = chain.release_files(); units = {}
    for name in _recipes(recipe):
        r = chain.load_recipe(name)
        for a in r['archives']:
            u = dict(file=a['file'], kind='mf', model=a['model'], fold=a['fold'], size=a['size'], views=list(a['views']),
                     cost=len(a['views']) * (a['size'] / SIZE) ** 2)
            if units.setdefault(u['file'], u) != u:raise ValueError('Two recipes describe one archive differently: ' + u['file'])
        if r['unet'] == 'fold5mean':
            for k in range(5):
                for v in UNET_VIEWS:
                    u = dict(file=f'unet_f{k}_{v}_test.npz', kind='unet', model=f'unetmodel_f{k}_{v}.pt', fold=k, view=v, cost=UNET_COST[v])
                    units.setdefault(u['file'], u)
    for u in units.values():
        if u['file'] not in pins or u['model'] not in pins:raise ValueError('Not a release file: ' + u['file'])
    if names is not None:
        missing = [n for n in names if n not in units]
        if missing:raise ValueError(f'{missing} not in the plan of {_recipes(recipe)}')
        return [u for u in units.values() if u['file'] in set(names)]
    return list(units.values())


def derived(recipe):
    """The 5-fold-mean U-Net files of the recipes that read them (built from the re-inferred fold maps)."""
    out = []
    for name in _recipes(recipe):
        if chain.load_recipe(name)['unet'] == 'fold5mean':
            out += [f for f in chain.UNET_FILES['fold5mean'].values() if f not in out]
    return out


def split(units, n):
    """n lists covering the units once, balanced by cost (longest first onto the least loaded); plan order kept inside."""
    order = sorted(range(len(units)), key=lambda i: -units[i]['cost']); bins = [[] for _ in range(n)]; load = [0.0] * n
    for i in order:
        b = min(range(n), key=lambda j: load[j]); bins[b].append(i); load[b] += units[i]['cost']
    return [[units[i] for i in sorted(b)] for b in bins]


# ---------------------------------------------------------------- receipts
@contextlib.contextmanager
def _locked(output):
    import fcntl
    with open(Path(output) / LOCK, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=str)); os.replace(tmp, path)


def _receipt(output, row):
    with _locked(output):
        _save(Path(output) / RECEIPTS / (row['file'] + '.json'), row)
        collect(output)


def collect(output):
    """chain_maps_result.json from the receipts of every process (call under the lock)."""
    output = Path(output); d = output / RECEIPTS
    archives = {p.name[:-5]: json.loads(p.read_text()) for p in sorted(d.glob('*.npz.json'))}
    envs = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob('environment_*.json'))}
    planned = json.loads((d / 'plan.json').read_text()) if (d / 'plan.json').exists() else None
    present = {f: (output / f).exists() for f in (planned['files'] if planned else archives)}
    result = dict(status='complete' if planned and all(present.values()) else 'partial', plan=planned, environments=envs, archives=archives,
                  missing=[f for f, ok in present.items() if not ok], model_fits=0, submission_api_calls=0)
    _save(output / RESULT, result)
    return result


def _process():
    return dict(pid=os.getpid(), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))


class _Deadline(Exception):
    pass


# ---------------------------------------------------------------- the GPU loop (device-agnostic; stand-ins in the tests)
def run_units(units, keys, read, output, loaders, device='cuda', deadline=None, process=None):
    """Infer each unit's archive for the sorted keys (read(key) -> uint8 2048x2048 image) into output/<file>.
    loaders: mf(unit) -> network, unet(unit) -> (network, view spec), unet_map(network, raw, spec) -> uint8 map.
    A unit whose file exists is skipped; deadline (time.time()) is checked before every image."""
    import cv2
    import torch
    from .determinism import archive_content_sha, save_npz
    output = Path(output); output.mkdir(parents=True, exist_ok=True); pins = chain.release_files(); keys = sorted(keys)
    cuda = str(device).startswith('cuda'); rows = []; stopped = False
    for u in units:
        path = output / u['file']; part = path.with_name(path.name + '.partial')
        base = dict(file=u['file'], kind=u['kind'], model=u['model'], model_sha256=pins[u['model']]['sha256'], process=process or _process(),
                    **({'size': u['size'], 'views': u['views']} if u['kind'] == 'mf' else {'fold': u['fold'], 'view': u['view']}))
        if stopped:
            row = dict(base, status='not_started'); rows.append(row); _receipt(output, row); continue
        if path.exists():
            rows.append(dict(file=u['file'], kind=u['kind'], status='skipped_existing')); continue
        cv2.setNumThreads(1 if u['kind'] == 'unet' else 2)
        tick = time.monotonic(); timer = {}
        if cuda:torch.cuda.reset_peak_memory_stats()
        try:
            if u['kind'] == 'mf':
                model = loaders['mf'](u); load_seconds = time.monotonic() - tick; w = MapWriter(part)
                try:
                    for i, key in enumerate(keys):
                        if deadline is not None and time.time() > deadline:raise _Deadline
                        w.write(key, mf_archive_map(model, read(key), u['size'], u['views'], device, timer))
                        if (i + 1) % 20 == 0:print(json.dumps(dict(stage='chain_maps', file=u['file'], done=i + 1, total=len(keys), seconds=time.monotonic() - tick)), flush=True)
                finally:
                    w.close()
            else:
                (model, spec) = loaders['unet'](u); load_seconds = time.monotonic() - tick; maps = {}
                for i, key in enumerate(keys):
                    if deadline is not None and time.time() > deadline:raise _Deadline
                    maps[key] = loaders['unet_map'](model, read(key), spec)
                    if (i + 1) % 30 == 0:print(json.dumps(dict(stage='chain_maps', file=u['file'], done=i + 1, total=len(keys), seconds=time.monotonic() - tick)), flush=True)
                save_npz(part, maps)
        except _Deadline:
            part.unlink(missing_ok=True); stopped = True
            row = dict(base, status='stopped_at_deadline', seconds=time.monotonic() - tick); rows.append(row); _receipt(output, row); continue
        except BaseException:
            part.unlink(missing_ok=True); raise
        os.replace(part, path); seconds = time.monotonic() - tick; sha = _sha(path)
        row = dict(base, status='complete', images=len(keys), seconds=seconds, load_seconds=load_seconds, bytes=path.stat().st_size, sha256=sha,
                   content_sha256=archive_content_sha(path), equal_to_release=sha == pins[u['file']]['sha256'],
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                   peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None)
        if u['kind'] == 'mf':
            row.update(view_seconds=timer, seconds_per_view=sum(timer.values()) / (len(keys) * len(u['views'])))
        else:
            row.update(seconds_per_image=(seconds - load_seconds) / len(keys))
        rows.append(row); _receipt(output, row)
        print(json.dumps({k: row[k] for k in ('file', 'status', 'seconds', 'equal_to_release')}), flush=True)
        del model
        if cuda:torch.cuda.empty_cache()
    return rows


def write_fold5mean(output, recipe, keys=None):
    """unet_fold5mean_{view}.npz from the five re-inferred unet_f{k}_{view}_test.npz, for each view whose five files are
    all there (and whose mean is not yet written). Returns the files written."""
    from .determinism import archive_content_sha, save_npz
    output = Path(output); pins = chain.release_files(); made = []
    if not derived(recipe):return made
    with _locked(output):
        for v, f in chain.UNET_FILES['fold5mean'].items():
            path = output / f; inputs = [f'unet_f{k}_{v}_test.npz' for k in range(5)]
            if path.exists() or not all((output / i).exists() for i in inputs):continue
            tick = time.monotonic(); zs = [np.load(output / i, allow_pickle=False) for i in inputs]
            ks = sorted(zs[0].files)
            if any(sorted(z.files) != ks for z in zs):raise ValueError('The fold maps cover different images: ' + v)
            if keys is not None and sorted(keys) != ks:raise ValueError('The fold maps do not cover the asked images: ' + v)
            part = path.with_name(path.name + '.partial'); save_npz(part, {k: unet_level_mean([z[k] for z in zs]) for k in ks}); os.replace(part, path)
            sha = _sha(path)
            row = dict(file=f, kind='unet_fold5mean', status='complete', inputs=inputs, input_sha256={i: _sha(output / i) for i in inputs}, images=len(ks),
                       rule='uint8(rint(sum_k q_k / 5)) (np.rint, half to even); package determinism.save_npz', bytes=path.stat().st_size, sha256=sha,
                       content_sha256=archive_content_sha(path), equal_to_release=sha == pins[f]['sha256'], seconds=time.monotonic() - tick, process=_process())
            _save(output / RECEIPTS / (f + '.json'), row); made.append(f)
        collect(output)
    return made


def infer_recipe_maps(images, release_root, output, recipe, names=None, gpu=0, keys=None, deadline=None, deterministic=True, image_set=None):
    """GPU (Kaggle T4): the recipe's archives (or the named subset) re-inferred from the release models into output, under
    the release names, then the 5-fold-mean U-Net maps when their five fold maps are there. images: the competition test
    images (a folder or the archive, pkg pipeline.prepare_images); keys: a subset of the test images (smoke)."""
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Neural inference requires the authorized GPU runtime')
    if image_set is None and 'T4' not in torch.cuda.get_device_name(gpu):raise RuntimeError('This validation lane is restricted to Kaggle T4')
    import cv2
    from .determinism import configure, environment
    from .pipeline import prepare_images, read_image
    units = plan(recipe, names); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any(u['kind'] == 'mf' for u in units) and not torch.cuda.is_bf16_supported():raise RuntimeError('Installed PyTorch cannot execute original MF BF16')
    torch.cuda.set_device(gpu); device = f'cuda:{gpu}'
    files = prepare_images(images, output / 'test_images', image_set=image_set)   # image_set 'any': new images (pipeline.any_images)
    keys = sorted(files) if keys is None else sorted(keys)
    if not set(keys) <= set(files):raise ValueError('Unknown test image keys')
    torch.set_num_threads(2); cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    settings = configure(deterministic)
    process = dict(_process(), gpu=gpu)
    env = dict(environment(), determinism=settings, opencv=cv2.__version__, mf_precision='bf16 autocast', bf16_supported=torch.cuda.is_bf16_supported(),
               device_name=torch.cuda.get_device_name(gpu), process=process, files=[u['file'] for u in units], images=len(keys),
               started_at=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip())
    with _locked(output):
        _save(output / RECEIPTS / f"environment_{process['pid']}.json", env)
    loaders = dict(mf=lambda u: load_mf(release_root, u, device), unet=lambda u: load_unet(release_root, u, device), unet_map=unet_test_map)
    rows = run_units(units, keys, lambda k: read_image(files[k]), output, loaders, device=device, deadline=deadline, process=process)
    made = write_fold5mean(output, recipe, keys)
    return dict(status='complete' if all(r['status'] in ('complete', 'skipped_existing') for r in rows) else 'partial', archives=rows, fold5mean=made)


# ---------------------------------------------------------------- one image at a time (any 2048 x 2048 grey image)
def load_recipe_models(release_root, recipe, device='cuda', weights=None):
    """The networks one recipe needs, each loaded once (hash-checked release files): {'device', 'mf': {release model
    file: network}, 'unet': {(fold, view): (network, spec)} for a 5-fold-mean recipe, or {view: (network, spec)} for a
    recipe on the package U-Nets (loaded from the package weights, pkg models.load_network)}. For maps_for_image."""
    r = chain.load_recipe(recipe); out = dict(device=device, mf={}, unet={})
    for u in plan(recipe):
        if u['kind'] == 'mf' and u['model'] not in out['mf']:out['mf'][u['model']] = load_mf(release_root, u, device)
        elif u['kind'] == 'unet':out['unet'][(u['fold'], u['view'])] = load_unet(release_root, u, device)
    if r['unet'] == 'package':
        from .assets import resolve_assets
        from .models import load_network
        root, manifest = resolve_assets(weights)
        for v in UNET_VIEWS:
            out['unet'][v] = (load_network(v, root, manifest).to(device), json.loads((DATA / f'{v}_config.json').read_text())['view'])
    return out


def maps_for_image(raw, recipe, models):
    """The recipe's maps of one grey uint8 2048 x 2048 image (any image, not only the frozen test set):
    {archive name NEW{k}_{j}: uint16 levels / 510, 'full': uint8, 'local': uint8}, the archive writer's arithmetic
    (mf_archive_map; U-Net: the fold maps' unet_level_mean, or the package U-Net map). models: load_recipe_models."""
    raw = np.asarray(raw)
    if raw.dtype != np.uint8:raise ValueError('Expected a uint8 grey image')
    if raw.shape != (SIZE, SIZE):raise ValueError(f'Expected a 2048 x 2048 grey image, got {raw.shape}')
    r = chain.load_recipe(recipe); device = models['device']; unet_map = models.get('unet_map', unet_test_map); out = {}
    for a in r['archives']:
        if a['model'] not in models['mf']:raise KeyError(f"{a['model']} is not loaded")
        out[a['name']] = mf_archive_map(models['mf'][a['model']], raw, a['size'], a['views'], device)
    for v in UNET_VIEWS:
        if r['unet'] == 'fold5mean':
            out[v] = unet_level_mean([unet_map(models['unet'][(k, v)][0], raw, models['unet'][(k, v)][1]) for k in range(5)])
        else:
            net, spec = models['unet'][v]; out[v] = unet_map(net, raw, spec)
    return out


# ---------------------------------------------------------------- Kaggle launcher: pinned environments, one process per GPU
CHILD = '''import json,traceback
try:
 from filseg.reproduction.chain_maps import infer_recipe_maps
 infer_recipe_maps({images!r},{release!r},{output!r},{recipe!r},names={names!r},gpu=0,keys={keys!r},deadline={deadline!r},deterministic={deterministic!r},image_set={image_set!r})
except BaseException as e:
 open({failure!r},"w").write(json.dumps(dict(stage={stage!r},error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc())))
 raise
'''

FINISH = '''import json,traceback
try:
 from filseg.reproduction.chain_maps import write_fold5mean,compare_maps
 print(json.dumps(dict(fold5mean=write_fold5mean({output!r},{recipe!r},{keys!r}) if {keys!r} is None else [])),flush=True)
 if {image_set!r} is None:   # new images have no stored maps to compare with
  c=compare_maps({output!r},{release!r},{recipe!r},names={names!r},output={output!r})
  print(json.dumps(dict(all_equal=c['all_equal'],first_differing=c['first_differing'])),flush=True)
except BaseException as e:
 open({failure!r},"w").write(json.dumps(dict(stage="finish",error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc())))
 raise
'''


def launch(images, release_root, output, recipe, names=None, gpus=None, budget_seconds=39600, keys=None, deterministic=True, image_set=None):
    """Kaggle T4 (x2): install the pinned U-Net environment, infer the U-Net fold maps, install the pinned MF environment,
    infer the MF archives; each phase split over the GPUs (one fresh interpreter per GPU). Then the 5-fold means and the
    comparison with the release files (chain_maps_compare.json). budget_seconds bounds the whole call."""
    from .determinism import child_environment
    from .launch import MF, UNET
    if image_set is None and not Path('/kaggle/input').is_dir():raise RuntimeError('This delivery validation is restricted to Kaggle')
    names_gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).split('\n')
    names_gpu = [n.strip() for n in names_gpu if n.strip()]
    if not names_gpu or (image_set is None and not all('T4' in n for n in names_gpu)):raise RuntimeError('Only Kaggle T4 is authorized (new images: any CUDA GPU)')
    gpus = list(range(len(names_gpu))) if gpus is None else list(gpus)
    start = time.time(); deadline = start + budget_seconds - 300; output = Path(output); output.mkdir(parents=True, exist_ok=True)
    units = plan(recipe, names); source = str(Path(__file__).parents[2])
    with _locked(output):
        _save(output / RECEIPTS / 'plan.json', dict(recipes=_recipes(recipe), names=names, keys=keys, gpus=gpus, budget_seconds=budget_seconds,
                                                    files=[u['file'] for u in units] + (derived(recipe) if keys is None else []), units=units))
    stages = []
    for kind, requirements in (('unet', UNET), ('mf', MF)):
        part = [u for u in units if u['kind'] == kind]
        if not part:continue
        if time.time() > deadline - 900:
            stages.append(dict(kind=kind, status='not_started: budget')); continue
        tick = time.time()
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *requirements], check=True, timeout=900)
        if kind == 'unet':
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'nvidia-cudnn-cu13==9.27.0.42'], check=True, timeout=300)
        install_seconds = time.time() - tick; procs = []
        for g, group in zip(gpus, split(part, len(gpus))):
            if not group:continue
            failure = str(output / f'stage_failure_{kind}_gpu{g}.json')
            code = CHILD.format(images=str(images), release=str(release_root), output=str(output), recipe=_recipes(recipe), names=[u['file'] for u in group],
                                keys=keys, deadline=deadline, deterministic=bool(deterministic), failure=failure, stage=f'{kind}_gpu{g}', image_set=image_set)
            env = child_environment(dict(os.environ, PYTHONPATH=source, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                         CUDA_VISIBLE_DEVICES=str(g)), deterministic)
            log = open(output / f'chain_maps_{kind}_gpu{g}.log', 'w')
            procs.append(dict(gpu=g, files=[u['file'] for u in group], log=log, p=subprocess.Popen([sys.executable, '-u', '-c', code], env=env, stdout=log, stderr=subprocess.STDOUT)))
        last = 0
        while any(x['p'].poll() is None for x in procs):
            time.sleep(10)
            if time.time() - last > 300:
                last = time.time()
                print(json.dumps(dict(stage='chain_maps', kind=kind, running=[x['gpu'] for x in procs if x['p'].poll() is None], elapsed_seconds=time.time() - start,
                                      done=sorted(p.name for p in output.glob('*.npz')))), flush=True)
            if time.time() > deadline + 240:
                for x in procs:
                    if x['p'].poll() is None:x['p'].kill()
        for x in procs:x['log'].close()
        stages.append(dict(kind=kind, install_seconds=install_seconds, seconds=time.time() - tick,
                           processes=[dict(gpu=x['gpu'], files=x['files'], returncode=x['p'].returncode) for x in procs],
                           pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()))
    # The pip installs replaced the packages under this interpreter: the 5-fold means and the comparison run in a fresh one.
    env = dict(os.environ, PYTHONPATH=source, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    code = FINISH.format(output=str(output), release=str(release_root), recipe=_recipes(recipe), keys=keys, names=None if names is None else [u['file'] for u in units],
                         failure=str(output / 'stage_failure_finish.json'), image_set=image_set)
    with open(output / 'chain_maps_finish.log', 'w') as log:
        finish = subprocess.run([sys.executable, '-u', '-c', code], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    with _locked(output):
        result = collect(output)
    result.update(stages=stages, finish_returncode=finish, allocation_seconds=time.time() - start, elapsed_seconds=time.time() - start,
                  driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'], text=True).strip())
    if (output / COMPARE).exists():
        cmp = json.loads((output / COMPARE).read_text()); result.update(all_equal=cmp['all_equal'], first_differing=cmp['first_differing'])
    _save(output / RESULT, result)
    return result


# ---------------------------------------------------------------- CPU: the re-inferred maps against the stored ones
def compare_maps(reinferred_dir, release_root, recipe, names=None, output=None):
    """Per archive (MF archives in recipe order, U-Net fold maps, 5-fold means): images equal, changed pixels, max and
    mean |level difference| (over every compared pixel), the histogram of nonzero differences and the first differing
    image; only the images both archives hold are compared. first_differing: the first artefact with a changed pixel."""
    reinferred_dir = Path(reinferred_dir); tick = time.monotonic()
    kinds = {u['file']: u['kind'] for u in plan(recipe)}; kinds.update({f: 'unet_fold5mean' for f in derived(recipe)})
    order = [f for f in kinds if names is None or f in set(names)]
    rows = []
    for f in order:
        a = reinferred_dir / f
        try:b = chain.find(f, [release_root])
        except FileNotFoundError:b = None
        if not a.exists() or b is None:
            rows.append(dict(file=f, kind=kinds[f], present=False, reinferred=a.exists(), release=b is not None)); continue
        with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
            ks = sorted(set(za.files) & set(zb.files))
            equal = changed = total = mx = 0; hist = {}; first = worst = None; dtypes = set()
            for k in ks:
                x, y = za[k], zb[k]; dtypes.add((x.dtype.str, y.dtype.str))
                if x.shape != y.shape:raise ValueError(f'{f} {k}: shapes differ')
                d = np.abs(x.astype(np.int32) - y.astype(np.int32)); n = int(np.count_nonzero(d))
                if n == 0:equal += 1; continue
                m = int(d.max()); changed += n; total += int(d.sum()); mx = max(mx, m)
                for lv, c in enumerate(np.bincount(d[d > 0])):
                    if c:hist[lv] = hist.get(lv, 0) + int(c)
                first = first or k
                if worst is None or n > worst['changed_pixels']:worst = dict(key=k, changed_pixels=n, max_abs_level_diff=m)
        pixels = len(ks) * SIZE * SIZE
        rows.append(dict(file=f, kind=kinds[f], present=True, keys_compared=len(ks), images_equal=equal, images_differing=len(ks) - equal,
                         changed_pixels=changed, changed_fraction=changed / pixels if pixels else None, max_abs_level_diff=mx,
                         mean_abs_level_diff=total / pixels if pixels else None, level_diff_histogram={str(k): hist[k] for k in sorted(hist)},
                         first_differing_image=first, worst_image=worst, dtypes=sorted(dtypes), file_sha256_equal=_sha(a) == _sha(b)))
    present = [r for r in rows if r['present']]
    first = next((r for r in present if r['changed_pixels']), None)
    result = dict(recipes=_recipes(recipe), reinferred=str(reinferred_dir), release=str(release_root),
                  first_differing=None if first is None else {k: first[k] for k in ('file', 'kind', 'first_differing_image', 'images_differing', 'keys_compared',
                                                                                   'changed_pixels', 'changed_fraction', 'max_abs_level_diff', 'mean_abs_level_diff')},
                  all_equal=bool(present) and first is None, compared=len(present),
                  summary={r['file']: (None if not r['present'] else dict(images_equal=r['images_equal'], keys=r['keys_compared'], changed_pixels=r['changed_pixels'],
                                                                         max_abs_level_diff=r['max_abs_level_diff'])) for r in rows},
                  artefacts=rows, elapsed_seconds=time.monotonic() - tick)
    if output is not None:_save(Path(output) / COMPARE, result)
    return result
