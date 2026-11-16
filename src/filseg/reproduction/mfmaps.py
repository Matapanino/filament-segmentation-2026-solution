"""MF-map variants: D4 test-time augmentation and a tiled native-scale reading of the three MF fold models (GPU), and
the MF probability map each variant gives the CSV stage (CPU).

GPU (Kaggle T4 only, inference only): for every test image and each of MF0, MF1 and MF2,
  tta    the mean of the 8 D4 views at the network's 256-px output, each inverse-transformed (mf_d4.py, a byte-identical
         copy of detect_20261113/mftta/d4.py); the identity view is the package MF map and is checked against it;
  tiled  nine 1024x1024 tiles of the native 2048 image at stride 512, the foreground of each tile Hann-blended on a
         512-px canvas (mf_tile.py, a byte-identical copy of detect_20261113/mfhires/tile.py);
both upsampled to 2048 (bilinear) and quantised like the package maps (rint(x 510), uint16). The forward pass and the
foreground formula max_q(sigmoid(mask_q) x score_q) are the cross-validation code's (and filseg.reproduction.inference.mf_map's).
  views  (opt-in, views=('tta', 'views'); the MF-var mode) the 8 inverse-transformed D4 views of the TTA pass itself, each
         upsampled and quantised like the TTA map, one (8, 2048, 2048) uint16 array per image in mf_d4.TRANSFORMS order
         (detect_20261113/mfvar/gpu_views.py); view 0 is the identity view. The TTA archives of that pass are the ones
         written without views, bit for bit (the same calls in the same order).
The GPU stage reads the TTA by default (views=('tta',)); the tiled reading is opt-in (views=('tta', 'tiled')).
Archives: MF{k}_tta.npz (and MF{k}_tiled.npz, MF{k}_views.npz) next to the package maps, receipts in mf_views_result.json.
The package maps (full, local, MF0, MF1, MF2; map set d7aa6a41...) are not rewritten. MF{k}_views.npz is written image by
image (mf_views_io.py, a byte-identical copy of detect_20261113/mfvar/views_io.py; deflate level 1).

CPU: the MF probability map of a variant is the level mean of its archives (arithmetic.mf_levels_mean):
  mftta    (T0 + T1 + T2) / 1530                         the three-fold mean of the TTA maps;
  mfhires  (M0 + M1 + M2 + H0 + H1 + H2) / 3060          the three-fold mean of (package map + tiled map) / 2;
  mfcomb   (T0 + T1 + T2 + H0 + H1 + H2) / 3060          the three-fold mean of (TTA map + tiled map) / 2.
Cross-validation reads one out-of-fold model per image as (a + b) / 1020; at test the three fold models are averaged, as
for the package MF mean. Everything downstream (blend .25 MF + .375 full + .375 local, SHIP_R3, features, selector,
P-cand repair) reads this map in place of the package MF mean.
"""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import mf_d4 as D4
from . import mf_tile as T

RESULT = 'mf_views_result.json'
FOLDS = ('MF0', 'MF1', 'MF2')
TTA = tuple(f'{n}_tta' for n in FOLDS)
TILED = tuple(f'{n}_tiled' for n in FOLDS)
VIEWS = tuple(f'{n}_views' for n in FOLDS)
KINDS = ('tta', 'tiled', 'views')
MF_ARCHIVES = dict(mftta=TTA, mfhires=FOLDS + TILED, mfcomb=TTA + TILED)
MAP_SETS = dict(package=('full', 'local') + FOLDS, **{v: ('full', 'local') + a for v, a in MF_ARCHIVES.items()})
# Map-set sha256 (sha256 of the JSON of the archive sha256, sorted keys, as for the package map set) expected from the
# package maps and the cross-validation runs' T4 test outputs (mftta test_tta_MF{k}.npz, mfhires test_tiled_MF{k}.npz).
EXPECTED_MAP_SETS = dict(package='d7aa6a41fafc26b07ce89b306b884f6667ebb68b86289ea4745ee44c0b0d195f',
                         mftta='44ee26c7d4284b02103f005935611b0911d8f1fe20e1693bf73f0eb21fff9cc2',
                         mfhires='6f97d421381f6470ff7748726bad27fcc89fde0c74fdf5dc2f9610125e54255e',
                         mfcomb='969adfb0f90fce9589424936423a64edfb8e7715dd191e9e779ac688254cbda1')


def map_set_sha256(archives):
    """archives: {archive name: archive file sha256}."""
    return hashlib.sha256(json.dumps(dict(archives), sort_keys=True).encode()).hexdigest()


def variant_mf(maps, variant):
    """The variant's MF probability map (float32) from the uint16 archives in maps."""
    from .arithmetic import mf_levels_mean
    if variant not in MF_ARCHIVES:raise ValueError('Unknown MF-map variant: ' + repr(variant))
    return mf_levels_mean(*(maps[n] for n in MF_ARCHIVES[variant]))


def load_view_archives(root, keys):
    """[(name, archive)] of MF0_views, MF1_views, MF2_views (MF-var), each checked against mf_views_result.json."""
    from .audit import verify
    root = Path(root); result = json.loads((root / RESULT).read_text()); out = []
    if result.get('status') != 'complete':raise ValueError('MF-view inference is incomplete')
    for name in VIEWS:
        if name not in result['networks']:raise ValueError('MF-view inference has no per-view archive: ' + name)
        row = result['networks'][name]; path = root / row['file']; verify(path, row)
        z = np.load(path, allow_pickle=False)
        if sorted(z.files) != sorted(keys):raise ValueError('Map key coverage differs: ' + name)
        out.append((name, z))
    return out


def load_variant_archives(root, variant, keys):
    """The variant's added archives (those not in the package map set), each checked against mf_views_result.json."""
    from .audit import verify
    if variant not in MF_ARCHIVES:raise ValueError('Unknown MF-map variant: ' + repr(variant))
    root = Path(root); result = json.loads((root / RESULT).read_text()); out = {}
    if result.get('status') != 'complete':raise ValueError('MF-view inference is incomplete')
    for name in MF_ARCHIVES[variant]:
        if name in FOLDS:continue
        row = result['networks'][name]; path = root / row['file']; verify(path, row)
        out[name] = np.load(path, allow_pickle=False)
        if sorted(out[name].files) != sorted(keys):raise ValueError('Map key coverage differs: ' + name)
    return out


# ---------------------------------------------------------------- GPU
def foreground(model):
    """detect_20261113/mftta/gpu_tta.py foreground(): BF16 autocast forward, max over queries of sigmoid(mask) x class score."""
    import torch

    def fg(x):
        with torch.autocast('cuda', dtype=torch.bfloat16): out = model(x)
        scores = out.class_queries_logits[0].float().softmax(-1)[:, 0]; logits = out.masks_queries_logits[0].float()
        return (logits.sigmoid() * scores[:, None, None]).amax(0)
    return fg


def model_input(raw, device='cuda'):
    """The package MF view: 1024 px INTER_AREA, / 255, [1, 1, 1024, 1024] float32."""
    import torch
    return torch.from_numpy((cv2.resize(raw, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)[None, None].copy()).to(device)


def image_views(fg, raw, device='cuda', views=('tta', 'tiled')):
    """uint16 identity and TTA maps of one image, the tiled map when 'tiled' is in views and the 8 quantised D4 views when
    'views' is in views (fg: [1,1,1024,1024] -> [256,256] foreground)."""
    if 'views' in views:
        # detect_20261113/mfvar/gpu_views.views_of: the same d4.tta pass keeps its 8 inverse-transformed views
        ident, mean, each = D4.tta(fg, model_input(raw, device), keep_views=True)
    else:
        ident, mean = D4.tta(fg, model_input(raw, device))
    out = dict(identity=D4.quantise(D4.upsample(ident)), tta=D4.quantise(D4.upsample(mean)))
    if 'views' in views:
        out['views'] = np.stack([D4.quantise(D4.upsample(v)) for v in each])
        if out['views'].shape != (8, 2048, 2048) or not np.array_equal(out['views'][0], out['identity']):raise RuntimeError('View 0 must be the identity view')
    if 'tiled' in views:out['tiled'] = T.quantise(T.upsample(T.tiled(fg, T.native_input(raw, device))))
    return out


def run_mf_view_maps(images, weights, output, budget_seconds=6600, names=FOLDS, deterministic=True, views=('tta',)):
    """T4: MF{k}_tta.npz (and MF{k}_tiled.npz when 'tiled' is in views, MF{k}_views.npz when 'views' is in views) of the
    180 test images; the identity view is compared with MF{k}.npz when present. The default reads the D4 TTA only."""
    views = tuple(views)
    if 'tta' not in views or not set(views) <= set(KINDS):raise ValueError('views: tta, optionally with tiled and/or views')
    import torch
    from .assets import resolve_assets
    from .audit import array_sha, identity
    from .determinism import archive_content_sha, configure, environment, save_npz
    from .models import load_network
    from .pipeline import prepare_images, read_image, save
    from .mf_views_io import ViewsWriter
    if not torch.cuda.is_available():raise RuntimeError('Neural inference requires the authorized GPU runtime')
    if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('This validation run is restricted to Kaggle T4')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Installed PyTorch cannot execute original MF BF16')
    start = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    root, manifest = resolve_assets(weights); files = prepare_images(images, output / 'test_images')
    torch.set_num_threads(2); cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    settings = configure(deterministic)
    env = dict(environment(), determinism=settings, opencv=cv2.__version__, mf_precision='bf16')
    path = output / RESULT
    result = json.loads(path.read_text()) if path.exists() else dict(networks={}, identity_check={})
    result.update(status='running', environment=env, images=len(files), model_fits=0, submission_api_calls=0,
                  views=dict(tta=[t[0] for t in D4.TRANSFORMS], **({'tiled': dict(offsets=T.offsets(), tiles=len(T.tiles()), tile=T.TILE, stride=T.STRIDE)} if 'tiled' in views else {}),
                             **({'views': dict(order=[t[0] for t in D4.TRANSFORMS], shape=[8, 2048, 2048], denominator=D4.DENOMINATOR)} if 'views' in views else {})))
    save(path, result)
    for name in names:
        if any(f'{name}_{kind}' in result['networks'] for kind in views):raise RuntimeError('Refusing to repeat an already completed network')
        tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        model = load_network(name, root, manifest).cuda(); fg = foreground(model)
        reference = np.load(output / (name + '.npz'), allow_pickle=False) if (output / (name + '.npz')).exists() else None
        tta, tiled, rows = {}, {}, {}
        writer = ViewsWriter(output / f'{name}_views.npz') if 'views' in views else None
        with torch.inference_mode():
            for i, (key, image) in enumerate(sorted(files.items())):
                if time.monotonic() - start > budget_seconds - 120:raise TimeoutError('Inference allocation budget reached')
                got = image_views(fg, read_image(image), views=views)
                tta[key] = got['tta']
                if 'tiled' in views:tiled[key] = got['tiled']
                if writer is not None:writer.write(key, got['views'])
                rows[key] = dict(identity_sha256=array_sha(got['identity']), **{f'{kind}_sha256': array_sha(got[kind]) for kind in views},
                                 identity_equals_package=None if reference is None else bool(np.array_equal(got['identity'], reference[key])))
                if (i + 1) % 20 == 0:print(json.dumps(dict(stage=name + '_views', done=i + 1, total=len(files), elapsed_seconds=time.monotonic() - start)), flush=True)
        for kind, arrays in (('tta', tta), ('tiled', tiled)):
            if kind not in views:continue
            p = output / f'{name}_{kind}.npz'; save_npz(p, arrays)
            result['networks'][f'{name}_{kind}'] = dict(file=p.name, **identity(p), content_sha256=archive_content_sha(p))
        if writer is not None:
            writer.close(); p = output / f'{name}_views.npz'
            if writer.keys != sorted(files):raise RuntimeError('Per-view archive keys differ from the test images')
            result['networks'][f'{name}_views'] = dict(file=p.name, **identity(p), content_sha256=archive_content_sha(p))
        result['identity_check'][name] = dict(reference=None if reference is None else dict(file=name + '.npz', **identity(output / (name + '.npz'))),
                                              equal_images=None if reference is None else sum(r['identity_equals_package'] for r in rows.values()), images=len(rows))
        result.setdefault('per_image', {})[name] = rows
        result.setdefault('seconds', {})[name] = dict(elapsed_seconds=time.monotonic() - tick, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
        save(path, result); del model, fg, tta, tiled; torch.cuda.empty_cache()
    complete = all(n in result['networks'] for n in TTA + (TILED if 'tiled' in views else ()) + (VIEWS if 'views' in views else ()))
    result.update(status='complete' if complete else 'partial', elapsed_seconds=time.monotonic() - start)
    package = output / 'inference_result.json'
    if complete and package.exists():
        archives = {n: r['sha256'] for n, r in json.loads(package.read_text())['networks'].items()}
        archives.update({n: r['sha256'] for n, r in result['networks'].items()})
        result['map_sets'] = {v: dict(sha256=map_set_sha256({n: archives[n] for n in MAP_SETS[v]}), expected=EXPECTED_MAP_SETS[v]) for v in MAP_SETS if all(n in archives for n in MAP_SETS[v])}
        for row in result['map_sets'].values():row['equal'] = row['sha256'] == row['expected']
    save(path, result)
    return result
