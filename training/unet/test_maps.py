"""Test-set U-Net maps in the format the chain's test stage reads (order UN1).

Format (package `pipeline.run_maps` / chain2 `csv_worker.py`): one archive per view, `full.npz` / `local.npz`, written with
the package's `determinism.save_npz` (sorted members, fixed timestamps); members = the 180 test image stems; each array
uint8 (2048, 2048) = rint(p * 255), p = sigmoid of the network on the view's tiles, restored to 2048 by
`preprocess.restore_tiles`. A multi-model map is the per-pixel level mean of the models' uint8 maps, rounded once:
uint8(rint(sum_m q_m / n)) (np.rint, half to even).

Two subcommands:
  infer  (GPU) checkpoints -> maps with the package's own `inference.unet_map` (FP16 autocast, tile batch 2) under the
         package's deterministic settings (`determinism.configure(True)`); also writes each model's own archive.
      python -B test_maps.py infer --view full --images $DATA/test/test_images --out $R/runs/unet_test/<label> \
             --checkpoints <run dir>/last.pt [<run dir 2>/last.pt ...]
  mean   (CPU) average existing per-model test archives (e.g. the as-run `test_prob.npz` of the five fold models):
      python -B test_maps.py mean --view full --out <dir> --archives f0/test_prob.npz f1/test_prob.npz ...
Both write <out>/<view>.npz and <out>/<view>_maps.json (sources with sha256, per-image array sha256, archive sha256).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SIZE = 2048


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def level_mean(arrays):
    """uint8(rint(sum / n)) of uint8 (2048, 2048) maps; n = 1 returns the map unchanged."""
    for q in arrays:
        if q.dtype != np.uint8 or q.shape != (SIZE, SIZE): raise ValueError('U-Net maps must be uint8 2048x2048')
    if len(arrays) == 1: return arrays[0]
    s = np.zeros((SIZE, SIZE), np.uint32)
    for q in arrays: s += q
    return np.rint(s / np.float64(len(arrays))).astype(np.uint8)


def write(out, view, per_model, sources, keys):
    from filseg.reproduction.determinism import save_npz
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    maps = {k: level_mean([m[k] for m in per_model]) for k in keys}
    path = out / f'{view}.npz'; save_npz(path, maps)
    rec = dict(view=view, n_models=len(per_model), sources=sources, n_images=len(maps), archive=path.name, archive_sha256=sha_file(path),
               arrays={k: hashlib.sha256(np.ascontiguousarray(maps[k]).tobytes()).hexdigest() for k in sorted(maps)},
               mean_rule='uint8(rint(sum_m q_m / n))')
    (out / f'{view}_maps.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ('view', 'n_models', 'n_images', 'archive_sha256')}))
    return rec


def cmd_mean(a):
    per_model, sources, keysets = [], [], []
    for p in a.archives:
        with np.load(p, allow_pickle=False) as z:
            per_model.append({k: z[k] for k in z.files}); keysets.append(sorted(z.files))
        sources.append(dict(archive=str(p), sha256=sha_file(p)))
    if any(k != keysets[0] for k in keysets): raise ValueError('The archives cover different images')
    if a.expect_images is not None and len(keysets[0]) != a.expect_images: raise ValueError(f'{len(keysets[0])} images, expected {a.expect_images}')
    return write(a.out, a.view, per_model, sources, keysets[0])


def cmd_infer(a):
    import cv2
    import torch
    from filseg.reproduction.determinism import configure
    from filseg.reproduction.inference import unet_map
    from train.unet.oof_maps import load_model
    from train.unet.train_unet import load_config
    if not torch.cuda.is_available(): raise RuntimeError('infer needs a GPU (the package unet_map is CUDA-only)')
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    configure(True)
    spec = load_config(a.view)['view']
    images = sorted(Path(a.images).glob('*.jpeg'))
    if a.expect_images is not None and len(images) != a.expect_images: raise ValueError(f'{len(images)} images, expected {a.expect_images}')
    keys = [p.stem for p in images]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_model, sources = [], []
    from filseg.reproduction.determinism import save_npz
    for i, ck in enumerate(a.checkpoints):
        model = load_model(ck, load_config(a.view)).cuda().eval(); maps = {}
        for n, p in enumerate(images):
            raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if raw is None or raw.shape != (SIZE, SIZE): raise ValueError(f'Bad test image {p}')
            maps[p.stem] = unet_map(model, raw, spec)[0]
            if (n + 1) % 30 == 0: print(json.dumps(dict(model=i, done=n + 1, total=len(images))), flush=True)
        save_npz(out / f'{a.view}_m{i}.npz', maps)
        per_model.append(maps); sources.append(dict(checkpoint=str(ck), sha256=sha_file(ck), own_archive=f'{a.view}_m{i}.npz'))
        del model; torch.cuda.empty_cache()
    return write(out, a.view, per_model, sources, keys)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]); s = p.add_subparsers(dest='cmd', required=True)
    i = s.add_parser('infer'); i.add_argument('--view', choices=('full', 'local'), required=True); i.add_argument('--images', required=True)
    i.add_argument('--out', required=True); i.add_argument('--checkpoints', nargs='+', required=True); i.add_argument('--expect-images', type=int, default=180)
    m = s.add_parser('mean'); m.add_argument('--view', choices=('full', 'local'), required=True); m.add_argument('--out', required=True)
    m.add_argument('--archives', nargs='+', required=True); m.add_argument('--expect-images', type=int, default=180)
    a = p.parse_args(argv)
    return cmd_infer(a) if a.cmd == 'infer' else cmd_mean(a)


if __name__ == '__main__':
    main()
