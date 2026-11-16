"""Scale TTA of the Mask2Former foreground map (order MS1): a copy of a1native/export_tta.py with one change — the views.

  python export_scale.py --run-dir RUN --out-dir OUT --size 2560 --views 2 [--weights W --expect-sha S] [--limit-val N --no-test]

Views: the 2048 native view (raw / 255, float32) is resized to --size x --size (torch bilinear, align_corners=False; skipped
when --size is 2048), then for each view in {identity, flip_W}[:views] the model is run (BF16 autocast, as eval_ckpt and
export_tta), fg = max_q(sigmoid(mask_q) * score_q) is computed at the network output (size/4 px), flipped back, the views
are averaged in float32, then bilinear to 2048 (align_corners=False) and rint(fg * 510) -> uint16, once (export_tta's
level_map). --size must be a multiple of 32 (every output cell maps onto an input 4x4 block and the flip is exact on the
output grid). With --size 2048 --views 1 the computation is export_tta's --views 1 = the plain export, bit for bit.

Writes OUT/<subdir>/MF_f<k>_val.npz, MF_f<k>_test.npz, eval_scale<size>_v<views>_step<N>.json and export_manifest.json in
collect.py / chainF stage.py format (same map_format keys, denominator 510). Fold runs only (no full-data runs).
--weights: a weights file to use instead of RUN/final_ema.safetensors (linked or copied into OUT), checked against
--expect-sha (sha256) before the model is built.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R = HERE.parent
for _p in (R / 'code', R / 'a1native'):          # host: filament/code ; Mac: filament-main/a1native
    if _p.is_dir():
        sys.path.insert(0, str(_p))

VIEWS = [('id', False, 0), ('flipW', True, 0)]
RUN_FILES = ('final_ema.safetensors', 'model_config.json', 'train_done.json', 'train.log')


def forward(x, t):
    _, flip, k = t
    y = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(y, k, dims=(-2, -1))


def inverse(y, t):
    _, flip, k = t
    x = torch.rot90(y, -k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def scaled_input(x, size):
    """x [1, 1, 2048, 2048] float32 -> [1, 1, size, size] (bilinear, align_corners=False); unchanged at size 2048."""
    if size % 32:
        raise ValueError(f'size {size} is not a multiple of 32')
    if size == x.shape[-1]:
        return x
    return torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)


def fg_lowres(model, x, size=2560, views=2):
    """x [1, 1, H, W] on the model's device -> mean over `views` views of the flipped-back fg, float32 [size/4, size/4]."""
    if views not in (1, 2):
        raise ValueError(f'views must be 1 or 2, got {views}')
    import decode as D
    xs = scaled_input(x, size)
    total = None
    for t in VIEWS[:views]:
        xv = xs if t[0] == 'id' else forward(xs, t).contiguous()
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16):
            o = model(xv)
        scores = D.query_scores(o.class_queries_logits[0])
        fg = inverse((o.masks_queries_logits[0].float().sigmoid() * scores[:, None, None]).amax(0), t)
        total = fg if total is None else total + fg
        del o, xv
    return total if views == 1 else total / views


def level_map(fg):
    """float [h, w] fg -> int64 [2048, 2048] levels rint(bilinear_2048(fg) * 510) (eval_ckpt's expression)."""
    return np.rint(torch.nn.functional.interpolate(fg[None, None], size=(2048, 2048), mode='bilinear',
                                                   align_corners=False)[0, 0].cpu().numpy() * 510).astype(np.int64)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--size', type=int, default=2560)
    ap.add_argument('--views', type=int, choices=(1, 2), default=2)
    ap.add_argument('--weights', default=None)
    ap.add_argument('--expect-sha', default=None)
    ap.add_argument('--export-subdir', default='export')
    ap.add_argument('--limit-val', type=int, default=None, help='test runs: only the first N val images')
    ap.add_argument('--no-test', action='store_true', help='test runs: skip the 180 test images')
    ap.add_argument('--torch-threads', type=int, default=1)
    a = ap.parse_args()
    import cv2
    from pycocotools import mask as mu
    from safetensors.torch import load_file
    import eval_ckpt as E
    import pqscore
    from model import config_model, sha256_file, view_native
    from train import annotation_sets
    cv2.setNumThreads(2)
    torch.set_num_threads(min(a.torch_threads, 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = Path(a.run_dir).resolve()
    out = Path(a.out_dir).resolve()
    out.mkdir(exist_ok=True)
    wsrc = Path(a.weights).resolve() if a.weights else run / 'final_ema.safetensors'
    if a.expect_sha and sha256_file(wsrc) != a.expect_sha:
        raise SystemExit(f'weights {wsrc}: sha256 differs from --expect-sha')
    link_or_copy(wsrc, out / 'final_ema.safetensors')
    for n in RUN_FILES[1:]:
        if (run / n).exists():
            link_or_copy(run / n, out / n)
    data = R / 'data' / 'MAGFiLO_1.0_Kaggle_2026'
    done = json.loads((out / 'train_done.json').read_text())
    step, cfg = done['step'], done['cfg']
    k = int(cfg['fold'])
    E.check_fold(cfg, k)
    assert cfg.get('input_scale', 1.0) == 1.0
    model = config_model(out / 'model_config.json')
    model.load_state_dict(load_file(str(out / 'final_ema.safetensors')), strict=True)
    model = model.cuda().eval()
    folds = json.loads((R / ('code' if (R / 'code').is_dir() else 'a1native') / 'fold_table_time24h_k5_seed42.json').read_text())
    files = sorted(f for f, kk in folds.items() if kk == k)[:a.limit_val]
    groups = annotation_sets(json.loads((data / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text()))
    exp = out / a.export_subdir
    exp.mkdir(exist_ok=True)
    hist = [np.zeros(E.LEVELS) for _ in range(3)]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def maps(raw):
        x = torch.from_numpy(view_native(raw)[None, None].copy()).cuda()
        return level_map(fg_lowres(model, x, a.size, a.views))

    val_name, test_name = f'MF_f{k}_val.npz', f'MF_f{k}_test.npz'
    vz = E.MapWriter(exp / val_name)
    for i, f in enumerate(files):
        raw = cv2.imread(str(data / 'train' / 'train_images' / (f + '.jpeg')), 0)
        lv = maps(raw)
        gt_union = np.zeros((2048, 2048), bool)
        for e in (pqscore.annotation_rles(s) for s in groups[f]):
            if e:
                gt_union |= mu.decode(mu.merge(e)).astype(bool)
        vz.write(f, lv.astype(np.uint16))
        for h, add in zip(hist, E.pixel_hists(lv, raw, gt_union)):
            h += add
        if (i + 1) % 20 == 0 or i == 0:
            print(json.dumps(dict(time=now(), split='val', done=i + 1, total=len(files), seconds=time.monotonic() - started,
                                  peak_gpu_alloc_bytes=torch.cuda.max_memory_allocated())), flush=True)
    vz.close()
    val_seconds = time.monotonic() - started
    tests = []
    if not a.no_test:
        test_dir = data / 'test' / 'test_images'
        tests = sorted(p.stem for p in test_dir.glob('*.jpeg'))
        assert len(tests) == 180
        tz = E.MapWriter(exp / test_name)
        for i, f in enumerate(tests):
            tz.write(f, maps(cv2.imread(str(test_dir / (f + '.jpeg')), 0)).astype(np.uint16))
            if (i + 1) % 60 == 0:
                print(json.dumps(dict(time=now(), split='test', done=i + 1, total=len(tests),
                                      seconds=time.monotonic() - started)), flush=True)
        tz.close()
    hp, hn, hs = hist
    defn = ('solar disk (Otsu, largest component, filled); positives = GT union of all sets; small = GT-union '
            'components < 1024 px')
    pix = dict(disk_ap=E.ap_from_hist(hp, hn), small_ap=E.ap_from_hist(hs, hn), positives=int(hp.sum()),
               negatives=int(hn.sum()), small_positives=int(hs.sum()), definition=defn)
    view_desc = 'native 2048' if a.size == 2048 else f'native 2048 resized to {a.size} (torch bilinear, align_corners=False)'
    tta_desc = (f'scale {a.size}/2048, {a.views} view(s) ' + ('(identity, flip_W), fg flipped back and averaged at the '
                'network output before the bilinear resize to 2048' if a.views == 2 else '(identity only)'))
    res = dict(time=now(), source=str(out / 'final_ema.safetensors'),
               source_sha256=sha256_file(out / 'final_ema.safetensors'), step=step, fold=k, images=len(files),
               test_images=len(tests), size=a.size, views=a.views, cfg=cfg, tta=tta_desc, pixel_ap=pix,
               pixel_ap_identity_view=None, seconds=time.monotonic() - started, val_seconds=val_seconds,
               peak_gpu_alloc_bytes=torch.cuda.max_memory_allocated())
    ev = exp / f'eval_scale{a.size}_v{a.views}_step{step}.json'
    ev.write_text(json.dumps(res, indent=1))
    files_out = {n: out / n for n in RUN_FILES if (out / n).exists()}
    files_out.update({ev.name: ev, val_name: exp / val_name, **({} if a.no_test else {test_name: exp / test_name})})
    man = dict(time=now(), run_dir=str(out), source_run_dir=str(run), fold=k, step=step, cfg=cfg,
               cfg_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
               metrics=dict(pq=dict(pq=None, note='not computed for the scale-TTA map'), pixel_ap=pix,
                            pixel_ap_identity_view=None, tta_views=a.views, size=a.size),
               map_format=dict(dtype='uint16', shape=[2048, 2048], denominator=510, keys='image stem',
                               formula='max_q(sigmoid(mask_q)*score_q), bilinear to 2048 (pkg inference.mf_map)',
                               view=view_desc, tta=tta_desc),
               files={n: dict(path=str(p), bytes=p.stat().st_size, sha256=sha256_file(p)) for n, p in files_out.items()},
               large=['final_ema.safetensors', val_name] + ([] if a.no_test else [test_name]))
    (exp / 'export_manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(dict(export=str(exp / 'export_manifest.json'), size=a.size, views=a.views, fold=k,
                          seconds=res['seconds'], val_seconds=val_seconds, disk_ap=pix['disk_ap'], small_ap=pix['small_ap'],
                          peak_gpu_alloc_bytes=res['peak_gpu_alloc_bytes'])), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
