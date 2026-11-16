"""D4 test-time augmentation of the Mask2Former foreground map, exported like `eval_ckpt.py --export` (order TT1).

  python export_tta.py --run-dir runs/screen/RID_d4 [--out-dir OUT] [--views 8|1] [--export-subdir export]

The run dir holds final_ema.safetensors, model_config.json, train_done.json, train.log (a finished screening run, or a
copy of one). OUT defaults to the run dir; if it differs, those four files are hard-linked (or copied) into OUT so that
`collect.py RUN_ID OUT` works unchanged.

Map: for each of the 8 D4 views (pkg `mf_d4.TRANSFORMS`: identity, rot90/180/270, flip_W + rot90^k; the same set as
identity, rot90/180/270 and their transposes) the view is transformed, the model is run (BF16 autocast, as in
eval_ckpt), fg = max_q(sigmoid(mask_q) * score_q) is computed at the network output (512 px for the native view, 256 px
for `view_1024`; both inputs are multiples of 4, so every D4 transform maps output cells onto output cells exactly),
inverse-transformed, and the 8 fg maps are averaged in float32; then bilinear to 2048 (align_corners=False) and
rint(fg * 510) -> uint16, once. With --views 1 the computation is the plain export's, bit for bit.

Writes OUT/<subdir>/MF_f<k>_val.npz (the run's held-out fold k = cfg['fold'], checked by eval_ckpt.check_fold),
OUT/<subdir>/MF_f<k>_test.npz (180 test images), OUT/<subdir>/eval_tta<views>_step<N>.json (solar-disk / small-object
pixel AP of the TTA map and of the identity view, eval_ckpt's definitions) and OUT/<subdir>/export_manifest.json
(collect.py format; metrics.pq is not computed for the TTA map and is null). Views are run one at a time.

Full-data runs (cfg['fold'] == 'all', train.py --fold all, order RF1) have seen every training image: only the 180
test maps are written, as OUT/<subdir>/MF_all_test.npz (same format), with eval_tta<views>_step<N>.json (no pixel AP)
and export_manifest.json (fold 'all', large = weights + MF_all_test.npz). No MF_f<k>_val.npz is ever opened for them.
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
sys.path.insert(0, str(HERE))
R = HERE.parent

# pkg/src/filseg/reproduction/mf_d4.py (w8_tta's transform definition): y = rot90(flip_W(x) if flip else x, k).
TRANSFORMS = [('id', False, 0), ('rot90', False, 1), ('rot180', False, 2), ('rot270', False, 3),
              ('flipW', True, 0), ('flipW_rot90', True, 1), ('flipW_rot180', True, 2), ('flipW_rot270', True, 3)]
RUN_FILES = ('final_ema.safetensors', 'model_config.json', 'train_done.json', 'train.log')


def forward(x, t):
    _, flip, k = t
    y = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(y, k, dims=(-2, -1))


def inverse(y, t):
    _, flip, k = t
    x = torch.rot90(y, -k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def fg_lowres(model, x, views=8):
    """x [1, 1, H, W] on the model's device -> (identity fg, mean of the inverse-transformed fg over `views` views),
    both float32 [h, w] at the network output resolution. views=1 -> the identity view only."""
    if views not in (1, 8):
        raise ValueError(f'views must be 1 or 8, got {views}')
    import decode as D
    ident = total = None
    for t in TRANSFORMS[:views]:
        xv = x if t[0] == 'id' else forward(x, t).contiguous()
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16):
            o = model(xv)
        scores = D.query_scores(o.class_queries_logits[0])
        fg = inverse((o.masks_queries_logits[0].float().sigmoid() * scores[:, None, None]).amax(0), t)
        if t[0] == 'id':
            ident = fg
        total = fg if total is None else total + fg
        del o, xv
    return ident, (total if views == 1 else total / views)


def level_map(fg):
    """float [h, w] fg -> int64 [2048, 2048] levels rint(bilinear_2048(fg) * 510) (eval_ckpt's expression)."""
    return np.rint(torch.nn.functional.interpolate(fg[None, None], size=(2048, 2048), mode='bilinear',
                                                   align_corners=False)[0, 0].cpu().numpy() * 510).astype(np.int64)


def run_fold(cfg):
    """The run's held-out fold (int), or 'all' for a full-data run."""
    if 'fold' not in cfg:
        raise SystemExit('run config carries no fold: refusing to export an OOF map')
    return 'all' if str(cfg['fold']) == 'all' else int(cfg['fold'])


def export_plan(cfg):
    """Map files of the export: fold k -> MF_f<k>_val.npz + MF_f<k>_test.npz; full data -> MF_all_test.npz only."""
    k = run_fold(cfg)
    if k == 'all':
        return dict(tag='all', fold='all', val=None, test='MF_all_test.npz',
                    large=['final_ema.safetensors', 'MF_all_test.npz'])
    return dict(tag=f'f{k}', fold=k, val=f'MF_f{k}_val.npz', test=f'MF_f{k}_test.npz',
                large=['final_ema.safetensors', f'MF_f{k}_val.npz', f'MF_f{k}_test.npz'])


def open_val_writer(exp, cfg):
    """The held-out fold's map writer; refuses (before creating any file) for a full-data model."""
    import eval_ckpt as E
    plan = export_plan(cfg)
    if plan['val'] is None:
        raise SystemExit('full-data model (fold all) has seen every training image: refusing to write a val/OOF map')
    E.check_fold(cfg, plan['fold'])
    return E.MapWriter(Path(exp) / plan['val'])


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
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--views', type=int, choices=(1, 8), default=8)
    ap.add_argument('--export-subdir', default='export')
    ap.add_argument('--torch-threads', type=int, default=2)
    a = ap.parse_args()
    import cv2
    from pycocotools import mask as mu
    from safetensors.torch import load_file
    import eval_ckpt as E
    import pqscore
    from model import config_model, sha256_file, view_1024, view_native
    from train import annotation_sets
    cv2.setNumThreads(2)
    torch.set_num_threads(min(a.torch_threads, 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = Path(a.run_dir).resolve()
    out = Path(a.out_dir).resolve() if a.out_dir else run
    out.mkdir(exist_ok=True)
    if out != run:
        for n in RUN_FILES:
            link_or_copy(run / n, out / n)
    data = R / 'data' / 'MAGFiLO_1.0_Kaggle_2026'
    done = json.loads((out / 'train_done.json').read_text())
    step, cfg = done['step'], done['cfg']
    plan = export_plan(cfg)
    k, full = plan['fold'], plan['val'] is None
    if not full:
        E.check_fold(cfg, k)
    model = config_model(out / 'model_config.json')
    model.load_state_dict(load_file(str(out / 'final_ema.safetensors')), strict=True)
    model = model.cuda().eval()
    input_scale = cfg.get('input_scale', 1.0)
    assert input_scale in (1.0, 0.5)
    view = view_native if input_scale == 1.0 else view_1024
    folds = json.loads((R / 'code' / 'fold_table_time24h_k5_seed42.json').read_text())
    files = [] if full else sorted(f for f, kk in folds.items() if kk == k)
    groups = annotation_sets(json.loads((data / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text()))
    exp = out / a.export_subdir
    exp.mkdir(exist_ok=True)
    hist = {m: [np.zeros(E.LEVELS) for _ in range(3)] for m in ('tta', 'identity')}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def maps(raw):
        x = torch.from_numpy(view(raw)[None, None].copy()).cuda()
        ident, mean = fg_lowres(model, x, a.views)
        return level_map(ident), level_map(mean)

    vz = None if full else open_val_writer(exp, cfg)
    for i, f in enumerate(files):
        raw = cv2.imread(str(data / 'train' / 'train_images' / (f + '.jpeg')), 0)
        lid, lmean = maps(raw)
        gt_union = np.zeros((2048, 2048), bool)
        for e in (pqscore.annotation_rles(s) for s in groups[f]):
            if e:
                gt_union |= mu.decode(mu.merge(e)).astype(bool)
        vz.write(f, lmean.astype(np.uint16))
        for m, lv in (('tta', lmean), ('identity', lid)):
            for h, add in zip(hist[m], E.pixel_hists(lv, raw, gt_union)):
                h += add
        if (i + 1) % 20 == 0:
            print(json.dumps(dict(time=now(), split='val', done=i + 1, total=len(files),
                                  seconds=time.monotonic() - started)), flush=True)
    if vz is not None:
        vz.close()
    val_seconds = time.monotonic() - started
    test_dir = data / 'test' / 'test_images'
    tests = sorted(p.stem for p in test_dir.glob('*.jpeg'))
    assert len(tests) == 180
    tz = E.MapWriter(exp / plan['test'])
    for i, f in enumerate(tests):
        _, lmean = maps(cv2.imread(str(test_dir / (f + '.jpeg')), 0))
        tz.write(f, lmean.astype(np.uint16))
        if (i + 1) % 60 == 0:
            print(json.dumps(dict(time=now(), split='test', done=i + 1, total=len(tests),
                                  seconds=time.monotonic() - started)), flush=True)
    tz.close()

    def ap(m):
        hp, hn, hs = hist[m]
        return dict(disk_ap=E.ap_from_hist(hp, hn), small_ap=E.ap_from_hist(hs, hn), positives=int(hp.sum()),
                    negatives=int(hn.sum()), small_positives=int(hs.sum()))

    defn = ('solar disk (Otsu, largest component, filled); positives = GT union of all sets; small = GT-union '
            'components < 1024 px')
    tta_desc = (f'{a.views} views' + (' (D4: identity, rot90/180/270, flip_W + rot90^k), fg inverse-transformed and '
                                      'averaged at the network output before the bilinear resize' if a.views == 8 else
                                      ' (identity only = plain export)'))
    res = dict(time=now(), source=str(out / 'final_ema.safetensors'),
               source_sha256=sha256_file(out / 'final_ema.safetensors'), step=step, fold=k, images=len(files),
               test_images=len(tests), input_scale=input_scale, cfg=cfg, tta=tta_desc,
               pixel_ap=None if full else dict(**ap('tta'), definition=defn),
               pixel_ap_identity_view=None if full else dict(**ap('identity'), definition=defn),
               **(dict(note='full-data model: no held-out images, test maps only') if full else {}),
               seconds=time.monotonic() - started, val_seconds=val_seconds,
               peak_gpu_alloc_bytes=torch.cuda.max_memory_allocated())
    ev = exp / f'eval_tta{a.views}_step{step}.json'
    ev.write_text(json.dumps(res, indent=1))
    files_out = {n: out / n for n in RUN_FILES}
    files_out.update({ev.name: ev, **({} if full else {plan['val']: exp / plan['val']}), plan['test']: exp / plan['test']})
    man = dict(time=now(), run_dir=str(out), source_run_dir=str(run), fold=k, step=step, cfg=cfg,
               cfg_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
               metrics=dict(pq=dict(pq=None, note='not computed for the TTA map'), pixel_ap=res['pixel_ap'],
                            pixel_ap_identity_view=res['pixel_ap_identity_view'], tta_views=a.views),
               map_format=dict(dtype='uint16', shape=[2048, 2048], denominator=510, keys='image stem',
                               formula='max_q(sigmoid(mask_q)*score_q), bilinear to 2048 (pkg inference.mf_map)',
                               view='native 2048' if input_scale == 1.0 else '1024 INTER_AREA', tta=tta_desc),
               files={n: dict(path=str(p), bytes=p.stat().st_size, sha256=sha256_file(p)) for n, p in files_out.items()},
               large=plan['large'])
    (exp / 'export_manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(dict(export=str(exp / 'export_manifest.json'), views=a.views, fold=k, seconds=res['seconds'],
                          **({} if full else dict(disk_ap=res['pixel_ap']['disk_ap'], small_ap=res['pixel_ap']['small_ap'],
                                                   disk_ap_identity=res['pixel_ap_identity_view']['disk_ap'],
                                                   small_ap_identity=res['pixel_ap_identity_view']['small_ap'])),
                          peak_gpu_alloc_bytes=res['peak_gpu_alloc_bytes'])), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
