"""Screening evaluator for any checkpoint (host GPU for the forward pass, host CPU for scoring).

  python eval_ckpt.py --run-dir runs/X [--which final|ckpt] [--fold 2] [--limit N]
  python eval_ckpt.py --shipped MF2 --out-name shipped_MF2          (the shipped model, same metrics)

Reports on the held-out fold (default 2, 141 images):
- own-instance PQ (SQ, RQ, TP/FP/FN) with decoder D at score >= 0.5 (+ the 0.3-0.7 grid), scored by `pqscore.py`
  (the as-run host-parity PQ);
- solar-disk pixel AP of the foreground map (package formula max_q(sigmoid(mask_q) * score_q), bilinear to 2048,
  quantised to 0..510): positives = union of all GT instances of all annotation sets; negatives = other pixels of the
  solar disk (Otsu threshold, largest connected component, holes filled; the mfhires lane's definition); AP pooled over
  images from per-level histograms (exact for the 511 quantised levels). Small-object AP: positives = pixels of GT-union
  connected components < 1024 px, negatives as above (other positives excluded).
The view follows the run's input_scale (1.0 -> the native 2048 image, 0.5 -> the package 1024 INTER_AREA view).
Output: <run-dir>/eval_<which>_step<N>.json (or eval/<out-name>.json for --shipped).

--export (final weights only): also writes, in the package MF map format (uint16 [2048, 2048] = rint(fg * 510), one
<image stem>.npy member per image, sorted, deflate, fixed 1980-01-01 timestamps, as determinism.save_npz), the
held-out fold's maps <run-dir>/export/MF_f<k>_val.npz and the 180 test-image maps <run-dir>/export/MF_f<k>_test.npz,
plus <run-dir>/export/export_manifest.json (sha256 + bytes of the weights, config, metrics and maps) for collect.py.

Full-data runs (train.py --fold all, cfg['fold'] == 'all', order RF1) saw every training image: they are never evaluated
and never get an MF_f<k>_val.npz. `--which final --export` on such a run (what queue_gpu.sh issues, with --fold 2 by
default) is replaced by `export_tta.py --run-dir RUN --views 8` (D4 test maps only, MF_all_test.npz); anything else
exits with an error.
"""
import hashlib
import io
import zipfile
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = HERE.parent
GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
LEVELS = 511


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def disk_mask(raw):
    import cv2
    _, th = cv2.threshold(raw, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.ones_like(raw, bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    flood = m.copy()
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    return (m | (1 - flood)).astype(bool)


def ap_from_hist(pos, neg):
    """AP of a score with integer levels: pos/neg = counts per level. Thresholds descend over levels."""
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    if tp[-1] == 0:
        return None
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / tp[-1]
    drec = np.diff(np.concatenate([[0.0], rec]))
    return float((prec * drec).sum())


def pixel_hists(level_map, raw, gt_union):
    import cv2
    disk = disk_mask(raw)
    pos = gt_union & disk
    neg = disk & ~gt_union
    n, lab, stats, _ = cv2.connectedComponentsWithStats(gt_union.astype(np.uint8), connectivity=8)
    small_ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < 1024)
    small_ids = small_ids[small_ids > 0]
    small = np.isin(lab, small_ids) & disk
    h = lambda m: np.bincount(level_map[m].ravel(), minlength=LEVELS)[:LEVELS]  # noqa: E731
    return h(pos), h(neg), h(small)


class MapWriter:
    """Image-by-image npz writer, byte-compatible with pkg determinism.save_npz when keys arrive sorted."""

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


def is_full_data(cfg):
    return 'fold' in cfg and str(cfg['fold']) == 'all'


def check_fold(cfg, fold):
    """A trained run may only be evaluated/exported on the fold it held out (shipped models carry no fold); a full-data
    model ('all') has no held-out fold at all."""
    if is_full_data(cfg):
        raise SystemExit(f'full-data model (fold all) has seen every training image: refusing to evaluate/export fold {fold}')
    if 'fold' in cfg and int(cfg['fold']) != int(fold):
        raise SystemExit(f'fold mismatch: model trained with fold {cfg["fold"]} held out, asked to evaluate fold {fold}')


def full_data_redirect(a):
    """None for fold runs / shipped models; for a full-data run: the argv of the D4 test-only export when a final export
    was asked (the queue's post-training call), else SystemExit (there is nothing to evaluate)."""
    if a['shipped'] or not a['run_dir'] or a['which'] != 'final':
        return None  # --which ckpt: check_fold refuses a full-data checkpoint after loading it
    run = Path(a['run_dir'])
    if not is_full_data(json.loads((run / 'train_done.json').read_text())['cfg']):
        return None
    if not a['export']:
        raise SystemExit('full-data model (fold all): no held-out evaluation; use --which final --export (test maps only)')
    return [sys.executable, str(HERE / 'export_tta.py'), '--run-dir', str(run), '--views', '8', '--torch-threads', '1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--which', choices=('final', 'ckpt'), default='final')
    ap.add_argument('--shipped', default=None, help='MF2: evaluate the shipped weights instead of a run')
    ap.add_argument('--out-name', default=None)
    ap.add_argument('--fold', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--export', action='store_true')
    a = ap.parse_args()
    assert not (a.export and (a.shipped or a.which != 'final' or a.limit)), '--export needs --which final, all images'
    redirect = full_data_redirect(vars(a))
    if redirect:
        print(json.dumps(dict(time=now(), full_data_run=a.run_dir, ignored_fold=a.fold, exec=redirect)), flush=True)
        os.execv(redirect[0], redirect)
    import cv2
    from pycocotools import mask as mu
    import decode as D
    import pqscore
    from model import config_model, load_shipped, sha256_file, view_1024, view_native
    from train import annotation_sets
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = R / 'data' / 'MAGFiLO_1.0_Kaggle_2026'
    if a.shipped:
        w = R / 'weights' / f'{a.shipped}.safetensors'
        row = json.loads((R / 'weights' / 'reproduction_weights.json').read_text())['networks'][a.shipped]
        model = load_shipped(w, R / 'weights' / 'mf_config.json', row)
        input_scale, step, src = 0.5, None, str(w)
        out_path = R / 'eval' / f'{a.out_name or "shipped_" + a.shipped}.json'
        out_path.parent.mkdir(exist_ok=True)
        cfg = dict(input_scale=0.5, shipped=a.shipped)
    else:
        run = Path(a.run_dir)
        model = config_model(run / 'model_config.json')
        if a.which == 'final':
            from safetensors.torch import load_file
            done = json.loads((run / 'train_done.json').read_text())
            state, step, cfg = load_file(str(run / 'final_ema.safetensors')), done['step'], done['cfg']
            src = str(run / 'final_ema.safetensors')
        else:
            c = torch.load(run / 'ckpt_latest.pt', map_location='cpu', weights_only=False)
            state, step, cfg = c['ema'], c['step'], c['config']
            src = str(run / 'ckpt_latest.pt') + ' (EMA)'
            del c
        model.load_state_dict(state, strict=True)
        input_scale = cfg.get('input_scale', 1.0)
        out_path = run / f'eval_{a.which}_step{step}.json'
    check_fold(cfg, a.fold)
    model = model.cuda().eval()
    view = view_native if input_scale == 1.0 else view_1024
    assert input_scale in (1.0, 0.5)
    folds = json.loads((R / 'code' / 'fold_table_time24h_k5_seed42.json').read_text())
    files = sorted(f for f, k in folds.items() if k == a.fold)[: a.limit or None]
    groups = annotation_sets(json.loads((data / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text()))
    tags = [f'D_t{t:.2f}' for t in GRID]
    counts = {t: np.zeros((len(files), 6)) for t in tags}
    hp = np.zeros(LEVELS); hn = np.zeros(LEVELS); hs = np.zeros(LEVELS)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    k = a.fold
    if a.export:
        exp = run / 'export'
        exp.mkdir(exist_ok=True)
        val_zip = MapWriter(exp / f'MF_f{k}_val.npz')
    for i, f in enumerate(files):
        raw = cv2.imread(str(data / 'train' / 'train_images' / (f + '.jpeg')), 0)
        x = torch.from_numpy(view(raw)[None, None].copy()).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x)
        scores = D.query_scores(o.class_queries_logits[0])
        logits = o.masks_queries_logits[0].float()
        fg = (logits.sigmoid() * scores[:, None, None]).amax(0)[None, None]
        level = np.rint(torch.nn.functional.interpolate(fg, size=(2048, 2048), mode='bilinear', align_corners=False)[0, 0]
                        .cpu().numpy() * 510).astype(np.int64)
        ids = torch.nonzero(scores >= min(GRID)).flatten()
        masks = D.native_masks(logits, ids, 2048)
        entries = [e for e in (pqscore.annotation_rles(s) for s in groups[f]) if e]
        gt_union = np.zeros((2048, 2048), bool)
        for e in entries:
            gt_union |= mu.decode(mu.merge(e)).astype(bool)
        if a.export:
            val_zip.write(f, level.astype(np.uint16))
        a_pos, a_neg, a_small = pixel_hists(level, raw, gt_union)
        hp += a_pos; hn += a_neg; hs += a_small
        for t in GRID:
            inst = D.decode(scores, masks, ids, t, **D.D_RULE)
            pred = [mu.encode(np.asfortranarray(m.cpu().numpy().astype(np.uint8))) for _, _, m in inst]
            counts[f'D_t{t:.2f}'][i], _ = pqscore.image_counts(entries, pred)
        del o, logits, masks, fg, x
        if (i + 1) % 20 == 0:
            print(json.dumps(dict(time=now(), done=i + 1, total=len(files), seconds=time.monotonic() - started)), flush=True)
    res = dict(time=now(), source=src, source_sha256=sha256_file(src.split(' ')[0]), step=step, fold=a.fold,
               images=len(files), input_scale=input_scale, cfg=cfg,
               pq=dict(decoder='D', primary='D_t0.50', **{t: pqscore.metric(counts[t].sum(0)) for t in tags}),
               pixel_ap=dict(disk_ap=ap_from_hist(hp, hn), small_ap=ap_from_hist(hs, hn), positives=int(hp.sum()),
                             negatives=int(hn.sum()), small_positives=int(hs.sum()),
                             definition='solar disk (Otsu, largest component, filled); positives = GT union of all '
                                        'sets; small = GT-union components < 1024 px'),
               per_file_counts_D_t0_50={f: counts['D_t0.50'][i].tolist() for i, f in enumerate(files)},
               seconds=time.monotonic() - started, peak_gpu_alloc_bytes=torch.cuda.max_memory_allocated())
    out_path.write_text(json.dumps(res, indent=1))
    if a.export:
        val_zip.close()
        test_dir = data / 'test' / 'test_images'
        tests = sorted(p.stem for p in test_dir.glob('*.jpeg'))
        assert len(tests) == 180
        tz = MapWriter(exp / f'MF_f{k}_test.npz')
        for f in tests:
            raw = cv2.imread(str(test_dir / (f + '.jpeg')), 0)
            x = torch.from_numpy(view(raw)[None, None].copy()).cuda()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(x)
            scores = D.query_scores(o.class_queries_logits[0])
            fg = (o.masks_queries_logits[0].float().sigmoid() * scores[:, None, None]).amax(0)[None, None]
            tz.write(f, np.rint(torch.nn.functional.interpolate(fg, size=(2048, 2048), mode='bilinear', align_corners=False)
                                [0, 0].cpu().numpy() * 510).astype(np.uint16))
            del o, fg, x
        tz.close()
        files_out = {'final_ema.safetensors': run / 'final_ema.safetensors', 'model_config.json': run / 'model_config.json',
                     'train_done.json': run / 'train_done.json', 'train.log': run / 'train.log', out_path.name: out_path,
                     f'MF_f{k}_val.npz': exp / f'MF_f{k}_val.npz', f'MF_f{k}_test.npz': exp / f'MF_f{k}_test.npz'}
        man = dict(time=now(), run_dir=str(run), fold=k, step=step, cfg=cfg,
                   cfg_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                   metrics=dict(pq=res['pq']['D_t0.50'], pixel_ap=res['pixel_ap']),
                   map_format=dict(dtype='uint16', shape=[2048, 2048], denominator=510, keys='image stem',
                                   formula='max_q(sigmoid(mask_q)*score_q), bilinear to 2048 (pkg inference.mf_map)',
                                   view='native 2048' if input_scale == 1.0 else '1024 INTER_AREA'),
                   files={n: dict(path=str(pth), bytes=pth.stat().st_size, sha256=sha256_file(pth))
                          for n, pth in files_out.items()},
                   large=['final_ema.safetensors', f'MF_f{k}_val.npz', f'MF_f{k}_test.npz'])
        (exp / 'export_manifest.json').write_text(json.dumps(man, indent=1))
        print(json.dumps(dict(export=str(exp / 'export_manifest.json'), seconds=time.monotonic() - started)), flush=True)
    print(json.dumps(dict(out=str(out_path), step=step, pq=res['pq']['D_t0.50']['pq'],
                          disk_ap=res['pixel_ap']['disk_ap'], small_ap=res['pixel_ap']['small_ap'])), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
