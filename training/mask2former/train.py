"""A1 arm T: Mask2Former (Swin-T, COCO-instance init) trained at native resolution on the four time folds
other than the held-out one (--holdout-fold k, default 2 = the gated science arm).

Copied/modified from the as-run worker (campaign 20261108_architecture_mask2former, snapshot 91f8ca62...,
`gpu_worker.py`): optimizer groups (encoder = `net.model.model.pixel_level_module.encoder`, AdamW lr 1e-5 / 1e-4,
wd 0.01, foreach=False), BF16 autocast without GradScaler, clip 1.0, EMA update rule (float tensors EMA, others
copied), one uniformly drawn annotation set per draw, HF Mask2Former native Hungarian set loss, payload contents
(model, EMA, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG). Changed per order A1: native-resolution 512 crops with
rotation/scale jitter/flips and instance-centred sampling (`geom.py`), 17,040 steps, per-step cosine with 500
warm-up steps, EMA 0.999, checkpoint every 1,000 steps (latest only, df guard), STOP-file polling, resume.

Run: python train.py --run-dir RUN --data-root DATA --fold-table FOLD_TABLE.json   (relaunch = resume)

Full-data mode (order RF1): `--fold all` (or `--holdout-fold all`) trains on all 707 images of the fold table and writes
cfg['fold'] = 'all'; such a run has no held-out fold (eval_ckpt refuses it; export_tta writes test maps only).
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULTS = dict(seed=42, fold=2, total_steps=17040, batch=8, warmup=500, lr=1e-4, encoder_lr=1e-5, weight_decay=0.01,
                clip=1.0, ema_decay=0.999, ckpt_every=1000, log_every=20, min_free_bytes=10 * 1024 ** 3,
                df_path=os.environ.get('MF_DF_PATH', '.'),  # disk-free guard path (was the training host's home)
                crop=512, p_instance=2 / 3, jitter=128.0, scale_lo=0.8, scale_hi=1.25,
                num_workers=4, debug_stop_after=0, keep_final_ckpt=0,
                # screening switches (defaults = the A1 recipe; the A1 fold runs use these defaults)
                max_steps=0, input_scale=1.0, rotate=1, flip=1, backbone='swin-tiny', accum=1, torch_threads=2,
                # label_mode 'random' = one uniformly drawn annotation set per draw (A1 recipe); 'consensus_mix' = with
                # probability consensus_p a draw on a >= 2-set image uses the majority set (geom.consensus_sets)
                label_mode='random', consensus_p=0.5, consensus_iou=0.3,
                consensus_rule='ceil_half')  # keep a cluster with >= ceil(n/2) sets; 'strict_majority' = floor(n/2)+1
LABEL_MODES = ('random', 'consensus_mix')
# Model-config overrides: None = keep the pretrained checkpoint's value (num_queries 100, class 2.0, mask 5.0, dice 5.0,
# no-object 0.1, train_num_points 12544 for swin-tiny-coco-instance).
MODEL_OVERRIDES = dict(num_queries=int, class_weight=float, mask_weight=float, dice_weight=float,
                       no_object_weight=float, train_num_points=int)
# GC1 opt-in memory/speed switches: written to cfg only when set (default off = the cfg JSON and the code path unchanged).
# grad_ckpt: '' (off) | 'all' (bare --grad-ckpt) | comma list of model.CKPT_PARTS; tf32: allow TF32 matmul/cudnn;
# channels_last: conv weights channels_last; fused_adam: AdamW(fused=True) + foreach grad clipping.
# Fine-tune switches (coordinator ~21:1x): init_from = start from a finished model's EMA weights (safetensors, strict);
# freeze_backbone = Swin in eval() under no_grad, only the pixel decoder + transformer decoder train; whole_image = every
# draw is the full native 2048 image (scale 1, no shift), one of the 8 D4 views (make_whole below); needs --crop 2048.
OPT_IN = dict(grad_ckpt='', tf32=0, channels_last=0, fused_adam=0, init_from='', freeze_backbone=0, whole_image=0)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, p)


def lr_factor(step, total, warmup):
    """Multiplier for the update that follows `step` completed updates: linear warm-up, then cosine to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, msg):
        line = f'[{now()}] {msg}'
        with self.path.open('a') as f:
            f.write(line + '\n')
        print(line, flush=True)


class Trainer:
    """Model-agnostic optimisation state: optimizer, per-step scheduler, EMA, step counter, checkpointing."""

    def __init__(self, cfg, model, encoder_params, run_dir, log):
        self.cfg, self.model, self.run_dir, self.log = cfg, model, Path(run_dir), log
        enc_ids = {id(p) for p in encoder_params}
        other = [p for p in model.parameters() if id(p) not in enc_ids]
        groups = [{'params': list(encoder_params), 'lr': cfg['encoder_lr']}, {'params': other, 'lr': cfg['lr']}]
        if cfg.get('fused_adam'):
            self.opt = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'], fused=True)
        else:
            self.opt = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'], foreach=False)
        total, warmup = cfg['total_steps'], cfg['warmup']
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_factor(s, total, warmup))
        self.ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.step = 0
        self.ckpt = self.run_dir / 'ckpt_latest.pt'

    @torch.no_grad()
    def ema_update(self):
        d = self.cfg['ema_decay']
        for k, v in self.model.state_dict().items():
            if v.is_floating_point():
                self.ema[k].mul_(d).add_(v, alpha=1 - d)
            else:
                self.ema[k].copy_(v)

    def train_step(self, loss_fn, batch):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        accum = int(self.cfg.get('accum', 1))
        total = 0.0
        for mb in (batch if accum > 1 else [batch]):
            loss = loss_fn(self.model, mb)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            (loss / accum).backward() if accum > 1 else loss.backward()
            total += float(loss.detach()) / accum
        loss = torch.tensor(total)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['clip'],
                                              foreach=True if self.cfg.get('fused_adam') else False)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        self.opt.step()
        self.sched.step()
        self.ema_update()
        self.step += 1
        return float(loss.detach())

    def payload(self):
        return dict(model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    ema={k: v.detach().cpu() for k, v in self.ema.items()}, optimizer=self.opt.state_dict(),
                    scheduler=self.sched.state_dict(), step=self.step, config=self.cfg,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    def free_ok(self, what):
        free = shutil.disk_usage(self.cfg['df_path']).free
        if free < self.cfg['min_free_bytes']:
            self.log(f'DF_GUARD skip {what}: free {free} B < {self.cfg["min_free_bytes"]} B at step {self.step}')
            with (self.run_dir / 'DF_GUARD_SKIPS.log').open('a') as f:
                f.write(json.dumps(dict(time=now(), step=self.step, what=what, free_bytes=free)) + '\n')
            return False
        return True

    def save(self, reason):
        if not self.free_ok('checkpoint'):
            return False
        tmp = self.ckpt.with_name(self.ckpt.name + '.tmp')
        torch.save(self.payload(), tmp)
        os.replace(tmp, self.ckpt)
        self.log(f'checkpoint step {self.step} ({reason}) -> {self.ckpt.name} {self.ckpt.stat().st_size} B')
        return True

    def load(self):
        c = torch.load(self.ckpt, map_location='cpu', weights_only=False)
        check_resume_config(c.get('config') or {}, self.cfg)
        self.model.load_state_dict(c['model'])
        dev = next(self.model.parameters()).device
        self.ema = {k: v.to(dev) for k, v in c['ema'].items()}
        self.opt.load_state_dict(c['optimizer'])
        self.sched.load_state_dict(c['scheduler'])
        self.step = int(c['step'])
        random.setstate(c['python_rng']); np.random.set_state(c['numpy_rng']); torch.set_rng_state(c['torch_rng'])
        if c['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(c['cuda_rng'])
        self.log(f'resumed from {self.ckpt.name} at step {self.step}')


def run(trainer, batches, loss_fn, total, stop_file, log, progress_path=None, extra=None):
    """Optimise until `total` steps; poll the stop file(s) before every step. Returns 'done' or 'stopped'.

    `stop_file`: one path or a list (per-fold runs/f{k}/STOP first, then the global filament/STOP); the debug hook
    writes the first one.

    `batches(start)` yields the batches for steps start, start+1, ... (deterministic in the step index).
    """
    cfg = trainer.cfg
    stops = [Path(p) for p in (stop_file if isinstance(stop_file, (list, tuple)) else [stop_file])]
    it = iter(batches(trainer.step))
    t0, s0, window = time.monotonic(), trainer.step, []
    while trainer.step < total:
        seen = [str(p) for p in stops if p.exists()]
        if seen:
            log(f'STOP file {seen} seen before step {trainer.step + 1}; checkpointing and exiting 0')
            trainer.save('stop')
            return 'stopped'
        tick = time.monotonic()
        batch = next(it)
        loss = trainer.train_step(loss_fn, batch)
        window.append((time.monotonic() - tick, loss))
        if trainer.step % cfg['ckpt_every'] == 0 and trainer.step < total:
            trainer.save('periodic')
        if cfg.get('debug_stop_after') and trainer.step == cfg['debug_stop_after']:
            stops[0].write_text('debug_stop_after\n')
        if trainer.step % cfg['log_every'] == 0 or trainer.step == total:
            sps = (trainer.step - s0) / max(1e-9, time.monotonic() - t0)
            eta_s = (total - trainer.step) / sps if sps > 0 else None
            mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            row = dict(time=now(), step=trainer.step, total=total, loss_mean_window=float(np.mean([w[1] for w in window])),
                       sec_per_step_window=float(np.mean([w[0] for w in window])), steps_per_sec_since_start=sps,
                       eta_seconds=eta_s, eta_local=(time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time() + eta_s))
                                                     if eta_s is not None else None),
                       lr=[g['lr'] for g in trainer.opt.param_groups], peak_gpu_alloc_bytes=mem, **(extra or {}))
            log(json.dumps(row))
            if progress_path:
                atomic_json(progress_path, row)
            window = []
    return 'done'


# ------------------------------------------------------------------ data
def file_base(name):
    return name.rsplit('/', 1)[-1].split('-')[-1].rsplit('.', 1)[0]


def annotation_sets(coco):
    """{file: [annotation list per annotation set]} (pkg coco_io.annotation_sets, sets in image order)."""
    by_image = {im['id']: (file_base(im['file_name']), []) for im in coco['images']}
    for a in coco['annotations']:
        if a['image_id'] in by_image:
            by_image[a['image_id']][1].append(a)
    out = {}
    for f, anns in by_image.values():
        out.setdefault(f, []).append(anns)
    return out


class CropBatches(torch.utils.data.Dataset):
    """Item `step` = the batch of that optimizer step: uint8 [B, crop, crop] images + list of uint8 [n, crop, crop]."""

    def __init__(self, images, groups, files, cfg):
        self.images, self.groups, self.files, self.cfg = images, groups, list(files), cfg
        self.label_mode = cfg.get('label_mode', 'random')
        if self.label_mode not in LABEL_MODES:
            raise ValueError(f'label_mode {self.label_mode!r} not in {LABEL_MODES}')
        self.consensus = {}  # file -> [consensus annotation list per drawn set index], precomputed once
        if self.label_mode == 'consensus_mix':
            from geom import consensus_sets
            t0 = time.monotonic()
            for f in self.files:
                if len(groups[f]) >= 2:
                    self.consensus[f] = consensus_sets(groups[f], iou_thr=cfg['consensus_iou'],
                                                       rule=cfg.get('consensus_rule', 'ceil_half'))
            self.consensus_init_seconds = time.monotonic() - t0

    def __len__(self):
        return self.cfg['total_steps']

    @lru_cache(maxsize=4)
    def _perm(self, cycle):
        return np.random.default_rng([self.cfg['seed'], 7, cycle]).permutation(len(self.files))

    def file_for_draw(self, j):
        n = len(self.files)
        return self.files[int(self._perm(j // n)[j % n])]

    def draw(self, j):
        from geom import make_crop
        c = self.cfg
        f = self.file_for_draw(j)
        rng = np.random.default_rng([c['seed'], 11, j])
        sets = self.groups[f]
        s = int(rng.integers(len(sets)))
        anns = sets[s]
        if c.get('whole_image'):
            img, masks, p = make_whole(self.images[f], anns, rng, flip=c.get('flip', 1))
            return img, masks, dict(p, file=f)
        label = None
        if self.label_mode == 'consensus_mix':
            # separate per-draw stream: the geometry stream `rng` is consumed exactly as in 'random' mode
            label = 'set'
            if len(sets) >= 2 and np.random.default_rng([c['seed'], 13, j]).random() < c['consensus_p']:
                anns, label = self.consensus[f][s], 'consensus'
        img, masks, p = make_crop(self.images[f], anns, rng, crop=c['crop'], p_instance=c['p_instance'],
                                  jitter=c['jitter'], scale_range=(c['scale_lo'], c['scale_hi']),
                                  rotate=c.get('rotate', 1), flip=c.get('flip', 1),
                                  input_scale=c.get('input_scale', 1.0))
        return img, masks, (dict(p, file=f) if label is None else dict(p, file=f, set=s, label=label))

    def __getitem__(self, step):
        b = self.cfg['batch']
        rows = [self.draw(step * b + k) for k in range(b)]
        return np.stack([r[0] for r in rows]), [r[1] for r in rows], [r[2] for r in rows]


def whole_params(rng, flip=1, size=2048):
    """GC1 --whole-image: the full native image, scale 1, no shift; one of the 8 D4 views (rot90^k, then flips)."""
    k = int(rng.integers(4))
    hflip = bool(rng.random() < 0.5)
    vflip = bool(rng.random() < 0.5)
    if not flip:
        hflip = vflip = False
    return dict(theta=90.0 * k, scale=1.0, hflip=hflip, vflip=vflip, cx=size / 2, cy=size / 2, mode='whole')


def make_whole(img, annotations, rng, flip=1):
    """One whole-image draw: (uint8 [H, H] image, uint8 [n, H, H] masks, params); right-angle maps are exact permutations."""
    size = img.shape[0]
    assert img.shape == (size, size)
    from geom import continuous_matrix, warp_image, warp_instances
    p = whole_params(rng, flip=flip, size=size)
    m = continuous_matrix(p, size)
    return warp_image(img, m, size), warp_instances(annotations, m, size), p


def _worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


FULL_DATA = 'all'


def fold_arg(s):
    """--fold / --holdout-fold value: 'all' (full-data mode, no fold held out) or a time fold 0-4 (int)."""
    if s == FULL_DATA:
        return FULL_DATA
    k = int(s)
    if k not in range(5):
        raise argparse.ArgumentTypeError(f'fold must be 0-4 or {FULL_DATA!r}, got {s!r}')
    return k


def train_files_for(folds, fold):
    """Training files: every file of the fold table in full-data mode, else the files outside the held-out fold."""
    if fold == FULL_DATA:
        return sorted(folds)
    return sorted(f for f, k in folds.items() if k != fold)


def init_weights(model, path):
    """--init-from: strict load of a finished run's final_ema.safetensors; returns the file's sha256."""
    from safetensors.torch import load_file
    import hashlib
    model.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_resume_config(saved, cfg):
    """Resuming must not change the held-out fold (the weights would already have seen the new fold's labels), nor
    switch between full-data mode ('all') and a fold."""
    if 'fold' in saved and str(saved['fold']) != str(cfg['fold']):
        raise SystemExit(f'resume fold mismatch: checkpoint fold {saved["fold"]}, requested fold {cfg["fold"]}')


def grad_ckpt_arg(s):
    if s == '':
        return s
    from model import ckpt_parts
    try:
        ckpt_parts(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return s


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--fold-table', required=True)
    ap.add_argument('--holdout-fold', type=fold_arg, default=None,
                    help="time fold held out (0-4) or 'all' (full data); alias of --fold")
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=fold_arg if k == 'fold' else type(v), default=v)
    for k, t in MODEL_OVERRIDES.items():
        ap.add_argument('--' + k.replace('_', '-'), type=t, default=None)
    ap.add_argument('--grad-ckpt', type=grad_ckpt_arg, nargs='?', const='all', default='')
    for k in ('tf32', 'channels_last', 'fused_adam', 'freeze_backbone', 'whole_image'):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=0, choices=(0, 1))
    ap.add_argument('--init-from', default='')
    return ap


def config_from_args(a):
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    overrides = {k: getattr(a, k) for k in MODEL_OVERRIDES if getattr(a, k) is not None}
    if overrides:
        cfg['model_overrides'] = overrides
    if a.holdout_fold is not None:
        cfg['fold'] = a.holdout_fold
    for k, off in OPT_IN.items():
        if getattr(a, k) != off:
            cfg[k] = getattr(a, k)
    if cfg.get('whole_image') and (cfg['crop'] != 2048 or cfg['input_scale'] != 1.0):
        raise SystemExit('--whole-image 1 needs --crop 2048 and --input-scale 1.0')
    assert cfg['fold'] == FULL_DATA or cfg['fold'] in range(5)
    return cfg


def main():
    a = build_parser().parse_args()
    cfg = config_from_args(a)
    overrides = cfg.get('model_overrides', {})
    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / 'train.log')
    stop_file = [run_dir / 'STOP', Path(os.environ.get('A1_STOP_FILE', str(HERE.parent / 'STOP')))]
    if any(p.exists() for p in stop_file):
        log(f'STOP file present at launch ({[str(p) for p in stop_file if p.exists()]}): not training (remove it to resume)')
        return 0
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(cfg['torch_threads'])
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get('tf32'))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get('tf32'))
    from model import build_pretrained
    import transformers
    log(f'start pid {os.getpid()} torch {torch.__version__} transformers {transformers.__version__} '
        f'GPU {torch.cuda.get_device_name(0)} CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} cfg {json.dumps(cfg)}')
    assert transformers.__version__ == '4.57.1'
    folds = json.loads(Path(a.fold_table).read_text())
    train_files = train_files_for(folds, cfg['fold'])
    data = Path(a.data_root)
    ann_path = data / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    from model import sha256_file
    assert sha256_file(ann_path) == '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
    groups = annotation_sets(json.loads(ann_path.read_text()))
    assert set(train_files) <= set(groups)
    images = {}
    for f in train_files:
        im = cv2.imread(str(data / 'train' / 'train_images' / (f + '.jpeg')), 0)
        assert im.shape == (2048, 2048)
        images[f] = im
    log(f'loaded {len(images)} training images (full data: no fold held out)' if cfg['fold'] == FULL_DATA else
        f'loaded {len(images)} training images (fold {cfg["fold"]} held out)')
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    model = build_pretrained(cfg['backbone'], overrides)
    if cfg.get('init_from'):
        log(f'GC1 init-from {cfg["init_from"]} sha256 {init_weights(model, cfg["init_from"])}')
    model = model.cuda()
    if any(k in cfg for k in OPT_IN):
        from model import apply_opt_in
        log(f'GC1 opt-in {apply_opt_in(model, cfg)}')
    model.net.model.config.to_json_file(str(run_dir / 'model_config.json'))
    log(f'model {cfg["backbone"]} params {sum(p.numel() for p in model.parameters())} overrides {overrides} '
        f'pretrained {json.dumps(model.pretrained_receipt)}')
    trainer = Trainer(cfg, model, list(model.net.encoder.parameters()), run_dir, log)
    if trainer.ckpt.exists():
        trainer.load()
    ds = CropBatches(images, {f: groups[f] for f in train_files}, train_files, cfg)
    if ds.consensus:
        log(f'label_mode {cfg["label_mode"]}: consensus sets for {len(ds.consensus)} multi-set images in '
            f'{ds.consensus_init_seconds:.1f} s (p {cfg["consensus_p"]}, iou > {cfg["consensus_iou"]}, '
            f'rule {cfg["consensus_rule"]})')

    def batches(start):
        dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=range(start, cfg['total_steps']),
                                         num_workers=cfg['num_workers'], worker_init_fn=_worker_init,
                                         prefetch_factor=4 if cfg['num_workers'] else None,
                                         persistent_workers=False)
        acc = cfg['accum']
        for imgs, masks, _ in dl:
            x = torch.as_tensor(imgs).cuda(non_blocking=True).float().div_(255)[:, None]
            ys = [torch.as_tensor(m).cuda(non_blocking=True).float() for m in masks]
            if acc == 1:
                yield x, ys
            else:  # gradient accumulation: the step's `batch` crops in `accum` micro-batches (same crops, same order)
                mb = len(ys) // acc
                assert mb * acc == len(ys)
                yield [(x[i * mb:(i + 1) * mb], ys[i * mb:(i + 1) * mb]) for i in range(acc)]

    def loss_fn(m, batch):
        x, ys = batch
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return m(x, ys).loss

    end = min(cfg['total_steps'], cfg['max_steps']) if cfg['max_steps'] else cfg['total_steps']
    status = run(trainer, batches, loss_fn, end, stop_file, log, run_dir / 'progress.json',
                 extra=dict(pid=os.getpid()))
    if status == 'stopped':
        return 0
    # Final: the EMA weights at the last step are the model.
    while not trainer.free_ok('final export'):
        if any(p.exists() for p in stop_file):
            log('STOP while waiting for disk space at the final export; exiting 0 without the export')
            return 0
        time.sleep(60)
    if cfg['keep_final_ckpt']:
        trainer.save('final')
    from safetensors.torch import save_file
    final = run_dir / 'final_ema.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in trainer.ema.items()}, str(final))
    atomic_json(run_dir / 'train_done.json', dict(time=now(), step=trainer.step, final_ema=final.name,
                                                  sha256=sha256_file(final), bytes=final.stat().st_size, cfg=cfg))
    log(f'done: {final.name} written')
    if not cfg['keep_final_ckpt']:
        # the final EMA (hash recorded above) is the model; drop the 0.76 GB resume checkpoint to save host disk
        trainer.ckpt.unlink(missing_ok=True)
        log('removed ckpt_latest.pt after the verified final export (keep_final_ckpt=0)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
