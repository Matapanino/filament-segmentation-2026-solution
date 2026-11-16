#!/usr/bin/env python
"""M1 baseline — semantic U-Net @1024 + connected-component instancing, E2E.

Kaggle GPU batch kernel (ADR-0005 M1). Sources it expects attached:
  - competition data: filament-segmentation-2026 (path glob-discovered)
  - dataset masayakawamata/filseg-lib: the filseg package + THE pinned fold
    table (configs/cv_folds_k5_seed0.json) — all experiments share those folds.

Recipe (deliberately plain; every knob is a later, measured lever):
  input 2048 JPEG -> INTER_AREA downscale to 1024 (F-002: median filament
  width 11 px, ~5.5 px at 1024 — the ceiling of downscale before barbs die);
  one training sample per annotation-image ENTRY (ADR-0004 §2, per-annotator);
  U-Net EfficientNet-B0 (ImageNet), in_channels=1; BCE + soft Dice; AMP;
  hflip/vflip aug (rotations deferred: physical sanity unchecked, ADR-0005);
  per-fold best-val-loss checkpoint; OOF + test sigmoid maps saved as uint8
  PNGs @1024 for local analysis; instancing = threshold -> bilinear upsample
  to 2048 -> 8-connected components, min area 8 px (F-002: smallest GT
  annotation is 9 px, so this removes nothing a GT instance could match);
  OOF threshold sweep scored with the ADR-0004 evaluator (both GT protocols);
  submission at THR_SUBMIT from the 5-fold mean test map.

Outputs: submission.csv, oof_eval.json, unet_m1_fold{K}.pt,
         oof_prob/<file>.png, test_prob/<file>.png, folds_used.json
"""
SMOKE = False  # smoke copy is pushed under a separate slug with True

import glob
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

t0 = time.time()

# --- filseg (shipped as a Kaggle dataset; mount root has moved before, so
# glob-discover the package rather than hardcoding /kaggle/input/<slug>) -----
LIB = glob.glob("/kaggle/input/**/filseg/__init__.py", recursive=True)
assert LIB, "filseg-lib dataset not attached / not found under /kaggle/input"
sys.path.insert(0, str(Path(LIB[0]).parents[1]))
from filseg.coco_io import annotation_sets, load_coco, mask_to_rle_counts  # noqa: E402
from filseg.evaluate import evaluate_image  # noqa: E402
from filseg.guard import TEST_FILENAMES, assert_clean, file_base  # noqa: E402
from filseg.splits import entries_for_files, fold_files, load_assignment  # noqa: E402

try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "segmentation_models_pytorch"],
        check=True,
    )
    import segmentation_models_pytorch as smp

# --- config -----------------------------------------------------------------
SEED = 42
SIZE = 1024
FOLDS = [0] if SMOKE else [0, 1, 2, 3, 4]
EPOCHS = 2 if SMOKE else 30
PATIENCE = 8
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
THR_SWEEP = [0.5] if SMOKE else [0.30, 0.40, 0.45, 0.50, 0.55, 0.60]
THR_SUBMIT = 0.5
MIN_AREA = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# --- locate competition data (path layout has moved before; never hardcode) -
hits = glob.glob(
    "/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True
)
assert hits, "competition train JSON not found under /kaggle/input"
TRAIN_JSON = Path(hits[0])
ROOT = TRAIN_JSON.parents[1]  # .../MAGFiLO_1.0_Kaggle_2026
TRAIN_IMG = ROOT / "train" / "train_images"
TEST_IMG = ROOT / "test" / "test_images"
tables = glob.glob("/kaggle/input/**/cv_folds_k5_seed0.json", recursive=True)
assert tables, "pinned fold table not found under /kaggle/input"
FOLD_TABLE = tables[0]
print(f"data root: {ROOT}")

coco = load_coco(TRAIN_JSON)
sets_by_file = annotation_sets(coco)
assignment = load_assignment(FOLD_TABLE)
assert set(assignment) == set(sets_by_file), "fold table / JSON file mismatch"

test_files = sorted(file_base(p.name) for p in TEST_IMG.glob("*.jpeg"))
assert set(test_files) == set(TEST_FILENAMES), "test dir vs pinned guard mismatch"

# --- RAM caches: images (uint8@1024) and per-entry union masks (uint8@1024) -
def load_gray_1024(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (2048, 2048), path
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


from pycocotools import mask as maskutils  # noqa: E402


def entry_mask_1024(anns: list[dict]) -> np.ndarray:
    """Union of the entry's filament polygons, rasterized directly at 1024
    (coords scaled by 0.5 — crisper than downsampling a 2048 raster)."""
    m = np.zeros((SIZE, SIZE), np.uint8)
    for a in anns:
        polys = [np.asarray(p, np.float64) * (SIZE / 2048.0) for p in a["segmentation"]]
        rles = maskutils.frPyObjects([p.tolist() for p in polys], SIZE, SIZE)
        m |= maskutils.decode(maskutils.merge(rles))
    return m


print("caching train images...", flush=True)
img_cache: dict[str, np.ndarray] = {
    f: load_gray_1024(TRAIN_IMG / f"{f}.jpeg") for f in sorted(sets_by_file)
}
print(f"  {len(img_cache)} images  {time.time() - t0:.0f}s", flush=True)

mask_cache: dict[str, np.ndarray] = {}
entry_file: dict[str, str] = {}
for f, sets in sorted(sets_by_file.items()):
    for s in sets:
        mask_cache[s.image_id] = entry_mask_1024(s.annotations)
        entry_file[s.image_id] = f
print(f"  {len(mask_cache)} entry masks  {time.time() - t0:.0f}s", flush=True)


class EntryDataset(Dataset):
    def __init__(self, entry_ids: list[str], train: bool):
        self.ids = entry_ids
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        eid = self.ids[i]
        img = img_cache[entry_file[eid]]
        msk = mask_cache[eid]
        if self.train:
            if random.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5:
                img, msk = img[::-1, :], msk[::-1, :]
        x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(np.ascontiguousarray(msk)).float().unsqueeze(0)
        return x, y


def make_model():
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )


bce = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode="binary", from_logits=True)


def criterion(logits, target):
    return bce(logits, target) + dice(logits, target)


@torch.no_grad()
def predict_files(model, files: list[str], cache: dict[str, np.ndarray]) -> dict:
    """file -> sigmoid map (float32 @1024), batched."""
    model.eval()
    out = {}
    for i in range(0, len(files), BATCH):
        chunk = files[i : i + BATCH]
        x = torch.stack(
            [torch.from_numpy(cache[f]).float().unsqueeze(0) / 255.0 for f in chunk]
        ).to(DEVICE)
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            p = torch.sigmoid(model(x)).float().cpu().numpy()[:, 0]
        for f, m in zip(chunk, p):
            out[f] = m
    return out


# --- train folds ------------------------------------------------------------
oof_prob: dict[str, np.ndarray] = {}
test_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((SIZE, SIZE), np.float32))
fold_log = {}

if SMOKE:  # keep the smoke fast but still E2E
    test_files = test_files[:8]
test_cache = {f: load_gray_1024(TEST_IMG / f"{f}.jpeg") for f in test_files}

for k in FOLDS:
    tr_files = fold_files(assignment, k, "train")
    va_files = fold_files(assignment, k, "val")
    assert_clean(tr_files + va_files, f"fold {k} assembly")
    tr_ids = entries_for_files(coco, tr_files)
    va_ids = entries_for_files(coco, va_files)
    if SMOKE:
        tr_ids = tr_ids[:64]
    dl_tr = DataLoader(
        EntryDataset(tr_ids, train=True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    dl_va = DataLoader(
        EntryDataset(va_ids, train=False),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = make_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        tr_loss, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(x)
            n += len(x)
        sched.step()

        model.eval()
        va_loss, m = 0.0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                    loss = criterion(model(x), y)
                va_loss += loss.item() * len(x)
                m += len(x)
        tr_loss, va_loss = tr_loss / max(n, 1), va_loss / max(m, 1)
        star = va_loss < best_val
        print(
            f"fold {k} ep {ep:02d} train {tr_loss:.4f} val {va_loss:.4f}"
            f"{' *' if star else ''}  {time.time() - t0:.0f}s",
            flush=True,
        )
        if star:
            best_val, bad = va_loss, 0
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"fold {k} early stop at ep {ep}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, f"unet_m1_fold{k}.pt")
    fold_log[k] = {"best_val_loss": best_val, "n_train": len(tr_ids), "n_val": len(va_ids)}

    oof_prob.update(predict_files(model, va_files, img_cache))
    for f, p in predict_files(model, test_files, test_cache).items():
        test_sum[f] += p
    del model
    torch.cuda.empty_cache()

test_mean = {f: test_sum[f] / len(FOLDS) for f in test_files}

# --- persist probability maps (two npz archives: Kaggle caps the output file
# count well below 707+180 loose files) --------------------------------------
np.savez_compressed(
    "oof_prob.npz", **{f: (p * 255).astype(np.uint8) for f, p in oof_prob.items()}
)
np.savez_compressed(
    "test_prob.npz", **{f: (p * 255).astype(np.uint8) for f, p in test_mean.items()}
)
print(f"saved {len(oof_prob)} OOF + {len(test_mean)} test maps  {time.time() - t0:.0f}s")

# --- instancing -------------------------------------------------------------
def upsample_prob(prob1024: np.ndarray) -> np.ndarray:
    return cv2.resize(prob1024, (2048, 2048), interpolation=cv2.INTER_LINEAR)


def instances_from_up(up: np.ndarray, thr: float) -> list[np.ndarray]:
    binm = (up >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < MIN_AREA:
            continue
        m = np.zeros((2048, 2048), np.uint8)
        m[y : y + h, x : x + w] = lab[y : y + h, x : x + w] == i
        out.append(m)
    return out


def ann_to_mask_2048(a: dict) -> np.ndarray:
    rles = maskutils.frPyObjects(a["segmentation"], 2048, 2048)
    return maskutils.decode(maskutils.merge(rles))


# --- OOF evaluation: ADR-0004 evaluator, both protocols, threshold sweep ----
print("evaluating OOF...", flush=True)
agg = {
    p: {t: defaultdict(float) for t in THR_SWEEP} for p in ("per_set", "concat")
}
for idx, f in enumerate(sorted(oof_prob)):
    gt_sets = [
        [ann_to_mask_2048(a) for a in s.annotations] for s in sets_by_file[f]
    ]
    concat = [m for s in gt_sets for m in s]
    up = upsample_prob(oof_prob[f])
    for thr in THR_SWEEP:
        preds = instances_from_up(up, thr)
        for proto, gt_lists in (("per_set", gt_sets), ("concat", [concat])):
            scores, evs = [], []
            for gm in gt_lists:
                ev = evaluate_image(gm, preds)
                evs.append(ev)
                if ev.score is not None:
                    scores.append(ev.score)
            a = agg[proto][thr]
            if scores:
                a["score_sum"] += float(np.mean(scores))
                a["n_files"] += 1
            a["dice_sum"] += sum(sum(e.matched_dice) for e in evs)
            a["n_matched"] += sum(len(e.matched_dice) for e in evs)
            a["un_gt"] += sum(e.n_unmatched_gt for e in evs)
            a["un_pred"] += sum(e.n_unmatched_pred for e in evs)
            a["o2m"] += sum(e.one_to_many for e in evs)
            a["m2o"] += sum(e.many_to_one for e in evs)
    if (idx + 1) % 100 == 0:
        print(f"  ... {idx + 1}/{len(oof_prob)}  {time.time() - t0:.0f}s", flush=True)

oof_eval = {"fold_log": fold_log, "thresholds": {}}
for proto, by_thr in agg.items():
    for thr, a in by_thr.items():
        key = f"{proto}@{thr:.2f}"
        oof_eval["thresholds"][key] = {
            "mean_image_score": a["score_sum"] / a["n_files"] if a["n_files"] else None,
            "matched_dice_mean": a["dice_sum"] / a["n_matched"] if a["n_matched"] else None,
            "n_matched": int(a["n_matched"]),
            "unmatched_gt": int(a["un_gt"]),
            "unmatched_pred": int(a["un_pred"]),
            "one_to_many": int(a["o2m"]),
            "many_to_one": int(a["m2o"]),
        }
        print(key, json.dumps(oof_eval["thresholds"][key]))
Path("oof_eval.json").write_text(json.dumps(oof_eval, indent=2))
Path("folds_used.json").write_text(json.dumps({"folds": FOLDS, "table": FOLD_TABLE}))

# --- submission (streamed: one instance mask alive at a time — an unconverged
# model can emit thousands of components per file, and holding every full-size
# mask for all files at once is an OOM, measured in the smoke run) -----------
import csv  # noqa: E402

n_rows = 0
with open("submission.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["filament_id", "segmentation_rle"])
    for f in test_files:
        up = upsample_prob(test_mean[f])
        binm = (up >= THR_SUBMIT).astype(np.uint8)
        ncc, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        k = 0
        for i in range(1, ncc):
            x, y, w, h, a = stats[i]
            if a < MIN_AREA:
                continue
            m = np.zeros((2048, 2048), np.uint8)
            m[y : y + h, x : x + w] = lab[y : y + h, x : x + w] == i
            k += 1
            wr.writerow([f"{f}_{k}", mask_to_rle_counts(m)])
            n_rows += 1
print(
    f"submission.csv: {n_rows} instances over {len(test_files)} files "
    f"(thr={THR_SUBMIT}, min_area={MIN_AREA})  total {time.time() - t0:.0f}s"
)
print("DONE" + (" (SMOKE)" if SMOKE else ""))
