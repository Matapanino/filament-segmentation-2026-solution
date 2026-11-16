"""Per-fold OOF report + MSIoU diagnostic from a kernel's saved probability maps.

Consumes the training kernel's oof_prob.npz (uint8 sigmoid maps @1024), the
pinned fold table and the Kaggle train JSON; reproduces the kernel's instancing
(threshold -> bilinear upsample to 2048 -> 8-connected components -> min-area)
and scores with the ADR-0004 evaluator, reported PER FOLD (the fold-level std
calibrates the adoption bar, ADR-0004 §4) plus the organizers' MSIoU on
matched pairs (the prize-metric view the kernel skips).

Usage:
    python scripts/oof_report.py \
        --annotations ../data/kaggle/.../MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --oof ../.scratch/prod_out/oof_prob.npz \
        --folds configs/cv_folds_k5_seed0.json \
        [--thr 0.5] [--min-area 8] [--out outputs/oof_report_m1.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.coco_io import annotation_sets, load_coco  # noqa: E402
from filseg.evaluate import evaluate_image  # noqa: E402
from filseg.guard import assert_clean  # noqa: E402
from filseg.msiou import pair_msiou  # noqa: E402
from filseg.splits import load_assignment  # noqa: E402
from pycocotools import mask as maskutils  # noqa: E402


def instances(
    prob1024: np.ndarray, thr: float, min_area: int, close_px: int = 0
) -> list[np.ndarray]:
    up = cv2.resize(prob1024, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    binm = (up >= thr).astype(np.uint8)
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        m = np.zeros((2048, 2048), np.uint8)
        m[y : y + h, x : x + w] = lab[y : y + h, x : x + w] == i
        out.append(m)
    return out


def ann_mask(a: dict) -> np.ndarray:
    rles = maskutils.frPyObjects(a["segmentation"], 2048, 2048)
    return maskutils.decode(maskutils.merge(rles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=8)
    ap.add_argument("--close-px", type=int, default=0)
    ap.add_argument("--out", default="outputs/oof_report_m1.json")
    args = ap.parse_args()

    coco = load_coco(args.annotations)
    sets_by_file = annotation_sets(coco)
    assignment = load_assignment(args.folds)
    z = np.load(args.oof)
    assert_clean(z.files, "OOF map set")

    per_fold_scores: dict[int, list[float]] = {k: [] for k in range(5)}
    msious: list[float] = []
    dices: list[float] = []
    for idx, f in enumerate(sorted(z.files)):
        prob = z[f].astype(np.float32) / 255.0
        preds = instances(prob, args.thr, args.min_area, args.close_px)
        set_scores = []
        for s in sets_by_file[f]:
            gt = [ann_mask(a) for a in s.annotations]
            ev = evaluate_image(gt, preds)
            if ev.score is not None:
                set_scores.append(ev.score)
            dices += ev.matched_dice
            for i, j in ev.matched_pairs:
                msious.append(pair_msiou(gt[i], preds[j]))
        if set_scores:
            per_fold_scores[assignment[f]].append(float(np.mean(set_scores)))
        if (idx + 1) % 100 == 0:
            print(f"  ... {idx + 1}/{len(z.files)}", flush=True)

    folds = {
        k: {
            "n_files": len(v),
            "mean_image_score": float(np.mean(v)),
            "median": float(np.median(v)),
        }
        for k, v in per_fold_scores.items()
    }
    fold_means = [folds[k]["mean_image_score"] for k in sorted(folds)]
    d, m = np.array(dices), np.array(msious)
    out = {
        "thr": args.thr,
        "min_area": args.min_area,
        "per_fold": folds,
        "fold_mean_of_means": float(np.mean(fold_means)),
        "fold_std_of_means": float(np.std(fold_means, ddof=1)),
        "matched_dice_mean": float(d.mean()),
        "matched_msiou": {
            "n": int(m.size),
            "mean": float(m.mean()),
            "median": float(np.median(m)),
            "p10": float(np.percentile(m, 10)),
            "p90": float(np.percentile(m, 90)),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
