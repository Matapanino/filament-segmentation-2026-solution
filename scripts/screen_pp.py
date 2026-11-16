"""R1 post-processing screen on the frozen M1 OOF maps (no retraining).

F-008 localized the image-score gap in false positives (5,141 unmatched
preds) and fragmentation (one-to-many ~4x the human rate) while boundary
quality is at the ceiling — so the first screen is pure decision-rule work on
the SAVED probability maps: threshold extension (F-008: monotone up to 0.6,
untested above), min-area, per-component confidence filtering, and
morphological closing (gap-closing measured safe: F-004, 1.7% endpoint
collisions within 20 px).

Paired protocol (ADR-0004): every arm is evaluated on the identical files and
GT; deltas are reported per fold vs the in-run baseline arm; adoption bar =
paired mean delta >= +0.005 with the sign consistent on >= 4/5 folds.

Structure: one rasterization + one upsample per file, all arms evaluated
inside the file loop (cheap); per_set is the primary protocol, concat is
aggregated alongside.

Runs anywhere the inputs exist; intended venue = a Kaggle CPU kernel with the
maps dataset attached (workspace directive: no heavy loops on the laptop).

Usage:
    python scripts/screen_pp.py \
        --annotations .../MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --oof .../oof_prob.npz --folds configs/cv_folds_k5_seed0.json \
        [--out outputs/screen_pp_r1.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.coco_io import annotation_sets, load_coco  # noqa: E402
from filseg.evaluate import evaluate_image  # noqa: E402
from filseg.guard import assert_clean  # noqa: E402
from filseg.splits import load_assignment  # noqa: E402
from pycocotools import mask as maskutils  # noqa: E402


@dataclass(frozen=True)
class Arm:
    name: str
    thr: float = 0.5
    min_area: int = 8
    conf_cut: float = 0.0  # drop components whose MEAN upsampled prob < cut
    close_px: int = 0      # morphological closing kernel (0 = off), pre-CC


# R2 (2026-07-26): R1 (F-009) found min_area monotone through 128 (+0.0586,
# 5/5) and closing monotone through 15 px; threshold CLOSED. R2 extends both
# to find the interior optimum and crosses the two independent mechanisms
# (FP removal x fragment merging); conf rides once on top of the area winner.
ARMS = [
    Arm("baseline"),
    Arm("area128", min_area=128),  # R1 winner = R2 reference
    # min-area upper end (GT area p25=664 — F-002; expect an interior optimum)
    Arm("area192", min_area=192),
    Arm("area256", min_area=256),
    Arm("area320", min_area=320),
    # closing upper end
    Arm("close21", close_px=21),
    Arm("close31", close_px=31),
    # area x closing grid
    Arm("a128c15", min_area=128, close_px=15),
    Arm("a192c15", min_area=192, close_px=15),
    Arm("a256c15", min_area=256, close_px=15),
    Arm("a128c21", min_area=128, close_px=21),
    Arm("a192c21", min_area=192, close_px=21),
    Arm("a256c21", min_area=256, close_px=21),
    # marginal value of confidence after area filtering
    Arm("a128conf065", min_area=128, conf_cut=0.65),
]


def instances_for_arm(up: np.ndarray, arm: Arm) -> list[np.ndarray]:
    binm = (up >= arm.thr).astype(np.uint8)
    if arm.close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (arm.close_px, arm.close_px))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < arm.min_area:
            continue
        comp = lab[y : y + h, x : x + w] == i
        if arm.conf_cut > 0.0:
            if float(up[y : y + h, x : x + w][comp].mean()) < arm.conf_cut:
                continue
        m = np.zeros(up.shape, np.uint8)
        m[y : y + h, x : x + w] = comp
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
    ap.add_argument("--out", default="outputs/screen_pp_r2.json")
    args = ap.parse_args()

    coco = load_coco(args.annotations)
    sets_by_file = annotation_sets(coco)
    assignment = load_assignment(args.folds)
    z = np.load(args.oof)
    assert_clean(z.files, "screen_pp OOF maps")

    t0 = time.time()
    # per arm: fold -> [per-file per_set scores]; plus concat + relation sums
    acc = {
        a.name: {
            "fold_scores": {k: [] for k in range(5)},
            "concat_scores": [],
            "un_pred": 0,
            "un_gt": 0,
            "o2m": 0,
            "m2o": 0,
            "n_pred": 0,
        }
        for a in ARMS
    }

    files = sorted(z.files)
    for idx, f in enumerate(files):
        prob = z[f].astype(np.float32) / 255.0
        up = cv2.resize(prob, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        gt_sets = [
            [ann_mask(a) for a in s.annotations] for s in sets_by_file[f]
        ]
        concat = [m for s in gt_sets for m in s]
        fold = assignment[f]
        for arm in ARMS:
            preds = instances_for_arm(up, arm)
            st = acc[arm.name]
            st["n_pred"] += len(preds)
            set_scores = []
            for gm in gt_sets:
                ev = evaluate_image(gm, preds)
                if ev.score is not None:
                    set_scores.append(ev.score)
                st["un_pred"] += ev.n_unmatched_pred
                st["un_gt"] += ev.n_unmatched_gt
                st["o2m"] += ev.one_to_many
                st["m2o"] += ev.many_to_one
            if set_scores:
                st["fold_scores"][fold].append(float(np.mean(set_scores)))
            evc = evaluate_image(concat, preds)
            if evc.score is not None:
                st["concat_scores"].append(evc.score)
        if (idx + 1) % 50 == 0:
            print(
                f"  ... {idx + 1}/{len(files)}  {time.time() - t0:.0f}s", flush=True
            )

    base_folds = {
        k: float(np.mean(v)) for k, v in acc["baseline"]["fold_scores"].items()
    }
    results = []
    for arm in ARMS:
        st = acc[arm.name]
        folds = {k: float(np.mean(v)) for k, v in st["fold_scores"].items()}
        deltas = [folds[k] - base_folds[k] for k in sorted(folds)]
        mean_delta = float(np.mean(deltas))
        n_pos = sum(1 for d in deltas if d > 0)
        results.append(
            {
                "arm": asdict(arm),
                "mean_image_score": float(np.mean(list(folds.values()))),
                "fold_scores": folds,
                "mean_delta": mean_delta,
                "sign_consistency": f"{n_pos}/5",
                "PASS_bar": bool(mean_delta >= 0.005 and n_pos >= 4),
                "concat_mean": float(np.mean(st["concat_scores"])),
                "n_pred_total": int(st["n_pred"]),
                "unmatched_pred": int(st["un_pred"]),
                "unmatched_gt": int(st["un_gt"]),
                "one_to_many": int(st["o2m"]),
                "many_to_one": int(st["m2o"]),
            }
        )

    results.sort(key=lambda r: -r["mean_delta"])
    out = {"n_files": len(files), "arms": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for r in results:
        a = r["arm"]
        print(
            f"{a['name']:10s} score={r['mean_image_score']:.4f} "
            f"delta={r['mean_delta']:+.4f} sign={r['sign_consistency']} "
            f"{'PASS' if r['PASS_bar'] else '----'} "
            f"npred={r['n_pred_total']} o2m={r['one_to_many']} m2o={r['many_to_one']}"
        )
    print(f"\nwritten: {args.out}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
