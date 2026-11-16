"""Materialize THE fold table (ADR-0004 §1) into configs/.

Run once against the Kaggle train JSON; the committed table is the single
source of folds for every experiment (paired comparisons require it).

Usage:
    python scripts/make_folds.py \
        --annotations ../data/kaggle/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        [--k 5] [--seed 0] [--out configs/cv_folds_k5_seed0.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.coco_io import load_coco  # noqa: E402
from filseg.splits import fold_assignment, fold_summary, save_assignment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"configs/cv_folds_k{args.k}_seed{args.seed}.json"
    coco = load_coco(args.annotations)
    assignment = fold_assignment(coco, k=args.k, seed=args.seed)
    save_assignment(assignment, out)
    summary = fold_summary(assignment, coco)
    print(json.dumps(summary, indent=2))
    print(f"\nfiles={len(assignment)} folds={args.k} seed={args.seed}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
