#!/usr/bin/env python
"""Measure the inter-annotator ceiling on TRAIN observations (ADR-0004).

For every training observation with >=2 independent annotator sets, evaluate
one set against another (all unordered pairs; the ADR-0004 image score is
symmetric). The resulting distribution is the practical upper bound for any
model judged by the same protocol: a model that agrees with a human annotator
as well as humans agree with each other has exhausted the label signal.

Test observations are excluded by the pinned guard, and the kept file list is
re-asserted clean before any mask is rasterized.

Usage:
    python scripts/interannotator_ceiling.py \
        --annotations ../data/external/magfilo_2024_v1.0.json \
        [--theta 0.1] [--out outputs/interannotator.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.coco_io import ann_to_mask, annotation_sets, load_coco  # noqa: E402
from filseg.evaluate import evaluate_image  # noqa: E402
from filseg.guard import TEST_FILENAMES, assert_clean  # noqa: E402
from filseg.msiou import pair_msiou  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument(
        "--msiou",
        action="store_true",
        help="also score matched pairs with the organizers' Multiscale IoU "
        "(vendored reference implementation; adds a few minutes)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    coco = load_coco(args.annotations)
    grouped = annotation_sets(coco)
    multi_train = {
        f: sets
        for f, sets in grouped.items()
        if f not in TEST_FILENAMES and len(sets) >= 2
    }
    assert_clean(multi_train.keys(), "inter-annotator ceiling")
    print(f"train files with >=2 annotator sets: {len(multi_train)}")

    scores: list[float] = []
    dices: list[float] = []
    ious: list[float] = []
    msious: list[float] = []
    o2m = m2o = un_a = un_b = n_pairs = n_instances = 0

    for k, (file, sets) in enumerate(sorted(multi_train.items())):
        masks_per_set = [
            [ann_to_mask(a) for a in s.annotations] for s in sets
        ]
        for ia, ib in itertools.combinations(range(len(sets)), 2):
            ev = evaluate_image(masks_per_set[ia], masks_per_set[ib], args.theta)
            n_pairs += 1
            n_instances += ev.n_gt + ev.n_pred
            if ev.score is not None:
                scores.append(ev.score)
            dices += ev.matched_dice
            ious += ev.matched_iou
            if args.msiou:
                # MSIoU is GT-normalized (asymmetric); an annotator pair has no
                # privileged GT side, so record both directions per match.
                ma, mb = masks_per_set[ia], masks_per_set[ib]
                for i, j in ev.matched_pairs:
                    msious.append(pair_msiou(ma[i], mb[j]))
                    msious.append(pair_msiou(mb[j], ma[i]))
            o2m += ev.one_to_many
            m2o += ev.many_to_one
            un_a += ev.n_unmatched_gt
            un_b += ev.n_unmatched_pred
        if (k + 1) % 50 == 0:
            print(f"  ... {k + 1}/{len(multi_train)} files")

    s = np.array(scores)
    d = np.array(dices)
    n_matched = d.size
    unmatched = un_a + un_b
    result = {
        "n_files": len(multi_train),
        "n_pairs": n_pairs,
        "n_scored_pairs": int(s.size),
        "image_score_mean": float(s.mean()) if s.size else None,
        "image_score_median": float(np.median(s)) if s.size else None,
        "image_score_p10": float(np.percentile(s, 10)) if s.size else None,
        "image_score_p90": float(np.percentile(s, 90)) if s.size else None,
        "matched_dice_mean": float(d.mean()) if d.size else None,
        "matched_dice_median": float(np.median(d)) if d.size else None,
        "matched_dice_p10": float(np.percentile(d, 10)) if d.size else None,
        "matched_dice_p90": float(np.percentile(d, 90)) if d.size else None,
        "matched_iou_mean": float(np.mean(ious)) if ious else None,
        "matched_msiou_mean": float(np.mean(msious)) if msious else None,
        "matched_msiou_median": float(np.median(msious)) if msious else None,
        "matched_msiou_p10": float(np.percentile(msious, 10)) if msious else None,
        "matched_msiou_p90": float(np.percentile(msious, 90)) if msious else None,
        "n_matched_instance_pairs": int(n_matched),
        "n_instances_total": int(n_instances),
        "instance_match_rate": float(2 * n_matched / n_instances)
        if n_instances
        else None,
        "unmatched_instances": int(unmatched),
        "one_to_many_per_pair": o2m / n_pairs if n_pairs else None,
        "many_to_one_per_pair": m2o / n_pairs if n_pairs else None,
        "relation_iou_theta": args.theta,
    }

    print(json.dumps(result, indent=2))
    print(
        f"\nCEILING (mean pairwise image score, train files) = "
        f"{result['image_score_mean']:.4f}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"written: {out}")


if __name__ == "__main__":
    main()
