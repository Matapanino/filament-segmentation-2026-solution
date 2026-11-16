"""Build a submission CSV from a saved test probability archive.

Applies the screened decision rule (filseg.instancing) to the fold-mean test
maps written by the training kernel, streaming one instance at a time.

Usage:
    python scripts/make_submission.py \
        --test ../.scratch/prod_out/test_prob.npz \
        --thr 0.5 --min-area 256 --close-px 21 \
        --out ../.scratch/submission_a256c21.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.coco_io import mask_to_rle_counts  # noqa: E402
from filseg.guard import TEST_FILENAMES  # noqa: E402
from filseg.instancing import iter_instances  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, required=True)
    ap.add_argument("--close-px", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.test)
    files = sorted(z.files)
    assert set(files) == set(TEST_FILENAMES), "test archive != pinned test set"

    n_rows = n_empty = 0
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filament_id", "segmentation_rle"])
        for f in files:
            prob = z[f].astype(np.float32) / 255.0
            k = 0
            for m in iter_instances(prob, args.thr, args.min_area, args.close_px):
                k += 1
                wr.writerow([f"{f}_{k}", mask_to_rle_counts(m)])
            n_rows += k
            n_empty += k == 0
    print(
        f"{args.out}: {n_rows} instances over {len(files) - n_empty}/{len(files)} "
        f"files (thr={args.thr}, min_area={args.min_area}, close={args.close_px})"
    )


if __name__ == "__main__":
    main()
