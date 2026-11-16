"""Build a Kaggle kernel (one-cell notebook + metadata) into a staging dir.

Kaggle's API accepts our pushes reliably as kernel_type=notebook with
kernelspec metadata (script-type pushes error with "Notebook not found";
papermill rejects notebooks without a kernel name), so we wrap a python
script verbatim into a single code cell.

Usage:
    # M1 trainer (default script), production / smoke:
    python build_kernel.py --stage <dir>
    python build_kernel.py --stage <dir> --smoke

    # any other script, e.g. the R1 post-processing screen on CPU:
    python build_kernel.py --stage <dir> --script screen_pp_kernel.py \
        --slug filament-r1-screen-pp --cpu --dataset masayakawamata/filament-m1-maps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
USER = "masayakawamata"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--script", default="train_m1.py", help="file in this dir")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--smoke", action="store_true", help="flip SMOKE=False -> True")
    ap.add_argument("--cpu", action="store_true", help="no GPU")
    ap.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="extra dataset source(s); filseg-lib is always attached",
    )
    args = ap.parse_args()

    src = (HERE / args.script).read_text()
    if args.smoke:
        src = src.replace("SMOKE = False", "SMOKE = True", 1)
    slug = args.slug or (
        "filament-m1-smoke" if args.smoke else "filament-m1-unet-5fold"
    )
    datasets = [f"{USER}/filseg-lib"] + (args.dataset or [])

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nb.cells = [nbformat.v4.new_code_cell(src)]
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, stage / f"{slug}.ipynb")
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not args.cpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"staged {slug} -> {stage} (gpu={not args.cpu}, datasets={datasets})")


if __name__ == "__main__":
    main()
