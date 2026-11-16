#!/usr/bin/env python
"""R1 post-processing screen — thin Kaggle CPU-kernel wrapper.

The screen logic lives in ONE place (filseg-lib dataset: scripts/screen_pp.py);
this wrapper only glob-discovers the inputs and runs it. Attach:
  - competition filament-segmentation-2026
  - dataset masayakawamata/filseg-lib   (package + scripts + fold table)
  - dataset masayakawamata/filament-m1-maps (frozen M1 OOF/test maps)
"""
import glob
import subprocess
import sys

def one(pattern: str) -> str:
    hits = glob.glob(pattern, recursive=True)
    assert hits, f"not found: {pattern}"
    return hits[0]

cmd = [
    sys.executable,
    one("/kaggle/input/**/scripts/screen_pp.py"),
    "--annotations", one("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
    "--oof", one("/kaggle/input/**/oof_prob.npz"),
    "--folds", one("/kaggle/input/**/cv_folds_k5_seed0.json"),
    "--out", "screen_pp_r2.json",
]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print("WRAPPER DONE")
