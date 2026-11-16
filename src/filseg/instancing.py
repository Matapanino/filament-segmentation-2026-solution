"""Probability map -> filament instances: the pipeline's decision rule.

The screened post-processing chain (F-009/F-010): bilinear upsample of the
1024px sigmoid map to full 2048px resolution, threshold, optional
morphological closing (merges fragments; gap-closing measured safe, F-004),
8-connected components, minimum-area filter (FP control; GT areas start at
9 px but p5 = 314, F-002). A per-component confidence filter was screened and
found fully redundant with min-area (R2: identical outputs) — deliberately
not implemented.

``iter_instances`` yields one full-frame mask at a time so submission writers
never hold more than one instance in memory (an unconverged model can emit
thousands of components per file — the all-at-once version OOMed, measured).
"""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

FULL = 2048


def binarize(prob1024: np.ndarray, thr: float, close_px: int = 0) -> np.ndarray:
    """Upsample a [0,1] sigmoid map to 2048 and threshold (+ optional closing)."""
    up = cv2.resize(
        prob1024.astype(np.float32), (FULL, FULL), interpolation=cv2.INTER_LINEAR
    )
    binm = (up >= thr).astype(np.uint8)
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    return binm


def iter_instances(
    prob1024: np.ndarray, thr: float, min_area: int, close_px: int = 0
) -> Iterator[np.ndarray]:
    """Yield one uint8 full-frame instance mask per connected component."""
    binm = binarize(prob1024, thr, close_px)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        m = np.zeros((FULL, FULL), np.uint8)
        m[y : y + h, x : x + w] = lab[y : y + h, x : x + w] == i
        yield m


def instances(
    prob1024: np.ndarray, thr: float, min_area: int, close_px: int = 0
) -> list[np.ndarray]:
    return list(iter_instances(prob1024, thr, min_area, close_px))
