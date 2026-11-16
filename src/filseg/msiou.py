"""Multiscale IoU diagnostics on top of the vendored reference implementation.

MSIoU (arXiv:2105.14572) is the organizers' own fine-structure metric: Sobel
contours, box-counting at cell sizes Delta = {2^n, n < 10}, intersection
normalized by the GROUND-TRUTH box count (recall-flavoured: missing GT contour
detail — barbs — is punished; extra predicted detail is not), and the area
under the ratio-vs-cellsize curve. We report it alongside the ADR-0004 image
score wherever instances have been matched; it is a diagnostic, not the
selection compass.

``pair_msiou`` is numerically identical to running the vendored code on the
full frame; when both frame sides are divisible by the largest box size it
first crops to a grid-aligned window around the two masks (grid phase and
Sobel neighbourhoods preserved — asserted by tests), which makes full-dataset
sweeps tractable.
"""
from __future__ import annotations

import numpy as np

from .evaluate import ImageEval
from .vendor.miou import MIoU

#: The paper's / walk-through's grid: cell sizes 1, 2, 4, ..., 512.
DEFAULT_BOXSIZES: np.ndarray = 2 ** np.arange(10)


def _grid_aligned_window(
    gt_mask: np.ndarray, pred_mask: np.ndarray, step: int, margin: int = 2
) -> tuple[slice, slice] | None:
    """Smallest window covering both masks (+margin), snapped to the box grid.

    Snapping keeps every cell boundary of every divisor box size where the
    full-frame computation puts it; the margin keeps Sobel 3x3 neighbourhoods
    of all foreground pixels inside the window. Returns None when either mask
    is empty (MSIoU is only defined for matched, overlapping pairs anyway).
    """
    stack = np.logical_or(gt_mask, pred_mask)
    rows = np.flatnonzero(stack.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(stack.any(axis=0))
    h, w = gt_mask.shape
    r0 = max(rows[0] - margin, 0) // step * step
    c0 = max(cols[0] - margin, 0) // step * step
    r1 = min(-(-(rows[-1] + 1 + margin) // step) * step, h)
    c1 = min(-(-(cols[-1] + 1 + margin) // step) * step, w)
    return slice(r0, r1), slice(c0, c1)


def pair_msiou(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    boxsizes: np.ndarray = DEFAULT_BOXSIZES,
) -> float:
    """MSIoU of one (GT, prediction) mask pair — GT is the reference side.

    Inputs are normalized to int64, the dtype the upstream ``mask_loader``
    produces. This is load-bearing: a uint8 mask sends ``np.hypot`` (inside
    the Sobel step) to float16, whose 2-ulp spacing above 2048 miscounts
    boxes on large contours — ratios can exceed 1 (see tests).
    """
    h, w = gt_mask.shape
    step = int(max(boxsizes))
    if h % step == 0 and w % step == 0:
        win = _grid_aligned_window(gt_mask, pred_mask, step)
        if win is None:
            return 0.0
        gt_mask = gt_mask[win]
        pred_mask = pred_mask[win]
    return float(
        MIoU(boxsizes, edge_only=True).measure(
            gt_mask.astype(np.int64), pred_mask.astype(np.int64)
        )
    )


def image_msiou(
    gt_masks,
    pred_masks,
    ev: ImageEval,
    boxsizes: np.ndarray = DEFAULT_BOXSIZES,
) -> list[float]:
    """MSIoU for each matched pair recorded by ``evaluate_image``."""
    return [
        pair_msiou(np.asarray(gt_masks[i]), np.asarray(pred_masks[j]), boxsizes)
        for i, j in ev.matched_pairs
    ]
