"""Post-processing mechanisms screened in R3 (see docs/adr/0006).

R1/R2 (F-009/F-010) established that the decision rule dominates: a single
minimum-area constant plus isotropic closing recovered +0.0847 image score for
zero GPU-hours. Both of those levers are now at their interior optimum, so R3
adds mechanisms that are *structurally* different rather than more of the same
constant:

- ``hysteresis``: grow the mask at a LOW threshold but keep only components
  that contain a HIGH-confidence core. A single global threshold has to trade
  barb recall against false positives (F-009 closed it: monotone, flat);
  hysteresis buys both, because the two thresholds serve different jobs.
- ``link_collinear``: join two components only when the gap between their tips
  is short AND aligned with both of their principal axes. This is the
  anisotropic merge the geometry asks for (median width 11 px, median spine
  110 px, F-002), and unlike closing it cannot merge a filament with a
  neighbour it merely runs beside.
- ``oriented_close``: closing by a union of oriented LINE structuring elements.
  MEASURED NOT to be orientation-selective and kept only as a reach control: a
  pixel is filled as soon as ONE orientation is blocked in every translate, so
  a lateral channel between two parallel bars is filled by the perpendicular
  element (720 px added vs 702 px for isotropic closing of the same span). The
  intuition that a line SE "only bridges collinear fragments" is wrong for the
  union over orientations; that is what ``link_collinear`` is for.
- ``fill_holes``: closes interior background pockets left by thresholding.

All functions take and return full-resolution uint8 masks (0/1) and are pure —
no state, no I/O — so the screen can compose them per arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-12


def hysteresis(prob: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Threshold at ``lo``, keep only components whose peak probability >= ``hi``.

    ``prob`` is a [0,1] map at the resolution the instances will live at.
    """
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"need 0 <= lo <= hi <= 1, got lo={lo}, hi={hi}")
    low = (prob >= lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(low, connectivity=8)
    if n <= 1:
        return low
    # peak probability per label in one pass (label 0 = background)
    peak = np.zeros(n, dtype=np.float32)
    np.maximum.at(peak, lab.ravel(), prob.ravel().astype(np.float32))
    keep = peak >= hi
    keep[0] = False
    return keep[lab].astype(np.uint8)


def oriented_close(
    binm: np.ndarray, length: int, n_angles: int = 12
) -> np.ndarray:
    """Union of closings with line SEs of ``length`` px at ``n_angles`` angles.

    ``length`` is the line's EUCLIDEAN span, so the bridging reach is the same
    in every direction (a diagonal line covers fewer pixels than an axis-
    aligned one of the same span). NOTE the union over orientations is close to
    isotropic closing of the same span, NOT an orientation-selective merge (see
    the module docstring); use ``link_collinear`` for that.

    Closing is increasing per SE and the union over orientations is still >=
    the input (closing never removes foreground), so this can only bridge gaps,
    never erode structure. The frame is zero-padded by ``length`` first:
    OpenCV's default morphology border makes erosion treat outside-the-frame as
    foreground, which smears any structure within one SE of the edge (measured:
    265 spurious pixels around a segment 10 px from the border). Padding makes
    the operator border-neutral, i.e. equal to closing on an infinite frame.
    """
    if length < 3:
        raise ValueError(f"length must be >= 3, got {length}")
    if n_angles < 1:
        raise ValueError(f"n_angles must be >= 1, got {n_angles}")
    p = length
    pad = cv2.copyMakeBorder(binm, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    out = pad.copy()
    for k in range(n_angles):
        se = line_se(length, 180.0 * k / n_angles)
        out |= cv2.morphologyEx(pad, cv2.MORPH_CLOSE, se)
    return np.ascontiguousarray(out[p:-p, p:-p])


def line_se(length: int, angle_deg: float) -> np.ndarray:
    """A ``length``-px line structuring element at ``angle_deg`` (uint8 0/1).

    The element is its own tight bounding box rounded UP to odd sides, not a
    length x length square. Two measured reasons:

    - drawing a tilted line inside a square and clamping the overhang folds the
      ends onto the box edges; such an SE closed 907 spurious pixels on the
      unit-test scene against ~25 legitimate bridge pixels;
    - the element must be point-symmetric about its centre, and OpenCV erodes
      with the SE exactly as given (it does not reflect it). An even side puts
      the anchor half a pixel off centre and the "closing" then returns the
      structure UNION a 1-px-shifted copy of itself — a fake parallel filament
      beside every real one. Worse, a corner-to-corner ``cv2.line`` is not
      always symmetric under Bresenham rounding, and an asymmetric SE makes
      MORPH_CLOSE *remove* foreground (measured: -2 px at 75 deg / 105 deg).
      Stepping along the major axis with ``rint`` (an odd function) is
      symmetric by construction.
    """
    a = np.deg2rad(angle_deg % 180.0)
    dx, dy = np.cos(a), np.sin(a)
    w = 2 * int(round(abs(dx) * (length - 1) / 2)) + 1
    h = 2 * int(round(abs(dy) * (length - 1) / 2)) + 1
    se = np.zeros((h, w), np.uint8)
    cx, cy = w // 2, h // 2
    if w >= h:  # |dx| >= |dy|: one pixel per column
        i = np.arange(-cx, cx + 1)
        xs = cx + i
        ys = cy - (np.rint(i * (dy / dx)).astype(int) if dx else 0)
    else:       # one pixel per row (y grows downward)
        j = np.arange(-cy, cy + 1)
        ys = cy + j
        xs = cx - (np.rint(j * (dx / dy)).astype(int) if dy else 0)
    se[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)] = 1
    return se


@dataclass(frozen=True)
class _Piece:
    """One component reduced to what the linker needs.

    Each tip carries its OWN local axis. A filament is curved, so the global
    principal axis is a poor proxy for the direction the structure leaves a tip
    in — measured on the R3 dry run: with global axes the 20 deg budget fired
    zero links across 12 files, while the same budget on tip-local axes links
    the fragment pairs the eye calls collinear.
    """

    axis: tuple[float, float]                        # global principal axis
    tips: tuple[tuple[int, int], tuple[int, int]]    # extreme points along it
    tip_axes: tuple[tuple[float, float], tuple[float, float]]
    elong: float


def _axis_of(pts: np.ndarray) -> tuple[tuple[float, float], float, float]:
    """Principal axis (unit vector) and the two eigenvalues of a (2, n) cloud."""
    if pts.shape[1] < 2:
        return (1.0, 0.0), 0.0, 0.0
    vals, vecs = np.linalg.eigh(np.atleast_2d(np.cov(pts)))
    v = vecs[:, -1]
    return (float(v[0]), float(v[1])), float(vals[-1]), float(max(vals[0], 0.0))


def _pieces(binm: np.ndarray, min_area: int, tip_window: int) -> list[_Piece]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    out: list[_Piece] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ys, xs = np.nonzero(lab[y : y + h, x : x + w] == i)
        xs = xs + x
        ys = ys + y
        pts = np.stack([xs, ys]).astype(np.float64)
        axis, l1, l2 = _axis_of(pts)
        t = pts.T @ np.array(axis)
        tip_idx = (int(np.argmin(t)), int(np.argmax(t)))
        tips, tip_axes = [], []
        for k in tip_idx:
            tx, ty = int(xs[k]), int(ys[k])
            tips.append((tx, ty))
            near = (xs - tx) ** 2 + (ys - ty) ** 2 <= tip_window**2
            local, _, _ = _axis_of(pts[:, near]) if near.sum() >= 3 else (axis, 0, 0)
            tip_axes.append(local)
        out.append(
            _Piece(
                axis=axis,
                tips=(tips[0], tips[1]),
                tip_axes=(tip_axes[0], tip_axes[1]),
                elong=(l1 / max(l2, EPS)) ** 0.5,
            )
        )
    return out


def link_collinear(
    binm: np.ndarray,
    max_gap: int,
    max_angle_deg: float = 20.0,
    thickness: int = 5,
    min_area: int = 32,
    min_elong: float = 1.0,
    tip_window: int = 30,
) -> np.ndarray:
    """Bridge component tips that are close AND aligned with both axes.

    One filament broken into two pieces shows up as an ``one_to_many`` relation
    (378 on the R2 ship config, ~2.7x the human rate, F-010) and costs the score
    twice: the extra piece is an unmatched prediction and the matched piece
    loses Dice. Isotropic closing fixes some of that but pays for it in
    ``many_to_one`` (97 -> 182 at 21 px) because it bridges whatever is nearby.
    This linker only draws a bridge when the gap direction lies within
    ``max_angle_deg`` of the local axis at BOTH tips being joined, i.e. when the
    two pieces look like one interrupted curvilinear structure. ``tip_window``
    sets how much of each piece counts as "local" (px around the tip).

    ``thickness`` is the bridge width in px (median filament width is 11 px, so
    the default 5 keeps the added area small while staying physically plausible
    for the rubric's morphology inspection).
    """
    if max_gap < 1:
        raise ValueError(f"max_gap must be >= 1, got {max_gap}")
    if not 0.0 <= max_angle_deg <= 90.0:
        raise ValueError(f"max_angle_deg must be in [0,90], got {max_angle_deg}")
    if tip_window < 2:
        raise ValueError(f"tip_window must be >= 2, got {tip_window}")
    pieces = _pieces(binm, min_area, tip_window)
    if len(pieces) < 2:
        return binm
    cos_min = float(np.cos(np.deg2rad(max_angle_deg)))
    out = binm.copy()
    for a in range(len(pieces)):
        pa = pieces[a]
        if pa.elong < min_elong:
            continue
        for b in range(a + 1, len(pieces)):
            pb = pieces[b]
            if pb.elong < min_elong:
                continue
            ka, kb = min(
                ((i, j) for i in (0, 1) for j in (0, 1)),
                key=lambda ij: (pa.tips[ij[0]][0] - pb.tips[ij[1]][0]) ** 2
                + (pa.tips[ij[0]][1] - pb.tips[ij[1]][1]) ** 2,
            )
            (ax, ay), (bx, by) = pa.tips[ka], pb.tips[kb]
            dx, dy = bx - ax, by - ay
            gap = (dx * dx + dy * dy) ** 0.5
            if gap < 1.0 or gap > max_gap:
                continue
            ux, uy = dx / gap, dy / gap
            va, vb = pa.tip_axes[ka], pb.tip_axes[kb]
            if abs(ux * va[0] + uy * va[1]) < cos_min:
                continue
            if abs(ux * vb[0] + uy * vb[1]) < cos_min:
                continue
            cv2.line(out, (ax, ay), (bx, by), 1, thickness)
    return out


def fill_holes(binm: np.ndarray) -> np.ndarray:
    """Fill background components that do not touch the image border."""
    h, w = binm.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = binm
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0).astype(np.uint8)
    return (binm | holes).astype(np.uint8)


def ellipse_close(binm: np.ndarray, close_px: int) -> np.ndarray:
    """The R2 ship mechanism: isotropic closing with an elliptical SE.

    Deliberately NOT border-padded like ``oriented_close``: this is the exact
    operator measured in F-010 (+0.0847, 5/5) and reproduced by the R2 parity
    check, and the solar disk sits far from the 2048 px frame edge, so the
    padding question is a documented micro-decision, not a live bug.
    """
    if not close_px:
        return binm
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    return cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
