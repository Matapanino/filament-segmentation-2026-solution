"""Connected components + their acceptance features (R3; docs/adr/0006).

The evaluation protocol charges one denominator unit for every predicted
instance, matched or not (ADR-0004), so each component is an independent bet:
keep it only if its probability of matching a GT filament beats the derived
break-even (ADR-0006 §2). ``min_area`` is one crude feature of that bet; this
module computes the cheap ones that are available at inference time on both
train and test, from nothing but the probability map and the binary mask (so
the features are fold-independent by construction — no leakage channel).

Feature order is FROZEN by ``FEATURE_NAMES``: the acceptance model persists as
coefficients over these names, and a reordering would silently mis-apply them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import cv2
import numpy as np

EPS = 1e-6

FEATURE_NAMES: tuple[str, ...] = (
    "log_area",          # size: the R1/R2 lever, in log units
    "mean_prob",         # confidence: mean sigmoid inside the component
    "max_prob",          # peak confidence (the hysteresis core signal)
    "p90_prob",
    "std_prob",
    "log_prob_mass",     # sum of probability = "how much model mass" it holds
    "elongation",        # sqrt(lambda1/lambda2) of the equivalent ellipse
    "log_major_axis",    # length scale of the ellipse (spine proxy)
    "width_proxy",       # area / major axis = mean width (F-002: median 11 px)
    "extent",            # area / bbox area
    "solidity",          # area / convex hull area
    "compactness",       # perimeter / sqrt(area) (thin structures score high)
    "radius_norm",       # centroid distance from frame centre / half-frame
    "log_n_comp",        # image context: how many components this frame has
    "area_rank_frac",    # rank of this area within the frame (1 = largest)
    "area_frac_of_max",  # area / largest area in the frame
    "log_image_area",    # total predicted area in the frame
)


@dataclass(frozen=True)
class Component:
    """One connected component: label id, bbox, area, centroid."""

    label: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float

    @property
    def slice(self) -> tuple[slice, slice]:
        return (slice(self.y, self.y + self.h), slice(self.x, self.x + self.w))


def components(binm: np.ndarray, min_area: int = 0) -> tuple[np.ndarray, list[Component]]:
    """8-connected components of a 0/1 mask, filtered by ``min_area``."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(binm, connectivity=8)
    comps = [
        Component(
            label=i,
            x=int(stats[i, 0]),
            y=int(stats[i, 1]),
            w=int(stats[i, 2]),
            h=int(stats[i, 3]),
            area=int(stats[i, 4]),
            cx=float(cent[i, 0]),
            cy=float(cent[i, 1]),
        )
        for i in range(1, n)
        if stats[i, 4] >= min_area
    ]
    return lab, comps


def iter_masks(
    labels: np.ndarray, comps: Sequence[Component]
) -> Iterator[np.ndarray]:
    """Yield one full-frame uint8 mask per component (bbox-local write)."""
    for c in comps:
        m = np.zeros(labels.shape, np.uint8)
        sl = c.slice
        m[sl] = labels[sl] == c.label
        yield m


def features(
    labels: np.ndarray, comps: Sequence[Component], prob: np.ndarray
) -> np.ndarray:
    """Acceptance features for every component, shape (n, len(FEATURE_NAMES))."""
    n = len(comps)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    if n == 0:
        return X

    areas = np.array([c.area for c in comps], dtype=np.float64)
    order = np.argsort(-areas)          # rank 0 = largest
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(n, dtype=np.float64)
    area_max = float(areas.max())
    area_sum = float(areas.sum())
    half = 0.5 * labels.shape[0]

    for k, c in enumerate(comps):
        sl = c.slice
        comp = labels[sl] == c.label
        pv = prob[sl][comp].astype(np.float64)
        cm = comp.astype(np.uint8)

        mu = cv2.moments(cm, binaryImage=True)
        m00 = max(mu["m00"], EPS)
        cxx, cyy, cxy = mu["mu20"] / m00, mu["mu02"] / m00, mu["mu11"] / m00
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
        l1, l2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 0.0)
        major = 4.0 * l1**0.5
        elong = (l1 / max(l2, EPS)) ** 0.5

        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea) if cnts else None
        perim = cv2.arcLength(cnt, True) if cnt is not None else 0.0
        hull_area = (
            cv2.contourArea(cv2.convexHull(cnt)) if cnt is not None else 0.0
        )

        X[k] = (
            np.log1p(c.area),
            pv.mean(),
            pv.max(),
            float(np.percentile(pv, 90)),
            pv.std(),
            np.log1p(pv.sum()),
            elong,
            np.log1p(major),
            c.area / max(major, EPS),
            c.area / max(c.w * c.h, 1),
            c.area / max(hull_area, EPS) if hull_area > 0 else 1.0,
            perim / max(c.area, 1) ** 0.5,
            ((c.cx - half) ** 2 + (c.cy - half) ** 2) ** 0.5 / half,
            np.log1p(n),
            rank[k] / max(n - 1, 1),
            c.area / max(area_max, EPS),
            np.log1p(area_sum),
        )
    return X
