"""Probability map -> filament instances: the pipeline's decision rule.

In this competition the decision rule, not the network, is where the score is:
R1/R2 (F-009/F-010) recovered **+0.0847** image score — more than half the
baseline-to-human-ceiling gap — by changing two constants applied to frozen
probability maps, for zero GPU-hours.

A ``Rule`` is that decision rule as one immutable object, applied in ONE place
so a screened arm, an OOF safety report and a submission cannot drift apart
(measured risk: the same chain used to live in three files). The stages, in the
order they are applied:

1. bilinear upsample of the 1024 px sigmoid map to full 2048 px resolution;
2. **binarize** — a single threshold, or ``hysteresis`` when ``hys_hi > 0``
   (grow at ``thr``, keep only components with a core above ``hys_hi``);
3. optional ``oriented_close`` (line-SE span, a reach control — see postproc);
4. optional isotropic ``ellipse_close`` (the R2 ship mechanism, 21 px);
5. optional ``link_collinear`` (bridge tips aligned with both tip-local axes);
6. optional ``fill_holes``;
7. 8-connected components, then the acceptance test: ``min_area``, and/or an
   ``AcceptanceModel`` probability above ``accept_tau`` (ADR-0006: the
   break-even is derived from the metric, not tuned).

A per-component *mean-probability* filter was screened and found fully
redundant with ``min_area`` (R2: identical outputs) — deliberately absent.

``iter_instances`` yields one full-frame mask at a time so submission writers
never hold more than one instance in memory (an unconverged model can emit
thousands of components per file — the all-at-once version OOMed, measured).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import cv2
import numpy as np

from . import compfeat, postproc
from .acceptance import AcceptanceModel

FULL = 2048


@dataclass(frozen=True)
class Rule:
    """The post-processing decision rule, as screened (see ADR-0006)."""

    thr: float = 0.5
    min_area: int = 8
    close_px: int = 0            # isotropic closing (R2 ship mechanism)
    hys_hi: float = 0.0          # >0: thr becomes the LOW threshold
    line_px: int = 0             # >0: oriented (line-SE) closing span
    n_angles: int = 12
    link_gap: int = 0            # >0: collinear tip linking, max gap in px
    link_angle: float = 20.0
    link_window: int = 30        # px around a tip that define its local axis
    link_min_area: int = 32      # pieces smaller than this never link
    fill: bool = False
    accept_tau: float = 0.0      # >0: require AcceptanceModel p(match) >= tau

    def with_(self, **kw) -> "Rule":
        return replace(self, **kw)


#: The rule adopted in R2 (F-010): OOF image score 0.4701, 5/5 folds. Kept for
#: lineage — the report documents how the decision rule evolved, and every
#: later delta is quoted against this configuration.
SHIP_R2 = Rule(thr=0.5, min_area=256, close_px=21)

#: The rule adopted in R3 (F-012): OOF image score **0.4792** (+0.0091 over
#: SHIP_R2, 5/5 folds). Adds anisotropic tip linking, which cuts fragmentation
#: (one-to-many 378 -> 235) and false positives (2,012 -> 1,693) at flat matched
#: Dice, and PASSES the MSIoU safety check with room to spare: mean +0.0071,
#: p10 (the barb-sensitive weak tail) **+0.0274**, p90 unchanged — merging a
#: fragmented filament into one object makes fine structure score better, not
#: worse. Cost: many-to-one 182 -> 266 and fold std 0.0076 -> 0.0092.
SHIP_R3 = Rule(
    thr=0.5, min_area=256, close_px=21, link_gap=60, link_angle=45.0
)

#: What the pipeline ships today.
SHIP = SHIP_R3


def upsample(prob: np.ndarray) -> np.ndarray:
    """Bilinear upsample of a [0,1] map to the full 2048 px frame.

    A map that is ALREADY full-resolution (M2 onward: native-res tiles stitched
    at 2048) is returned unchanged rather than round-tripped through
    ``cv2.resize`` at scale 1 — the two agree numerically on the versions we
    tested, but the decision rule should not depend on that.
    """
    prob = prob.astype(np.float32, copy=False)
    if prob.shape == (FULL, FULL):
        return prob
    return cv2.resize(prob, (FULL, FULL), interpolation=cv2.INTER_LINEAR)


def binarize(prob_full: np.ndarray, rule: Rule) -> np.ndarray:
    """Apply stages 2-6 of ``rule`` to an already-upsampled probability map."""
    if rule.hys_hi > 0.0:
        binm = postproc.hysteresis(prob_full, rule.thr, rule.hys_hi)
    else:
        binm = (prob_full >= rule.thr).astype(np.uint8)
    if rule.line_px:
        binm = postproc.oriented_close(binm, rule.line_px, rule.n_angles)
    if rule.close_px:
        binm = postproc.ellipse_close(binm, rule.close_px)
    if rule.link_gap:
        binm = postproc.link_collinear(
            binm,
            rule.link_gap,
            rule.link_angle,
            min_area=rule.link_min_area,
            tip_window=rule.link_window,
        )
    if rule.fill:
        binm = postproc.fill_holes(binm)
    return binm


def iter_instances(
    prob1024: np.ndarray, rule: Rule, model: AcceptanceModel | None = None
) -> Iterator[np.ndarray]:
    """Yield one uint8 full-frame instance mask per accepted component."""
    up = upsample(prob1024)
    binm = binarize(up, rule)
    labels, comps = compfeat.components(binm, rule.min_area)
    if rule.accept_tau > 0.0:
        if model is None:
            raise ValueError("accept_tau > 0 requires an AcceptanceModel")
        if comps:
            keep = model.predict_proba(compfeat.features(labels, comps, up))
            comps = [c for c, p in zip(comps, keep) if p >= rule.accept_tau]
    elif model is not None:
        raise ValueError("an AcceptanceModel was passed but accept_tau is 0")
    yield from compfeat.iter_masks(labels, comps)


def instances(
    prob1024: np.ndarray, rule: Rule, model: AcceptanceModel | None = None
) -> list[np.ndarray]:
    return list(iter_instances(prob1024, rule, model))
