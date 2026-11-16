"""Instance-level evaluation: the campaign's compass (ADR-0004).

The hidden competition evaluator is unknown and demonstrably quirky (an
all-empty submission scores ~0.93 on the public LB). We therefore define an
explicit, documented protocol and use it for ALL modeling decisions:

    image score = sum(matched per-filament Dice)
                  / (n_matched + n_unmatched_gt + n_unmatched_pred)

with maximum-weight (Hungarian) matching on the pairwise Dice matrix; only
overlapping pairs are eligible matches. Perfect predictions give the mean
per-filament Dice; every spurious or missed instance decays the score
symmetrically.

Two GT protocols against multi-annotator ground truth:

- ``per_set``: evaluate against each annotator set separately, then average
  (primary protocol until evidence about the hidden GT says otherwise);
- ``concat``: all sets' instances form one GT list (duplicates retained;
  mirrors a hidden GT that keeps duplicate annotator instances).

Empty semantics are explicit: no-GT + no-pred images are excluded from means
(counted separately); no-GT + preds scores 0; GT + no-preds scores 0. We
deliberately refuse to copy the LB's empty-inflation behavior.

Rubric-shaped diagnostics (per-filament Dice/IoU distributions, one-to-many and
many-to-one relation counts at a configurable IoU threshold) are emitted
alongside every aggregate, mirroring the organizers' final rubric.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class _MaskBox:
    mask: np.ndarray
    area: int
    box: tuple[int, int, int, int] | None  # (r0, r1, c0, c1); None if empty


def _prep(mask: np.ndarray) -> _MaskBox:
    m = mask.astype(bool)
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return _MaskBox(m, 0, None)
    cols = np.flatnonzero(m.any(axis=0))
    return _MaskBox(m, int(m.sum()), (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1))


def _intersection(a: _MaskBox, b: _MaskBox) -> int:
    if a.box is None or b.box is None:
        return 0
    r0 = max(a.box[0], b.box[0])
    r1 = min(a.box[1], b.box[1])
    c0 = max(a.box[2], b.box[2])
    c1 = min(a.box[3], b.box[3])
    if r0 >= r1 or c0 >= c1:
        return 0
    return int(np.logical_and(a.mask[r0:r1, c0:c1], b.mask[r0:r1, c0:c1]).sum())


@dataclass
class ImageEval:
    n_gt: int
    n_pred: int
    matched_dice: list[float] = field(default_factory=list)
    matched_iou: list[float] = field(default_factory=list)
    # (gt_idx, pred_idx) per match, indexing the ORIGINAL gt_masks/pred_masks
    # argument lists — lets diagnostics (e.g. MSIoU) revisit matched pairs.
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)
    n_unmatched_gt: int = 0
    n_unmatched_pred: int = 0
    one_to_many: int = 0  # GT instances overlapped (IoU>theta) by >=2 preds
    many_to_one: int = 0  # pred instances overlapping (IoU>theta) >=2 GTs

    @property
    def score(self) -> float | None:
        denom = len(self.matched_dice) + self.n_unmatched_gt + self.n_unmatched_pred
        if denom == 0:
            return None  # empty vs empty: excluded from means, counted upstream
        return sum(self.matched_dice) / denom


def intersection_matrix(gt_masks, pred_masks):
    """Positive-area GT/pred index lists + their pairwise intersection matrix.

    Returns ``(inter, g_areas, p_areas, gt_idx, pr_idx)``; the index lists map
    matrix rows/columns back to positions in the input lists. Everything the
    protocol needs is a function of these four arrays (no masks) — which is
    what lets ``evaluate_from_intersections`` re-score arbitrary SUBSETS of the
    same predictions for free (the screen's acceptance-threshold sweeps).
    """
    gt_idx, gt, pr_idx, pr = [], [], [], []
    for i, m in enumerate(gt_masks):
        g = _prep(m)
        if g.area > 0:
            gt_idx.append(i)
            gt.append(g)
    for j, m in enumerate(pred_masks):
        p = _prep(m)
        if p.area > 0:
            pr_idx.append(j)
            pr.append(p)
    inter = np.zeros((len(gt), len(pr)), dtype=np.int64)
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            inter[i, j] = _intersection(g, p)
    return (
        inter,
        np.array([g.area for g in gt], dtype=np.int64),
        np.array([p.area for p in pr], dtype=np.int64),
        gt_idx,
        pr_idx,
    )


def _evaluate_matrix(
    inter, g_areas, p_areas, gt_idx, pr_idx, relation_iou_theta: float
) -> ImageEval:
    n_g, n_p = len(gt_idx), len(pr_idx)
    ev = ImageEval(n_gt=n_g, n_pred=n_p)
    if n_g == 0 or n_p == 0:
        ev.n_unmatched_gt = n_g
        ev.n_unmatched_pred = n_p
        return ev

    ga = np.asarray(g_areas, dtype=np.int64)[:, None]
    pa = np.asarray(p_areas, dtype=np.int64)[None, :]
    dice = 2.0 * inter / (ga + pa)
    iou = inter / (ga + pa - inter)

    gi, pj = linear_sum_assignment(-dice)
    for i, j in zip(gi, pj):
        if inter[i, j] > 0:  # zero-overlap assignments are not matches
            ev.matched_dice.append(float(dice[i, j]))
            ev.matched_iou.append(float(iou[i, j]))
            ev.matched_pairs.append((gt_idx[i], pr_idx[j]))
    n_match = len(ev.matched_dice)
    ev.n_unmatched_gt = n_g - n_match
    ev.n_unmatched_pred = n_p - n_match
    ev.one_to_many = int(((iou > relation_iou_theta).sum(axis=1) >= 2).sum())
    ev.many_to_one = int(((iou > relation_iou_theta).sum(axis=0) >= 2).sum())
    return ev


def evaluate_image(
    gt_masks, pred_masks, relation_iou_theta: float = 0.1
) -> ImageEval:
    """Score one image's predicted instances against one GT instance list."""
    inter, g_areas, p_areas, gt_idx, pr_idx = intersection_matrix(
        gt_masks, pred_masks
    )
    return _evaluate_matrix(
        inter, g_areas, p_areas, gt_idx, pr_idx, relation_iou_theta
    )


def evaluate_from_intersections(
    inter,
    g_areas,
    p_areas,
    keep=None,
    gt_idx=None,
    pr_idx=None,
    relation_iou_theta: float = 0.1,
) -> ImageEval:
    """Re-score a cached intersection matrix, optionally keeping a pred subset.

    ``keep`` is a boolean/index array over the matrix COLUMNS (predictions).
    Identical output to ``evaluate_image`` on the corresponding mask subset —
    the protocol only ever reads intersections and areas (parity-tested).
    """
    inter = np.asarray(inter, dtype=np.int64)
    p_areas = np.asarray(p_areas, dtype=np.int64)
    n_g, n_p = inter.shape
    gt_idx = list(range(n_g)) if gt_idx is None else list(gt_idx)
    pr_idx = list(range(n_p)) if pr_idx is None else list(pr_idx)
    if keep is not None:
        cols = np.asarray(keep)
        if cols.dtype == bool:
            cols = np.flatnonzero(cols)
        inter = inter[:, cols]
        p_areas = p_areas[cols]
        pr_idx = [pr_idx[c] for c in cols]
    return _evaluate_matrix(
        inter, g_areas, p_areas, gt_idx, pr_idx, relation_iou_theta
    )


@dataclass
class Report:
    protocol: str
    n_files: int
    n_scored: int
    n_empty_pairs: int
    mean_score: float | None
    matched_dice: np.ndarray
    matched_iou: np.ndarray
    one_to_many: int
    many_to_one: int
    n_unmatched_gt: int
    n_unmatched_pred: int

    def summary(self) -> str:
        md = self.matched_dice
        score_line = (
            f"mean image score = {self.mean_score:.4f}"
            if self.mean_score is not None
            else "mean image score = n/a (nothing scored)"
        )
        dice_line = (
            f"matched filaments: n={md.size}  dice mean={md.mean():.4f} "
            f"median={float(np.median(md)):.4f}"
            if md.size
            else "matched filaments: none"
        )
        return "\n".join(
            [
                f"protocol={self.protocol}  files={self.n_files} "
                f"scored={self.n_scored} empty-pairs={self.n_empty_pairs}",
                score_line,
                dice_line,
                f"unmatched instances: gt={self.n_unmatched_gt} "
                f"pred={self.n_unmatched_pred}",
                f"relations (IoU>theta): one_to_many={self.one_to_many} "
                f"many_to_one={self.many_to_one}",
            ]
        )


def evaluate_files(
    gt_sets_by_file: dict[str, list[list[np.ndarray]]],
    preds_by_file: dict[str, list[np.ndarray]],
    protocol: str = "per_set",
    relation_iou_theta: float = 0.1,
) -> tuple[Report, dict[str, float | None]]:
    """Evaluate predictions for many files under a GT protocol.

    ``gt_sets_by_file`` maps an observation file to its annotator sets, each a
    list of instance masks. ``preds_by_file`` maps a file to predicted masks.
    """
    per_file: dict[str, float | None] = {}
    evals: list[ImageEval] = []
    for file, sets in gt_sets_by_file.items():
        preds = preds_by_file.get(file, [])
        if protocol == "concat":
            sets = [[m for s in sets for m in s]]
        elif protocol != "per_set":
            raise ValueError(f"unknown protocol: {protocol}")
        set_scores = []
        for s in sets:
            ev = evaluate_image(s, preds, relation_iou_theta)
            evals.append(ev)
            if ev.score is not None:
                set_scores.append(ev.score)
        per_file[file] = float(np.mean(set_scores)) if set_scores else None

    scored = [v for v in per_file.values() if v is not None]
    report = Report(
        protocol=protocol,
        n_files=len(per_file),
        n_scored=len(scored),
        n_empty_pairs=sum(1 for v in per_file.values() if v is None),
        mean_score=float(np.mean(scored)) if scored else None,
        matched_dice=np.array([d for e in evals for d in e.matched_dice]),
        matched_iou=np.array([d for e in evals for d in e.matched_iou]),
        one_to_many=sum(e.one_to_many for e in evals),
        many_to_one=sum(e.many_to_one for e in evals),
        n_unmatched_gt=sum(e.n_unmatched_gt for e in evals),
        n_unmatched_pred=sum(e.n_unmatched_pred for e in evals),
    )
    return report, per_file
