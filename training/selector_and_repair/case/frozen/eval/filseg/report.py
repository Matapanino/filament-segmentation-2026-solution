"""Per-fold OOF scoring — the one loop every report script runs.

`scripts/oof_report.py` prices ONE model's probability maps;
`scripts/ensemble_report.py` prices an average of several. They must score the
same way or their deltas are not comparable, so the loop lives here and both
call it — the same one-place-for-the-rule discipline `filseg.cli` exists for.

Everything is reported PER FOLD (ADR-0004 §4): an aggregate computed over a
different fold set than its baseline is a contrast against the wrong baseline,
which is the trap that inflated `link60_a45_a320` in F-013 §3.

The relation counts (one-to-many, many-to-one, unmatched) are reported
alongside the score because the final rubric scores their *distributions*
directly, not only the mean Dice they feed into (F-011).
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .acceptance import AcceptanceModel
from .evaluate import evaluate_image
from .instancing import Rule, instances
from .msiou import pair_msiou

#: file -> probability map in [0, 1]; the map may be 1024 or already 2048 px
#: (`instancing.upsample` handles both), which is what lets an ensemble of
#: mixed-resolution members be scored by this identical loop.
ProbSource = Mapping[str, np.ndarray]


def msiou_stats(v: Sequence[float]) -> dict:
    m = np.asarray(v, dtype=np.float64)
    if m.size == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": int(m.size),
        "mean": float(m.mean()),
        "median": float(np.median(m)),
        "p10": float(np.percentile(m, 10)),
        "p90": float(np.percentile(m, 90)),
    }


def fold_report(
    probs: ProbSource,
    files: Sequence[str],
    sets_by_file: Mapping[str, list],
    assignment: Mapping[str, int],
    rule: Rule,
    gt_masks: Callable[[dict], np.ndarray],
    model: AcceptanceModel | None = None,
    n_folds: int = 5,
    progress_every: int = 100,
    log: Callable[[str], None] = print,
) -> dict:
    """Score `files` through `rule`, per fold, with the ADR-0004 evaluator."""
    scores: dict[int, list[float]] = {k: [] for k in range(n_folds)}
    msiou: dict[int, list[float]] = {k: [] for k in range(n_folds)}
    dices: dict[int, list[float]] = {k: [] for k in range(n_folds)}
    rel: dict[int, dict[str, int]] = {
        k: {"one_to_many": 0, "many_to_one": 0, "unmatched_gt": 0,
            "unmatched_pred": 0, "n_pred": 0}
        for k in range(n_folds)
    }

    for idx, f in enumerate(files):
        k = assignment[f]
        preds = instances(probs[f], rule, model)
        rel[k]["n_pred"] += len(preds)
        set_scores = []
        for s in sets_by_file[f]:
            gt = [gt_masks(a) for a in s.annotations]
            ev = evaluate_image(gt, preds)
            if ev.score is not None:
                set_scores.append(ev.score)
            dices[k] += ev.matched_dice
            for i, j in ev.matched_pairs:
                msiou[k].append(pair_msiou(gt[i], preds[j]))
            rel[k]["one_to_many"] += ev.one_to_many
            rel[k]["many_to_one"] += ev.many_to_one
            rel[k]["unmatched_gt"] += ev.n_unmatched_gt
            rel[k]["unmatched_pred"] += ev.n_unmatched_pred
        if set_scores:
            scores[k].append(float(np.mean(set_scores)))
        if progress_every and (idx + 1) % progress_every == 0:
            log(f"  ... {idx + 1}/{len(files)}")

    # A partial map set (e.g. the first M2 folds, before the rest are funded)
    # leaves some folds empty — report the folds that are present rather than
    # emitting nan and poisoning the mean.
    folds = {
        k: {
            "n_files": len(v),
            "mean_image_score": float(np.mean(v)),
            "median": float(np.median(v)),
            "matched_dice_mean": float(np.mean(dices[k])) if dices[k] else None,
            "matched_msiou": msiou_stats(msiou[k]),
            "relations": rel[k],
        }
        for k, v in scores.items()
        if v
    }
    fold_means = [folds[k]["mean_image_score"] for k in sorted(folds)]
    d = np.array([x for k in folds for x in dices[k]])
    m = [x for k in folds for x in msiou[k]]
    return {
        "rule": {k: getattr(rule, k) for k in rule.__dataclass_fields__},
        "per_fold": folds,
        "fold_mean_of_means": float(np.mean(fold_means)),
        "fold_std_of_means": (
            float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else None
        ),
        "matched_dice_mean": float(d.mean()) if d.size else None,
        "matched_msiou": msiou_stats(m),
        "relations": {
            key: sum(folds[k]["relations"][key] for k in folds)
            for key in ("one_to_many", "many_to_one", "unmatched_gt",
                        "unmatched_pred", "n_pred")
        },
    }
