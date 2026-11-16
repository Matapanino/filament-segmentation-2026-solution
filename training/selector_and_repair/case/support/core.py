"""Shared data partition and exact pooled-PQ evaluation; no model fitting."""
import json
from pathlib import Path

import numpy as np

FIELDS = ("iou_sum", "tp", "fp", "fn", "split", "merge")
THRESHOLDS = np.arange(0, 1.001, .025).round(6)


def counts(iou):
    iou = np.asarray(iou, dtype=np.float64)
    assert iou.ndim == 2 and np.isfinite(iou).all()
    hit, overlap = iou > .5, iou > 0
    return np.array([iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(),
                     (hit.sum(1) == 0).sum(), (overlap.sum(1) > 1).sum(),
                     (overlap.sum(0) > 1).sum()], dtype=np.float64)


def report(c):
    den = c[1] + .5 * (c[2] + c[3])
    return dict(zip(FIELDS, [float(c[0]), *map(int, c[1:])]),
                pq=float(c[0] / den) if den else 0.,
                sq=float(c[0] / c[1]) if c[1] else 0.,
                rq=float(c[1] / den) if den else 0.)


def image_counts(records, files, keep):
    result = {}
    for f in files:
        item = records[f]
        selection = keep[np.asarray(item["candidate_ids"], dtype=int)]
        total = np.zeros(6)
        for entry in item["annotations"]:
            iou = np.asarray(entry["iou"], dtype=float).reshape(entry["n_gt"], len(selection))
            total += counts(iou[:, selection])
        result[f] = total
    return result


def pooled(records, files, keep):
    return report(sum(image_counts(records, files, keep).values(), start=np.zeros(6)))


def tune_threshold(records, files, probabilities):
    ids = [i for f in files for i in records[f]["candidate_ids"]]
    assert np.isfinite(probabilities[ids]).all()
    sweep = [{"threshold": float(t), **pooled(records, files, probabilities >= t)}
             for t in THRESHOLDS]
    # Stable tie rule: retain more candidates when pooled PQ is identical.
    best = max(sweep, key=lambda x: (x["pq"], -x["threshold"]))
    return best["threshold"], sweep


def make_plan(all_files, validation, temporal_groups, assign_groups):
    groups = temporal_groups(all_files, 24)
    group_id = {f: min(g) for g in groups for f in g}
    chosen = [g for g in groups if set(g) & set(validation)]
    assert set(f for g in chosen for f in g) == set(validation)
    outer = assign_groups(chosen, k=3, seed=20261029)
    jobs, outer_folds = [], []
    for k in range(3):
        train = sorted(f for f in validation if outer[f] != k)
        valid = sorted(set(validation) - set(train))
        inner_groups = [g for g in chosen if g[0] in train]
        inner = assign_groups(inner_groups, k=3, seed=20261100 + k)
        for j in range(3):
            fit = sorted(f for f in train if inner[f] != j)
            query = sorted(set(train) - set(fit))
            jobs.append(dict(id=f"outer{k}_inner{j}", outer=k, inner=j, train=fit, query=query))
        jobs.append(dict(id=f"outer{k}_final", outer=k, inner=None, train=train, query=valid))
        outer_folds.append(dict(fold=k, train=train, validation=valid))
    for job in jobs:
        assert not {group_id[f] for f in job["train"]} & {group_id[f] for f in job["query"]}
    return dict(groups={f: group_id[f] for f in validation}, jobs=jobs, outer=outer_folds,
                thresholds=THRESHOLDS.tolist(), seed=20261029, group_gap_hours=24)


def job_arrays(table, labels, job, features):
    fit_labels = labels[labels.candidate_id.map(table.file).isin(job["train"])]
    query_ids = table.index[table.file.isin(job["query"])].to_numpy()
    train_ids = fit_labels.candidate_id.to_numpy()
    assert not set(train_ids) & set(query_ids)
    return (table.loc[train_ids, features].reset_index(drop=True),
            fit_labels.target.to_numpy(),
            table.loc[query_ids, features].reset_index(drop=True), query_ids)


def evaluate_method(records, plan, predictions, n_candidates):
    keep, outer_probability = np.ones(n_candidates, dtype=bool), np.full(n_candidates, np.nan)
    rows = []
    for fold in plan["outer"]:
        k = fold["fold"]
        inner = np.full(n_candidates, np.nan)
        for j in range(3):
            p = predictions[f"outer{k}_inner{j}"]
            inner[np.asarray(p["ids"], dtype=int)] = p["probabilities"]
        threshold, sweep = tune_threshold(records, fold["train"], inner)
        final = predictions[f"outer{k}_final"]
        ids = np.asarray(final["ids"], dtype=int)
        expected = sorted(i for f in fold["validation"] for i in records[f]["candidate_ids"])
        assert sorted(ids.tolist()) == expected
        values = np.asarray(final["probabilities"])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        keep[ids] = values >= threshold
        outer_probability[ids] = values
        baseline = pooled(records, fold["validation"], np.ones(n_candidates, dtype=bool))
        score = pooled(records, fold["validation"], keep)
        rows.append(dict(fold=k, threshold=threshold, calibration_sweep=sweep,
                         baseline=baseline, score=score, delta=score["pq"] - baseline["pq"]))
    assert np.isfinite(outer_probability).all()
    files = sorted(records)
    before = image_counts(records, files, np.ones(n_candidates, dtype=bool))
    after = image_counts(records, files, keep)
    group_names = sorted(set(plan["groups"].values()))
    a, b = [], []
    for g in group_names:
        members = [f for f in files if plan["groups"][f] == g]
        a.append(sum((before[f] for f in members), start=np.zeros(6)))
        b.append(sum((after[f] for f in members), start=np.zeros(6)))
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(2000):
        indices = rng.integers(len(a), size=len(a))
        deltas.append(report(b[indices].sum(0))["pq"] - report(a[indices].sum(0))["pq"])
    baseline, score = report(a.sum(0)), report(b.sum(0))
    return dict(baseline=baseline, score=score, delta=score["pq"] - baseline["pq"],
                paired_group_bootstrap_95=np.quantile(deltas, [.025, .975]).tolist(),
                positive_folds=sum(row["delta"] > 0 for row in rows), folds=rows,
                candidates_kept=int(keep.sum()), candidates_total=n_candidates,
                by_file={f: dict(baseline=report(before[f]), score=report(after[f])) for f in files},
                outer_predictions=dict(probabilities=outer_probability.tolist(), keep=keep.tolist()),
                warning="Exploratory adaptive base fold0; bootstrap conditions on fitted meta models. "
                        "Independent base fold/seed confirmation is required before adoption.")
