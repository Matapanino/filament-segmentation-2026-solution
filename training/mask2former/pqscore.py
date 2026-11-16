"""Host-parity PQ, copied from the as-run evaluation code.

Sources (campaign archive `detect_20261113/repo_final/campaigns/20261108_architecture_mask2former.tar.gz`, snapshot
91f8ca627224aa2c...):
- `frozen/pq_metric.py` ("RLE-based implementation of the host's 2026-10-24 self-evaluation notebook"):
  `counts_from_iou`, `annotation_rles`, `evaluate_rles` (strict IoU > .5; all qualifying pairs; counts pooled over
  annotator-image entries);
- `evaluate_cpu.py` `image()` / `metric()`: one image is scored against each of its non-empty annotation sets and the
  six counts [iou_sum, tp, fp, fn, split, merge] are summed over the sets; PQ = iou_sum / (tp + .5 (fp + fn)) on the
  counts pooled over images.

The order cites `pkg/train/evaluation/{pq.py, cv_protocol.py, score_csv.py}`; those files are not in the package
clone (see README "Corrections to the order"), so this file is the scorer actually used.
"""
import gzip
import json

import numpy as np
from pycocotools import mask as mu

FIELDS = ['iou_sum', 'tp', 'fp', 'fn', 'split', 'merge']
AREA_BINS = [('lt1024', 0, 1024), ('1024to4096', 1024, 4096), ('gt4096', 4096, float('inf'))]


def counts_from_iou(iou):
    """pq_metric.counts_from_iou, returned as a float vector in FIELDS order."""
    iou = np.asarray(iou, dtype=np.float64)
    if iou.ndim != 2 or not np.isfinite(iou).all():
        raise ValueError('Expected a finite [n_gt, n_pred] matrix')
    hits = iou > .5
    overlap = iou > 0
    return np.array([float(iou[hits].sum()), int(hits.sum()), int((hits.sum(0) == 0).sum()),
                     int((hits.sum(1) == 0).sum()), int((overlap.sum(1) > 1).sum()),
                     int((overlap.sum(0) > 1).sum())], np.float64)


def annotation_rles(annotations, height=2048, width=2048):
    return [mu.merge(mu.frPyObjects(a['segmentation'], height, width)) for a in annotations]


def iou_matrix(gt, pred):
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    return np.asarray(mu.iou(gt, pred, [0] * len(pred)), np.float64).reshape(len(gt), len(pred))


def metric(c):
    """evaluate_cpu.metric."""
    c = np.asarray(c, np.float64)
    den = c[1] + .5 * (c[2] + c[3])
    return dict(zip(FIELDS, map(float, c)), pq=float(c[0] / den) if den else 0.,
                sq=float(c[0] / c[1]) if c[1] else 0., rq=float(c[1] / den) if den else 0.)


def _bin(area):
    for name, lo, hi in AREA_BINS:
        if lo <= area < hi:
            return name
    raise ValueError(area)


def image_counts(entries, pred):
    """entries: list of GT RLE lists (one per non-empty annotation set); pred: list of RLEs.

    Returns (counts[6], bins) where bins holds, per GT-area bin, [n_gt, n_gt_matched, iou_sum_matched] and, per
    prediction-area bin, [n_pred, n_pred_unmatched], summed over the sets (descriptive only).
    """
    c = np.zeros(6)
    bins = {f'gt_{n}': np.zeros(3) for n, _, _ in AREA_BINS}
    bins.update({f'pred_{n}': np.zeros(2) for n, _, _ in AREA_BINS})
    pred_areas = mu.area(pred) if pred else []
    for gt in entries:
        q = iou_matrix(gt, pred)
        c += counts_from_iou(q)
        hits = q > .5
        for i, a in enumerate(mu.area(gt)):
            row = bins['gt_' + _bin(a)]
            row[0] += 1
            if hits[i].any():
                row[1] += 1
                row[2] += q[i][hits[i]].sum()
        for j, a in enumerate(pred_areas):
            row = bins['pred_' + _bin(a)]
            row[0] += 1
            row[1] += 0 if hits[:, j].any() else 1
    return c, bins


def gt_entries(groups, files):
    """{file: [GT RLE list per non-empty annotation set]} (evaluate_cpu: `entries=[a for a in entries if a]`)."""
    out = {}
    for f in files:
        es = [annotation_rles(s.annotations) for s in groups[f]]
        out[f] = [e for e in es if e]
    return out


def read_instances(path):
    """jsonl.gz rows {file, instances: [{size, counts, ...}]} -> {file: [RLE]}."""
    out = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            out[row['file']] = [dict(size=r['size'], counts=r['counts'].encode()) for r in row['instances']]
    return out


def score_set(entries_by_file, preds_by_file, files):
    """Per-file counts [n, 6] (files order), pooled metric and pooled area bins."""
    per = np.zeros((len(files), 6))
    pooled_bins = None
    for i, f in enumerate(files):
        pred = preds_by_file[f]
        c, bins = image_counts(entries_by_file[f], pred)
        per[i] = c
        if pooled_bins is None:
            pooled_bins = {k: v.copy() for k, v in bins.items()}
        else:
            for k, v in bins.items():
                pooled_bins[k] += v
    bins_out = {}
    for k, v in pooled_bins.items():
        if k.startswith('gt_'):
            bins_out[k] = dict(n_gt=int(v[0]), n_matched=int(v[1]), recall=float(v[1] / v[0]) if v[0] else None,
                               sq=float(v[2] / v[1]) if v[1] else None)
        else:
            bins_out[k] = dict(n_pred=int(v[0]), n_unmatched=int(v[1]),
                               unmatched_frac=float(v[1] / v[0]) if v[0] else None)
    return per, metric(per.sum(0)), bins_out


def pq_of(counts):
    """Pooled PQ for count arrays [..., 6] (vectorised over leading axes)."""
    den = counts[..., 1] + .5 * (counts[..., 2] + counts[..., 3])
    return np.where(den > 0, counts[..., 0] / np.where(den > 0, den, 1), 0.)


def paired_bootstrap(t_counts, c_counts, draws=10000, seed=0):
    """Resample images with replacement (same indices for both arms); pooled PQ(T) - PQ(C) per draw."""
    t_counts = np.asarray(t_counts)
    c_counts = np.asarray(c_counts)
    n = len(t_counts)
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for d in range(draws):
        ix = rng.integers(0, n, size=n)
        deltas[d] = pq_of(t_counts[ix].sum(0)) - pq_of(c_counts[ix].sum(0))
    lo, hi = np.quantile(deltas, [.025, .975])
    return dict(delta=float(pq_of(t_counts.sum(0)) - pq_of(c_counts.sum(0))), ci95=[float(lo), float(hi)],
                draws=draws, seed=seed, unit='image', positive_fraction=float((deltas > 0).mean()))
