"""RLE-based implementation of the host's 2026-10-24 self-evaluation notebook.

Reference: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
Strict IoU > .5; all qualifying pairs; pool counts over annotator-image entries.
This reproduces the published notebook, not undisclosed leaderboard checks.
"""
from dataclasses import dataclass, asdict
import numpy as np
from pycocotools import mask as mu


@dataclass
class Counts:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    split: int = 0
    merge: int = 0

    def add(self, other):
        for key in asdict(self):
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def result(self):
        den = self.tp + .5 * (self.fp + self.fn)
        return {**asdict(self), 'pq': self.iou_sum / den if den else 0.,
                'sq': self.iou_sum / self.tp if self.tp else 0.,
                'rq': self.tp / den if den else 0.}


def counts_from_iou(iou):
    iou = np.asarray(iou, dtype=np.float64)
    if iou.ndim != 2 or not np.isfinite(iou).all():
        raise ValueError('Expected a finite [n_gt, n_pred] matrix')
    hits = iou > .5
    overlap = iou > 0
    return Counts(float(iou[hits].sum()), int(hits.sum()),
                  int((hits.sum(0) == 0).sum()), int((hits.sum(1) == 0).sum()),
                  int((overlap.sum(1) > 1).sum()), int((overlap.sum(0) > 1).sum()))


def annotation_rles(annotations, height=2048, width=2048):
    return [mu.merge(mu.frPyObjects(a['segmentation'], height, width)) for a in annotations]


def evaluate_rles(gt, pred):
    if not gt or not pred:
        return counts_from_iou(np.zeros((len(gt), len(pred))))
    return counts_from_iou(mu.iou(gt, pred, [0] * len(pred)))


def prediction_rles(prob, rule):
    from filseg.instancing import iter_instances
    return [mu.encode(np.asfortranarray(m, dtype=np.uint8)) for m in iter_instances(prob, rule)]
