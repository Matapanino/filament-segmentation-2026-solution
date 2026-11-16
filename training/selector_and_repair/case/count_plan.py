"""Exact historical threshold arithmetic; extracted without model/controller imports."""
import time,hashlib
import numpy as np
from core import counts,report
METRIC_SEMANTICS='frozen_core_all_pairs_strict_gt_0.5_v2'
METRIC_CORE_SHA256=hashlib.sha256(open(counts.__code__.co_filename,'rb').read()).hexdigest()

class CountPlan:
    """Prepared host-count inputs; a safe sparse path plus exact dense fallback."""
    def __init__(self,n):
        self.n=n;self.images={}
        self.stats=dict(annotation_entries=0,fast_entries=0,fallback_entries=0,images_with_fallback=0,
            gt_rows_multiple_qualifying=0,prediction_columns_multiple_qualifying=0,
            gt_rows_multiple_positive_overlap=0,evaluation_calls=0,
            fast_entry_calls=0,fallback_entry_calls=0,evaluation_seconds=0.)
    def provenance(self):
        return dict(semantics=METRIC_SEMANTICS,core_sha256=METRIC_CORE_SHA256,
                    matching='all qualifying pairs; no assignment',
                    fast_path='each GT row has at most one strictly positive IoU',**self.stats)

def count_coefficients(records,n):
    """Prepare exact all-pairs counts, preserving annotation and float-sum order.

    IoU/TP/FP alone are additive. FN and split depend on the retained subset.
    The sparse path is used only when every GT row has <=1 positive overlap;
    arbitrary overlap entries retain the frozen counts(q[:, selected]) path.
    """
    tick=time.monotonic();plan=CountPlan(n);ngt={};seen=set()
    for f,r in records.items():
        ids=np.asarray(r['candidate_ids'],dtype=int)
        assert len(set(ids.tolist()))==len(ids) and ((ids>=0)&(ids<n)).all()
        assert not seen.intersection(ids.tolist());seen.update(ids.tolist())
        entries=[];ngt[f]=0
        for a in r['annotations']:
            q=np.asarray(a['iou'],dtype=np.float64).reshape(a['n_gt'],len(ids))
            assert np.isfinite(q).all()
            hit=q>.5;overlap=q>0
            plan.stats['annotation_entries']+=1
            plan.stats['gt_rows_multiple_qualifying']+=int((hit.sum(1)>1).sum())
            plan.stats['prediction_columns_multiple_qualifying']+=int((hit.sum(0)>1).sum())
            plan.stats['gt_rows_multiple_positive_overlap']+=int((overlap.sum(1)>1).sum())
            if (overlap.sum(1)<=1).all():
                # q[hit] is in GT-row-major order, exactly as in frozen counts.
                # Candidate filtering preserves that order; do not sum by column.
                entry=dict(fast=True,n_gt=a['n_gt'],hit_values=q[hit],hit_columns=np.nonzero(hit)[1],
                           fp=(hit.sum(0)==0),merge=(overlap.sum(0)>1))
                plan.stats['fast_entries']+=1
            else:
                q.setflags(write=False);entry=dict(fast=False,iou=q)
                plan.stats['fallback_entries']+=1
            entries.append(entry);ngt[f]+=a['n_gt']
        plan.stats['images_with_fallback']+=int(any(not entry['fast'] for entry in entries))
        plan.images[f]=(ids,entries)
    plan.stats['preparation_seconds']=time.monotonic()-tick
    return plan,ngt

def evaluate(plan,ngt,records,files,keep):
    tick=time.monotonic();keep=np.asarray(keep,dtype=bool)
    assert keep.shape==(plan.n,)
    rows=[]
    for f in files:
        ids,entries=plan.images[f];selected=keep[ids];total=np.zeros(6,dtype=np.float64)
        for entry in entries:
            if entry['fast']:
                hit_keep=selected[entry['hit_columns']];tp=int(hit_keep.sum())
                value=np.array([entry['hit_values'][hit_keep].sum(),tp,entry['fp'][selected].sum(),
                                entry['n_gt']-tp,0,entry['merge'][selected].sum()],dtype=np.float64)
                plan.stats['fast_entry_calls']+=1
            else:
                value=counts(entry['iou'][:,selected]);plan.stats['fallback_entry_calls']+=1
            total+=value # Preserve frozen core.image_counts annotation order.
        rows.append(total)
    plan.stats['evaluation_calls']+=1;plan.stats['evaluation_seconds']+=time.monotonic()-tick
    return np.asarray(rows,dtype=np.float64).reshape(len(rows),6)

def sum_counts(rows):
    # core.pooled uses a sequential per-image sum, including for threshold ties.
    return sum(rows,start=np.zeros(6,dtype=np.float64))

def threshold_sweep(plan,ngt,records,files,probabilities,grid):
    ids=np.asarray([i for f in files for i in records[f]['candidate_ids']],dtype=int)
    assert np.isfinite(probabilities[ids]).all()
    sweep=[dict(threshold=float(t),**report(sum_counts(evaluate(plan,ngt,records,files,probabilities>=t)))) for t in grid]
    chosen=max(sweep,key=lambda row:(row['pq'],-row['threshold']))
    return chosen['threshold'],sweep
