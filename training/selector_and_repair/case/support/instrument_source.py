"""Remote CPU only: preregistered 5x3 nested group measurement, no lever search."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from core import counts, report
from make_time_folds import temporal_groups, assign_groups
from resume_adapter import atomic,digest


def load_bank(root,folds,seed):
    tables=[];labels=[];records={};offset=0;features=None
    for fold in folds:
        p=Path(root)/f'fold{fold}_{seed}'/'V00_R3'
        m=json.loads((p/'manifest.json').read_text());names=m['base_features']
        assert len(names)==39 and (features is None or names==features);features=names
        t=pd.read_csv(p/'features.csv').set_index('candidate_id');t.index+=offset
        l=pd.read_csv(p/'labels.csv');l['candidate_id']+=offset
        r=json.loads((p/'iou_records.json').read_text())
        for f,item in r.items():
            assert f not in records;item['candidate_ids']=[x+offset for x in item['candidate_ids']];records[f]=item
        tables.append(t);labels.append(l);offset+=len(t)
    table=pd.concat(tables);lab=pd.concat(labels,ignore_index=True)
    assert np.isfinite(table[features].to_numpy()).all()
    return table,lab,records,features


METRIC_SEMANTICS='frozen_core_all_pairs_strict_gt_0.5_v2'
METRIC_CORE_SHA256=digest(counts.__code__.co_filename,'sha256')


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


def predict(table,labels,features,train,query,selector,params,path):
    query_ids=table.index[table.file.isin(query)].to_numpy()
    if path.exists():
        r=json.loads(path.read_text());assert r['train']==train and r['query']==query and r['selector']==selector and r['ids']==query_ids.tolist()
        return query_ids,np.array(r['probabilities']),r['train_rows']
    labs=labels[labels.candidate_id.map(table.file).isin(train)];ids=labs.candidate_id.to_numpy()
    assert not set(ids)&set(query_ids)
    y=labs.target.to_numpy();assert len(y)>0
    if len(np.unique(y))==1:prob=np.full(len(query_ids),float(y[0]))
    else:
        if selector=='logistic':
            p=dict(params);p.pop('preprocessing');model=make_pipeline(StandardScaler(),LogisticRegression(**p))
        else:model=LGBMClassifier(**params)
        model.fit(table.loc[ids,features].to_numpy(),y)
        prob=model.predict_proba(table.loc[query_ids,features].to_numpy())[:,1]
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic(path,dict(train=train,query=query,selector=selector,ids=query_ids.tolist(),probabilities=prob.tolist(),train_rows=len(y)))
    if globals().get('FIT_CHECKPOINT'):FIT_CHECKPOINT(path)
    return query_ids,prob,len(y)


def run(root,folds,seeds,protocol,destination,deadline,progress):
    assert Path('/content').is_dir(),'No real-data aggregation on Mac'
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    version='four_fold' if len(folds)==4 else 'five_fold'
    out=destination/version;out.mkdir(exist_ok=True)
    allrows=[];summaries={};banks={};by_arm={}
    cfg=protocol['phase_C']
    for seed in seeds:
        table,labels,records,features=load_bank(root,folds,seed);files=sorted(records)
        banks[seed]=(table,labels,records,features)
        groups=temporal_groups(files,24);gid={f:min(g) for g in groups for f in g}
        if seed==seeds[0]:shared_files=files
        else:assert files==shared_files
        coef,ngt=count_coefficients(records,len(table));baseline=evaluate(coef,ngt,records,files,np.ones(len(table),bool))
        baseline_report=report(sum_counts(baseline))
        for partition_seed in cfg['partition_seeds']:
            outer=assign_groups(groups,k=5,seed=partition_seed)
            for nfolds in cfg['training_fold_counts']:
                for selector,params in cfg['selectors'].items():
                    key=(seed,nfolds,selector);by_arm.setdefault(key,[])
                    result_path=out/f'{seed}_p{partition_seed}_n{nfolds}_{selector}.json'
                    if result_path.exists():
                        row=json.loads(result_path.read_text())
                        assert row.get('metric_semantics')==METRIC_SEMANTICS and row.get('metric_core_sha256')==METRIC_CORE_SHA256,'Cached partition counts use different metric semantics'
                        allrows.append(row);by_arm[key].append(np.array(row['per_file_counts']));continue
                    keep=np.ones(len(table),bool);outer_rows=[]
                    for k in range(5):
                        if time.time()>deadline:raise TimeoutError('CPU runtime cap reached')
                        selected=[(k+i)%5 for i in range(1,nfolds+1)]
                        train=sorted(f for f in files if outer[f] in selected);valid=sorted(f for f in files if outer[f]==k)
                        assert not {gid[f] for f in train}&{gid[f] for f in valid}
                        inner_groups=[g for g in groups if g[0] in train];inner=assign_groups(inner_groups,k=3,seed=partition_seed+10000+k)
                        probs=np.full(len(table),np.nan);fitdir=out/'fits'/f'{seed}_p{partition_seed}_n{nfolds}_{selector}_outer{k}'
                        for j in range(3):
                            fit=sorted(f for f in train if inner[f]!=j);query=sorted(f for f in train if inner[f]==j)
                            ids,p,_=predict(table,labels,features,fit,query,selector,params,fitdir/f'inner{j}.json');probs[ids]=p
                        ids=np.array([i for f in train for i in records[f]['candidate_ids']],int)
                        assert np.isfinite(probs[ids]).all()
                        tau,sweep=threshold_sweep(coef,ngt,records,train,probs,cfg['threshold_grid'])
                        query_ids,p,nrows=predict(table,labels,features,train,valid,selector,params,fitdir/'final.json');keep[query_ids]=p>=tau
                        if globals().get('FIT_WAVE_CHECKPOINT'):FIT_WAVE_CHECKPOINT(fitdir)
                        b=evaluate(coef,ngt,records,valid,np.ones(len(table),bool));a=evaluate(coef,ngt,records,valid,keep)
                        br,ar=report(sum_counts(b)),report(sum_counts(a))
                        outer_rows.append(dict(outer=k,train_images=len(train),train_rows=nrows,validation_images=len(valid),threshold=tau,keepall=br,selector=ar,delta=ar['pq']-br['pq']))
                    after=evaluate(coef,ngt,records,files,keep);ar=report(sum_counts(after))
                    row=dict(metric_semantics=METRIC_SEMANTICS,metric_core_sha256=METRIC_CORE_SHA256,partition_seed=partition_seed,map_seed=seed,selector=selector,training_folds=nfolds,keepall=baseline_report,
                             score=ar,delta=ar['pq']-baseline_report['pq'],outer_folds=outer_rows,files=files,per_file_counts=after.tolist(),baseline_per_file_counts=baseline.tolist())
                    atomic(result_path,row)
                    if globals().get('PARTITION_CHECKPOINT'):PARTITION_CHECKPOINT(result_path)
                    allrows.append(row);by_arm[key].append(after)
                    progress('instrument',version=version,map_seed=seed,partition_seed=partition_seed,selector=selector,training_folds=nfolds)
        summaries[seed]=dict(metric_evaluation=coef.provenance(),n_images=len(files),n_candidates=len(table),n_annotation_rows=len(labels),keepall=baseline_report,
                             per_base_fold={str(f):dict(feature_means=table[table.file.isin(json.loads((Path(root)/f'fold{f}_{seed}'/'V00_R3/iou_records.json').read_text()))][features].mean().to_dict()) for f in folds})
    spreads={};paired={};composition={}
    for seed in seeds:
        for nf in cfg['training_fold_counts']:
            for selector in cfg['selectors']:
                rows=[r for r in allrows if r['map_seed']==seed and r['training_folds']==nf and r['selector']==selector]
                d=np.array([r['delta'] for r in rows]);spreads[f'{seed}|{nf}|{selector}']=dict(mean=float(d.mean()),sd=float(d.std(ddof=1)),positive_fraction=float((d>0).mean()),train_images=[x['train_images'] for r in rows for x in r['outer_folds']],train_rows=[x['train_rows'] for r in rows for x in r['outer_folds']])
            lr=[r for r in allrows if r['map_seed']==seed and r['training_folds']==nf and r['selector']=='logistic']
            gb=[r for r in allrows if r['map_seed']==seed and r['training_folds']==nf and r['selector']=='lightgbm']
            assert [r['partition_seed'] for r in lr]==[r['partition_seed'] for r in gb]
            d=np.array([b['score']['pq']-a['score']['pq'] for a,b in zip(lr,gb)])
            paired[f'{seed}|{nf}']=dict(deltas=d.tolist(),mean=float(d.mean()),sd=float(d.std(ddof=1)),MDD=float(2*d.std(ddof=1)),positive_fraction=float((d>0).mean()),outer_fold_deltas=[[b['selector']['pq']-a['selector']['pq'] for a,b in zip(x['outer_folds'],y['outer_folds'])] for x,y in zip(lr,gb)])
        rows=[r for r in allrows if r['map_seed']==seed and r['training_folds']==4 and r['selector']=='logistic']
        d=[float(np.std([x['keepall']['pq'] for x in r['outer_folds']],ddof=1)) for r in rows]
        composition[seed]=dict(per_partition_outer_fold_sd=d,mean=float(np.mean(d)),minimum=float(min(d)),maximum=float(max(d)),pooled_keepall_partition_invariant=True)
    seed_axis={}
    for a in seeds:
        for b in seeds:
            if a>=b:continue
            for selector in cfg['selectors']:
                ra=[r for r in allrows if r['map_seed']==a and r['training_folds']==4 and r['selector']==selector]
                rb=[r for r in allrows if r['map_seed']==b and r['training_folds']==4 and r['selector']==selector]
                d=[y['delta']-x['delta'] for x,y in zip(ra,rb)]
                seed_axis[f'{b}-{a}|{selector}']=dict(gain_deltas=d,mean=float(np.mean(d)),sd=float(np.std(d,ddof=1)),
                    keepall_pq_deltas=[y['keepall']['pq']-x['keepall']['pq'] for x,y in zip(ra,rb)],
                    selector_pq_deltas=[y['score']['pq']-x['score']['pq'] for x,y in zip(ra,rb)],
                    outer_fold_gain_deltas=[[v['delta']-u['delta'] for u,v in zip(x['outer_folds'],y['outer_folds'])] for x,y in zip(ra,rb)],
                    outer_fold_keepall_deltas=[[v['keepall']['pq']-u['keepall']['pq'] for u,v in zip(x['outer_folds'],y['outer_folds'])] for x,y in zip(ra,rb)])
            dd=np.array(paired[f'{b}|4']['deltas'])-np.array(paired[f'{a}|4']['deltas'])
            seed_axis[f'{b}-{a}|lightgbm-minus-logistic']=dict(deltas=dd.tolist(),mean=float(dd.mean()),sd=float(dd.std(ddof=1)),
                outer_fold_deltas=(np.array(paired[f'{b}|4']['outer_fold_deltas'])-np.array(paired[f'{a}|4']['outer_fold_deltas'])).tolist())
    result=dict(metric_semantics=METRIC_SEMANTICS,metric_core_sha256=METRIC_CORE_SHA256,status='complete',version=version,R=cfg['R'],selectors=list(cfg['selectors']),map_seeds=seeds,
                summaries=summaries,partitions=[{k:v for k,v in row.items() if k not in ('per_file_counts','baseline_per_file_counts','files')} for row in allrows],composition_spread=composition,selector_gain_spread_and_SD_vs_rows=spreads,
                lightgbm_minus_logistic=paired,training_seed_axis=seed_axis,protocol=protocol,
                no_sqrt_R_scaling=True,resumed_fits='populated by remote_cpu from actual Phase A manifests')
    atomic(out/'instrument.json',result)
    return result
