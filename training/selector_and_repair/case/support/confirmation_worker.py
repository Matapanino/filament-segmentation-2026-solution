"""RemoteCPU fit0 frozen100 scorers, newMF282 dual confirmation only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import glob,hashlib,io,json,multiprocessing as mp,signal,sys,tarfile,time
from pathlib import Path
import cv2,numpy as np,pandas as pd
from scipy.special import expit
from pycocotools import mask as mu
sys.path.insert(0,'frozen')
from extract_cpu import features_for_mask
from core import image_counts
from make_time_folds import temporal_groups,assign_groups
G={}
def read(p):return json.loads(Path(p).read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pq(c):
 d=c[1]+.5*(c[2]+c[3]);return float(c[0]/d) if d else 0.

def unique(pattern,scope=''):
 v=[p for p in glob.glob('/kaggle/input/**/'+pattern,recursive=True) if scope in p];assert len(v)==1,(pattern,scope,v);return Path(v[0])

def checked(path,row):
 assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],('input identity',path)
 assert hashlib.md5(path.read_bytes()).hexdigest()==row['md5'];return path

def init(spec,groups,root,cfg,folds):
 cv2.setNumThreads(1);G.update(z={fold:{k:np.load(v['local'],allow_pickle=False) for k,v in views.items()} for fold,views in spec.items()},spec=spec,groups=groups,root=root,cfg=cfg,folds=folds)

def image(f):
 from filseg.instancing import iter_instances,SHIP_R3
 from pq_metric import annotation_rles
 raw=cv2.imread(str(G['root']/(f+'.jpeg')),0);assert raw.shape==(2048,2048);fold=G['folds'][f]
 native={k:z[f] for k,z in G['z'][fold].items()}
 for k,value in native.items():assert value.shape==(2048,2048) and np.issubdtype(value.dtype,np.unsignedinteger) and value.max()<=G['spec'][fold][k]['denominator']
 maps={k:value.astype(np.float32)/G['spec'][fold][k]['denominator'] for k,value in native.items()};probs={k:maps[k] for k in ('full','local')};probs['blend']=(probs['full']+probs['local'])*.5
 gts=[(e.image_id,annotation_rles(e.annotations)) for e in G['groups'][f]];gts=[(i,g) for i,g in gts if g];result={}
 for name,w in G['cfg']['endpoints'].items():
  p=sum(np.float32(v)*maps[k] for k,v in w.items());masks=[np.asarray(m,bool) for m in iter_instances(p,SHIP_R3)];pred=[mu.encode(np.asfortranarray(m,dtype=np.uint8)) for m in masks];features=[features_for_mask(m,probs,raw) for m in masks];entries=[]
  for iid,gt in gts:
   q=np.asarray(mu.iou(gt,pred,[0]*len(pred))) if pred else np.zeros((len(gt),0));entries.append(dict(image_id=iid,n_gt=len(gt),iou=q.tolist()))
  result[name]=dict(features=features,annotations=entries)
 return f,result

def project(m,x):
 z=(x-np.asarray(m['scaler_mean']))/np.asarray(m['scaler_scale']);return expit(z@np.asarray(m['coef']).T+np.asarray(m['intercept'])).ravel()

def load_bank(inv):
 path=checked(unique('bank.bin','filament-matchedquality-input-1109'),inv['archive']);bank=Path('source');bank.mkdir(exist_ok=True)
 with tarfile.open(path) as tar:
  for name,row in inv['members'].items():
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   b=tar.extractfile(name).read();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];dest=bank/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 tables=[];labels=[];records={};features=None;offset=0
 for fold in range(5):
  d=bank/f'bank/fold{fold}_s42/V00_R3';manifest=read(d/'manifest.json');assert len(manifest['base_features'])==39 and (features is None or features==manifest['base_features']);features=manifest['base_features'];t=pd.read_csv(d/'features.csv').set_index('candidate_id');t.index+=offset;l=pd.read_csv(d/'labels.csv');l.candidate_id+=offset;r=read(d/'iou_records.json')
  for f,v in r.items():assert f not in records;v['candidate_ids']=[i+offset for i in v['candidate_ids']];records[f]=v
  tables.append(t);labels.append(l);offset+=len(t)
 table=pd.concat(tables);labs=pd.concat(labels,ignore_index=True);files=sorted(records);assert len(files)==707 and len(table)==5148 and len(labs)==8402
 assert np.array_equal(table.index,np.arange(len(table))) and set(labs.candidate_id)<=set(table.index)
 return bank,table,labs,records,features,files

def old_partition_records(cfg):
 inventory=read('frozen/strong_inventory.json');mount=unique('inputs.json','filament-publicv4confirm-inputs-1108');mounted=read(mount);out={}
 for seed in cfg['partition_seeds']:
  row=next(v for v in inventory.values() if v['partition_seed']==seed);native=next(v for v in mounted.values() if v['partition_seed']==seed);path=mount.parent/native['filename'];assert sha(path)==row['sha256']
  with tarfile.open(path) as tar:b=tar.extractfile(row['partition_member']).read()
  assert hashlib.sha256(b).hexdigest()==row['partition_sha256'];out[seed]=json.loads(b)
 return out

def artifact(path):
 path=Path(path)
 with path.open('rb') as stream:sha256=hashlib.file_digest(stream,'sha256').hexdigest()
 with path.open('rb') as stream:md5=hashlib.file_digest(stream,'md5').hexdigest()
 return dict(bytes=path.stat().st_size,sha256=sha256,md5=md5)

def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True))
def score_pair(original,orecords,MFtable,MFrecords,features,query,files,groups,old,new,cfg):
 result={}
 for seed in cfg['partition_seeds']:
  assignment=assign_groups(groups,5,seed);keep={a:np.zeros(len(original) if a.startswith('original') else len(MFtable),bool) for a in cfg['arms']}
  for k in range(5):
   valid=sorted(f for f in files if assignment[f]==k);train=sorted(set(files)-set(valid));mo=old[(seed,k)];mn=new[(seed,k)];assert mn['train']==train and mn['query']==valid
   assert mo['train_identity_sha256']==hashlib.sha256(json.dumps(train,separators=(',',':')).encode()).hexdigest()
   for g in groups:assert not (set(g)&set(train) and set(g)&set(valid))
   for m in (mo,mn):assert m['feature_order']==features and m['classes']==[0,1]
   for kind,t in [('original',original),('MF',MFtable)]:
    ids=t.index[t.file.isin(set(valid)&set(query))].to_numpy();values=t.loc[ids,features].to_numpy(float);po=project(mo,values);pn=project(mn,values);assert np.isfinite(po).all() and np.isfinite(pn).all();keep[kind+'_old_nested'][ids]=po>=mo['threshold'];keep[kind+'_new_nested'][ids]=pn>=mn['threshold']
    if kind=='MF':keep['MF_old_global035'][ids]=po>=cfg['old_global_cut'];keep['MF_new_global030'][ids]=pn>=cfg['new_global_cut']
  result[seed]={a:np.asarray(list(image_counts(orecords if a.startswith('original') else MFrecords,query,v).values())) for a,v in keep.items()};print(json.dumps(dict(stage='projection',query_images=len(query),partitions=len(result),fits=0)),flush=True)
 return result

def paired_summary(per,query,groups,cfg):
 arms=cfg['arms'];means={a:np.mean([per[s][a] for s in cfg['partition_seeds']],axis=0) for a in arms};scores={a:pq(v.sum(0)) for a,v in means.items()};gid={f:min(g) for g in groups for f in g};ug=sorted({gid[f] for f in query});ixs=[[i for i,f in enumerate(query) if gid[f]==g] for g in ug];mat={a:np.stack([v[ix].sum(0) for ix in ixs]) for a,v in means.items()};rng=np.random.default_rng(cfg['bootstrap_seed']);boot={a:[] for a in arms}
 for _ in range(cfg['bootstrap_draws']):
  ix=rng.integers(len(ug),size=len(ug))
  for a in arms:boot[a].append(pq(mat[a][ix].sum(0)))
 contrasts={}
 pairs={**cfg['primary_contrasts'],'pipeline_context':['MF_new_nested','original_old_nested'],'original_shift':['original_new_nested','original_old_nested']}
 for name,(a,b) in pairs.items():
  ds=[pq(per[s][a].sum(0))-pq(per[s][b].sum(0)) for s in cfg['partition_seeds']];delta=scores[a]-scores[b];mean=float(np.mean(ds));positive=sum(d>0 for d in ds);ci=np.quantile(np.asarray(boot[a])-np.asarray(boot[b]),[.025,.975]).tolist();contrasts[name]=dict(arms=[a,b],delta=delta,mean_paired_partition_delta=mean,positive_of20=positive,partition_deltas=ds,paired_group_CI95=ci,passes=bool(delta>0 and mean>0 and positive>=16 and ci[0]>0))
 return dict(images=len(query),groups=len(ug),scores=scores,contrasts=contrasts)

def dual_gate(reports):
 assert reports['new282']['images']==282 and reports['fold3']['images']==reports['fold4']['images']==141
 return bool(all(reports['new282']['contrasts'][arm]['passes'] and all(reports['fold'+str(f)]['contrasts'][arm]['delta']>0 and reports['fold'+str(f)]['contrasts'][arm]['mean_paired_partition_delta']>0 for f in (3,4)) for arm in ('A','B')))

def native_original(cfg,pins,root,fold,view):
 declared=cfg['original_map_inventory'][f'fold{fold}_s42'][view];transport=pins['original_transport'][str(fold)][view];scope=cfg['original_kernel_sources'][str(fold)]
 if transport['mode']=='kernel':
  pattern=f'original_{"full_raw_1024" if view=="full" else "local_raw_clahe_detail"}_f{fold}_s42/oof_prob.npz';paths=[Path(p) for p in glob.glob('/kaggle/input/**/'+pattern,recursive=True) if scope in p];assert len(paths)==1;(p,)=paths;p=checked(p,declared['map']);mrow=declared['manifest'];direct=[Path(p) for p in glob.glob('/kaggle/input/**/'+mrow['member'],recursive=True) if scope in p]
  if direct:m=read(checked(direct[0],mrow));assert len(direct)==1
  else:
   archive=unique('diagnostics.tar.gz',scope)
   with tarfile.open(archive) as tar:payload=tar.extractfile(mrow['member']).read()
   assert len(payload)==mrow['bytes'] and hashlib.sha256(payload).hexdigest()==mrow['sha256'] and hashlib.md5(payload).hexdigest()==mrow['md5'];m=json.loads(payload)
 else:
  assert transport['mode']=='opaque';p=checked(root/transport['map_filename'],declared['map']);m=read(checked(root/transport['manifest_filename'],declared['manifest']))
 return dict(declared['map'],local=str(p),denominator=255),m

def main():
 assert Path.cwd().as_posix().startswith('/kaggle/working/'),'Remote CPU only';cfg=read('protocol.json');auth=read('runtime_authorization.json');assert auth['root_review_verified'] and auth['action']=='mf_quality_confirmation_CPU' and auth['enable_gpu'] is False and auth['enable_tpu'] is False
 for name,value in auth['source_sha256'].items():assert sha(name)==value
 assert {'worker.py','protocol.json','input_identity.json',*cfg['required_frozen_filenames']}<=set(auth['source_sha256']);deadline=float(os.environ['JOB_DEADLINE']);started=float(os.environ['CASE_STARTED_AT']);pins=read('input_identity.json');assert auth['case_started_at']==started and deadline<=min(started+900,pins['producer']['global_absolute_deadline'],cfg['producer_global_deadline_upper_bound']);assert time.time()<deadline-30;signal.alarm(max(1,int(deadline-time.time())));start=time.monotonic()
 assert np.__version__==cfg['environment']['numpy'] and cv2.__version__==cfg['environment']['opencv'] and pd.__version__==cfg['environment']['pandas']
 for module in ['scipy','sklearn','skimage']:assert __import__(module).__version__==cfg['environment'][module]
 producer=pins['producer'];assert producer['status']=='complete' and producer['GPU_stopped_verified'] is True and producer['verified_artifacts']==29 and producer['started_at']==cfg['producer_started_at'] and producer['global_absolute_deadline']<=cfg['producer_global_deadline_upper_bound'];assert producer['science_identity']==cfg['producer_science_identity'] and producer['source_payload']==cfg['producer_source_payload'] and set(producer['logical_artifacts'])==set(cfg['producer_expected_logical_artifacts'])
 assert set(producer['artifacts'])==set(cfg['producer_expected_logical_artifacts'])
 for r in producer['artifacts'].values():assert r['verified'] is True and r['owner']=='<owner-email>' and r['bytes']>0 and len(r['sha256'])==64 and len(r['md5'])==32
 assert producer['cut_lock_sha256']==cfg['fixed_cut_identity']['sha256'] and cfg['cut_lock_completed_at']<producer['started_at']
 root=unique('input_identity.json','filament-mfqualityconfirm-input-1109').parent;assert read(root/'input_identity.json')==pins
 fixed=read(checked(unique('fixed_cut.json',cfg['cut_kernel']),cfg['fixed_cut_identity']));assert fixed['source_new_models_sha256']==cfg['input_rows']['mixed_quality_models.json']['sha256'] and fixed['source_MF_features_sha256']==cfg['input_rows']['MF_features.csv']['sha256'] and fixed['source_MF_IoU_sha256']==cfg['input_rows']['MF_iou_records.json']['sha256']
 assert fixed['new_global_cut']==cfg['new_global_cut']==.30 and fixed['old_global_cut']==cfg['old_global_cut']==.35 and fixed['new_MF282_input_used'] is False and fixed['query_images']==425
 inputs={n:checked(unique(n,cfg['refit_kernel']),r) for n,r in cfg['input_rows'].items()};reference=read(inputs['result.json']);cutcounts=read(checked(unique('counts425.json',cfg['cut_kernel']),cfg['cut_counts_identity']));bank,table,_,records,features,files=load_bank(read('frozen/bank_inventory.json'));folds=read('frozen/fold_table.json');groups=temporal_groups(files,24);query=sorted(f for f in files if folds[f] in (3,4));oldquery=reference['query'];assert len(query)==282 and len(oldquery)==425 and not set(query)&set(oldquery) and set(query)|set(oldquery)==set(files)
 old={(m['partition_seed'],m['outer']):m for m in read(checked(unique('quality_models.json',cfg['old_model_kernel']),cfg['old_models_identity']))['models']};new={(m['partition_seed'],m['outer']):m for m in read(inputs['mixed_quality_models.json'])['models']};expected={(s,k) for s in cfg['partition_seeds'] for k in range(5)};assert set(old)==set(new)==expected
 MFtable=pd.read_csv(inputs['MF_features.csv']);MFrecords=read(inputs['MF_iou_records.json']);assert set(MFrecords)==set(oldquery) and set(MFtable.columns)=={'file',*features};controls=score_pair(table,records,MFtable,MFrecords,features,oldquery,files,groups,old,new,cfg);controlmap={'original_old_nested':'original_oldC5','original_new_nested':'original_refitC5','MF_old_nested':'MF_oldC5','MF_new_nested':'MF_refitC5'}
 for s in cfg['partition_seeds']:
  for a,b in controlmap.items():assert np.array_equal(controls[s][a],np.asarray(reference['per_partition_counts'][str(s)][b])),'Development425 exact count replay'
  for a,b in [('MF_old_global035','MF_old_global035'),('MF_new_global030','MF_new_global')]:assert np.array_equal(controls[s][a],np.asarray(cutcounts['per_partition_counts'][str(s)][b])),'Fixed global425 exact replay'
 # Full707 old model/probability/cut/count replay from immutable old partition evidence.
 refs=old_partition_records(cfg);oldfull=[];new_original_full=[]
 for s in cfg['partition_seeds']:
  assignment=assign_groups(groups,5,s);keep=np.zeros(len(table),bool);newkeep=keep.copy()
  for k in range(5):
   m=old[(s,k)];ids=table.index[table.file.map(assignment)==k].to_numpy();cache=read(bank/f'instrument_reads/five_fold/fits/s42_p{s}_n4_logistic_outer{k}/final.json');assert cache['ids']==ids.tolist();p=project(m,table.loc[ids,features].to_numpy(float));assert np.max(np.abs(p-np.asarray(cache['probabilities'])))<cfg['probability_max_abs_error'] and np.array_equal(p>=m['threshold'],np.asarray(cache['probabilities'])>=m['threshold']);keep[ids]=p>=m['threshold'];newkeep[ids]=project(new[(s,k)],table.loc[ids,features].to_numpy(float))>=new[(s,k)]['threshold']
  new_original_full.append(np.asarray(list(image_counts(records,files,newkeep).values())))
  c=np.asarray(list(image_counts(records,files,keep).values()));assert refs[s]['files']==files and np.array_equal(c,np.asarray(refs[s]['arms']['raw_h']['per_file_counts']));oldfull.append(c)
 assert abs(pq(np.mean(oldfull,axis=0).sum(0))-cfg['expected_C5_full707'])<1e-12;write('old_controls.json',dict(status='passed',old707_exact=True,old425_four_nested_exact=True,old425_two_global_exact=True))
 # Only after frozen controls pass may new MF282 arrays/features be decoded.
 spec={3:{},4:{}}
 for fold in (3,4):
  expectedfiles=sorted(f for f in files if folds[f]==fold)
  for view in ('full','local'):
   row,m=native_original(cfg,pins,root,fold,view);assert m['config']['fold']==fold and m['seed']==42 and sorted(m['validation_files'])==expectedfiles and sorted(m['train_files'])==sorted(set(files)-set(expectedfiles)) and not m['smoke_subset'];spec[fold][view]=row
  row=pins['MF'][str(fold)];assert all(row[k]==producer['artifacts'][f'm2f_f{fold}_oof.npz'][k] for k in ('bytes','md5','sha256')) and all(row['manifest'][k]==producer['artifacts'][f'm2f_f{fold}_result.json'][k] for k in ('bytes','md5','sha256'));p=checked(root/row['filename'],row);m=read(checked(root/row['manifest_filename'],row['manifest']));assert m['fold']==fold and m['epochs']==12 and m['epochs_completed']==12 and m['init_policy']=='public_COCO_only' and m['initial_state_sha256']==cfg['expected_MF_initial_state_sha256'] and m['domain_warmstart'] is False and m['worker_sha256']==cfg['producer_science_identity']['gpu_worker.py'] and m['protocol_sha256']==cfg['producer_science_identity']['protocol.json'] and sorted(m['train_files'])==sorted(set(files)-set(expectedfiles)) and sorted(m['validation_files'])==expectedfiles;spec[fold]['m2f_public']=dict(row,local=str(p));assert row['denominator']==510
  for row in spec[fold].values():
   with np.load(row['local'],allow_pickle=False) as z:assert sorted(z.files)==expectedfiles
 raw=checked(unique('raw_inputs.tar.gz',cfg['raw']['kernel']),cfg['raw']);img=Path('images');img.mkdir(exist_ok=True);annotations=None
 with tarfile.open(raw) as tar:
  for member in tar:
   name=Path(member.name).name
   if name.endswith('.jpeg') and name[:-5] in query:(img/name).write_bytes(tar.extractfile(member).read())
   elif name=='MAGFiLO_1.0_Annotations_kaggle2026_train.json':annotations=tar.extractfile(member).read()
 assert len(list(img.glob('*.jpeg')))==282 and hashlib.sha256(annotations).hexdigest()==cfg['annotation_sha256'];from filseg.coco_io import annotation_sets
 annotationgroups=annotation_sets(json.loads(annotations));extracted={}
 with mp.get_context('fork').Pool(4,initializer=init,initargs=(spec,annotationgroups,img,cfg,folds)) as pool:
  for f,v in pool.imap(image,query):
   extracted[f]=v
   if len(extracted)%20==0:print(json.dumps(dict(stage='new_features',done=len(extracted),total=282,fits=0)),flush=True)
 newtables={};featureerror=0.
 for arm in cfg['endpoints']:
  rows=[];rec={}
  for f in query:
   value=extracted[f][arm];ids=list(range(len(rows),len(rows)+len(value['features'])));rec[f]=dict(candidate_ids=ids,annotations=value['annotations']);rows.extend(dict(file=f,**x) for x in value['features'])
   if arm=='original':
    a=table.loc[records[f]['candidate_ids'],features].to_numpy();b=pd.DataFrame(value['features'])[features].to_numpy() if value['features'] else np.empty((0,len(features)));assert a.shape==b.shape;featureerror=max(featureerror,float(np.max(np.abs(a-b))) if a.size else 0.);assert featureerror<cfg['feature_max_abs_error'] and records[f]['annotations']==value['annotations']
  t=pd.DataFrame(rows,columns=['file',*features]);newtables[arm]=(t,rec);t.to_csv('features_'+arm+'_282.csv',index=False)
 fresh=score_pair(*newtables['original'],*newtables['MF25_original75'],features,query,files,groups,old,new,cfg)
 for i,s in enumerate(cfg['partition_seeds']):
  ix=[files.index(f) for f in query];assert np.array_equal(fresh[s]['original_old_nested'],oldfull[i][ix]) and np.array_equal(fresh[s]['original_new_nested'],new_original_full[i][ix]),'Original282 feature/projection transport exact'
 reports={'new282':paired_summary(fresh,query,groups,cfg)}
 for fold in (3,4):
  subset=[f for f in query if folds[f]==fold];ix=[query.index(f) for f in subset];part={s:{a:v[ix] for a,v in arms.items()} for s,arms in fresh.items()};reports['fold'+str(fold)]=paired_summary(part,subset,groups,cfg)
 # Old425 is descriptive only; never rescue a failed new282 gate.
 allquery=sorted(files);combined={s:{a:np.vstack([(controls[s][a][oldquery.index(f)] if f in set(oldquery) else fresh[s][a][query.index(f)]) for f in allquery]) for a in cfg['arms']} for s in cfg['partition_seeds']};reports['descriptive707']=paired_summary(combined,allquery,groups,cfg);passed=dual_gate(reports);write('counts282.json',dict(query=query,per_partition_counts={str(s):{a:c.tolist() for a,c in arms.items()} for s,arms in fresh.items()}));write('counts707_descriptive.json',dict(query=allquery,per_partition_counts={str(s):{a:c.tolist() for a,c in arms.items()} for s,arms in combined.items()}));write('new_MF282_iou_records.json',newtables['MF25_original75'][1]);outputs=['old_controls.json','features_original_282.csv','features_MF25_original75_282.csv','counts282.json','counts707_descriptive.json','new_MF282_iou_records.json']
 result=dict(status='complete',case='mf_quality_confirmation',model_fits=0,base_model_fits=0,quality_refits=0,deployment_fits=0,query_images=282,R=20,features=39,old707_control_exact=True,old425_four_nested_control_exact=True,old425_two_global_control_exact=True,original282_features_and_IoUs_exact=True,feature_max_abs_error=featureerror,new_global_cut=.30,old_global_cut=.35,fixed_cut_sha256=cfg['fixed_cut_identity']['sha256'],subsets=reports,qualifies_confirmation=passed,adopt=False,submission=False,rawGT_pristine=False,MF_candidate_distribution_unseen=True,MF707_refit=False,test_recipe_changed=False,artifacts={n:artifact(n) for n in outputs},protocol_sha256=sha('protocol.json'),input_identity_sha256=sha('input_identity.json'),elapsed_seconds=time.monotonic()-start,limitation='Original282GT historically used; each queried group excluded from its scorer fit/cut; not pristine rawGT. Full707 deployfit1 remains separate.');assert time.time()<deadline;write('result.json',result);print(json.dumps(dict(status='complete',qualifies_confirmation=passed,subsets=reports,seconds=result['elapsed_seconds'])),flush=True)
if __name__=='__main__':main()
