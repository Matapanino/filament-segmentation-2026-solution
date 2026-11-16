"""End-to-end historical-instrument replay and prospective comparison. CPU only."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,hashlib,tarfile,sys,time,signal,multiprocessing as mp
from pathlib import Path
import cv2,numpy as np
from scipy.special import expit
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parent/'frozen'
sys.path.insert(0,str(ROOT/'eval'))
from quality_features import features_for_mask
from make_time_folds import temporal_groups,assign_groups
from contract import mix,replay_pq,gate
from revision import quality_disjoint
G={}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def unique(name,scope):
    paths=[p for p in Path('/kaggle/input').rglob(name) if scope in str(p)];assert len(paths)==1,(name,scope,len(paths));return paths[0]
def checked(p,row):
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],('INPUT_IDENTITY',p.name)
    with p.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()==row['md5']
    return p
def counts(q):
    hit=q>.5;overlap=q>0
    return np.array([q[hit].sum(),hit.sum(),(hit.sum(0)==0).sum(),(hit.sum(1)==0).sum(),(overlap.sum(1)>1).sum(),(overlap.sum(0)>1).sum()],float)
def init(spec,groups,models,outer,folds,images,prospective):
    cv2.setNumThreads(1);G.update(spec=spec,groups=groups,models=models,outer=outer,folds=folds,images=images,prospective=prospective,z={k:{n:np.load(v['local'],allow_pickle=False) for n,v in rows.items()} for k,rows in spec.items()})
def image(f):
    from filseg.instancing import iter_instances,SHIP_R3
    from pq_metric import annotation_rles
    k=G['folds'][f];raw=cv2.imread(str(G['images']/(f+'.jpeg')),0);assert raw.shape==(2048,2048)
    maps={n:z[f].astype(np.float32)/G['spec'][k][n]['denominator'] for n,z in G['z'][k].items()}
    probs=dict(full=maps['full'],local=maps['local'],blend=(maps['full']+maps['local'])*.5)
    gt=[annotation_rles(e.annotations) for e in G['groups'][f]];gt=[g for g in gt if g];out={}
    for arm in (['incumbent','yolo25'] if k==2 or G['prospective'] else ['incumbent']):
        masks=[np.asarray(m,bool) for m in iter_instances(mix(maps,arm),SHIP_R3)]
        pred=[mu.encode(np.asfortranarray(m,dtype=np.uint8)) for m in masks];features=[features_for_mask(m,probs,raw) for m in masks]
        qs=[np.asarray(mu.iou(g,pred,[0]*len(pred))).reshape(len(g),len(pred)) if pred else np.zeros((len(g),0)) for g in gt];rows={}
        for seed,assignment in G['outer'].items():
            m=G['models'][(seed,assignment[f])];x=np.array([[v[n] for n in m['feature_order']] for v in features],float).reshape(len(features),len(m['feature_order']))
            prob=expit(((x-np.asarray(m['scaler_mean']))/np.asarray(m['scaler_scale']))@np.asarray(m['coef'])[0]+m['intercept'][0]);keep=prob>=m['threshold']
            rows[seed]=sum((counts(q[:,keep]) for q in qs),start=np.zeros(6)).tolist()
        out[arm]=rows
    return f,out
def prepare():
    assert Path('/kaggle/input').is_dir(),'KAGGLE_CPU_ONLY'
    start=time.monotonic();cfg=read(ROOT/'protocol.json');pins=read(ROOT/'evidence/mf_inputs.json');triple=read(ROOT/'evidence/triples_protocol.json');folds=read(ROOT/'evidence/fold_table.json');files=sorted(folds)
    assert all(r['verified'] for r in read(ROOT/'results/eval/MF_remote_manifest_verification.json').values())
    spec={0:{},1:{},2:{}};data=unique('input_identity.json','filament-crossfoldquality-input-1109').parent;assert read(data/'input_identity.json')==pins
    for k in (0,1):
        valid=sorted(f for f in files if folds[f]==k)
        for n in ('full','local'):
            row=pins['original'][f'f{k}_{n}_map'];spec[k][n]=dict(row,local=str(checked(data/row['filename'],row)))
            mr=pins['original'][f'f{k}_{n}_manifest'];m=read(checked(data/mr['filename'],mr));assert sorted(m['validation_files'])==valid and sorted(m['train_files'])==sorted(set(files)-set(valid)) and not m['smoke_subset']
        row=pins['MF'][str(k)];spec[k]['m2f_public']=dict(row,local=str(checked(data/row['filename'],row)))
        m=read(checked(data/row['manifest_filename'],row['manifest']));assert m['fold']==k and m['init_policy']=='public_COCO_only' and m['epochs']==12
        assert sorted(m['train_files'])==sorted(set(files)-set(valid)) and sorted(m['validation_files'])==valid
    for dataset,names in [('filament-archm2f-oof-1108',['full','local','m2f_public']),('filament-archyoloroi-oof-1108',['yolo_roi'])]:
        p=unique('map_specs.json',dataset);catalog=read(p)['maps']
        for name in names:
            expected=triple['new_maps'][name]['sha256'] if name in triple['new_maps'] else triple['reference_maps_sha256'][name]
            matches=[r for r in catalog.values() if r['sha256']==expected];assert len(matches)==1
            row=matches[0];spec[2][name]=dict(row,local=str(checked(p.parent/row['filename'],row)))
    prospective='--prospective' in sys.argv
    if prospective:
        producer=read(ROOT/'new_roi_inputs.json')
        for k in (0,1):
            row=producer[str(k)];p=unique('yolo_roi.npz',f'filament-yolo-xfold-fold{k}-1110');spec[k]['yolo_roi']=dict(row,local=str(checked(p,row)),denominator=510)
    for k,views in spec.items():
        for n,row in views.items():
            with np.load(row['local'],allow_pickle=False) as z:assert sorted(z.files)==sorted(f for f in files if folds[f]==k)
    modelpath=checked(unique('quality_models.json','filament-matchedquality-cpu-v2-1109'),pins['old_quality_models']);saved=read(modelpath)['models'];assert len(saved)==100
    groups24=temporal_groups(files,24);outer={s:assign_groups(groups24,5,s) for s in cfg['partition_seeds']};models={}
    for m in saved:
        s,k=m['partition_seed'],m['outer'];train=sorted(f for f in files if outer[s][f]!=k);query=sorted(set(files)-set(train));quality_disjoint(train,query)
        assert hashlib.sha256(json.dumps(train,separators=(',',':')).encode()).hexdigest()==m['train_identity_sha256']
        assert len(m['feature_order'])==39 and m['classes']==[0,1];models[s,k]=m
    assert set(models)=={(s,k) for s in cfg['partition_seeds'] for k in range(5)}
    query=sorted(f for f in files if folds[f] in (0,1,2));raw=checked(unique('raw_inputs.tar.gz','filament-h1-radial-prepare-cpu-1104'),pins['raw']);images=ROOT/'images';images.mkdir(exist_ok=True);ann=None
    with tarfile.open(raw) as tar:
        for member in tar:
            if not member.isfile():continue
            name=Path(member.name).name
            if '/train_images/' in '/'+member.name and name.endswith('.jpeg') and name[:-5] in query:(images/name).write_bytes(tar.extractfile(member).read())
            elif name=='MAGFiLO_1.0_Annotations_kaggle2026_train.json':ann=tar.extractfile(member).read()
    assert len(list(images.glob('*.jpeg')))==425 and hashlib.sha256(ann).hexdigest()==pins['annotation_sha256']
    from filseg.coco_io import annotation_sets
    groups=annotation_sets(json.loads(ann));per={}
    return spec,groups,models,outer,folds,images,query
