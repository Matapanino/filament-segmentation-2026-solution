"""Two-runtime demo: neural maps on GPU, all CSV/replay diagnostics on CPU."""
import json,os,platform,resource,subprocess,tarfile,time
from pathlib import Path
import cv2,numpy as np
from .audit import array_sha,file_hash,identity,verify,compare_arrays
from .assets import resolve_assets
DATA=Path(__file__).parent/'data'
EXPECTED_SHA='4befa16df080ad4f782d50253ab2aa93ab88563614ae45b3040e37ad398a675f'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def test_keys():
    return sorted((Path(__file__).parents[1]/'data/test_filenames.txt').read_text().split())


IMAGE_SUFFIXES=('.jpeg','.jpg','.png')


def any_images(source):
    """{stem: path} of every .jpeg/.jpg/.png under source (new images: no frozen names, no pinned hashes; read_image
    still requires a 2048x2048 grayscale image)."""
    files={}
    for p in sorted(Path(source).rglob('*')):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            if p.stem in files:raise ValueError('Image file stems must be unique (they name the CSV rows): '+p.stem)
            files[p.stem]=p
    if not files:raise ValueError('No .jpeg, .jpg or .png image under '+str(source))
    return files


def prepare_images(source,destination,image_set=None):
    """The test images: the frozen 180 competition test images (hash-checked; image_set None), or image_set='any': every
    image of a folder of new images (any_images)."""
    if image_set=='any':return any_images(source)
    if image_set is not None:raise ValueError("image_set must be None (the competition test set) or 'any'")
    source=Path(source);destination=Path(destination);keys=test_keys()
    if source.is_dir():
        files={p.stem:p for p in source.rglob('*.jpeg') if p.stem in set(keys)}
    else:
        verify(source,dict(bytes=705047904,sha256='0eb760107a522d4485e3d0b26db44fcf5fae826ea6f9f613d383fb2566a8584a'))
        destination.mkdir(parents=True,exist_ok=True);files={}
        with tarfile.open(source) as tar:
            for member in tar:
                name=Path(member.name).name
                if member.isfile() and member.name.startswith('test/test_images/') and name.endswith('.jpeg') and name[:-5] in keys:
                    p=destination/name;p.write_bytes(tar.extractfile(member).read());files[p.stem]=p
    if sorted(files)!=keys:raise ValueError('Test image coverage must be the exact frozen key set')
    image_pins=json.loads((DATA/'test_image_manifest.json').read_text())
    for key,path in files.items():verify(path,image_pins[key])
    return files


def read_image(path):
    raw=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape!=(2048,2048):raise ValueError('Expected native grayscale image')
    return raw


def run_maps(images,weights,output,mf_precision='bf16',budget_seconds=7200,names=None,deterministic=True,image_set=None):
    import torch
    from .models import load_network
    from .inference import unet_map,mf_map
    from .determinism import configure,environment,save_npz,archive_content_sha,module_trace
    if not torch.cuda.is_available():raise RuntimeError('Neural inference requires the authorized GPU runtime')
    if image_set is None and 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('This validation run is restricted to Kaggle T4')
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root,manifest=resolve_assets(weights);files=prepare_images(images,output/'test_images',image_set=image_set)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    settings=configure(deterministic)
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'GPU':torch.cuda.get_device_name(0),'numpy':np.__version__,'opencv':cv2.__version__,'mf_precision':mf_precision,'historical_mf_precision':'bf16','hardware_native_bf16':torch.cuda.get_device_capability()[0]>=8,'determinism':settings,'runtime':environment()}
    result=dict(status='running',environment=env,networks={},preprocessing={},images=len(files),model_fits=0,submission_api_calls=0,started_at=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip())
    if image_set is not None:result['image_set']=image_set
    names=names or ('full','local','MF0','MF1','MF2')
    previous=output/'inference_result.json'
    if previous.exists():
        old=json.loads(previous.read_text());result['networks']=old['networks'];result['preprocessing']=old['preprocessing'];result['environment_by_stage']=old.get('environment_by_stage',{})
    result.setdefault('environment_by_stage',{})
    save(output/'inference_result.json',result)
    for name in names:
        if name in result['networks']:raise RuntimeError('Refusing to repeat an already completed network')
        cv2.setNumThreads(1 if name in ('full','local') else 2)
        result['environment_by_stage'][name]=env
        tick=time.monotonic();torch.cuda.reset_peak_memory_stats();model=load_network(name,root,manifest).cuda();maps={};traces={}
        cfg=json.loads((DATA/(name+'_config.json')).read_text()) if name in ('full','local') else None
        for i,(key,path) in enumerate(sorted(files.items())):
            if time.monotonic()-start>budget_seconds-180:raise TimeoutError('Inference allocation budget reached')
            raw=read_image(path)
            if i==0:
                # Leaf-module output hashes of the first image localise any run-to-run divergence.
                with module_trace(model) as op_trace:q,trace=unet_map(model,raw,cfg['view']) if cfg else mf_map(model,raw,mf_precision)
                save(output/(name+'_module_trace.json'),dict(network=name,image=key,modules=op_trace))
            else:q,trace=unet_map(model,raw,cfg['view']) if cfg else mf_map(model,raw,mf_precision)
            maps[key]=q;traces[key]=dict(raw_file_sha256=file_hash(path),decoded_sha256=array_sha(raw),output_sha256=array_sha(q),**trace)
            if (i+1)%20==0:print(json.dumps(dict(stage=name,done=i+1,total=len(files),elapsed_seconds=time.monotonic()-start)),flush=True)
        path=output/(name+'.npz');save_npz(path,maps)
        result['networks'][name]=dict(file=path.name,**identity(path),content_sha256=archive_content_sha(path),module_trace=identity(output/(name+'_module_trace.json')),elapsed_seconds=time.monotonic()-tick,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
        result['preprocessing'][name]=traces;save(output/'inference_result.json',result)
        del maps,model;torch.cuda.empty_cache()
    result.update(status='complete' if len(result['networks'])==5 else 'partial',elapsed_seconds=sum(x['elapsed_seconds'] for x in result['networks'].values()),peak_host_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(output/'inference_result.json',result);return result


def image_stages(raw,maps,model,instance_stage=None,pnat=False,pcand=None,selector=None,mf_variant=None,views=None,chain=None):
    from .arithmetic import mf_mean,candidate_blend
    from .instancing import iter_instances,SHIP_R3
    from .quality import apply_quality
    from pycocotools import mask as mu
    # Future post-instance stages can wrap this boundary; incumbent has no P-nat.
    if pnat and pcand is not None:raise ValueError('P-nat and P-cand are alternative repair stages')
    if chain is not None:
        # Chain modes (chain.py): the MF map is the level mean of the named archives, blended with weight chain['weight'].
        if mf_variant is not None:raise ValueError('A chain recipe brings its own MF map: drop mf_variant')
        from .chain import chain_mf,weighted_blend
        mf=chain_mf(maps,chain['names'])
        blend=weighted_blend(maps['full'],maps['local'],mf,chain['weight'])
    elif mf_variant is None:
        mf=mf_mean(*(maps[n] for n in ('MF0','MF1','MF2')))
        blend=candidate_blend(*(maps[n] for n in ('full','local','MF0','MF1','MF2')))
    else:
        # MF-map variant (mfmaps.py): the variant's MF probability map replaces the three-fold MF mean everywhere below.
        from .arithmetic import blend_with_mf
        from .mfmaps import variant_mf
        mf=variant_mf(maps,mf_variant)
        blend=blend_with_mf(maps['full'],maps['local'],mf)
    instance_stage=instance_stage or (lambda probability: iter_instances(probability,SHIP_R3))
    masks=[np.asarray(m,bool) for m in instance_stage(blend)]
    occupied=np.zeros(raw.shape,bool);labels=np.zeros(raw.shape,np.int32)
    for i,m in enumerate(masks):
        if not m.any() or (occupied&m).any():raise ValueError('Empty or overlapping SHIP_R3 instance')
        occupied|=m;labels[m]=i+1
    rows,x,probability,keep=apply_quality(masks,maps['full'],maps['local'],raw,model)
    extra={}
    if selector is not None:
        # Optional selector stage: keep/drop from the 39 quality + 15 context features of the unrepaired candidates
        # (refit.py); the shipped LR39 keep is recorded. lr39(model) here gives the old keep bit for bit.
        from . import refit
        from .context_features import context_features
        refit.check(selector)
        context=None
        if refit.needs_context(selector):
            cmaps={'full':maps['full'].astype(np.float32)/255,'local':maps['local'].astype(np.float32)/255,'m2f_public':mf}
            context=[context_features(m,cmaps,raw) for m in masks]
        neighbour=None
        if refit.needs_neighbour(selector):
            # P-neigh: 6 neighbour features of the unrepaired candidates on the candidate blend (neighbour.py)
            from .neighbour import neighbour_features
            neighbour=neighbour_features(masks,blend)
        mfrich=None
        if refit.needs_mfrich(selector):
            # MF-rich: 8 MF-derived features of the unrepaired candidates on the three-fold MF mean (mfrich_features.py),
            # the map the 6 MF context features read
            from .mfrich_features import image_features
            mfrich=image_features(masks,mf)
        view_rows=None
        if refit.needs_views(selector):
            # MF-var: 5 view-disagreement features of the unrepaired candidates from each MF fold model's 8 D4 views
            # (views: MF0, MF1, MF2, each (8, 2048, 2048) uint16), averaged over the three (view_features.py)
            if views is None:raise ValueError('The per-view MF maps are required by this selector')
            from .view_features import test_image_features
            view_rows=test_image_features(masks,views)
        _,selected,new_keep=refit.apply_selector(rows,context,selector,neighbour,mfrich,view_rows)
        extra['selector']=dict(lr39_keep=keep.tolist(),probability=selected.tolist(),keep=new_keep.tolist(),context=context)
        if neighbour is not None:extra['selector']['neighbour']=neighbour
        if mfrich is not None:extra['selector']['mfrich']=mfrich
        if view_rows is not None:extra['selector']['views']=view_rows
        extra['selector_stage']=array_sha(selected)
        keep=new_keep
    if pnat:
        # Optional post-selection stage: quality was decided on the original candidates and is never rescored.
        from .pnat import repair
        kept=[int(j) for j in np.flatnonzero(keep)]
        repaired,stats=repair([masks[j] for j in kept],blend)
        occupied=np.zeros(raw.shape,bool)
        for m in repaired:
            if not m.any() or (occupied&m).any():raise ValueError('Empty or overlapping P-nat mask')
            occupied|=m
        masks=list(masks)
        for j,m in zip(kept,repaired):masks[j]=m
        extra['pnat']=stats
    if pcand is not None:
        # P-cand: the same post-selection repair with one threshold per kept candidate, predicted from the
        # 64 prediction-only features of each kept candidate; quality is never rescored.
        from . import pcand as P
        from .context_features import context_features
        kept=[int(j) for j in np.flatnonzero(keep)];taus=[]
        if kept:
            ctx={'full':maps['full'].astype(np.float32)/255,'local':maps['local'].astype(np.float32)/255,'m2f_public':mf}
            xp=[]
            for j in kept:
                row={**{n:float(rows[j][n]) for n in P.FEATURES[:39]},**context_features(masks[j],ctx,raw),**P.window_features(masks[j],blend)}
                xp.append([row[n] for n in P.FEATURES])
            xp=np.asarray(xp,float)
            if not np.isfinite(xp).all():raise ValueError('Nonfinite P-cand features')
            taus=[float(t) for t in pcand(xp)]
            if len(taus)!=len(kept) or any(t not in P.GRID for t in taus):raise ValueError('P-cand thresholds must be grid values, one per kept candidate')
        repaired,stats=P.repair_taus([masks[j] for j in kept],blend,taus)
        occupied=np.zeros(raw.shape,bool)
        for m in repaired:
            if not m.any() or (occupied&m).any():raise ValueError('Empty or overlapping P-cand mask')
            occupied|=m
        masks=list(masks)
        for j,m in zip(kept,repaired):masks[j]=m
        extra['pcand']=dict(stats,taus=taus)
    rles=[]
    for mask in masks:
        r=mu.encode(np.asfortranarray(mask,dtype=np.uint8))
        if not np.array_equal(mu.decode(r).astype(bool),mask):raise ValueError('RLE roundtrip failed')
        rles.append(r['counts'].decode())
    stages={'MF_mean':array_sha(mf),'candidate_blend':array_sha(blend),'SHIP_R3':array_sha(labels),'features39':array_sha(x),'LR39_probability':array_sha(probability),'keep':array_sha(keep),'RLE':fileless_sha(rles)}
    if 'selector_stage' in extra:stages['selector_probability']=extra.pop('selector_stage')
    return dict(stages=stages,rows=rows,features=x,scores=probability,keep=keep,rles=rles,candidates=len(masks),**extra)


def fileless_sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def compare_instances(actual,reference):
    from collections import Counter
    from scipy.optimize import linear_sum_assignment
    from pycocotools import mask as mu
    a=[dict(size=[2048,2048],counts=r.encode()) for r in actual]
    b=[dict(size=[2048,2048],counts=r.encode()) for r in reference]
    exact=sum((Counter(actual)&Counter(reference)).values())
    iou=mu.iou(a,b,[0]*len(b)) if a and b else np.zeros((len(a),len(b)))
    ii,jj=linear_sum_assignment(-iou) if a and b else ([],[])
    matched=[float(iou[i,j]) for i,j in zip(ii,jj)]
    union_a=mu.decode(mu.merge(a)).astype(bool) if a else np.zeros((2048,2048),bool)
    union_b=mu.decode(mu.merge(b)).astype(bool) if b else np.zeros((2048,2048),bool)
    union=int((union_a|union_b).sum());intersection=int((union_a&union_b).sum())
    return dict(actual_instances=len(a),reference_instances=len(b),exact_instances=exact,exact_reference_fraction=exact/len(b) if b else float(not a),foreground_IoU=intersection/union if union else 1.,assigned_instance_IoU=matched,assigned_IoU_with_unmatched_zero=sum(matched)/max(len(a),len(b)) if a or b else 1.,unmatched_actual=len(a)-len(matched),unmatched_reference=len(b)-len(matched))


def run_finalize(images,weights,maps_root,reference_root,output,historical_aux_root=None):
    """CPU-only paired replay of submitted maps and freshly inferred maps."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1',None):raise RuntimeError('Use a CPU runtime for formatting')
    from .quality import load_quality
    from .csv_io import write_rows,read_rows
    import pandas as pd
    start=time.monotonic();cv2.setNumThreads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root,manifest=resolve_assets(weights);model=load_quality(root/'quality_model.json')
    files=prepare_images(images,output/'test_images');pin=json.loads((DATA/'reference_manifest.json').read_text())
    inferred=json.loads((Path(maps_root)/'inference_result.json').read_text())
    if inferred['status']!='complete':raise ValueError('Inference is incomplete')
    new={};old={}
    for name in ('full','local','MF0','MF1','MF2'):
        path=Path(maps_root)/(name+'.npz');verify(path,inferred['networks'][name]);new[name]=np.load(path,allow_pickle=False)
        row=pin['maps'][name];matches=list(Path(reference_root).rglob(row['filename']));matches=[p for p in matches if p.stat().st_size==row['bytes'] and file_hash(p)==row['sha256']]
        if len(matches)!=1:raise ValueError('Unique exact historical map required: '+name)
        verify(matches[0],row);old[name]=np.load(matches[0],allow_pickle=False)
        if sorted(new[name].files)!=test_keys() or sorted(old[name].files)!=test_keys():raise ValueError('Map key coverage differs')
    result=dict(status='running',exact_csv_match=False,submission_api_calls=0,model_fits=0,images=len(files),per_image={},stage_reference_policy='Map archive hashes and selected CSV are historical. Per-image downstream array hashes are reconstructed from historical maps using frozen source. Aggregate feature/score CSV hashes are checked against historical receipts; preprocessing intermediates were not saved at submission time.',inference_environment=inferred.get('environment_by_stage',inferred['environment']),map_inference_seconds=inferred['elapsed_seconds'])
    rows={};keeps={};controls={};control_keeps={};features=[];scores=[];control_features=[];control_scores=[]
    selected={f:[] for f in files}
    for identifier,rle in read_rows(DATA/'reference_submission.csv'):selected[identifier.rsplit('_',1)[0]].append(rle)
    for i,(key,path) in enumerate(sorted(files.items())):
        raw=read_image(path);a={n:z[key] for n,z in new.items()};b={n:z[key] for n,z in old.items()}
        for group in (a,b):
            for n,q in group.items():
                if q.shape!=(2048,2048) or q.dtype!=(np.uint8 if n in ('full','local') else np.uint16) or q.max()>(255 if n in ('full','local') else 510):raise ValueError('Malformed probability map')
        actual=image_stages(raw,a,model);reference=image_stages(raw,b,model)
        per=dict(raw_image_sha256=file_hash(path),raw_image_matches_historical_sha256=True,preprocessing_historical_hashes_available=False,maps={n:compare_arrays(a[n],b[n]) for n in a},stages={n:dict(actual_sha256=actual['stages'][n],reference_sha256=reference['stages'][n],equal=actual['stages'][n]==reference['stages'][n]) for n in actual['stages']},candidates_actual=actual['candidates'],candidates_reference=reference['candidates'])
        kept=[r for r,k in zip(actual['rles'],actual['keep']) if k]
        per['instances']=compare_instances(kept,selected[key]);result['per_image'][key]=per
        rows[key]=actual['rles'];keeps[key]=actual['keep'];controls[key]=reference['rles'];control_keeps[key]=reference['keep']
        for value,table,scoretable in [(actual,features,scores),(reference,control_features,control_scores)]:
            table.extend(dict(file=key,index=j+1,**v) for j,v in enumerate(value['rows']))
            scoretable.extend(dict(file=key,index=j+1,probability=float(v),keep=bool(k)) for j,(v,k) in enumerate(zip(value['scores'],value['keep'])))
        if (i+1)%10==0:save(output/'reproduction_result.json',result);print(json.dumps(dict(stage='paired_CPU_replay',done=i+1,total=len(files))),flush=True)
    counts=write_rows(output/'submission.csv',sorted(files),rows,keeps)
    write_rows(output/'control_submission.csv',sorted(files),controls,control_keeps)
    for name,data in [('test_features.csv',features),('test_scores.csv',scores),('control_features.csv',control_features),('control_scores.csv',control_scores)]:pd.DataFrame(data).to_csv(output/name,index=False)
    actual_sha=file_hash(output/'submission.csv');control_sha=file_hash(output/'control_submission.csv')
    per=list(result['per_image'].values());stage_names=list(per[0]['stages'])
    result.update(actual_csv_sha256=actual_sha,expected_csv_sha256=EXPECTED_SHA,control_csv_sha256=control_sha,control_exact=control_sha==EXPECTED_SHA,exact_csv_match=actual_sha==EXPECTED_SHA,kept=sum(counts.values()),candidates=len(features),stage_match={n:dict(equal_images=sum(x['stages'][n]['equal'] for x in per),total_images=len(per)) for n in stage_names},map_match={n:dict(equal_images=sum(x['maps'][n]['equal'] for x in per),changed_pixels=sum(x['maps'][n]['changed_pixels'] for x in per),max_abs_difference=max(x['maps'][n]['max_abs_difference'] for x in per)) for n in new},exact_instance_fraction=sum(x['instances']['exact_instances'] for x in per)/sum(x['instances']['reference_instances'] for x in per),elapsed_seconds=time.monotonic()-start,peak_host_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,outputs={n:identity(output/n) for n in ['submission.csv','control_submission.csv','test_features.csv','test_scores.csv','control_features.csv','control_scores.csv']})
    result['map_archive_sha256']={name:dict(actual=inferred['networks'][name]['sha256'],historical=pin['maps'][name]['sha256'],equal=inferred['networks'][name]['sha256']==pin['maps'][name]['sha256']) for name in new}
    result['control_candidates']=len(control_features)
    result['historical_intermediate_audit']={}
    for historical,generated in [('test_features.csv','control_features.csv'),('test_scores.csv','control_scores.csv')]:
        expected=pin['historical_aux_outputs'][historical];actual=identity(output/generated)
        audit=dict(actual=actual,expected=expected,equal=actual==expected)
        if historical_aux_root is not None:
            matches=[p for p in Path(historical_aux_root).rglob(historical) if p.stat().st_size==expected['bytes'] and file_hash(p)==expected['sha256']]
            if len(matches)!=1:raise ValueError('Unique historical feature/score evidence required')
            left=pd.read_csv(output/generated);right=pd.read_csv(matches[0])
            audit['row_identity_equal']=bool(left[['file','index']].equals(right[['file','index']]))
            if audit['row_identity_equal']:
                audit['numeric_max_abs_difference']={col:float(np.max(np.abs(left[col].to_numpy(float)-right[col].to_numpy(float)))) for col in left if col not in ('file','index')}
        result['historical_intermediate_audit'][historical]=audit
    result['control_complete']=result['control_exact'] and len(control_features)==pin['incumbent']['candidate_count'] and all(x['equal'] for x in result['historical_intermediate_audit'].values())
    result['first_divergent_stage']=next((n for n in ('full','local','MF0','MF1','MF2') if result['map_match'][n]['equal_images']!=len(files)),None)
    if result['first_divergent_stage'] is None:result['first_divergent_stage']=next((n for n in stage_names if result['stage_match'][n]['equal_images']!=len(files)),None)
    if result['first_divergent_stage'] is None and not result['control_complete']:
        if len(control_features)!=pin['incumbent']['candidate_count']:result['first_divergent_stage']='historical_SHIP_R3_candidate_count'
        elif not result['historical_intermediate_audit']['test_features.csv']['equal']:result['first_divergent_stage']='historical_features39'
        elif not result['historical_intermediate_audit']['test_scores.csv']['equal']:result['first_divergent_stage']='historical_LR39_probability'
        else:result['first_divergent_stage']='CSV_formatting'
    result['status']='control_failure' if not result['control_complete'] else ('exact_match' if result['exact_csv_match'] else 'mismatch_stop')
    save(output/'reproduction_result.json',result)
    if result['status']!='exact_match':raise RuntimeError('Reproduction not accepted: '+result['status']+'; inspect reproduction_result.json')
    return result


def run_csv(images,weights,maps_root,output,pnat=False,pcand=None,selector=None,mf_variant=None,variant_maps=None,mf_views=False,image_set=None):
    """CPU-only: one submission CSV from one set of freshly inferred maps. pnat=True adds the P-nat stage.

    selector replaces the shipped LR39 keep (refit.py): 'lr39' runs the shipped LR39 through the selector stage
    (must give the same CSV as selector=None), (selector_json_path, sha256) applies a hash-checked refit selector,
    True means the pinned deployment refit selector in the package data, and another name ('pneigh') means the
    selector pinned under that name ('pneigh', 'mfrich', 'pcombo'; refit.deployed(name)).

    pcand adds the P-cand stage instead: a float gives every kept candidate that threshold (pnat.TAU must
    reproduce the P-nat CSV), (regressor_path, sha256) applies the hash-checked deployment regressor, and True
    means (regressor.pkl of the weights dataset, the pinned sha256), the submitted P-cand CSV.

    mf_variant ('mftta', 'mfhires', 'mfcomb'; mfmaps.py) reads the variant's MF map in place of the three-fold MF mean;
    its added archives (MF{k}_tta.npz, MF{k}_tiled.npz) are taken from variant_maps (default: maps_root) and checked
    against mf_views_result.json.

    mf_views=True (the MF-var mode, with mf_variant 'mftta') also reads MF{k}_views.npz, the 8 D4 views of each MF fold
    model, from the same directory (checked against mf_views_result.json) for the 5 view-disagreement selector features."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1',None):raise RuntimeError('Use a CPU runtime for formatting')
    if mf_views and mf_variant!='mftta':raise ValueError('The per-view maps are the views of the MF-TTA map: mf_views needs mf_variant mftta')
    from .quality import load_quality
    from .csv_io import write_rows
    root,manifest=resolve_assets(weights)
    predictor=None
    if pcand is not None:
        from . import pcand as P
        if pcand is True:pcand=P.deployed(root)
        predictor=P.uniform(pcand) if isinstance(pcand,float) else P.load_regressor(*pcand)
    start=time.monotonic();cv2.setNumThreads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=load_quality(root/'quality_model.json')
    chosen=None
    if selector is not None:
        from . import refit
        source=None if selector=='lr39' else refit.deployed() if selector is True else refit.deployed(selector) if isinstance(selector,str) else selector
        chosen=refit.lr39(model) if selector=='lr39' else refit.load_selector(*source)
    files=prepare_images(images,output/'test_images',image_set=image_set)
    keys=test_keys() if image_set is None else sorted(files)   # new images (image_set 'any'): the maps must cover exactly those
    inferred=json.loads((Path(maps_root)/'inference_result.json').read_text())
    if inferred['status']!='complete':raise ValueError('Inference is incomplete')
    maps={}
    for name in ('full','local','MF0','MF1','MF2'):
        path=Path(maps_root)/(name+'.npz');verify(path,inferred['networks'][name]);maps[name]=np.load(path,allow_pickle=False)
        if sorted(maps[name].files)!=keys:raise ValueError('Map key coverage differs: '+name)
    if mf_variant is not None:
        from .mfmaps import load_variant_archives
        maps.update(load_variant_archives(variant_maps or maps_root,mf_variant,keys))
    view_archives=None
    if mf_views:
        from .mfmaps import load_view_archives
        view_archives=load_view_archives(variant_maps or maps_root,keys)
    rows={};keeps={};per={}
    for i,(key,path) in enumerate(sorted(files.items())):
        raw=read_image(path);a={n:z[key] for n,z in maps.items()}
        for n,q in a.items():
            if q.shape!=(2048,2048) or q.dtype!=(np.uint8 if n in ('full','local') else np.uint16) or q.max()>(255 if n in ('full','local') else 510):raise ValueError('Malformed probability map')
        v=None
        if view_archives is not None:
            v=[z[key] for _,z in view_archives]
            if any(q.shape!=(8,2048,2048) or q.dtype!=np.uint16 or q.max()>510 for q in v):raise ValueError('Malformed per-view map')
        before=len(predictor.raw) if predictor is not None else 0
        value=image_stages(raw,a,model,pnat=pnat,pcand=predictor,selector=chosen,mf_variant=mf_variant,views=v)
        rows[key]=value['rles'];keeps[key]=value['keep']
        per[key]=dict(candidates=value['candidates'],kept=int(np.sum(value['keep'])),stages=value['stages'],**({'pnat':value['pnat']} if pnat else {}),**({'selector':value['selector']} if chosen is not None else {}))
        if predictor is not None:per[key]['pcand']=dict(value['pcand'],raw_predictions=[v for r in predictor.raw[before:] for v in r])
        if (i+1)%20==0:print(json.dumps(dict(stage='csv',pnat=pnat,done=i+1,total=len(files),elapsed_seconds=time.monotonic()-start)),flush=True)
    counts=write_rows(output/'submission.csv',sorted(files),rows,keeps)
    result=dict(status='complete',pnat=bool(pnat),pnat_tau=0.4016711682033202 if pnat else None,images=len(files),candidates=sum(x['candidates'] for x in per.values()),kept=sum(counts.values()),
        csv=identity(output/'submission.csv'),map_archives={n:dict(sha256=inferred['networks'][n]['sha256'],content_sha256=inferred['networks'][n].get('content_sha256')) for n in ('full','local','MF0','MF1','MF2')},
        inference_environment=inferred.get('environment'),per_image=per,model_fits=0,submission_api_calls=0,elapsed_seconds=time.monotonic()-start)
    if image_set is not None:result['image_set']=image_set
    if mf_variant is not None:
        from .mfmaps import MF_ARCHIVES,MAP_SETS,map_set_sha256
        added=json.loads((Path(variant_maps or maps_root)/'mf_views_result.json').read_text())['networks']
        archives={n:(added[n] if n in added else inferred['networks'][n])['sha256'] for n in MAP_SETS[mf_variant]}
        result['mf_variant']=dict(name=mf_variant,mf_archives=list(MF_ARCHIVES[mf_variant]),archives=archives,map_set_sha256=map_set_sha256(archives))
        if mf_views:
            views_sha={n:added[n]['sha256'] for n,_ in view_archives}
            result['mf_views']=dict(archives=views_sha,views_set_sha256=map_set_sha256(views_sha))
    if chosen is not None:
        result['selector']=dict(mode='lr39' if selector=='lr39' else 'file',role=chosen['role'],threshold=chosen['threshold'],sha256=None if selector=='lr39' else source[1],
            kept=sum(sum(x['selector']['keep']) for x in per.values()),lr39_kept=sum(sum(x['selector']['lr39_keep']) for x in per.values()))
    if pnat:result['pnat_totals']={k:sum(x['pnat'][k] for x in per.values()) for k in ('kept','fallback','changed','added','removed','seed_pixels')}
    if predictor is not None:
        from . import pcand as P
        taus=[t for x in per.values() for t in x['pcand']['taus']]
        result.update(pcand=dict(mode='uniform' if isinstance(pcand,float) else 'regressor',uniform_tau=pcand if isinstance(pcand,float) else None,
            regressor_sha256=None if isinstance(pcand,float) else pcand[1],grid=list(P.GRID)),
            pcand_totals={k:sum(x['pcand'][k] for x in per.values()) for k in ('kept','fallback','changed','added','removed','seed_pixels')},
            pcand_tau_counts={str(t):sum(v==t for v in taus) for t in P.GRID})
    save(output/'csv_result.json',result);return result
