"""Fresh interpreter per historical GPU environment; no model work on the host."""
import json,os,subprocess,sys,time
from pathlib import Path
UNET=['numpy==2.4.6','opencv-python-headless==4.14.0.94','torch==2.11.0','torchvision==0.26.0','segmentation-models-pytorch==0.5.0','timm==1.0.22','safetensors==0.6.2','kagglehub==1.0.3']
MF=['numpy==2.1.3','opencv-python-headless==4.14.0.94','torch==2.14.1','torchvision==0.29.1','transformers==4.57.1','segmentation-models-pytorch==0.5.0','timm==1.0.22','safetensors==0.6.2','kagglehub==1.0.3']


def run_gpu_pipeline(images,weights,output,mf_precision='bf16',budget_seconds=6600,deterministic=True,image_set=None):
    """image_set='any': new images (pipeline.any_images) on any CUDA GPU, also outside Kaggle; the pinned environments are
    installed into the running interpreter, so use a fresh virtual environment. None: the competition test set on a Kaggle T4."""
    from .determinism import child_environment
    if image_set is None and not Path('/kaggle/input').is_dir():raise RuntimeError('This delivery validation is restricted to Kaggle')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
    if image_set is None and 'T4' not in gpu:raise RuntimeError('Only Kaggle T4 is authorized')
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True);stages=[]
    source=str(Path(__file__).parents[2]);env=child_environment(dict(os.environ,PYTHONPATH=source,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),deterministic)
    for names,requirements in [(('full','local'),UNET),(('MF0','MF1','MF2'),MF)]:
        left=budget_seconds-(time.monotonic()-start)
        if left<1200:raise TimeoutError('Insufficient remaining allocation for environment and inference')
        subprocess.run([sys.executable,'-m','pip','install','-q',*requirements],check=True,timeout=min(900,left))
        if names[0]=='full':
            # Recorded source runtime used cuDNN 92700, overriding the wheel default.
            subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','nvidia-cudnn-cu13==9.27.0.42'],check=True,timeout=300)
        left=budget_seconds-(time.monotonic()-start)
        # A child failure (e.g. a nondeterministic CUDA op under deterministic mode) is saved with its message.
        failure=str(output/('stage_failure_'+names[0]+'.json'))
        code='import json,traceback\ntry:\n from filseg.reproduction.pipeline import run_maps; run_maps('+repr(str(images))+','+repr(str(weights) if weights else None)+','+repr(str(output))+',mf_precision='+repr(mf_precision)+',budget_seconds='+repr(left)+',names='+repr(names)+',deterministic='+repr(bool(deterministic))+(',image_set='+repr(image_set) if image_set is not None else '')+')\nexcept BaseException as e:\n open('+repr(failure)+',"w").write(json.dumps(dict(networks='+repr(list(names))+',error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc())))\n raise'
        subprocess.run([sys.executable,'-u','-c',code],check=True,env=env,timeout=max(1,left))
        stages.append(dict(networks=list(names),pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),elapsed_total_seconds=time.monotonic()-start))
    p=output/'inference_result.json';result=json.loads(p.read_text());result.update(allocation_pipeline_seconds=time.monotonic()-start,environment_installations=stages,deterministic=bool(deterministic),driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,name,memory.total','--format=csv,noheader'],text=True).strip())
    p.write_text(json.dumps(result,indent=2));return result


def run_gpu_mf_views(images,weights,output,mf_precision='bf16',budget_seconds=6600,deterministic=True,views=('tta',)):
    """MF-map variants (mfmaps.py) in a fresh interpreter of the pinned MF environment: MF{k}_tta.npz and MF{k}_tiled.npz.
    Run it after `infer` into the same output folder so that the identity view is checked against MF{k}.npz and the
    variant map-set sha256 are recorded; it never rewrites the package maps."""
    from .determinism import child_environment
    if mf_precision!='bf16':raise ValueError('The MF-map variants use the original MF BF16 only')
    if not Path('/kaggle/input').is_dir():raise RuntimeError('This delivery validation is restricted to Kaggle')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
    if 'T4' not in gpu:raise RuntimeError('Only Kaggle T4 is authorized')
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=str(Path(__file__).parents[2]);env=child_environment(dict(os.environ,PYTHONPATH=source,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),deterministic)
    subprocess.run([sys.executable,'-m','pip','install','-q',*MF],check=True,timeout=min(900,budget_seconds))
    left=budget_seconds-(time.monotonic()-start)
    if left<1200:raise TimeoutError('Insufficient remaining allocation for environment and inference')
    failure=str(output/'stage_failure_mfviews.json')
    code='import json,traceback\ntry:\n from filseg.reproduction.mfmaps import run_mf_view_maps; run_mf_view_maps('+repr(str(images))+','+repr(str(weights) if weights else None)+','+repr(str(output))+',budget_seconds='+repr(left)+',deterministic='+repr(bool(deterministic))+',views='+repr(tuple(views))+')\nexcept BaseException as e:\n open('+repr(failure)+',"w").write(json.dumps(dict(stage="mfviews",error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc())))\n raise'
    subprocess.run([sys.executable,'-u','-c',code],check=True,env=env,timeout=max(1,left))
    p=output/'mf_views_result.json';result=json.loads(p.read_text())
    result.update(allocation_pipeline_seconds=time.monotonic()-start,deterministic=bool(deterministic),pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
                  driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,name,memory.total','--format=csv,noheader'],text=True).strip())
    p.write_text(json.dumps(result,indent=2));return result
