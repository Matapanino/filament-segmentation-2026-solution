"""Freeze one chainF call (order INT1: chain2m = chain2r + --levels, plus chain2u U-Net sets, pcand refit) as a private Kaggle CPU kernel (sources embedded, hashes checked remotely; no credentials).

usage: render_kernel.py <label>      (reads stage/<label>/chain_inputs.json, writes .kernels/<label>/ and state/<label>_render.json)

Payload: case/ = the CV tree (derived from chain/case, CH1; worker.py runs pcombo + the weight arms w<NN> and the ship fits)
+ case/chain_inputs.json; package/ = the package source (pkg/src/filseg, branch repro-package-20261111 @ 3f6524cd);
csv_worker.py. Steps (each in its own interpreter): CV pins -> case tests -> frozen incumbent replay -> worker.py ->
CSV pins (the package's requirements-reproduce-cpu.txt) -> csv_worker.py. Adapted from mfcomb/render_kernel.py and
mfmap_ship/scripts/render_ship.py.
"""
import base64, hashlib, io, json, lzma, re, subprocess, sys, tarfile
from pathlib import Path

H = Path(__file__).resolve().parent
CASE = H / 'case'
PKG = H.parent / 'pkg'
USER = '<kaggle-user>'  # your Kaggle user name
CAP = 43000                      # Kaggle CPU session limit is 12 h; the steps below are capped inside this
FIT_PINS = ['numpy==2.1.3', 'scipy==1.16.3', 'scikit-learn==1.6.1', 'pandas==2.3.3', 'scikit-image==0.25.2', 'opencv-python-headless>=4.14.0,<4.14.1', 'pycocotools']
DATASETS = ['<kaggle-user>/filament-crossfoldquality-input-1109', '<kaggle-user>/filament-archm2f-oof-1108', '<kaggle-user>/filament-archyoloroi-oof-1108',
            '<kaggle-user>/filament-mfqualityconfirm-input-1109', '<kaggle-user>/filament-matchedquality-input-1109', '<kaggle-user>/filament-repro-weights-1111']
KERNELS = ['<kaggle-user>/filament-h1-radial-prepare-cpu-1104', '<kaggle-user>/filament-matchedquality-cpu-v2-1109', '<kaggle-user>/filament-w7s-oof-f3s42-gpu-1104',
           '<kaggle-user>/filament-w7s-oof-f4s42-gpu-1104', '<kaggle-user>/filament-pkgfinal-1113-gpu']
COMPETITION = 'filament-segmentation-2026'
CASE_FILES = ['worker.py', 'mt.py', 'mr.py', 'pc.py', 'mf_features.py', 'neigh.py', 'pmerge.py', 'pcand2.py', 'spatial.py', 'pcand_v1_reference.py', 'run_tests.py',
              'presel.py', 'pcand.py', 'count_plan.py', 'pnat.py', 'refine_ops.py', 'context_features.py', 'rules.py', 'bank_loader.py', 'load_inputs.py',
              'frozen_identity.json', 'protocol.json', 'rulespec.py', 'levels.py', 'pt.py', 'pworker.py']
EXCLUDE_TESTS = {'test_finish.py'}
LABEL = re.compile(r'^[a-z0-9][a-z0-9-]{0,20}$')     # <= 21 chars: the dataset slug filament-mainpc-chainf-<label>-maps stays <= 50


def slug(label, csv_only=False): return f'{USER}/filament-mainpc-chainf-{label}' + ('-csv' if csv_only else '')
def dataset_slug(label): return f'{USER}/filament-mainpc-chainf-{label}-maps'
def combined_dataset_slug(label): return f'{USER}/filament-mainpc-chainf-{label}-cmb'       # chain2s (ported after F0): the combined archives' private dataset


def payload(label, csv_only=False):
    out = {}
    if csv_only:
        for p in sorted((H / 'stage' / label / 'standin_ship').glob('*.json')): out['standin_ship/' + p.name] = p
    for n in CASE_FILES: out['case/' + n] = CASE / n
    for folder in ('frozen', 'tests', 'support', 'evidence'):
        for p in sorted((CASE / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in EXCLUDE_TESTS and 'images' not in p.relative_to(CASE).parts:
                out['case/' + str(p.relative_to(CASE))] = p
    out['case/chain_inputs.json'] = H / 'stage' / label / 'chain_inputs.json'
    for p in sorted((PKG / 'src/filseg').rglob('*')):
        if p.is_file() and p.suffix in ('.py', '.json', '.csv', '.txt') and '__pycache__' not in p.parts:
            out['package/' + str(p.relative_to(PKG / 'src'))] = p
    out['csv_worker.py'] = H / 'csv_worker.py'
    return out


def render(label, csv_only=False):
    """csv_only: the CSV stage alone, with stand-in ship selectors from stage/<label>/standin_ship/ (tools/standin_ship.py):
    a fast check of the test path on Kaggle before a full call; its CSVs are plumbing checks, never results."""
    assert LABEL.match(label), 'label: lowercase letters, digits, hyphens; at most 23 characters'
    pins = json.loads((H / 'stage' / label / 'chain_inputs.json').read_text()); assert pins['label'] == label
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PKG, text=True).strip()
    assert subprocess.check_output(['git', 'status', '--porcelain', 'src'], cwd=PKG, text=True) == '', 'package source must be the committed head'
    csv_pins = (PKG / 'requirements-reproduce-cpu.txt').read_text().split()
    mem = io.BytesIO(); hashes = {}
    with tarfile.open(fileobj=mem, mode='w') as tar:
        for arc, path in sorted(payload(label, csv_only).items()):
            b = Path(path).read_bytes(); hashes[arc] = hashlib.sha256(b).hexdigest()
            info = tarfile.TarInfo(arc); info.size = len(b); tar.addfile(info, io.BytesIO(b))
    data = base64.b64encode(lzma.compress(mem.getvalue(), preset=9 | lzma.PRESET_EXTREME)).decode()
    cap = 5400 if csv_only else CAP; deadline = cap - 600
    code = f'''import os,sys,time,subprocess,tarfile,io,lzma,base64,json,hashlib,traceback,signal,shutil
from pathlib import Path
start=time.time();work=Path('/kaggle/working');out=work/'out';out.mkdir(exist_ok=True);status='complete'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PIP_CACHE_DIR']=str(work/'pip_cache')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
def find(name,scope):
 p=[x for x in Path('/kaggle/input').rglob(name) if scope in str(x)];assert len(p)==1,(name,scope,len(p));return p[0]
def run(argv,cap,cwd,env=None):
 p=subprocess.Popen(argv,cwd=cwd,start_new_session=True,env=env)
 try:
  r=p.wait(timeout=min(cap,max(1,{deadline}-(time.time()-start))))
  if r:raise RuntimeError('Owned step failed: '+str(r)+' '+str(argv[-1]))
 except BaseException:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
  raise
def mark(stage):(out/'stages.json').write_text(json.dumps(dict(stage=stage,elapsed_seconds=time.time()-start)))
try:
 with tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b64decode({data!r})))) as tar:
  for m in tar:
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   p=work/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
 for name,digest in {hashes!r}.items():assert hashlib.sha256((work/name).read_bytes()).hexdigest()==digest,name
 if not {csv_only!r}:
  mark('pip_fit');run([sys.executable,'-m','pip','install','--disable-pip-version-check','-q',*{FIT_PINS!r}],900,work)
  mark('tests');run([sys.executable,'-B','run_tests.py'],2400,work/'case')
  mark('frozen_replay');run([sys.executable,'-B','-u',str(work/'case/frozen/evaluate.py')],1800,work/'case')
  (work/'case/out').mkdir(exist_ok=True);shutil.copy(work/'case/frozen/replay_result.json',work/'case/out/replay_result.json')
  mark('worker');run([sys.executable,'-B','-u','worker.py'],36000,work/'case')
  shutil.rmtree(work/'case/frozen/images',ignore_errors=True)
 ship=work/('standin_ship' if {csv_only!r} else 'case/out/ship')
 mark('pip_csv');run([sys.executable,'-m','pip','install','--disable-pip-version-check','-q',*{csv_pins!r}],900,work)
 weights=find('reproduction_weights.json','filament-repro-weights-1111').parent;maps=find('inference_result.json','filament-pkgfinal-1113-gpu').parent
 images=subprocess.check_output([sys.executable,'-c','from filseg.reproduction.paths import find_test_images;print(find_test_images())'],env=dict(os.environ,PYTHONPATH=str(work/'package')),text=True).strip()
 mark('csv');run([sys.executable,'-B','-u',str(work/'csv_worker.py'),str(work/'package'),images,str(weights),str(maps),'/kaggle/input',str(out/'csv'),str(ship),str(work/'case/chain_inputs.json')],14400,work,dict(os.environ,PYTHONPATH=str(work/'package')))
 mark('done')
except BaseException as e:
 status='failed';(out/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,reason=str(e)[:1000],traceback=traceback.format_exc())))
finally:
 def identity(p):
  b=p.read_bytes();return dict(bytes=len(b),md5=hashlib.md5(b).hexdigest(),sha256=hashlib.sha256(b).hexdigest())
 for junk in ('case/frozen/images','pip_cache','out/csv/unused_test_images'):shutil.rmtree(work/junk,ignore_errors=True)
 for d in (out/'csv').glob('*/test_images') if (out/'csv').is_dir() else []:shutil.rmtree(d,ignore_errors=True)
 files=[p for d in (out,work/'case/out') if d.is_dir() for p in sorted(d.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json']
 (out/'artifact_manifest.json').write_text(json.dumps(dict(status=status,label={label!r},elapsed_seconds=time.time()-start,segmentation_fits=0,segmentation_inference_calls=0,submission_calls=0,
  artifacts={{str(p.relative_to(work)):identity(p) for p in files}})))
assert status=='complete',status
'''
    compile(code, 'kernel', 'exec')
    name = label + ('-csv' if csv_only else '')
    folder = H / '.kernels' / name; folder.mkdir(parents=True, exist_ok=True)
    nb = dict(cells=[dict(cell_type='markdown', metadata={}, source=[f'# filament main-PC chainF {label}: pcombo CV + MF blend-weight arms (optional U-Net sets, levels, P-cand refit) + test CSVs (CPU)\n',
                                                                    'Private. No GPU, no segmentation inference, no submission.']),
                     dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(True))],
              metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')), nbformat=4, nbformat_minor=5)
    (folder / 'run.ipynb').write_text(json.dumps(nb, indent=1))
    datasets = DATASETS + [pins['dataset']]          # chain2r: the pinned dataset (a reused chain2 call's, or dataset_slug(label))
    if 'combined' in pins: datasets.append(pins['combined']['dataset'])     # chain2s: the call's combined archives (both attached)
    if 'test_unet' in pins and pins['test_unet']['dataset'] not in datasets: datasets.append(pins['test_unet']['dataset'])   # chainF: --test-unet (chain3's dataset)
    meta = dict(id=slug(label, csv_only), title=slug(label, csv_only).split('/')[-1], code_file='run.ipynb', language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                enable_tpu=False, enable_internet=True, dataset_sources=datasets, kernel_sources=KERNELS, competition_sources=[COMPETITION], model_sources=[])
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    record = dict(label=label, csv_only=csv_only, slug=meta['id'], cap_seconds=cap, package_head=head, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                  notebook_sha256=hashlib.sha256((folder / 'run.ipynb').read_bytes()).hexdigest(), notebook_bytes=(folder / 'run.ipynb').stat().st_size,
                  payload_files=hashes, fit_pins=FIT_PINS, csv_pins=csv_pins, metadata=meta)
    (H / 'state').mkdir(exist_ok=True); (H / 'state' / f'{name}_render.json').write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    r = render(sys.argv[1], '--csv-only' in sys.argv[2:]); print(json.dumps(dict(slug=r['slug'], files=len(r['payload_files']), notebook_bytes=r['notebook_bytes'], package_head=r['package_head'])))
