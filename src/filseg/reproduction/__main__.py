"""Command-line demo entry points; importing the CLI does not load models."""
import argparse,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['infer','csv','finalize','infer-mfviews','chain-maps','chain-maps-compare'])
    p.add_argument('--images',required=True);p.add_argument('--weights');p.add_argument('--output',required=True)
    p.add_argument('--maps');p.add_argument('--references');p.add_argument('--mf-precision',choices=['bf16','fp32'],default='bf16');p.add_argument('--budget-seconds',type=int,default=6600)
    p.add_argument('--nondeterministic',action='store_true',help='disable deterministic mode (enabled by default)');p.add_argument('--pnat',action='store_true',help='add the P-nat stage (off by default)')
    p.add_argument('--pcand',action='store_true',help='add the P-cand stage with the pinned regressor from the weights dataset: the submitted CSV')
    from .modes import MODES,MFMAP_MODES,CHAIN_MODES
    p.add_argument('--mode',choices=list(MODES)+list(MFMAP_MODES)+list(CHAIN_MODES),help='csv: build one named CSV (filseg.reproduction.modes; the notebooks default to pcombo) and check its sha256')
    p.add_argument('--views',nargs='+',choices=['tta','tiled','views'],default=['tta'],help='infer-mfviews: the D4 TTA (default), optionally the tiled native-scale reading and the 8 per-view maps (views; the MF-var mode)')
    p.add_argument('--variant-maps',help='csv with an MF-map mode (mftta, mfhires, mfcomb, mfvar): directory of MF{k}_tta/_tiled/_views.npz and mf_views_result.json (default: --maps)')
    p.add_argument('--any-images',action='store_true',help='infer/csv: --images is a folder of new 2048x2048 full-disk images (any names), not the competition test set; no CSV hash to match')
    p.add_argument('--release',help='csv with a chain mode (ms125, ...): folder of the release files (stored maps; data/chain/release_files.json); default: the repository copy in weights/final (the final submission only)')
    p.add_argument('--maps-from-models',action='store_true',help='csv with a chain mode: --release holds maps re-inferred from the models (same file names); their hashes are recorded instead of required')
    p.add_argument('--recipe',nargs='+',choices=list(CHAIN_MODES),help='chain-maps / chain-maps-compare: the chain recipe(s) whose maps are re-inferred from the release models (filseg.reproduction.chain_maps)')
    p.add_argument('--names',nargs='+',help='chain-maps / chain-maps-compare: only these release file names')
    p.add_argument('--image-keys',nargs='+',help='chain-maps: only these test images (smoke)')
    args=p.parse_args()
    if args.stage in ('chain-maps','chain-maps-compare'):
        if not args.recipe:p.error(args.stage+' needs --recipe')
        from . import chain_maps
        if not args.release:   # default: the models and stored maps of weights/final (the final submission)
            from .chain import repo_release
            found=[repo_release(r) for r in args.recipe]
            if len(args.recipe)!=1 or found[0] is None:p.error(args.stage+' needs --release (the repository holds the files of the final submission only)')
            args.release=str(found[0])
        if args.stage=='chain-maps':result=chain_maps.launch(images=args.images,release_root=args.release,output=args.output,recipe=args.recipe,names=args.names,keys=args.image_keys,budget_seconds=args.budget_seconds,deterministic=not args.nondeterministic,**({'image_set':'any'} if args.any_images else {}))
        else:
            if not args.maps:p.error('chain-maps-compare needs --maps (the folder of re-inferred files)')
            result=chain_maps.compare_maps(args.maps,args.release,args.recipe,names=args.names,output=args.output)
        print(json.dumps({k:v for k,v in result.items() if k in ['status','all_equal','first_differing','elapsed_seconds']},default=str));return
    if (args.release or args.maps_from_models) and args.mode not in CHAIN_MODES:p.error('--release applies to the chain modes')
    if args.variant_maps and args.mode not in MFMAP_MODES:p.error('--variant-maps applies to the MF-map modes')
    if 'tta' not in args.views:p.error('--views must include tta')
    if args.pnat and args.pcand:p.error('--pnat and --pcand are alternative repair stages')
    if args.mode and (args.pnat or args.pcand):p.error('--mode already sets the repair stage; drop --pnat/--pcand')
    if args.mode and args.stage!='csv':p.error('--mode applies to the csv stage')
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    extra={'image_set':'any'} if args.any_images else {}
    if args.stage=='infer':
        from .launch import run_gpu_pipeline
        result=run_gpu_pipeline(args.images,args.weights,args.output,args.mf_precision,args.budget_seconds,deterministic=not args.nondeterministic,**({'image_set':'any'} if args.any_images else {}))
    elif args.stage=='infer-mfviews':
        from .launch import run_gpu_mf_views
        result=run_gpu_mf_views(args.images,args.weights,args.output,args.mf_precision,args.budget_seconds,deterministic=not args.nondeterministic,views=tuple(dict.fromkeys(args.views)))
    elif args.stage=='csv':
        if not args.maps and args.mode not in CHAIN_MODES:p.error('--maps is required for csv')
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from .pipeline import run_csv
        if args.mode in CHAIN_MODES:
            from . import chain
            result=chain.run_chain_csv(images=args.images,weights=args.weights,release=args.release,output=args.output,recipe=args.mode,strict=not args.maps_from_models,**extra)
            result={**result,'mode':args.mode,'expected_csv_sha256':CHAIN_MODES[args.mode]['sha256'],'matches_expected':result['csv']['sha256']==CHAIN_MODES[args.mode]['sha256']}
        elif args.mode in MFMAP_MODES:
            row=MFMAP_MODES[args.mode];expected=row['sha256']
            result=run_csv(args.images,args.weights,args.maps,args.output,**row['kwargs'],**({'variant_maps':args.variant_maps} if args.variant_maps else {}),**extra)
            result={**result,'mode':args.mode,'expected_csv_sha256':expected,'matches_expected':None if expected is None else result['csv']['sha256']==expected}
        elif args.mode:
            expected=MODES[args.mode]['sha256'];result=run_csv(args.images,args.weights,args.maps,args.output,**MODES[args.mode]['kwargs'],**extra)
            result={**result,'mode':args.mode,'expected_csv_sha256':expected,'matches_expected':result['csv']['sha256']==expected}
        else:result=run_csv(args.images,args.weights,args.maps,args.output,pnat=args.pnat,pcand=True if args.pcand else None,**extra)
        if args.any_images and 'matches_expected' in result:result={**result,'expected_csv_sha256':None,'matches_expected':None}   # new images: no submitted CSV to match
        result={**result,'actual_csv_sha256':result['csv']['sha256']}
    else:
        if not args.maps:p.error('--maps is required for finalize')
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from .assets import resolve_assets
        from .pipeline import run_finalize
        root,_=resolve_assets(args.weights)
        result=run_finalize(args.images,root,args.maps,args.references or root,args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ['status','exact_csv_match','mode','actual_csv_sha256','expected_csv_sha256','matches_expected','elapsed_seconds','images']}))
if __name__=='__main__':main()
