"""Paired remote CUDA training with immutable original masks and reviewed cores."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument("job");args=parser.parse_args()
    assert Path("/kaggle/input").is_dir(),"Model fits are remote only"
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    from views import input_view,crop_resize,boxes,restore_tiles
    cfg=json.loads(Path(args.job).read_text());protocol=json.loads(Path("protocol.json").read_text())
    cache=json.loads(Path("cache_manifest.json").read_text());root=Path(cache["root"])
    assert torch.cuda.is_available()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    seed=cfg["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda:0");out=Path("artifacts")/cfg["id"];out.mkdir(parents=True,exist_ok=True)
    model=smp.Unet(encoder_name=cfg["encoder"],encoder_weights="imagenet",in_channels=3,classes=1).to(device)
    initial={k:v.detach().cpu() for k,v in model.state_dict().items()}
    initial_digest=hashlib.sha256()
    for key,value in sorted(initial.items()):
        initial_digest.update(key.encode());initial_digest.update(value.numpy().tobytes())
    if cfg.get("expected_initial_state_sha256"):
        assert initial_digest.hexdigest()==cfg["expected_initial_state_sha256"],"Paired initial weights changed"
    torch.save(dict(model=initial),out/"initial.pt")
    del initial
    # Dropout/stochastic-depth RNG is reset identically after initialization.
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"],foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg["epochs"])
    scaler=torch.amp.GradScaler("cuda")
    batch,accumulation=cfg["batch_size"],cfg["accumulation"]
    samples=math.ceil(cfg["samples_per_epoch"]/(batch*accumulation))*batch*accumulation
    spec=cfg["view"];size,extent=spec["size"],spec["extent"]
    corrected=cfg["arm"]=="corrected"
    correction_digest=cache.get("corrections_sha256")
    if corrected:assert correction_digest and cache["n_corrected_images"]>0
    manifest=dict(config=cfg,train_files=protocol["training_files"],validation_files=protocol["validation_files"],
        fold_table_sha256=protocol["fold_table_sha256"],source_sha256=json.loads(Path("wave.json").read_text())["source_sha256"],
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),initial_state_sha256=initial_digest.hexdigest(),
        corrections_sha256=correction_digest if corrected else None,n_corrected_images=cache["n_corrected_images"] if corrected else 0,
        torch=torch.__version__,smp=smp.__version__,device=torch.cuda.get_device_name(0),
        physical_device=os.environ.get("CUDA_VISIBLE_DEVICES"),precision="CUDA FP16 autocast with FP32 supervised losses",
        target="Original separate annotation losses"+(" plus reviewed core positives" if corrected else ""),
        epochs=cfg["epochs"],seed=seed,selection="Final fixed epoch; no validation selection",smoke_subset=False)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))

    @lru_cache(maxsize=4)
    def load(f):
        image=np.load(root/(f+"_image.npy"),mmap_mode="r")
        masks=np.load(root/(f+"_mask.npy"),mmap_mode="r")
        path=root/(f+"_core.npy")
        core=np.load(path,mmap_mode="r") if corrected and path.exists() else None
        return input_view(image,spec),masks,core

    def make_batch(files,rng):
        xs,ys,vs,cs=[],[],[],[]
        for f in files:
            image,masks,core=load(f);x0,y0=rng.integers(2048-extent+1,size=2);box=(x0,y0,x0+extent,y0+extent)
            x=crop_resize(image,box,size);y=np.stack([crop_resize(m,box,size,mask=True) for m in masks])
            c=crop_resize(core,box,size,mask=True) if core is not None else np.zeros((size,size),np.uint8)
            if rng.random()<.5:x,y,c=x[:,::-1],y[:,:,::-1],c[:,::-1]
            if rng.random()<.5:x,y,c=x[::-1],y[:,::-1],c[::-1]
            padded=np.zeros((cache["max_annotations"],size,size),np.float32);padded[:len(y)]=y
            valid=np.zeros(cache["max_annotations"],np.float32);valid[:len(y)]=1
            xs.append(np.moveaxis(x,-1,0).copy());ys.append(padded);vs.append(valid);cs.append(c.copy())
        return [torch.from_numpy(np.stack(v)).to(device) for v in (xs,ys,vs,cs)]

    started=time.monotonic();history=[]
    for epoch in range(cfg["epochs"]):
        tick=time.monotonic();rng=np.random.default_rng(seed+epoch*1009);order=[]
        while len(order)<samples:order.extend(rng.permutation(protocol["training_files"]).tolist())
        order=order[:samples];model.train();optimizer.zero_grad(set_to_none=True)
        totals=torch.zeros(3,device=device);nsteps=samples//batch
        for step in range(nsteps):
            draw=np.random.default_rng(seed+epoch*1000003+step)
            x,targets,valid,cores=make_batch(order[step*batch:(step+1)*batch],draw)
            with torch.autocast("cuda",dtype=torch.float16):logits=model(x)
            z=logits.float();expanded=z.expand_as(targets)
            bce=F.binary_cross_entropy_with_logits(expanded,targets,reduction="none").mean((-2,-1))
            p=expanded.sigmoid();dice=1-(2*(p*targets).sum((-2,-1))+1)/(p.sum((-2,-1))+targets.sum((-2,-1))+1)
            base=(((bce+dice)*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
            weights=cores.float();count=weights.sum((-2,-1));per=(F.softplus(-z[:,0])*weights).sum((-2,-1))/count.clamp_min(1)
            aux=(per*(count>0)).mean()
            loss=base+protocol["auxiliary_weight"]*aux if corrected else base
            scaler.scale(loss/accumulation).backward();totals+=torch.stack((loss.detach(),base.detach(),aux.detach()))
            if (step+1)%accumulation==0:
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if step==0 and epoch==0:
                assert torch.isfinite(loss).item()
                print(json.dumps(dict(job=cfg["id"],first_batch_passed=True,loss=float(loss.detach()),
                    memory_bytes=torch.cuda.max_memory_allocated(),core_pixels=int(count.sum()))),flush=True)
        torch.cuda.synchronize();values=(totals/nsteps).cpu().tolist();assert all(math.isfinite(x) for x in values)
        scheduler.step()
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,epoch=epoch+1),out/"last.pt")
        row=dict(epoch=epoch+1,loss=values[0],original_loss=values[1],core_loss=values[2],seconds=time.monotonic()-tick,
                 samples=samples,optimizer_steps=nsteps//accumulation,scaler=float(scaler.get_scale()))
        history.append(row);(out/"history.json").write_text(json.dumps(history,indent=2));print(json.dumps(dict(job=cfg["id"],**row)),flush=True)
    model.eval();tile_boxes=boxes(2048,extent)
    for kind,files in (("oof",protocol["validation_files"]),("test",cache["test_files"])):
        maps={}
        for number,f in enumerate(sorted(files)):
            if kind=="oof":image,_,_=load(f)
            else:
                raw=cv2.imread(str(Path(cache["test_root"])/(f+".jpeg")),cv2.IMREAD_GRAYSCALE)
                assert raw is not None and raw.shape==(2048,2048);image=input_view(raw,spec)
            tiles=[]
            with torch.no_grad():
                for start in range(0,len(tile_boxes),batch):
                    selected=tile_boxes[start:start+batch];x=[np.moveaxis(crop_resize(image,b,size),-1,0) for b in selected]
                    x+=[x[-1]]*(batch-len(x));xt=torch.from_numpy(np.stack(x).copy()).to(device)
                    with torch.autocast("cuda",dtype=torch.float16):logits=model(xt)
                    tiles.extend(logits.float().sigmoid().cpu().numpy()[:len(selected),0])
            q=np.rint(restore_tiles(tiles,tile_boxes,2048)*255).astype(np.uint8);maps[f]=q
            if (number+1)%30==0:print(json.dumps(dict(job=cfg["id"],kind=kind,done=number+1,total=len(files))),flush=True)
        np.savez_compressed(out/(kind+"_prob.npz"),**maps)
    result=dict(status="inference_complete",id=cfg["id"],arm=cfg["arm"],elapsed_seconds=time.monotonic()-started,
                n_oof=len(protocol["validation_files"]),n_test=len(cache["test_files"]),history=history,initial_state_sha256=manifest["initial_state_sha256"],
                torch=torch.__version__,smp=smp.__version__,worker_sha256=manifest["worker_sha256"])
    (out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=="__main__":main()
