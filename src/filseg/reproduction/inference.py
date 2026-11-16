"""GPU-only image inference. Original AMP/batch/restore/quantization are explicit."""
from contextlib import nullcontext
import cv2,numpy as np
from .preprocess import input_view,boxes,crop_resize,restore_tiles
from .audit import array_sha


def unet_map(model,raw,spec):
    import torch
    image=input_view(raw,spec);tile_boxes=boxes(2048,spec['extent']);tiles=[];hashes=[]
    with torch.inference_mode():
        for start in range(0,len(tile_boxes),2):
            selected=tile_boxes[start:start+2]
            x=[np.moveaxis(crop_resize(image,b,spec['size']),-1,0) for b in selected]
            x+=[x[-1]]*(2-len(x));batch=np.stack(x).copy();hashes.append(array_sha(batch))
            xt=torch.from_numpy(batch).cuda()
            with torch.autocast('cuda',dtype=torch.float16):logits=model(xt)
            tiles.extend(logits.float().sigmoid().cpu().numpy()[:len(selected),0])
    probability=restore_tiles(tiles,tile_boxes,2048)
    if not np.isfinite(probability).all():raise ValueError('Nonfinite U-Net output')
    return np.rint(probability*255).astype(np.uint8),dict(view_sha256=array_sha(image),batch_sha256=hashes,restored_sha256=array_sha(probability))


def mf_map(model,raw,precision='bf16'):
    import torch
    x=cv2.resize(raw,(1024,1024),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    if precision=='bf16':
        if not torch.cuda.is_bf16_supported():raise RuntimeError('Installed PyTorch cannot execute original MF BF16; FP32 diagnostic must be explicitly selected')
        context=torch.autocast('cuda',dtype=torch.bfloat16)
    elif precision=='fp32':context=nullcontext()
    else:raise ValueError('Allowed MF precision: bf16, fp32')
    with torch.inference_mode(),context:outputs=model(torch.from_numpy(x[None,None].copy()).cuda())
    scores=outputs.class_queries_logits[0].float().softmax(-1)[:,0]
    logits=outputs.masks_queries_logits[0].float()
    foreground=(logits.sigmoid()*scores[:,None,None]).amax(0)[None,None]
    probability=torch.nn.functional.interpolate(foreground,size=(2048,2048),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    if not np.isfinite(probability).all():raise ValueError('Nonfinite MF output')
    return np.rint(probability*510).astype(np.uint16),dict(input_sha256=array_sha(x),restored_sha256=array_sha(probability),precision=precision)
