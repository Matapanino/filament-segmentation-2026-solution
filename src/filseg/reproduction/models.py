"""Inference-only strict loaders. No pretrained downloads or training guards."""
import json
from pathlib import Path


def state_hash(state):
    import hashlib
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(x.dtype).encode());digest.update(str(tuple(x.shape)).encode());digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def load_network(name,root,manifest):
    import torch
    from safetensors.torch import load_file
    from .audit import verify
    row=manifest['networks'][name];path=Path(root)/row['file'];verify(path,row)
    state=load_file(str(path),device='cpu')
    if state_hash(state)!=row['tensor_state_sha256']:raise ValueError('Tensor identity changed')
    if name in ('full','local'):
        import segmentation_models_pytorch as smp
        model=smp.Unet(encoder_name='efficientnet-b0',encoder_weights=None,in_channels=3,classes=1)
    else:
        from transformers import Mask2FormerConfig,Mask2FormerForUniversalSegmentation
        config_path=Path(root)/'mf_config.json';verify(config_path,manifest['files']['mf_config.json'])
        cfg=json.loads(config_path.read_text())
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();config=Mask2FormerConfig.from_dict(cfg)
                config.num_labels=1;config.id2label={0:'filament'};config.label2id={'filament':0}
                self.model=Mask2FormerForUniversalSegmentation(config)
            def forward(self,x):
                x=x.repeat(1,3,1,1)
                mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
                std=x.new_tensor([.229,.224,.225])[None,:,None,None]
                return self.model(pixel_values=(x-mean)/std)
        class Dense(torch.nn.Module):
            def __init__(self):super().__init__();self.net=Net()
            def forward(self,x):return self.net(x)
        model=Dense()
    model.load_state_dict(state,strict=True)
    return model.eval()
