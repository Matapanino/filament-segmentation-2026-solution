"""Mask2Former builders for arms T (pretrained init) and C (shipped MF2), and the model inputs.

- `build_pretrained` copies `models.py: build()` of the as-run campaign (20261108_architecture_mask2former, snapshot
  91f8ca62...): Swin-T `facebook/mask2former-swin-tiny-coco-instance` at revision 22c4a2f1..., source weight sha256
  3550e8b2..., 1-class head, `ignore_mismatched_sizes=True`, grey image repeated to 3 channels, ImageNet mean/std.
  The wrapper (`Dense.net.model`) keeps the as-run state-dict key layout, so T's EMA weights load like the shipped ones.
- `load_shipped` copies the package loader `pkg/src/filseg/reproduction/models.py: load_network()` (Mask2FormerConfig
  from `weights/mf_config.json`, strict load, file sha256 and tensor-identity checks).
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

PRETRAINED = dict(repository='facebook/mask2former-swin-tiny-coco-instance',
                  revision='22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6',
                  source_weight_sha256='3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80')


class Net(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def encoder(self):
        return self.model.model.pixel_level_module.encoder

    def forward(self, x, targets=None):
        x = x.repeat(1, 3, 1, 1)
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        kw = {} if targets is None else dict(
            mask_labels=targets, class_labels=[torch.zeros(len(y), dtype=torch.long, device=y.device) for y in targets])
        return self.model(pixel_values=(x - mean) / std, **kw)


class Dense(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = Net(model)

    def forward(self, x, targets=None):
        return self.net(x, targets)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


BACKBONES = {'swin-tiny': 'facebook/mask2former-swin-tiny-coco-instance',
             'swin-small': 'facebook/mask2former-swin-small-coco-instance',
             'swin-base': 'facebook/mask2former-swin-base-coco-instance',
             'swin-base-in21k': 'facebook/mask2former-swin-base-IN21k-coco-instance'}


def build_pretrained(backbone='swin-tiny', overrides=None):
    """COCO-instance Mask2Former with a 1-class head. swin-tiny is pinned (revision + sha256, as-run); other backbones
    use the current 'main' revision, whose resolved commit and weight sha256 are recorded in `pretrained_receipt`.
    `overrides` (num_queries, loss weights, train_num_points) are applied to the config before loading."""
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import Mask2FormerForUniversalSegmentation
    repo = BACKBONES[backbone]
    if backbone == 'swin-tiny':
        rev = PRETRAINED['revision']
    else:
        rev = HfApi().model_info(repo).sha
    p = hf_hub_download(repo, 'model.safetensors', revision=rev)
    digest = sha256_file(p)
    if backbone == 'swin-tiny':
        assert digest == PRETRAINED['source_weight_sha256'], 'Pretrained source integrity'
    m = Mask2FormerForUniversalSegmentation.from_pretrained(
        repo, revision=rev, num_labels=1, id2label={0: 'filament'}, label2id={'filament': 0},
        ignore_mismatched_sizes=True, **(overrides or {}))
    d = Dense(m)
    d.pretrained_receipt = dict(repository=repo, revision=rev, weight_sha256=digest)
    return d


CKPT_PARTS = ('backbone', 'pixel', 'decoder')  # GC1: Swin stages, pixel-decoder deformable layers, masked-attn decoder


def ckpt_parts(spec):
    """'all' or a comma list of CKPT_PARTS -> the parts in canonical order."""
    parts = CKPT_PARTS if spec == 'all' else tuple(s.strip() for s in spec.split(',') if s.strip())
    bad = [p for p in parts if p not in CKPT_PARTS]
    if bad or not parts:
        raise ValueError(f'grad_ckpt parts {bad or spec!r} not in {CKPT_PARTS} (or "all")')
    return tuple(p for p in CKPT_PARTS if p in parts)


def _checkpointed(layer, fn):
    orig = layer.forward

    def forward(*args, **kwargs):
        if layer.training and torch.is_grad_enabled():
            return fn(orig, *args, **kwargs)
        return orig(*args, **kwargs)
    layer.forward = forward


def apply_opt_in(dense, cfg):
    """GC1 opt-in switches (absent from cfg = off = nothing touched). Returns a receipt for the log.

    grad_ckpt: activation checkpointing (non-reentrant torch.utils.checkpoint, RNG state preserved) of the selected parts.
      HF 4.57.1 Mask2FormerForUniversalSegmentation does not declare supports_gradient_checkpointing, so: backbone = the
      SwinBackbone's own gradient_checkpointing_enable (per SwinStage); decoder = the 10 GradientCheckpointingLayer
      decoder layers; pixel = the 6 deformable-attention encoder layers, wrapped per instance (state-dict keys unchanged).
    channels_last: conv weights in channels_last memory format (values unchanged)."""
    out = {}
    if cfg.get('grad_ckpt'):
        from functools import partial
        from torch.utils.checkpoint import checkpoint
        fn = partial(checkpoint, use_reentrant=False)
        m = dense.net.model.model
        n = {}
        for part in ckpt_parts(cfg['grad_ckpt']):
            if part == 'backbone':
                enc = m.pixel_level_module.encoder
                enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
                n[part] = sum(1 for x in enc.modules() if type(x).__name__ == 'SwinStage' and x.gradient_checkpointing)
            elif part == 'pixel':
                layers = m.pixel_level_module.decoder.encoder.layers
                for layer in layers:
                    _checkpointed(layer, fn)
                n[part] = len(layers)
            else:
                layers = m.transformer_module.decoder.layers
                for layer in layers:
                    layer.gradient_checkpointing, layer._gradient_checkpointing_func = True, fn
                n[part] = len(layers)
        out['grad_ckpt'] = n
    if cfg.get('freeze_backbone'):
        enc = dense.net.encoder
        for p in enc.parameters():
            p.requires_grad_(False)
        orig = enc.forward

        def frozen_forward(*args, **kwargs):  # eval() (drop_path off) and no_grad: no backbone activations are stored
            was = enc.training
            enc.eval()
            try:
                with torch.no_grad():
                    return orig(*args, **kwargs)
            finally:
                enc.train(was)
        enc.forward = frozen_forward
        out['freeze_backbone'] = sum(p.numel() for p in enc.parameters())
    if cfg.get('channels_last'):
        dense.to(memory_format=torch.channels_last)
        out['channels_last'] = True
    return out


def state_hash(state):
    """pkg models.state_hash."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(x.dtype).encode()); digest.update(str(tuple(x.shape)).encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def config_model(mf_config_path):
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    config = Mask2FormerConfig.from_dict(json.loads(Path(mf_config_path).read_text()))
    config.num_labels = 1; config.id2label = {0: 'filament'}; config.label2id = {'filament': 0}
    return Dense(Mask2FormerForUniversalSegmentation(config))


def load_shipped(weights_file, mf_config_path, row=None):
    """Strict load of a shipped (or T-exported) EMA state; `row` = manifest entry to verify sha256/tensor identity."""
    from safetensors.torch import load_file
    if row is not None:
        assert sha256_file(weights_file) == row['sha256'], 'weights file sha256'
    state = load_file(str(weights_file), device='cpu')
    if row is not None:
        assert state_hash(state) == row['tensor_state_sha256'], 'Tensor identity changed'
    m = config_model(mf_config_path)
    m.load_state_dict(state, strict=True)
    return m.eval()


def view_1024(raw):
    """The shipped MF view (pkg mfmaps.model_input): INTER_AREA 2048 -> 1024, / 255."""
    return cv2.resize(raw, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def view_native(raw):
    return raw.astype(np.float32) / 255
