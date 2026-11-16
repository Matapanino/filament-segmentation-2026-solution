"""D4 test-time augmentation of the Mask2Former foreground map (w8_tta's transform definition).

forward: y = rot90(flip_W(x) if flip else x, k) on the last two axes; inverse: x = flip_W(rot90(y, -k)) if flip.
The network reads the 1024-px image and returns masks at 1/4 resolution (256 px); 1024 is a multiple of 4, so every
D4 transform maps 4x4 input blocks onto 4x4 blocks and the inverse transform realigns the 256-px output exactly.
tta() returns the identity view (the model on the untouched image) and the mean of the 8 inverse-transformed views.
"""
import numpy as np
import torch

TRANSFORMS = [('id', False, 0), ('rot90', False, 1), ('rot180', False, 2), ('rot270', False, 3),
              ('flipW', True, 0), ('flipW_rot90', True, 1), ('flipW_rot180', True, 2), ('flipW_rot270', True, 3)]
DENOMINATOR = 510


def forward(x, t):
    _, flip, k = t
    y = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(y, k, dims=(-2, -1))


def inverse(y, t):
    _, flip, k = t
    x = torch.rot90(y, -k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def tta(model, x, keep_views=False):
    """model(x) -> 2-D float map. Returns (identity, mean of the 8 inverse-transformed views[, views])."""
    views = []; total = None; ident = None
    for t in TRANSFORMS:
        v = inverse(model(forward(x, t).contiguous()), t).float()
        if t[0] == 'id': ident = v
        total = v.clone() if total is None else total + v
        if keep_views: views.append(v)
    mean = total / len(TRANSFORMS)
    return (ident, mean, views) if keep_views else (ident, mean)


def upsample(m, size=2048):
    return torch.nn.functional.interpolate(m[None, None], size=(size, size), mode='bilinear', align_corners=False)[0, 0]


def quantise(p):
    a = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
    return np.rint(a * DENOMINATOR).astype(np.uint16)
