"""Native-scale tiled reading of the Mask2Former (MF) foreground map.

The producer reads MF from one 1024-px view of the 2048 image (INTER_AREA), so one output cell (MF returns 1/4 of its
input) covers 8 native px. Here the native 2048 image is cut into 1024x1024 tiles at stride 512 (offsets 0/512/1024,
9 tiles); each tile goes through the producer's forward pass and foreground formula max_q(sigmoid(mask_q) x score_q)
at the tile's 256-px output (one cell = 4 native px). Query sets differ between tiles, so the foreground (not the
per-query masks) is what is blended. The 9 foregrounds are Hann-blended on a 512x512 canvas (4 native px per cell):
1-D window sin^2(pi (i+.5)/256), with the half that faces the image border set to 1 on the first/last tile of each
axis, so the 9 weights sum to 1 at every canvas cell (partition of unity; the division by the weight sum is kept as a
guard). The canvas is upsampled to 2048 (bilinear, align_corners=False, as the producer's 256->2048 step) and
quantised like the stored maps (rint(x 510), uint16).
"""
import math
import numpy as np
import torch

N = 2048
TILE = 1024
STRIDE = 512
SCALE = 4                        # MF output = input / 4
TILE_OUT = TILE // SCALE         # 256
CANVAS = N // SCALE              # 512
DENOMINATOR = 510


def offsets(n=N, tile=TILE, stride=STRIDE):
    return list(range(0, n - tile + 1, stride))


def tiles():
    return [(oy, ox) for oy in offsets() for ox in offsets()]


def window_1d(first, last, length=TILE_OUT):
    i = torch.arange(length, dtype=torch.float64)
    w = torch.sin(math.pi * (i + .5) / length) ** 2
    if first: w[:length // 2] = 1.
    if last: w[length // 2:] = 1.
    return w.float()


def weights():
    o = offsets(); out = []
    for oy, ox in tiles():
        wy = window_1d(oy == o[0], oy == o[-1]); wx = window_1d(ox == o[0], ox == o[-1])
        out.append(((oy, ox), wy[:, None] * wx[None, :]))
    return out


def native_input(raw, device='cuda'):
    """The 2048 grayscale image as the producer scales it (/255), without the 1024 resize."""
    return torch.from_numpy((np.asarray(raw, np.uint8).astype(np.float32) / 255)[None, None].copy()).to(device)


def tiled(fg, x):
    """fg: [1,1,1024,1024] tile -> [256,256] foreground. x: [1,1,2048,2048]. Returns the blended 512x512 canvas (float32)."""
    assert tuple(x.shape[-2:]) == (N, N)
    acc = torch.zeros(CANVAS, CANVAS, dtype=torch.float32, device=x.device); wsum = torch.zeros_like(acc)
    for (oy, ox), w in weights():
        f = fg(x[..., oy:oy + TILE, ox:ox + TILE].contiguous()).float(); assert tuple(f.shape) == (TILE_OUT, TILE_OUT)
        w = w.to(x.device); cy, cx = oy // SCALE, ox // SCALE
        acc[cy:cy + TILE_OUT, cx:cx + TILE_OUT] += w * f; wsum[cy:cy + TILE_OUT, cx:cx + TILE_OUT] += w
    return acc / wsum


def upsample(m, size=N):
    return torch.nn.functional.interpolate(m[None, None].float(), size=(size, size), mode='bilinear', align_corners=False)[0, 0]


def quantise(p):
    a = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
    return np.rint(a * DENOMINATOR).astype(np.uint16)


def hires_mix(stored, tiled_q):
    """The treatment's MF map: mean of the stored OOF map (a) and the tiled map (b), both uint16 / 510, as float32 probability."""
    a = np.asarray(stored); b = np.asarray(tiled_q); assert a.dtype == np.uint16 and b.dtype == np.uint16 and a.shape == b.shape
    return (a.astype(np.float32) + b.astype(np.float32)) / np.float32(2 * DENOMINATOR)
