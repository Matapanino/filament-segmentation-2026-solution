"""GC1 fine-tune switches (coordinator 2026-11-14 ~21:1x): --init-from, --freeze-backbone, --whole-image.

All default off: the cfg JSON is byte-identical (hashes in test_gc1_flags.REF) and the draws are unchanged.
"""
import hashlib
import json

import numpy as np
import pytest
import torch

import geom
import train
from test_gc1_flags import BASE, REF, cfg_of
from test_geom import poly_l_shape, raster


@pytest.mark.parametrize('extra', [('--init-from', ''), ('--freeze-backbone', '0'), ('--whole-image', '0')])
def test_explicit_off_is_byte_identical(extra):
    assert hashlib.sha256(json.dumps(cfg_of(*extra)).encode()).hexdigest() == REF[()]


def test_flags_on_are_recorded():
    c = cfg_of('--init-from', '/x/final_ema.safetensors', '--freeze-backbone', '1', '--whole-image', '1',
               '--crop', '2048')
    assert c['init_from'] == '/x/final_ema.safetensors' and c['freeze_backbone'] == 1 and c['whole_image'] == 1


def test_whole_image_needs_crop_2048():
    with pytest.raises(SystemExit):
        cfg_of('--whole-image', '1')


def d4(img, k, hflip, vflip):
    """The whole-image transform as numpy ops: rotate by k*90 deg (geom's theta convention), then flips."""
    out = np.rot90(img, -k)  # theta>0 rotates x->y (clockwise on screen, y down)
    if hflip:
        out = out[:, ::-1]
    if vflip:
        out = out[::-1, :]
    return np.ascontiguousarray(out)


@pytest.mark.parametrize('seed', range(16))
def test_whole_image_draw_is_exact_d4_of_full_image(seed):
    rng0 = np.random.default_rng(100 + seed)
    img = rng0.integers(0, 256, size=(2048, 2048), dtype=np.uint8)
    segs = poly_l_shape(300 + 20 * seed, 1100)
    anns = [dict(segmentation=segs), dict(segmentation=poly_l_shape(1500, 200 + 10 * seed, 0.5))]
    out, masks, p = train.make_whole(img, anns, np.random.default_rng(seed))
    assert p['mode'] == 'whole' and p['scale'] == 1.0 and p['theta'] in (0.0, 90.0, 180.0, 270.0)
    k = int(p['theta'] // 90)
    assert out.shape == (2048, 2048) and np.array_equal(out, d4(img, k, p['hflip'], p['vflip']))
    assert masks.shape == (2, 2048, 2048)
    for i, a in enumerate(anns):
        ref = d4(raster(a['segmentation']), k, p['hflip'], p['vflip'])
        assert (masks[i] != ref).sum() <= 0.01 * ref.sum(), (p, int((masks[i] != ref).sum()), int(ref.sum()))


def test_whole_image_covers_all_eight_d4_views():
    seen = {tuple(train.make_whole(np.zeros((2048, 2048), np.uint8), [], np.random.default_rng(s))[2][k]
                  for k in ('theta', 'hflip', 'vflip')) for s in range(200)}
    assert len(seen) == 16  # 4 rotations x 2 x 2 flips (16 labels for the 8 distinct D4 elements)


def test_crop_batches_whole_image_mode(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, size=(2048, 2048), dtype=np.uint8)
    groups = {'a': [[dict(segmentation=poly_l_shape(400, 400))]], 'b': [[], [dict(segmentation=poly_l_shape(900, 900))]]}
    cfg = dict(train.DEFAULTS, crop=2048, batch=2, total_steps=3, whole_image=1)
    ds = train.CropBatches({'a': img, 'b': img}, groups, ['a', 'b'], cfg)
    x, ms, ps = ds[0]
    assert x.shape == (2, 2048, 2048) and all(p['mode'] == 'whole' for p in ps)
    x2, _, ps2 = ds[0]
    assert np.array_equal(x, x2) and ps == ps2
    # default mode is untouched by the new branch
    cfg0 = dict(train.DEFAULTS, crop=512, batch=2, total_steps=3)
    p0 = train.CropBatches({'a': img, 'b': img}, groups, ['a', 'b'], cfg0)[0][2]
    assert all(p['mode'] in ('instance', 'disk') for p in p0)


def test_init_from_loads_exact_weights(tmp_path):
    from safetensors.torch import save_file
    from test_train import Tiny
    torch.manual_seed(1)
    src = Tiny()
    f = tmp_path / 'final_ema.safetensors'
    save_file({k: v.detach().contiguous() for k, v in src.state_dict().items()}, str(f))
    torch.manual_seed(2)
    dst = Tiny()
    digest = train.init_weights(dst, f)
    assert digest == hashlib.sha256(f.read_bytes()).hexdigest()
    assert all(torch.equal(a, b) for a, b in zip(src.state_dict().values(), dst.state_dict().values()))
