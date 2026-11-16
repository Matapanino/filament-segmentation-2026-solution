"""D4 TTA export (export_tta.py): exact inverse transforms, 1-view path == plain export, same map format."""
import importlib.util
import io
import zipfile

import numpy as np
import pytest
import torch

import eval_ckpt
import export_tta as T

PKG_D4 = str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'src/filseg/reproduction/mf_d4.py')  # this repository's package (was an absolute path)


class Out:
    def __init__(self, cls, masks):
        self.class_queries_logits = cls
        self.masks_queries_logits = masks


class PoolStub(torch.nn.Module):
    """Output = a function of the input content at each 4x4 block (equivariant under D4 when H, W are multiples of 4).
    Queries differ by an affine map of the pooled content, so max_q and the scores matter."""

    def __init__(self, q=5):
        super().__init__()
        self.q = q

    def forward(self, x):
        p = torch.nn.functional.avg_pool2d(x[:, :1], 4)[0, 0]            # [H/4, W/4]
        masks = torch.stack([(p - 30.0 * k) / 50.0 for k in range(self.q)])[None]
        cls = torch.stack([torch.tensor([0.3 * k, -0.2 * k]) for k in range(self.q)])[None]
        return Out(cls, masks)


class FixedStub(torch.nn.Module):
    """Output independent of the input: a fixed asymmetric ramp (tests that inverses undo the forward transforms)."""

    def __init__(self, n):
        super().__init__()
        i, j = np.mgrid[:n, :n]
        self.ramp = torch.from_numpy(((i * 7 + j * 3) % 23 / 23.0 - 0.5).astype(np.float32) * 4)

    def forward(self, x):
        return Out(torch.tensor([[[2.0, -1.0]]]), self.ramp[None, None])


def coord_image(n=64):
    i, j = np.mgrid[:n, :n]
    return torch.from_numpy((i * n + j).astype(np.float32))[None, None]   # unique integer per pixel, exact sums


def test_transforms_match_package_d4():
    spec = importlib.util.spec_from_file_location('mf_d4', PKG_D4)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert T.TRANSFORMS == m.TRANSFORMS
    x = torch.arange(6 * 6, dtype=torch.float32).reshape(1, 1, 6, 6)
    for t in T.TRANSFORMS:
        assert torch.equal(T.forward(x, t), m.forward(x, t))
        assert torch.equal(T.inverse(T.forward(x, t), t), x)


def test_eight_distinct_views_are_the_d4_group():
    x = coord_image(8)
    views = [T.forward(x, t) for t in T.TRANSFORMS]
    keys = {tuple(v.flatten().tolist()) for v in views}
    assert len(keys) == 8
    a = x[0, 0].numpy()
    ref = {tuple(np.rot90(b, k).flatten().tolist()) for b in (a, a.T) for k in range(4)}
    assert keys == ref                                                      # identity, rot90/180/270 and transposes


def test_inverse_exact_equivariant_stub_gives_identity_map():
    x = coord_image(64)
    ident, mean = T.fg_lowres(PoolStub(), x, views=8)
    assert ident.shape == (16, 16)
    for t in T.TRANSFORMS:                                                   # every view, inverted, equals identity
        _, one = T.fg_lowres(PoolStub(), T.forward(x, t), views=1)
        assert torch.equal(T.inverse(one, t), ident)
    assert torch.allclose(mean, ident, rtol=0, atol=1e-6)
    d = np.abs(T.level_map(mean) - T.level_map(ident))                     # mean of 8 equal floats: 1-ulp rounding
    assert d.max() <= 1 and (d > 0).mean() < 1e-3


def test_inverse_direction_fixed_stub_is_orbit_mean():
    n = 16
    stub = FixedStub(n)
    _, mean = T.fg_lowres(stub, torch.zeros(1, 1, 4 * n, 4 * n), views=8)
    s = torch.tensor([2.0, -1.0]).softmax(-1)[0].item()
    f = (1 / (1 + np.exp(-stub.ramp.numpy().astype(np.float64)))) * s
    # inverse of y = rot90(fliplr^b(x), k) applied to the fixed output f: fliplr^b(rot90(f, -k))
    ref = np.mean([np.fliplr(np.rot90(f, -k)) if b else np.rot90(f, -k) for b in (0, 1) for k in range(4)], 0)
    assert np.abs(mean.numpy() - ref).max() < 1e-6
    assert not np.allclose(ref, f)                                           # the test can fail


def plain_level(model, x):
    """eval_ckpt.main's map, verbatim (export of the 180 test images), on the CPU."""
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16):
        o = model(x)
    import decode as D
    scores = D.query_scores(o.class_queries_logits[0])
    fg = (o.masks_queries_logits[0].float().sigmoid() * scores[:, None, None]).amax(0)[None, None]
    return np.rint(torch.nn.functional.interpolate(fg, size=(2048, 2048), mode='bilinear', align_corners=False)
                   [0, 0].cpu().numpy() * 510).astype(np.uint16)


@pytest.mark.parametrize('stub', ['pool', 'fixed'])
def test_one_view_reproduces_plain_export_bytes(tmp_path, stub):
    rng = np.random.default_rng(0)
    if stub == 'pool':
        model, n = PoolStub(), 2048
    else:
        model, n = FixedStub(512), 2048
    xs = {k: torch.from_numpy(rng.random((1, 1, n, n), dtype=np.float32) * 255) for k in ('b', 'a')}
    a = eval_ckpt.MapWriter(tmp_path / 'plain.npz'); b = eval_ckpt.MapWriter(tmp_path / 'tta1.npz')
    for k in sorted(xs):
        a.write(k, plain_level(model, xs[k]))
        ident, mean = T.fg_lowres(model, xs[k], views=1)
        assert torch.equal(ident, mean)
        b.write(k, T.level_map(mean).astype(np.uint16))
    a.close(); b.close()
    assert (tmp_path / 'plain.npz').read_bytes() == (tmp_path / 'tta1.npz').read_bytes()


def test_level_map_format():
    lv = T.level_map(torch.full((512, 512), 0.5))
    assert lv.shape == (2048, 2048) and lv.dtype == np.int64 and lv.min() == lv.max() == 255
    lv = T.level_map(torch.ones(512, 512))
    assert lv.max() == 510


def test_written_npz_members_and_dtype(tmp_path):
    w = eval_ckpt.MapWriter(tmp_path / 'm.npz')
    _, mean = T.fg_lowres(PoolStub(), coord_image(64), views=8)
    w.write('img_a', T.level_map(mean).astype(np.uint16)); w.close()
    with zipfile.ZipFile(tmp_path / 'm.npz') as z:
        assert z.namelist() == ['img_a.npy']
        arr = np.load(io.BytesIO(z.read('img_a.npy')))
    assert arr.dtype == np.uint16 and arr.shape == (2048, 2048)
    with np.load(tmp_path / 'm.npz') as d:
        assert list(d.keys()) == ['img_a']


def test_views_argument_validated():
    with pytest.raises(ValueError):
        T.fg_lowres(PoolStub(), coord_image(8), views=4)


def test_fold_comes_from_cfg():
    assert T.run_fold({'fold': 3}) == 3
    with pytest.raises(SystemExit):
        T.run_fold({})
