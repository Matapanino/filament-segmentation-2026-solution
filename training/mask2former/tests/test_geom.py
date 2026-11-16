"""Geometry tests: image and instance masks stay aligned under rotation, scale jitter, flips and cropping."""
import numpy as np
import pytest
from pycocotools import mask as mu

import geom


def poly_l_shape(x0, y0, s=1.0):
    # asymmetric L so that flips/rotations are distinguishable
    pts = np.array([[0, 0], [300, 0], [300, 60], [80, 60], [80, 200], [0, 200]], float) * s + [x0, y0]
    return [pts.reshape(-1).tolist()]


def raster(segs, h=2048, w=2048):
    return mu.decode(mu.merge(mu.frPyObjects(segs, h, w)))


def test_identity_crop_is_exact_slice():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8)
    p = dict(theta=0.0, scale=1.0, hflip=False, vflip=False, cx=1000.0, cy=800.0)
    out = geom.warp_image(img, geom.continuous_matrix(p))
    assert np.array_equal(out, img[800 - 256:800 + 256, 1000 - 256:1000 + 256])


@pytest.mark.parametrize('theta', [90.0, 180.0, 270.0])
@pytest.mark.parametrize('hflip,vflip', [(False, False), (True, False), (False, True), (True, True)])
def test_right_angle_warp_is_exact_permutation(theta, hflip, vflip):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8)
    p = dict(theta=theta, scale=1.0, hflip=hflip, vflip=vflip, cx=1024.0, cy=900.0)
    m = geom.continuous_matrix(p)
    out = geom.warp_image(img, m)
    mi = geom.image_matrix(m)
    inv = np.linalg.inv(np.vstack([mi, [0, 0, 1]]))
    v, u = np.mgrid[0:512:37, 0:512:41]
    xy = inv @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    x, y = np.rint(xy[0]).astype(int), np.rint(xy[1]).astype(int)
    assert np.allclose(xy[:2], np.rint(xy[:2]), atol=1e-6)
    assert np.array_equal(out[v.ravel(), u.ravel()], img[y, x])


@pytest.mark.parametrize('seed', range(8))
def test_image_and_masks_stay_aligned(seed):
    """A synthetic polygon drawn into the image must land on its own rasterised instance mask after any augmentation."""
    segs = poly_l_shape(900, 850)
    img = (raster(segs) * 255).astype(np.uint8)
    anns = [dict(segmentation=segs)]
    rng = np.random.default_rng(seed)
    out, masks, p = geom.make_crop(img, anns, rng, p_instance=1.0, jitter=40.0)
    assert masks.shape[0] == 1 and masks.dtype == np.uint8
    a = out > 127
    b = masks[0].astype(bool)
    iou = (a & b).sum() / (a | b).sum()
    assert iou > 0.97, (p, iou)
    # centroids agree to well under a pixel
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    assert abs(xa.mean() - xb.mean()) < 0.5 and abs(ya.mean() - yb.mean()) < 0.5
    # area scales with scale**2
    assert abs(b.sum() / raster(segs).sum() / p['scale'] ** 2 - 1) < 0.03


def test_flip_and_rotation_change_orientation():
    segs = poly_l_shape(900, 850)
    img = (raster(segs) * 255).astype(np.uint8)
    anns = [dict(segmentation=segs)]
    base = dict(theta=0.0, scale=1.0, hflip=False, vflip=False, cx=1050.0, cy=950.0)
    m0 = geom.warp_instances(anns, geom.continuous_matrix(base))[0]
    mh = geom.warp_instances(anns, geom.continuous_matrix(dict(base, hflip=True)))[0]
    assert np.array_equal(mh, m0[:, ::-1])
    mv = geom.warp_instances(anns, geom.continuous_matrix(dict(base, vflip=True)))[0]
    assert np.array_equal(mv, m0[::-1, :])
    i0 = geom.warp_image(img, geom.continuous_matrix(base))
    ih = geom.warp_image(img, geom.continuous_matrix(dict(base, hflip=True)))
    assert np.array_equal(ih, i0[:, ::-1])


def test_disk_sampling_stays_on_disk_and_instance_sampling_hits_instance():
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = geom.sample_params(rng, [], p_instance=1.0)
        assert p['mode'] == 'disk'
        assert np.hypot(p['cx'] - geom.DISK_CX, p['cy'] - geom.DISK_CY) <= geom.DISK_R + 1e-9
        assert 0.8 <= p['scale'] <= 1.25 and 0 <= p['theta'] <= 360
    segs = poly_l_shape(200, 1500)
    anns = [dict(segmentation=segs)]
    modes = []
    for _ in range(300):
        p = geom.sample_params(rng, anns)
        modes.append(p['mode'])
        if p['mode'] == 'instance':
            masks = geom.warp_instances(anns, geom.continuous_matrix(p))
            assert masks.shape[0] == 1
    frac = modes.count('instance') / len(modes)
    assert 0.58 < frac < 0.75


def test_empty_annotation_set_gives_empty_target():
    rng = np.random.default_rng(0)
    img = np.zeros((2048, 2048), np.uint8)
    out, masks, p = geom.make_crop(img, [], rng)
    assert out.shape == (512, 512) and masks.shape == (0, 512, 512) and p['mode'] == 'disk'
