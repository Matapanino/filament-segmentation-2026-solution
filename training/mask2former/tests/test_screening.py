"""Screening switches keep the default draw unchanged; pixel AP and disk mask on synthetic data."""
import numpy as np

import eval_ckpt
import geom


def test_switches_default_identical_and_only_zero_their_part():
    anns = [dict(segmentation=[[900, 900, 1100, 900, 1100, 950, 900, 950]])]
    for seed in range(20):
        base = geom.sample_params(np.random.default_rng(seed), anns)
        same = geom.sample_params(np.random.default_rng(seed), anns, rotate=1, flip=1, input_scale=1.0)
        assert base == same
        nr = geom.sample_params(np.random.default_rng(seed), anns, rotate=0)
        assert nr['theta'] == 0 and {k: v for k, v in nr.items() if k != 'theta'} == {k: v for k, v in base.items() if k != 'theta'}
        nf = geom.sample_params(np.random.default_rng(seed), anns, flip=0)
        assert not nf['hflip'] and not nf['vflip'] and nf['theta'] == base['theta'] and nf['cx'] == base['cx']
        half = geom.sample_params(np.random.default_rng(seed), anns, input_scale=0.5)
        assert half['scale'] == base['scale'] * 0.5 and half['cx'] == base['cx']


def test_ap_from_hist():
    pos = np.zeros(511); neg = np.zeros(511)
    pos[500] = 10; neg[10] = 90           # perfect ranking
    assert eval_ckpt.ap_from_hist(pos, neg) == 1.0
    pos2 = np.zeros(511); neg2 = np.zeros(511)
    pos2[100] = 10; neg2[100] = 30        # all tied: AP = prevalence
    assert abs(eval_ckpt.ap_from_hist(pos2, neg2) - 0.25) < 1e-12
    assert eval_ckpt.ap_from_hist(np.zeros(511), neg2) is None


def test_disk_mask_fills_dark_features():
    import cv2
    img = np.zeros((2048, 2048), np.uint8)
    cv2.circle(img, (1024, 1024), 900, 160, -1)
    cv2.circle(img, (1000, 1000), 40, 5, -1)     # a dark filament-like hole inside the disk
    m = eval_ckpt.disk_mask(img)
    assert m[1000, 1000] and m[1024, 1024] and not m[10, 10]
    assert abs(m.sum() / (np.pi * 900 ** 2) - 1) < 0.01


def test_pixel_hists_small_components():
    img = np.zeros((2048, 2048), np.uint8)
    import cv2
    cv2.circle(img, (1024, 1024), 900, 160, -1)
    level = np.zeros((2048, 2048), np.int64)
    gt = np.zeros((2048, 2048), bool)
    gt[1000:1010, 1000:1010] = True       # 100 px: small
    gt[500:600, 1000:1100] = True         # 10000 px: large
    level[gt] = 400
    pos, neg, small = eval_ckpt.pixel_hists(level, img, gt)
    assert pos.sum() == 10100 and small.sum() == 100 and pos[400] == 10100 and neg[0] == neg.sum()


def test_map_writer_matches_package_save_npz(tmp_path):
    import sys
    from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))  # this repository's package (was an absolute path)
    import importlib.util
    spec = importlib.util.spec_from_file_location('det', str(Path(__file__).resolve().parents[3] / 'src/filseg/reproduction/determinism.py'))
    det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)
    rng = np.random.default_rng(0)
    arrays = {k: rng.integers(0, 511, size=(2048, 2048)).astype(np.uint16) for k in ('b', 'a', 'c')}
    det.save_npz(tmp_path / 'ref.npz', arrays)
    w = eval_ckpt.MapWriter(tmp_path / 'ours.npz')
    for k in sorted(arrays):
        w.write(k, arrays[k])
    w.close()
    assert (tmp_path / 'ref.npz').read_bytes() == (tmp_path / 'ours.npz').read_bytes()
