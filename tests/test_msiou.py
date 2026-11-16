"""Parity + equivalence tests for the vendored Multiscale IoU.

The golden values in ``tests/data/miou_golden.json`` were produced by running
the UPSTREAM bitbucket.org/gsudmlab/multiscale_iou code (commit e423f4e4,
sklearn MinMaxScaler and all) on the same synthetic masks; the generator
script is documented in the vendored module header. These tests pin our
vendored copy — and the grid-aligned crop fast path — to those numbers.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from filseg.evaluate import evaluate_image
from filseg.msiou import DEFAULT_BOXSIZES, image_msiou, pair_msiou
from filseg.vendor.miou import MIoU

GOLDEN = json.loads((Path(__file__).parent / "data" / "miou_golden.json").read_text())


def make_cases():
    """Identical construction to the upstream golden generator."""
    cases = {}
    yy, xx = np.mgrid[0:512, 0:512]
    circ = ((yy - 256) ** 2 + (xx - 256) ** 2 < 80**2).astype(np.uint8)
    cases["identical_circles_512"] = (circ, circ.copy())
    circ2 = ((yy - 276) ** 2 + (xx - 236) ** 2 < 80**2).astype(np.uint8)
    cases["shifted_circle_512"] = (circ, circ2)
    line = np.zeros((512, 512), np.uint8)
    for i in range(60, 450):
        line[i, i - 2 : i + 2] = 1
    line_fat = ndimage.binary_dilation(line, iterations=2).astype(np.uint8)
    cases["line_vs_fat_512"] = (line, line_fat)
    smooth = np.zeros((512, 512), np.uint8)
    smooth[250:262, 60:450] = 1
    barbed = smooth.copy()
    rng = np.random.default_rng(0)
    for x in rng.integers(70, 440, size=25):
        h = int(rng.integers(6, 18))
        if rng.random() < 0.5:
            barbed[250 - h : 250, x : x + 3] = 1
        else:
            barbed[262 : 262 + h, x : x + 3] = 1
    cases["gt_barbed_vs_smooth_512"] = (barbed, smooth)
    cases["gt_smooth_vs_barbed_512"] = (smooth, barbed)
    rng = np.random.default_rng(1)
    big = np.zeros((2048, 2048), np.uint8)
    for _ in range(6):
        r, c = rng.integers(200, 1800, size=2)
        h, w = rng.integers(20, 200, size=2)
        big[r : r + h, c : c + w] = 1
    big2 = np.roll(big, (7, -13), axis=(0, 1))
    cases["blobs_2048"] = (big, big2)
    small = np.zeros((100, 100), np.uint8)
    small[30:60, 20:80] = 1
    small2 = np.zeros((100, 100), np.uint8)
    small2[35:65, 25:85] = 1
    cases["padding_100"] = (small, small2)
    return cases


CASES = make_cases()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_vendored_matches_upstream_golden(name):
    """int64 inputs = the dtype the upstream mask_loader produces."""
    m1, m2 = CASES[name]
    metric = MIoU(DEFAULT_BOXSIZES, edge_only=True)
    area = metric.measure(m1.astype(np.int64), m2.astype(np.int64))
    g = GOLDEN[name]
    assert area == pytest.approx(g["miou"], abs=1e-12)
    assert metric.slope == pytest.approx(g["slope"], abs=1e-12)
    assert metric.box_counts_m1 == g["box_counts_m1"]
    assert metric.box_counts_intersect == g["box_counts_intersect"]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_crop_fast_path_equals_full_frame(name):
    m1, m2 = CASES[name]
    full = MIoU(DEFAULT_BOXSIZES, edge_only=True).measure(
        m1.astype(np.int64), m2.astype(np.int64)
    )
    assert pair_msiou(m1, m2) == pytest.approx(full, abs=1e-12)


def test_uint8_input_would_break_the_upstream_code():
    """Documents WHY pair_msiou normalizes to int64: uint8 masks promote to
    float16 inside Sobel/hypot, whose spacing above 2048 drops box counts —
    the identical-mask MSIoU then exceeds 1. pair_msiou must not reproduce
    this; the raw vendored code (fed the wrong dtype) does."""
    barbed, _ = CASES["gt_barbed_vs_smooth_512"]
    raw = MIoU(DEFAULT_BOXSIZES, edge_only=True).measure(barbed, barbed.copy())
    assert raw > 1.0  # the trap exists (uint8 in, float16 promotion)
    assert pair_msiou(barbed, barbed.copy()) == pytest.approx(1.0, abs=1e-12)


def test_msiou_prefers_the_barbed_prediction():
    """The metric's reason to exist: losing GT barbs must cost more than
    predicting extra ones (recall-flavoured, GT-normalized)."""
    barbed, smooth = CASES["gt_barbed_vs_smooth_512"]
    assert pair_msiou(barbed, smooth) < pair_msiou(smooth, barbed)
    assert pair_msiou(barbed, barbed) == pytest.approx(1.0, abs=1e-12)


def test_image_msiou_via_matched_pairs():
    """End-to-end: evaluate_image records matched pairs incl. original list
    indices; image_msiou scores exactly those pairs."""
    a = np.zeros((512, 512), np.uint8)
    a[100:120, 100:300] = 1
    b = np.zeros((512, 512), np.uint8)
    b[400:420, 100:300] = 1
    empty = np.zeros((512, 512), np.uint8)
    # gt list has an empty mask first: original indices must survive filtering
    ev = evaluate_image([empty, a, b], [b.copy(), a.copy()])
    assert sorted(ev.matched_pairs) == [(1, 1), (2, 0)]
    vals = image_msiou([empty, a, b], [b.copy(), a.copy()], ev)
    assert vals == pytest.approx([1.0, 1.0], abs=1e-12)
