import numpy as np

from filseg.instancing import binarize, instances


def synth_prob(spots):
    """1024px prob map with rectangular high-confidence blobs.

    spots: list of (r, c, h, w) in 1024px coordinates.
    """
    p = np.zeros((1024, 1024), np.float32)
    for r, c, h, w in spots:
        p[r : r + h, c : c + w] = 0.9
    return p


def test_components_and_min_area():
    # a 20x20 blob at 1024 becomes ~40x40=1600 px at 2048; a 2x2 becomes ~16 px
    p = synth_prob([(100, 100, 20, 20), (500, 500, 2, 2)])
    both = instances(p, thr=0.5, min_area=8)
    assert len(both) == 2
    big_only = instances(p, thr=0.5, min_area=256)
    assert len(big_only) == 1
    assert all(m.shape == (2048, 2048) and m.dtype == np.uint8 for m in both)


def test_closing_merges_nearby_fragments():
    # two blobs 4 px apart at 1024 (8 px at 2048): closing 21 bridges them
    p = synth_prob([(100, 100, 10, 10), (100, 114, 10, 10)])
    assert len(instances(p, thr=0.5, min_area=8)) == 2
    assert len(instances(p, thr=0.5, min_area=8, close_px=21)) == 1


def test_binarize_threshold():
    p = synth_prob([(100, 100, 10, 10)])
    assert binarize(p, thr=0.5).sum() > 0
    assert binarize(p, thr=0.95).sum() == 0
