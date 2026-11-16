import numpy as np
from pycocotools import mask as mu

import pqscore


def rle(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))


def test_counts_from_iou():
    q = np.array([[0.9, 0.0, 0.2], [0.0, 0.4, 0.0]])
    c = pqscore.counts_from_iou(q)
    assert c.tolist() == [0.9, 1, 2, 1, 1, 0]


def test_perfect_and_empty():
    m = np.zeros((2048, 2048), bool)
    m[10:50, 10:200] = True
    gt = [[rle(m)]]
    c, bins = pqscore.image_counts(gt, [rle(m)])
    assert pqscore.metric(c)['pq'] == 1.0
    assert bins['gt_1024to4096'][0] == 0 and bins['gt_gt4096'][1] == 1
    c2, _ = pqscore.image_counts(gt, [])
    assert c2.tolist() == [0, 0, 0, 1, 0, 0]


def test_two_sets_sum():
    a = np.zeros((2048, 2048), bool); a[0:40, 0:40] = True
    b = np.zeros((2048, 2048), bool); b[100:140, 100:140] = True
    c, _ = pqscore.image_counts([[rle(a)], [rle(b)]], [rle(a)])
    # set 1: tp 1; set 2: fp 1, fn 1
    assert c.tolist() == [1.0, 1, 1, 1, 0, 0]


def test_bootstrap_identical_arms_zero():
    rng = np.random.default_rng(0)
    c = rng.integers(1, 9, size=(20, 6)).astype(float)
    r = pqscore.paired_bootstrap(c, c, draws=200, seed=0)
    assert r['delta'] == 0 and r['ci95'] == [0.0, 0.0]
    t = c.copy(); t[:, 0] += 1
    r2 = pqscore.paired_bootstrap(t, c, draws=200, seed=0)
    assert r2['delta'] > 0 and r2['ci95'][0] > 0
