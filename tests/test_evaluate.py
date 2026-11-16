import numpy as np
import pytest

from filseg.evaluate import evaluate_files, evaluate_image


def block(r0, r1, c0, c1, shape=(200, 200)):
    m = np.zeros(shape, dtype=np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def test_perfect_match():
    gt = [block(10, 110, 10, 110)]
    ev = evaluate_image(gt, [gt[0].copy()])
    assert ev.score == pytest.approx(1.0)
    assert ev.matched_dice == [pytest.approx(1.0)]
    assert ev.n_unmatched_gt == 0 and ev.n_unmatched_pred == 0


def test_half_overlap_dice():
    gt = [block(0, 100, 0, 100)]
    pred = [block(0, 100, 50, 150)]  # overlap 100x50
    ev = evaluate_image(gt, pred)
    assert ev.score == pytest.approx(0.5)


def test_fragmentation_penalized_and_counted():
    gt = [block(0, 100, 0, 100)]
    preds = [block(0, 100, 0, 50), block(0, 100, 50, 100)]  # two halves
    ev = evaluate_image(gt, preds)
    # one half matches (dice = 2*5000/15000), the other stays unmatched
    assert ev.matched_dice == [pytest.approx(2 * 5000 / 15000)]
    assert ev.n_unmatched_pred == 1
    assert ev.score == pytest.approx((2 * 5000 / 15000) / 2)
    assert ev.one_to_many == 1  # the GT is covered by two preds at IoU>0.1
    assert ev.many_to_one == 0


def test_merging_counted():
    gts = [block(0, 100, 0, 45), block(0, 100, 55, 100)]
    pred = [block(0, 100, 0, 100)]  # one blob swallowing two GTs
    ev = evaluate_image(gts, pred)
    assert ev.many_to_one == 1
    assert ev.n_unmatched_gt == 1


def test_empty_semantics():
    assert evaluate_image([], []).score is None
    assert evaluate_image([], [block(0, 10, 0, 10)]).score == 0.0
    assert evaluate_image([block(0, 10, 0, 10)], []).score == 0.0


def test_evaluate_files_protocols():
    f = "20110101000000Bh"
    set_a = [block(0, 100, 0, 100)]
    set_b = [block(0, 100, 10, 110)]
    preds = {f: [block(0, 100, 0, 100)]}
    gt = {f: [set_a, set_b]}

    rep_ps, per_file_ps = evaluate_files(gt, preds, protocol="per_set")
    assert rep_ps.n_files == 1 and rep_ps.n_scored == 1
    # perfect vs set_a (1.0), imperfect vs set_b -> average in (0.5, 1.0)
    assert 0.5 < per_file_ps[f] < 1.0

    rep_c, per_file_c = evaluate_files(gt, preds, protocol="concat")
    # concat keeps both duplicates as GT: one matches, one goes unmatched
    assert per_file_c[f] < per_file_ps[f]
    assert rep_c.n_unmatched_gt == 1
