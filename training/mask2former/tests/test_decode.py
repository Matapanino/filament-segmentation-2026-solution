"""Decoder D on synthetic scores and logits."""
import torch

import decode


def box(h, w, r0, r1, c0, c1):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[r0:r1, c0:c1] = True
    return m


def test_query_scores_is_softmax_class0():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    s = decode.query_scores(logits)
    assert torch.allclose(s, torch.softmax(logits, -1)[:, 0])


def test_native_masks_upsamples_and_thresholds_at_zero():
    lg = torch.full((3, 4, 4), -5.0)
    lg[1, :2, :2] = 5.0
    m = decode.native_masks(lg, torch.tensor([1, 2]), size=16)
    assert m.shape == (2, 16, 16) and m.dtype == torch.bool
    assert m[0, :6, :6].all() and not m[0, 10:, 10:].any() and not m[1].any()


def test_greedy_ownership_and_drop_rules():
    H = W = 64
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.45, 0.95])
    masks = torch.stack([
        box(H, W, 0, 20, 0, 20),     # q0: 400 px, kept
        box(H, W, 10, 30, 10, 30),   # q1: 400 px, 100 overlap with q0 -> 300 new (75%) kept
        box(H, W, 0, 20, 5, 25),     # q2: 400 px, overlaps q0 (300) and q1 (50) -> 50 new (<50%, <64) dropped
        box(H, W, 40, 47, 40, 49),   # q3: 63 px, alone -> dropped by the 64 px rule
        box(H, W, 50, 60, 50, 60),   # q4: below threshold .5
        torch.zeros(H, W, dtype=torch.bool),  # q5: empty mask, highest score -> ignored
    ])
    ids = torch.arange(6)
    out = decode.decode(scores, masks, ids, thr=0.5, **decode.D_RULE)
    assert [q for q, _, _ in out] == [0, 1]
    assert int(out[1][2].sum()) == 300 and not (out[0][2] & out[1][2]).any()
    out3 = decode.decode(scores, masks, ids, thr=0.4, **decode.D_RULE)
    assert [q for q, _, _ in out3] == [0, 1, 4]
    # 50% rule alone: q1 with 60% overlap -> dropped
    m2 = torch.stack([box(H, W, 0, 20, 0, 20), box(H, W, 0, 20, 8, 28)])  # q1 new = 160/400 = 40%
    out4 = decode.decode(torch.tensor([.9, .8]), m2, torch.arange(2), thr=.5, **decode.D_RULE)
    assert [q for q, _, _ in out4] == [0]
    # as-run rule keeps it (160 < 256 -> dropped there too); a larger one survives as-run but not D
    m3 = torch.stack([box(H, W, 0, 40, 0, 40), box(H, W, 0, 40, 22, 62)])  # q1 new 40*22=880 of 1600 = 55% -> D keeps
    assert [q for q, _, _ in decode.decode(torch.tensor([.9, .8]), m3, torch.arange(2), .5, **decode.D_RULE)] == [0, 1]
    m4 = torch.stack([box(H, W, 0, 40, 0, 40), box(H, W, 0, 40, 30, 64)])  # q1 new 40*24=960 of 1360 = 70%
    m5 = torch.stack([box(H, W, 0, 40, 0, 40), box(H, W, 0, 40, 10, 50)])  # q1 new 400 of 1600 = 25%: D drops, as-run keeps
    assert [q for q, _, _ in decode.decode(torch.tensor([.9, .8]), m5, torch.arange(2), .5, **decode.D_RULE)] == [0]
    assert [q for q, _, _ in decode.decode(torch.tensor([.9, .8]), m5, torch.arange(2), .5, **decode.ASRUN_RULE)] == [0, 1]
    assert len(decode.decode(torch.tensor([.9, .8]), m4, torch.arange(2), .5, **decode.D_RULE)) == 2


def test_subset_query_ids_map_back():
    H = W = 32
    scores = torch.tensor([0.1, 0.9, 0.2, 0.7])
    keep = torch.tensor([1, 3])
    masks = torch.stack([box(H, W, 0, 16, 0, 16), box(H, W, 16, 32, 16, 32)])
    out = decode.decode(scores, masks, keep, thr=0.5, **decode.D_RULE)
    assert [q for q, _, _ in out] == [1, 3] and abs(out[1][1] - 0.7) < 1e-6
