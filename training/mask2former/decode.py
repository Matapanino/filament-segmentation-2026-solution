"""Decoder D (order A1) and the as-run decoder, shared by arms C and T.

D: per query, score = class probability of "filament" (softmax over [filament, no-object], index 0); keep queries
with score >= thr; mask = sigmoid(mask logits upsampled to the native 2048 image) > 0.5 (== logits > 0); overlaps are
resolved greedily by descending score (a pixel belongs to the first kept query); an instance is dropped if < 50% of
its own mask survives or it has < 64 px left.

The as-run decoder (campaign 20261108_architecture_mask2former gpu_worker.py, inference block): score >= .5, logits > 0,
confidence-order exclusive ownership, drop when fewer than 256 new pixels remain (no survival-fraction rule).
Used here only to check that arm C reproduces the recorded 0.349 (descriptive).
"""
import torch

D_RULE = dict(min_frac=0.5, min_px=64)
ASRUN_RULE = dict(min_frac=0.0, min_px=256)


def query_scores(class_queries_logits):
    """[Q, 2] logits -> [Q] probability of class 0 ("filament")."""
    return class_queries_logits.float().softmax(-1)[:, 0]


def native_masks(mask_logits, keep, size=2048):
    """Bilinear upsampling (align_corners=False) of the kept queries' logits to size x size, thresholded at 0."""
    if keep.numel() == 0:
        return torch.zeros((0, size, size), dtype=torch.bool, device=mask_logits.device)
    up = torch.nn.functional.interpolate(mask_logits[keep][None].float(), size=(size, size), mode='bilinear',
                                         align_corners=False)[0]
    return up > 0


def decode(scores, masks, query_ids, thr, min_frac, min_px):
    """scores: [Q] (all queries); masks: [K, H, W] bool for query_ids (K queries, a superset of those >= thr).

    Returns [(query id, score, bool mask [H, W])] in decreasing score order.
    """
    out = []
    if masks.shape[0] == 0:
        return out
    sel = scores[query_ids]
    order = torch.argsort(sel, descending=True, stable=True).tolist()
    occupied = torch.zeros_like(masks[0])
    for k in order:
        s = float(sel[k])
        if s < thr:
            break
        m = masks[k]
        area = int(m.sum())
        if area == 0:
            continue
        new = m & ~occupied
        n = int(new.sum())
        if n < min_px or n < min_frac * area:
            continue
        occupied |= new
        out.append((int(query_ids[k]), s, new))
    return out
