"""chain2m (order MT1): multi-threshold candidates, pure helpers (no fit, no transport).

A level set is a list of binarisation thresholds for SHIP_R3's otherwise-unchanged rule, e.g. [0.5, 0.4]. An arm's candidates
are the union, level by level in the given order, of SHIP_R3.with_(thr=t) candidates of the arm's mixture; with more than one
level each candidate carries the feature `level` (= t). Candidates of different levels overlap (a 0.4 component contains its
0.5 components), so after the selector the overlap is resolved by one greedy rule, identical in the CV and on test:

    highest selector probability first (ties: lower candidate index first); a candidate sharing a pixel with an already kept
    candidate is dropped.

Applied after the cutoff tau it keeps exactly {greedy survivors over all candidates} with p >= tau (every candidate that could
block one with p >= tau has a higher p), so the nested cutoff sweep scores survivors only (resolve_probabilities).
"""
import numpy as np

SHIP_LEVEL = 0.5


def _level(x):
    try: v = float(x)
    except (TypeError, ValueError): raise ValueError(f'level {x!r}: not a number') from None
    if not 0 < v < 1 or abs(v * 100 - round(v * 100)) > 1e-9: raise ValueError(f'level {v}: must be in (0, 1) on the 0.01 grid')
    return round(v, 2)


def parse_levelsets(spec):
    """'0.5;0.5,0.4' -> [[0.5], [0.5, 0.4]]: ';' separates level sets (one arm per weight and set), ',' the levels of a set
    (strictly descending). Sets must differ."""
    if not isinstance(spec, str) or not spec.strip(): raise ValueError('levels: e.g. 0.5,0.4 or 0.5;0.5,0.4')
    out = []
    for part in spec.split(';'):
        items = [x.strip() for x in part.split(',')]
        if not items or any(x == '' for x in items): raise ValueError(f'levels: empty item in {spec!r}')
        ls = [_level(x) for x in items]
        if any(a <= b for a, b in zip(ls, ls[1:])): raise ValueError(f'levels {ls}: must be strictly descending')
        if ls in out: raise ValueError(f'levels {ls}: duplicate set')
        out.append(ls)
    return out


def level_tag(levels):
    """Arm-name suffix: '' for None or [0.5] (the SHIP_R3 arm itself), else 'l50l40' etc."""
    if levels is None or list(levels) == [SHIP_LEVEL]: return ''
    return ''.join(f'l{int(round(t * 100)):02d}' for t in levels)


def survivors(p, pairs):
    """Greedy survivors (bool per candidate): highest p first, ties lower index first; dropped if it overlaps a kept one."""
    p = np.asarray(p, float); n = len(p); nb = [[] for _ in range(n)]
    for i, j in pairs: nb[i].append(j); nb[j].append(i)
    alive = np.zeros(n, bool)
    for i in sorted(range(n), key=lambda i: (-p[i], i)):
        if not any(alive[j] for j in nb[i]): alive[i] = True
    return alive


def resolve_keep(p, tau, pairs):
    """Kept candidate indices (ascending) of one image: p >= tau, then the greedy overlap rule."""
    p = np.asarray(p, float)
    return np.flatnonzero(survivors(p, pairs) & (p >= tau)).tolist()


def resolve_probabilities(prob, records, files, overlap):
    """The nested cutoff sweep's probabilities with every non-survivor set to -1 (below every grid cutoff), per image of files.
    records[f]['candidate_ids'] = the image's bank ids (in candidate order); overlap[f] = its overlapping pairs. Returns a copy."""
    out = np.array(prob, float, copy=True)
    for f in files:
        ids = np.asarray(records[f]['candidate_ids'], int)
        if len(ids) and overlap[f]:
            alive = survivors(out[ids], overlap[f]); out[ids[~alive]] = -1.
    return out


def overlap_pairs(masks):
    """(i, j), i < j, of masks sharing at least one pixel (bounding boxes first)."""
    boxes = []
    for m in masks:
        ys = np.flatnonzero(m.any(1)); xs = np.flatnonzero(m.any(0))
        boxes.append((ys[0], ys[-1], xs[0], xs[-1]) if len(ys) else None)
    out = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = boxes[i], boxes[j]
            if a is None or b is None or a[0] > b[1] or b[0] > a[1] or a[2] > b[3] or b[2] > a[3]: continue
            y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
            if (masks[i][y0:y1 + 1, x0:x1 + 1] & masks[j][y0:y1 + 1, x0:x1 + 1]).any(): out.append((i, j))
    return out


def nesting(masks, lev, levels, pairs=None):
    """Across consecutive levels: parent of a candidate of level levels[i] = the overlapping candidate of level levels[i + 1]
    with the most shared pixels (ties: lower index; -1 if none or last level); children = the inverse;
    identical_to_parent = the same pixels as its parent."""
    pairs = overlap_pairs(masks) if pairs is None else pairs
    pos = {t: i for i, t in enumerate(levels)}; n = len(masks)
    best = [(-1, 0)] * n
    for i, j in pairs:
        for c, q in ((i, j), (j, i)):
            if pos[lev[q]] == pos[lev[c]] + 1:
                inter = int((masks[c] & masks[q]).sum())
                if inter > best[c][1] or (inter == best[c][1] and best[c][0] != -1 and q < best[c][0]): best[c] = (q, inter)
    parent = [b[0] for b in best]; children = [[] for _ in range(n)]
    for c, q in enumerate(parent):
        if q >= 0: children[q].append(c)
    same = [bool(q >= 0 and np.array_equal(masks[c], masks[q])) for c, q in enumerate(parent)]
    return dict(parent=parent, children=children, identical_to_parent=same)
