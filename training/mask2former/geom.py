"""Native-resolution training crops: rotation about the solar-disk centre, scale jitter, h/v flips, crop sampling.

Derived from the as-run Mask2Former recipe (campaign 20261108_architecture_mask2former, snapshot 91f8ca62...):
`instance_geometry.crop_instances` rasterised the original polygons after scaling (0.5) and shifting them into the
512 crop with the COCO polygon rasteriser. Here the same idea is kept (polygons are transformed, then rasterised by
pycocotools), with scale 1.0 (native 2048 image), plus rotation, scale jitter and flips folded into one affine map.

Rotating the full image about the disk centre and then cropping a window centred on a point c' is the same map as
rotating about the crop centre c (the pre-image of c'), so one affine warp per crop is used instead of rotating the
whole 2048 image (identical geometry, much cheaper).

Coordinates: polygons use COCO continuous coordinates (pixel (i, j) covers [j, j+1) x [i, i+1)); OpenCV uses
integer pixel centres. `image_matrix` converts the continuous map to OpenCV's convention.
"""
import math

import cv2
import numpy as np
from pycocotools import mask as mu

# Disk geometry, measured on the training images by `measure_disk.py` (results/disk_geometry.json).
DISK_CX = 1023.5
DISK_CY = 1023.5
DISK_R = 904.0
CROP = 512


def sample_params(rng, annotations, crop=CROP, p_instance=2 / 3, jitter=128.0, scale_range=(0.8, 1.25),
                  disk=(DISK_CX, DISK_CY, DISK_R), rotate=1, flip=1, input_scale=1.0):
    """Random augmentation parameters for one draw (all randomness comes from `rng`).

    Screening switches: rotate=0 / flip=0 zero the drawn angle / flips (the same random numbers are still drawn, so the
    other parameters do not change); input_scale multiplies the scale (0.5 = the 1024 view of the as-run recipe,
    crop pixels then cover 2x the original pixels; jitter stays in original pixels).
    """
    theta = float(rng.uniform(0.0, 360.0))
    scale = float(rng.uniform(*scale_range))
    hflip = bool(rng.random() < 0.5)
    vflip = bool(rng.random() < 0.5)
    pick = float(rng.random())
    if annotations and pick < p_instance:
        a = annotations[int(rng.integers(len(annotations)))]
        pts = np.concatenate([np.asarray(f, np.float64).reshape(-1, 2) for f in a['segmentation']])
        cx, cy = pts.mean(0) + rng.uniform(-jitter, jitter, size=2)
        mode = 'instance'
    else:
        r = disk[2] * math.sqrt(float(rng.random()))
        phi = float(rng.uniform(0.0, 2 * math.pi))
        cx, cy = disk[0] + r * math.cos(phi), disk[1] + r * math.sin(phi)
        mode = 'disk'
    if not rotate:
        theta = 0.0
    if not flip:
        hflip = vflip = False
    return dict(theta=theta, scale=scale * input_scale, hflip=hflip, vflip=vflip, cx=float(cx), cy=float(cy), mode=mode)


def continuous_matrix(p, crop=CROP):
    """2x3 matrix M with q = M @ [x, y, 1]: original continuous coords -> crop continuous coords."""
    t = math.radians(p['theta'])
    c, s = math.cos(t), math.sin(t)
    a = p['scale'] * np.array([[c, -s], [s, c]], np.float64)
    if p['hflip']:
        a = np.array([[-1, 0], [0, 1]], np.float64) @ a
    if p['vflip']:
        a = np.array([[1, 0], [0, -1]], np.float64) @ a
    centre = np.array([p['cx'], p['cy']], np.float64)
    b = np.array([crop / 2, crop / 2], np.float64) - a @ centre
    return np.concatenate([a, b[:, None]], axis=1)


def image_matrix(m):
    """The same map in OpenCV's integer-pixel-centre convention: q_i = A p_i + (A @ .5 + b - .5)."""
    a, b = m[:, :2], m[:, 2]
    return np.concatenate([a, (a @ np.array([.5, .5]) + b - .5)[:, None]], axis=1)


def warp_image(img, m, crop=CROP):
    """uint8 [H, W] image -> uint8 [crop, crop] (bilinear, black outside the image)."""
    return cv2.warpAffine(img, image_matrix(m), (crop, crop), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_instances(annotations, m, crop=CROP):
    """Transform every annotation polygon by m and rasterise it in the crop; empty instances are dropped (as-run)."""
    out = []
    a, b = m[:, :2], m[:, 2]
    for ann in annotations:
        polys = []
        for flat in ann['segmentation']:
            pts = np.asarray(flat, np.float64).reshape(-1, 2)
            if len(pts) < 3:
                continue
            q = pts @ a.T + b
            polys.append(q.reshape(-1).tolist())
        if not polys:
            continue
        mask = mu.decode(mu.merge(mu.frPyObjects(polys, crop, crop)))
        if mask.any():
            out.append(mask)
    return np.stack(out) if out else np.zeros((0, crop, crop), np.uint8)


CONSENSUS_RULES = ('ceil_half', 'strict_majority')


def consensus_sets(sets, iou_thr=0.3, size=2048, rule='ceil_half'):
    """Majority label sets of one image: list with one annotation list per drawn set index.

    Every instance is rasterised at `size` (COCO RLE); instances of different sets are matched greedily by IoU >
    `iou_thr` (pairs by decreasing IoU; two clusters merge only if they share no set, so a cluster holds at most one
    instance per set). A cluster is kept if it has instances from >= ceil(n_sets / 2) sets (rule 'ceil_half') or
    >= floor(n_sets / 2) + 1 sets (rule 'strict_majority': on 2 sets both must agree). For drawn set s, a kept
    cluster contributes s's own instance if it has one, else the cluster's largest member (ties: lowest (set, index)).
    The lists hold the original annotation dicts, ordered by the cluster's first member (set, index).
    """
    if rule not in CONSENSUS_RULES:
        raise ValueError(f'consensus rule {rule!r} not in {CONSENSUS_RULES}')
    inst = []  # (set index, index in set, rle, area)
    for s, anns in enumerate(sets):
        for i, a in enumerate(anns):
            polys = [list(map(float, p)) for p in a['segmentation'] if len(p) >= 6]
            if not polys:
                continue
            rle = mu.merge(mu.frPyObjects(polys, size, size))
            area = float(mu.area(rle))
            if area > 0:
                inst.append((s, i, rle, area))
    n = len(inst)
    parent = list(range(n))
    members = [[k] for k in range(n)]
    set_of = [{inst[k][0]} for k in range(n)]

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k

    edges = []
    for u in range(n):
        others = [v for v in range(u + 1, n) if inst[v][0] != inst[u][0]]
        if not others:
            continue
        ious = mu.iou([inst[u][2]], [inst[v][2] for v in others], [0] * len(others))[0]
        edges += [(-float(x), u, v) for x, v in zip(ious, others) if x > iou_thr]
    for _, u, v in sorted(edges):
        ru, rv = root(u), root(v)
        if ru == rv or set_of[ru] & set_of[rv]:
            continue
        parent[rv] = ru
        members[ru] += members[rv]
        set_of[ru] |= set_of[rv]
    need = math.ceil(len(sets) / 2) if rule == 'ceil_half' else len(sets) // 2 + 1
    roots = sorted({root(k) for k in range(n)})
    clusters = sorted((sorted(members[r]) for r in roots if len(set_of[r]) >= need), key=lambda c: c[0])
    out = []
    for s in range(len(sets)):
        chosen = []
        for c in clusters:
            own = [k for k in c if inst[k][0] == s]
            k = own[0] if own else min(c, key=lambda k: (-inst[k][3], k))
            chosen.append(sets[inst[k][0]][inst[k][1]])
        out.append(chosen)
    return out


def make_crop(img, annotations, rng, crop=CROP, **kw):
    """One training draw: (uint8 [crop, crop] image, uint8 [n, crop, crop] instance masks, params)."""
    p = sample_params(rng, annotations, crop=crop, **kw)
    m = continuous_matrix(p, crop)
    return warp_image(img, m, crop), warp_instances(annotations, m, crop), p
