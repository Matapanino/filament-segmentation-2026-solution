"""Competition-format checks for a submission CSV and its difference from a reference CSV."""
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from .csv_io import read_rows

SIZE = [2048, 2048]


def _by_file(path):
    rows = read_rows(path)
    grouped = defaultdict(list)
    for identifier, rle in rows:
        f, k = identifier.rsplit('_', 1)
        grouped[f].append((int(k), rle))
    return rows, grouped


def _decode(rle):
    return mu.decode(dict(size=SIZE, counts=rle.encode())).astype(bool)


def validate_submission(path, keys):
    with open(path, newline='') as stream:
        header_ok = stream.readline().rstrip('\r\n') == 'filament_id,segmentation_rle'
    rows, grouped = _by_file(path)
    keys = set(keys)
    ids = [r[0] for r in rows]
    decoded = empty = 0
    overlapping = []
    contiguous = True
    for f, items in sorted(grouped.items()):
        contiguous &= [k for k, _ in items] == list(range(1, len(items) + 1))
        occupied = np.zeros(SIZE, bool)
        overlap = False
        for _, rle in items:
            m = _decode(rle)
            decoded += 1
            if not m.any():
                empty += 1
            overlap |= bool((occupied & m).any())
            occupied |= m
        if overlap:
            overlapping.append(f)
    unknown = sorted(set(grouped) - keys)
    result = dict(rows=len(rows), decoded_rows=decoded, empty_rows=empty, nonempty_rows=decoded - empty,
                  files_with_rows=len(grouped), files_without_rows=len(keys - set(grouped)), expected_files=len(keys),
                  unknown_files=unknown, overlapping_pixel_images=overlapping, header_ok=header_ok,
                  ids_unique=len(set(ids)) == len(ids), ids_contiguous=bool(contiguous),
                  sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), bytes=Path(path).stat().st_size)
    result['passed'] = bool(header_ok and decoded == len(rows) and empty == 0 and not unknown and not overlapping
                            and result['ids_unique'] and contiguous and len(rows) > 0)
    return result


def compare_submissions(path, reference, keys):
    _, a = _by_file(path)
    _, b = _by_file(reference)
    ious = []
    changed = 0
    for f in sorted(keys):
        ra = [r for _, r in a.get(f, [])]
        rb = [r for _, r in b.get(f, [])]
        if ra != rb:
            changed += 1
        ua = np.zeros(SIZE, bool)
        ub = np.zeros(SIZE, bool)
        for r in ra:
            ua |= _decode(r)
        for r in rb:
            ub |= _decode(r)
        union = int((ua | ub).sum())
        ious.append(float((ua & ub).sum()) / union if union else 1.)
    return dict(instances=sum(len(v) for v in a.values()), reference_instances=sum(len(v) for v in b.values()),
                images=len(keys), changed_images=changed, identical_images=len(keys) - changed,
                foreground_IoU_mean=float(np.mean(ious)), foreground_IoU_min=float(np.min(ious)),
                byte_identical=Path(path).read_bytes() == Path(reference).read_bytes())
