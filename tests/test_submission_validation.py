"""Submission-format checks and the per-image difference report. Synthetic only."""
import numpy as np
from pycocotools import mask as mu

from filseg.reproduction.csv_io import write_rows


def rle(y, x, h=10, w=20):
    a = np.zeros((2048, 2048), np.uint8)
    a[y:y + h, x:x + w] = 1
    return mu.encode(np.asfortranarray(a))['counts'].decode()


def write(path, rows):
    files = sorted(rows)
    write_rows(path, files, rows, {f: [True] * len(v) for f, v in rows.items()})


def test_valid_submission_passes(tmp_path):
    from filseg.reproduction.validate import validate_submission
    p = tmp_path / 's.csv'
    write(p, {'a': [rle(0, 0), rle(100, 100)], 'b': [rle(5, 5)], 'c': []})
    r = validate_submission(p, ['a', 'b', 'c'])
    assert r['passed'] and r['rows'] == 3 and r['nonempty_rows'] == 3
    assert r['files_with_rows'] == 2 and r['unknown_files'] == [] and r['overlapping_pixel_images'] == []
    assert r['decoded_rows'] == 3 and r['header_ok'] and r['ids_unique'] and r['ids_contiguous']


def test_overlap_empty_and_unknown_file_fail(tmp_path):
    from filseg.reproduction.validate import validate_submission
    p = tmp_path / 's.csv'
    empty = mu.encode(np.asfortranarray(np.zeros((2048, 2048), np.uint8)))['counts'].decode()
    write(p, {'a': [rle(0, 0), rle(5, 5)], 'b': [empty], 'z': [rle(1, 1)]})
    r = validate_submission(p, ['a', 'b'])
    assert not r['passed']
    assert r['overlapping_pixel_images'] == ['a'] and r['empty_rows'] == 1 and r['unknown_files'] == ['z']


def test_difference_report_counts_changed_images_and_iou(tmp_path):
    from filseg.reproduction.validate import compare_submissions
    ref = tmp_path / 'ref.csv'
    new = tmp_path / 'new.csv'
    write(ref, {'a': [rle(0, 0)], 'b': [rle(50, 50)]})
    write(new, {'a': [rle(0, 0)], 'b': [rle(50, 50, h=20)]})
    d = compare_submissions(new, ref, ['a', 'b', 'c'])
    assert d['instances'] == 2 and d['reference_instances'] == 2
    assert d['changed_images'] == 1 and d['identical_images'] == 2
    assert d['foreground_IoU_min'] == 0.5 and d['foreground_IoU_mean'] == (1 + .5 + 1) / 3
    assert d['byte_identical'] is False
    assert compare_submissions(ref, ref, ['a', 'b', 'c'])['byte_identical'] is True
