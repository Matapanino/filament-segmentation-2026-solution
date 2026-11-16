import numpy as np

from filseg.coco_io import (
    annotation_sets,
    mask_to_rle_counts,
    poly_to_mask,
    read_submission,
    rle_counts_to_mask,
    write_submission,
)


def test_poly_to_mask_square():
    poly = [[10.0, 10.0, 10.0, 60.0, 60.0, 60.0, 60.0, 10.0, 10.0, 10.0]]
    mask = poly_to_mask(poly, h=100, w=100)
    area = int(mask.sum())
    assert 2400 < area < 2700  # ~50x50 square, rasterization-dependent edges
    assert mask[35, 35] == 1
    assert mask[5, 5] == 0


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    mask = (rng.random((2048, 2048)) > 0.999).astype(np.uint8)
    counts = mask_to_rle_counts(mask)
    back = rle_counts_to_mask(counts)
    assert np.array_equal(mask, back)


def test_submission_roundtrip(tmp_path):
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[100:200, 300:500] = 1
    path = tmp_path / "submission.csv"
    write_submission({"20110120105534Ch": [mask, mask]}, path)
    back = read_submission(path)
    assert list(back) == ["20110120105534Ch"]
    assert len(back["20110120105534Ch"]) == 2
    assert np.array_equal(back["20110120105534Ch"][0], mask)


def test_annotation_sets_grouping():
    coco = {
        "images": [
            {"id": "010101-20110101000000Bh", "file_name": "20110101000000Bh.jpeg"},
            {"id": "010102-20110101000000Bh", "file_name": "20110101000000Bh.jpeg"},
            {"id": "010101-20120101000000Ch", "file_name": "20120101000000Ch.jpeg"},
        ],
        "annotations": [
            {"id": "a", "image_id": "010101-20110101000000Bh"},
            {"id": "b", "image_id": "010101-20110101000000Bh"},
            {"id": "c", "image_id": "010102-20110101000000Bh"},
        ],
    }
    grouped = annotation_sets(coco)
    assert set(grouped) == {"20110101000000Bh", "20120101000000Ch"}
    sets = {s.image_id: len(s.annotations) for s in grouped["20110101000000Bh"]}
    assert sets == {"010101-20110101000000Bh": 2, "010102-20110101000000Bh": 1}
    assert len(grouped["20120101000000Ch"]) == 1
