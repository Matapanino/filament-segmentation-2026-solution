import pytest

from filseg.guard import TEST_FILENAMES, ContaminationError
from filseg.splits import (
    entries_for_files,
    fold_assignment,
    fold_files,
    fold_summary,
    load_assignment,
    save_assignment,
)


def synth_coco(spec):
    """spec: list of (file, n_sets, filaments_per_set)."""
    images, annotations = [], []
    for file, n_sets, count in spec:
        for b in range(n_sets):
            iid = f"0101{b:02d}-{file}"
            images.append({"id": iid, "file_name": f"{file}.jpeg"})
            for a in range(count):
                annotations.append({"image_id": iid, "id": f"{iid}-{a}"})
    return {"images": images, "annotations": annotations}


def big_spec():
    # 40 files across four count strata, some multiply annotated
    spec = []
    for i in range(40):
        count = [2, 5, 9, 14][i % 4]
        n_sets = 1 + (i % 3 == 0) + (i % 7 == 0)
        spec.append((f"209901{i:02d}0000{'BCLMTU'[i % 6]}h", n_sets, count))
    return spec


def test_deterministic_and_seed_sensitive():
    coco = synth_coco(big_spec())
    a1 = fold_assignment(coco, k=5, seed=0)
    a2 = fold_assignment(coco, k=5, seed=0)
    a3 = fold_assignment(coco, k=5, seed=1)
    assert a1 == a2
    assert a1 != a3


def test_fold_sizes_within_one():
    coco = synth_coco(big_spec())
    a = fold_assignment(coco, k=5, seed=0)
    sizes = [list(a.values()).count(f) for f in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 40


def test_count_strata_spread_within_one():
    """Round-robin over the count-sorted order: every stratum lands on each
    fold either floor or ceil(m/k) times."""
    coco = synth_coco(big_spec())
    a = fold_assignment(coco, k=5, seed=0)
    by_count = {}
    for file, n_sets, count in big_spec():
        by_count.setdefault(count, []).append(file)
    for count, files in by_count.items():
        per_fold = [sum(1 for f in files if a[f] == k) for k in range(5)]
        assert max(per_fold) - min(per_fold) <= 1, (count, per_fold)


def test_grouping_is_by_file_and_partition_is_exact():
    coco = synth_coco(big_spec())
    a = fold_assignment(coco, k=5, seed=0)
    for fold in range(5):
        tr = fold_files(a, fold, "train")
        va = fold_files(a, fold, "val")
        assert set(tr) | set(va) == set(a)
        assert not set(tr) & set(va)
    with pytest.raises(ValueError):
        fold_files(a, 0, "test")


def test_entries_follow_their_file():
    coco = synth_coco([("20990101000000Bh", 3, 4), ("20990102000000Ch", 1, 2)])
    entries = entries_for_files(coco, ["20990101000000Bh"])
    assert entries == [
        "010100-20990101000000Bh",
        "010101-20990101000000Bh",
        "010102-20990101000000Bh",
    ]


def test_contamination_fails_loudly(tmp_path):
    test_file = sorted(TEST_FILENAMES)[0]
    coco = synth_coco([(test_file, 1, 3), ("20990101000000Bh", 1, 3)])
    with pytest.raises(ContaminationError):
        fold_assignment(coco, k=2, seed=0)
    with pytest.raises(ContaminationError):
        entries_for_files(synth_coco([("20990101000000Bh", 1, 1)]), [test_file])
    poisoned = {test_file: 0, "20990101000000Bh": 1}
    p = tmp_path / "folds.json"
    save_assignment(poisoned, p)
    with pytest.raises(ContaminationError):
        load_assignment(p)


def test_roundtrip_and_summary(tmp_path):
    coco = synth_coco(big_spec())
    a = fold_assignment(coco, k=5, seed=0)
    p = tmp_path / "folds.json"
    save_assignment(a, p)
    assert load_assignment(p) == a
    s = fold_summary(a, coco)
    assert sum(v["n_files"] for v in s.values()) == 40
    assert sum(v["n_filaments"] for v in s.values()) == sum(
        n * c for _, n, c in big_spec()
    )
    assert all(set(v["site"]) <= set("BCLMTU") for v in s.values())
