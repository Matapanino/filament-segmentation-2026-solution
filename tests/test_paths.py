"""Finding the competition's test images under either Kaggle mount layout, or failing with what was found."""
from pathlib import Path

import pytest


def make(root, rel):
    p = Path(root) / rel
    p.mkdir(parents=True)
    (p / 'x.jpeg').write_bytes(b'')
    return p


def test_current_mount_layout(tmp_path):
    from filseg.reproduction.paths import find_test_images
    want = make(tmp_path, 'competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images')
    assert find_test_images(tmp_path) == want


def test_older_mount_layout(tmp_path):
    from filseg.reproduction.paths import find_test_images
    want = make(tmp_path, 'filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images')
    assert find_test_images(tmp_path) == want


def test_the_path_the_notebook_used_to_hard_code(tmp_path):
    from filseg.reproduction.paths import find_test_images
    want = make(tmp_path, 'filament-segmentation-2026/test/test_images')
    assert find_test_images(tmp_path) == want


def test_known_paths_are_tried_in_order(tmp_path):
    from filseg.reproduction.paths import KNOWN_TEST_IMAGE_DIRS, find_test_images
    assert KNOWN_TEST_IMAGE_DIRS[0] == 'competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images'
    for rel in KNOWN_TEST_IMAGE_DIRS:
        make(tmp_path, rel)
    assert find_test_images(tmp_path) == tmp_path / KNOWN_TEST_IMAGE_DIRS[0]


def test_a_known_path_wins_over_other_test_images_directories(tmp_path):
    from filseg.reproduction.paths import find_test_images
    make(tmp_path, 'notebooks/someone/maps/test_images')
    want = make(tmp_path, 'competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images')
    assert find_test_images(tmp_path) == want


def test_one_unknown_test_images_directory_is_accepted(tmp_path):
    from filseg.reproduction.paths import find_test_images
    want = make(tmp_path, 'datasets/someone/copy/test/test_images')
    assert find_test_images(tmp_path) == want


def test_nothing_found_fails_listing_what_is_there(tmp_path):
    from filseg.reproduction.paths import find_test_images
    (tmp_path / 'datasets/someone/other').mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as e:
        find_test_images(tmp_path)
    message = str(e.value)
    assert 'competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images' in message
    assert str(tmp_path / 'datasets/someone') in message


def test_two_unknown_candidates_fail_listing_both(tmp_path):
    from filseg.reproduction.paths import find_test_images
    a = make(tmp_path, 'datasets/a/test_images')
    b = make(tmp_path, 'datasets/b/test_images')
    with pytest.raises(FileNotFoundError) as e:
        find_test_images(tmp_path)
    assert str(a) in str(e.value) and str(b) in str(e.value)


def test_missing_input_root_fails(tmp_path):
    from filseg.reproduction.paths import find_test_images
    with pytest.raises(FileNotFoundError):
        find_test_images(tmp_path / 'absent')
