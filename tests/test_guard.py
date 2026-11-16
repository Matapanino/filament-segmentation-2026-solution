import pytest

from filseg.guard import (
    TEST_FILENAMES,
    ContaminationError,
    assert_clean,
    file_base,
    is_test_file,
)

A_TEST_FILE = sorted(TEST_FILENAMES)[0]  # '20110120105534Ch'


def test_pinned_set_size():
    assert len(TEST_FILENAMES) == 180


def test_file_base_normalization():
    assert file_base(A_TEST_FILE) == A_TEST_FILE
    assert file_base(A_TEST_FILE + ".jpeg") == A_TEST_FILE
    assert file_base("010101-" + A_TEST_FILE) == A_TEST_FILE
    assert file_base("some/dir/" + A_TEST_FILE + ".jpg") == A_TEST_FILE


def test_is_test_file():
    assert is_test_file(A_TEST_FILE)
    assert is_test_file("010102-" + A_TEST_FILE)
    assert not is_test_file("19990101000000Bh")


def test_assert_clean_passes_on_train_names():
    assert_clean(["19990101000000Bh", "20000101000000Ch.jpeg"], "unit test")


def test_assert_clean_raises_on_test_names():
    with pytest.raises(ContaminationError):
        assert_clean(["19990101000000Bh", "010103-" + A_TEST_FILE], "unit test")
