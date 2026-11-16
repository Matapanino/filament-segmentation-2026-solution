"""Machine-enforced isolation of the competition test set.

Campaign policy (ADR-0002): no annotation attached to any of the 180 test image
files may enter training, tuning, validation, model selection, or
post-processing calibration. This module is the single enforcement point; the
pinned filename list ships with the package so the guard cannot drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

_DATA = Path(__file__).parent / "data" / "test_filenames.txt"

TEST_FILENAMES: frozenset[str] = frozenset(
    line.strip() for line in _DATA.read_text().splitlines() if line.strip()
)


class ContaminationError(RuntimeError):
    """Raised when test-set files are about to leak into a fitting/tuning path."""


def file_base(name: str) -> str:
    """Normalize an image reference to its bare observation name.

    Accepts a bare name ('20110120105534Ch'), a filename with extension, a path,
    or an annotation-image id of the form '<annotatorBatch>-<name>'
    (e.g. '010101-20110120105534Ch').
    """
    name = name.rsplit("/", 1)[-1]
    if "-" in name:
        name = name.split("-", 1)[1]
    for suffix in (".jpeg", ".jpg"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def is_test_file(name: str) -> bool:
    return file_base(name) in TEST_FILENAMES


def assert_clean(names: Iterable[str], context: str) -> None:
    """Raise ContaminationError if any name refers to a test-set file.

    `context` names the pipeline stage (e.g. 'train-fold assembly') so a
    failure is self-explaining in logs and CI.
    """
    offenders = sorted({file_base(n) for n in names} & TEST_FILENAMES)
    if offenders:
        raise ContaminationError(
            f"{context}: {len(offenders)} test-set file(s) would leak in, "
            f"e.g. {offenders[:5]}"
        )
