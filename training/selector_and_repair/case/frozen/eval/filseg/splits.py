"""Group-aware cross-validation splits (ADR-0004 §1).

K=5, grouped by unique observation FILE — never by annotation-image entry:
44% of files carry 2-3 independent annotator sets re-exposed as separate
``images[]`` entries, and an entry-level split would put the same pixels in
train and validation. Folds are stratified by per-file filament count (median
across the file's annotator sets). Deterministic under a fixed seed.

Isolation: every public entry point re-asserts the pinned 180-file test set
(``guard.assert_clean``), so a contaminated annotation source fails loudly
before any fold is materialized.

The shipped fold table lives in ``configs/cv_folds_k5_seed0.json`` (written by
``scripts/make_folds.py``); all experiments must consume that one table so
scores stay paired across models and sessions.
"""
from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .coco_io import annotation_sets
from .guard import assert_clean, file_base


def fold_assignment(coco: dict, k: int = 5, seed: int = 0) -> dict[str, int]:
    """Assign every observation file in ``coco`` to one of ``k`` folds.

    Stratification: shuffle files (seed), then stable-sort by the file's
    median filament count and deal round-robin — every count stratum spreads
    across folds to within one file, fold sizes differ by at most one, and
    ties are broken randomly but reproducibly.
    """
    grouped = annotation_sets(coco)
    assert_clean(grouped.keys(), "CV fold assignment")
    med_count = {
        f: statistics.median(len(s.annotations) for s in sets)
        for f, sets in grouped.items()
    }
    files = sorted(grouped)
    random.Random(seed).shuffle(files)
    files.sort(key=lambda f: med_count[f])  # stable: shuffle survives within a stratum
    return {f: i % k for i, f in enumerate(files)}


def fold_files(assignment: dict[str, int], fold: int, role: str) -> list[str]:
    """The train or val file list of one fold — the loader's single entry point.

    ``role`` is ``"train"`` (files outside ``fold``) or ``"val"`` (files in
    ``fold``). Re-asserts test isolation on the returned list.
    """
    if role not in ("train", "val"):
        raise ValueError(f"role must be 'train' or 'val', got {role!r}")
    want_val = role == "val"
    files = sorted(f for f, g in assignment.items() if (g == fold) == want_val)
    assert_clean(files, f"fold {fold} {role} file list")
    return files


def entries_for_files(coco: dict, files: list[str] | set[str]) -> list[str]:
    """All annotation-image entry ids belonging to ``files`` (training samples
    are per-entry — one sample per annotator set, ADR-0004 §2)."""
    keep = {file_base(f) for f in files}
    assert_clean(keep, "entry selection")
    return sorted(
        im["id"] for im in coco["images"] if file_base(im["file_name"]) in keep
    )


def save_assignment(assignment: dict[str, int], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(assignment, indent=0, sort_keys=True))


def load_assignment(path: str | Path) -> dict[str, int]:
    assignment = json.loads(Path(path).read_text())
    assert_clean(assignment.keys(), f"fold table {path}")
    return assignment


def fold_summary(assignment: dict[str, int], coco: dict) -> dict:
    """Post-hoc balance check (ADR-0004: observatory/year checked, not forced)."""
    grouped = annotation_sets(coco)
    per_fold: dict[int, dict] = defaultdict(
        lambda: {
            "n_files": 0,
            "n_entries": 0,
            "n_filaments": 0,
            "site": Counter(),
            "year": Counter(),
        }
    )
    for f, fold in assignment.items():
        s = per_fold[fold]
        s["n_files"] += 1
        sets = grouped.get(f, [])
        s["n_entries"] += len(sets)
        s["n_filaments"] += sum(len(x.annotations) for x in sets)
        s["site"][f[-2]] += 1  # ...<site>h -> site letter
        s["year"][f[:4]] += 1
    return {
        fold: {
            "n_files": v["n_files"],
            "n_entries": v["n_entries"],
            "n_filaments": v["n_filaments"],
            "site": dict(sorted(v["site"].items())),
            "year": dict(sorted(v["year"].items())),
        }
        for fold, v in sorted(per_fold.items())
    }
