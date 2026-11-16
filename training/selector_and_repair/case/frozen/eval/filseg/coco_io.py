"""COCO-style I/O for MAGFiLO annotations and competition submissions.

MAGFiLO specifics (verified on the public v1.0 release):
- ``images[].id`` is a string ``<annotatorBatch>-<observationName>``; the same
  observation file may appear as 1-3 independent annotation-image entries.
- ``annotations[].segmentation`` is a single closed polygon per filament.
- Submission masks are COCO *compressed* RLE ``counts`` strings only; the mask
  size is fixed at 2048x2048 for every image.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pycocotools import mask as maskutils

from .guard import file_base

IMG_H = 2048
IMG_W = 2048


def load_coco(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


@dataclass
class AnnotationSet:
    """One annotator's complete labeling of one observation."""

    image_id: str  # e.g. '010101-20110120105534Ch'
    file: str      # e.g. '20110120105534Ch'
    annotations: list[dict] = field(default_factory=list)


def annotation_sets(coco: dict) -> dict[str, list[AnnotationSet]]:
    """Group a COCO dict into ``{observation file -> [AnnotationSet, ...]}``."""
    by_image: dict[str, AnnotationSet] = {}
    for im in coco["images"]:
        by_image[im["id"]] = AnnotationSet(
            image_id=im["id"], file=file_base(im["file_name"])
        )
    for ann in coco["annotations"]:
        entry = by_image.get(ann["image_id"])
        if entry is not None:
            entry.annotations.append(ann)
    grouped: dict[str, list[AnnotationSet]] = defaultdict(list)
    for entry in by_image.values():
        grouped[entry.file].append(entry)
    return dict(grouped)


def poly_to_mask(segmentation, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Rasterize a COCO polygon segmentation to a uint8 {0,1} mask."""
    rles = maskutils.frPyObjects(segmentation, h, w)
    rle = maskutils.merge(rles)
    return maskutils.decode(rle)


def ann_to_mask(ann: dict, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    return poly_to_mask(ann["segmentation"], h, w)


def mask_to_rle_counts(mask: np.ndarray) -> str:
    """Encode a binary mask as the competition's RLE ``counts`` string."""
    rle = maskutils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def rle_counts_to_mask(counts: str, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    rle = {"size": [h, w], "counts": counts.encode("ascii")}
    return maskutils.decode(rle)


def write_submission(
    masks_by_file: dict[str, list[np.ndarray]], path: str | Path
) -> None:
    """Write the competition CSV: one row per predicted filament instance."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for file, masks in masks_by_file.items():
            for k, m in enumerate(masks, start=1):
                writer.writerow([f"{file}_{k}", mask_to_rle_counts(m)])


def read_submission(path: str | Path) -> dict[str, list[np.ndarray]]:
    """Read a submission CSV back into ``{observation file -> [mask, ...]}``."""
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            file = row["filament_id"].rsplit("_", 1)[0]
            out[file].append(rle_counts_to_mask(row["segmentation_rle"]))
    return dict(out)
