"""Gate (ii) audit: Kaggle competition train JSON vs public MAGFiLO v1.0.

ADR-0002 gate (ii): before building on the Kaggle annotations, measure whether
they are identical to, or a revision of, the public MAGFiLO v1.0 release —
entry counts, annotation identity (matched by the shared UUID annotation ids),
category schema (the Kaggle data documents a 4th category "Ambiguous" absent
from public v1.0), and the geometry facts F-002/F-003/F-004 recomputed on the
Kaggle JSON with the same definitions used on the public JSON:

- effective width  = area / polyline length of ``spine``
- seam-cut rate    = fraction of filaments whose bbox straddles a tile-grid line
- gap-closing risk = fraction of spine endpoints with another filament's spine
  point within r px (same annotation entry), + median NN centroid distance

Usage:
    python scripts/gate2_audit.py \
        --kaggle ../data/kaggle/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --public ../data/external/magfilo_2024_v1.0.json \
        --train-images ../data/kaggle/MAGFiLO_1.0_Kaggle_2026/train/train_images \
        --out outputs/gate2_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.guard import TEST_FILENAMES, assert_clean, file_base  # noqa: E402


def polyline_length(flat_xy: list[float]) -> float:
    p = np.asarray(flat_xy, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def pctiles(x) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    qs = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "mean": float(x.mean()),
        "p5": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": int(x.size),
    }


def seam_cut_rate(anns: list[dict], tile: int) -> float:
    """Fraction of filaments whose bbox straddles a tile boundary (grid anchored
    at (0,0)); bbox straddling ~= the mask is cut, since filaments span their
    bbox lengthwise."""
    cut = 0
    for a in anns:
        x, y, w, h = a["bbox"]
        if int(x // tile) != int((x + w) // tile) or int(y // tile) != int((y + h) // tile):
            cut += 1
    return cut / len(anns) if anns else float("nan")


def endpoint_proximity(coco: dict) -> dict:
    """Per annotation entry: spine endpoints of each filament vs OTHER
    filaments' spine points; + NN centroid distances."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    n_endpoints = 0
    near = {20: 0, 40: 0}
    nn_centroid: list[float] = []
    for anns in by_image.values():
        spines = [np.asarray(a["spine"], dtype=float).reshape(-1, 2) for a in anns]
        cents = [s.mean(axis=0) for s in spines]
        if len(anns) >= 2:
            ct = cKDTree(np.array(cents))
            d, _ = ct.query(np.array(cents), k=2)
            nn_centroid.extend(d[:, 1].tolist())
        for i, s in enumerate(spines):
            others = [p for j, p in enumerate(spines) if j != i]
            if not others:
                continue
            tree = cKDTree(np.vstack(others))
            for ep in (s[0], s[-1]):
                n_endpoints += 1
                dist = tree.query(ep)[0]
                for r in near:
                    if dist <= r:
                        near[r] += 1
    return {
        "n_endpoints_with_neighbors": n_endpoints,
        "frac_endpoint_within_20px": near[20] / n_endpoints if n_endpoints else None,
        "frac_endpoint_within_40px": near[40] / n_endpoints if n_endpoints else None,
        "nn_centroid_distance": pctiles(nn_centroid),
    }


def geometry(coco: dict) -> dict:
    anns = coco["annotations"]
    areas = [a["area"] for a in anns]
    spine_len = [polyline_length(a["spine"]) for a in anns]
    width = [ar / sl for ar, sl in zip(areas, spine_len) if sl > 0]
    bw = [a["bbox"][2] for a in anns]
    bh = [a["bbox"][3] for a in anns]
    per_entry = Counter(a["image_id"] for a in anns)
    n_entries = len(coco["images"])
    fil_per_entry = [per_entry.get(im["id"], 0) for im in coco["images"]]
    return {
        "area_px": pctiles(areas),
        "spine_length_px": pctiles(spine_len),
        "effective_width_px": pctiles(width),
        "bbox_w_px": pctiles(bw),
        "bbox_h_px": pctiles(bh),
        "filaments_per_entry": pctiles(fil_per_entry),
        "n_zero_filament_entries": int(sum(1 for c in fil_per_entry if c == 0)),
        "foreground_fraction": float(sum(areas) / (n_entries * 2048 * 2048)),
        "category_counts": dict(Counter(a["category_id"] for a in anns)),
        "seam_cut_rate_512": seam_cut_rate(anns, 512),
        "seam_cut_rate_1024": seam_cut_rate(anns, 1024),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--train-images", default=None)
    ap.add_argument("--out", default="outputs/gate2_audit.json")
    args = ap.parse_args()

    with open(args.kaggle) as f:
        kag = json.load(f)
    with open(args.public) as f:
        pub = json.load(f)

    out: dict = {"kaggle_path": args.kaggle, "public_path": args.public}

    # --- 0. schema & counts -------------------------------------------------
    out["schema"] = {
        "kaggle": {
            "n_image_entries": len(kag["images"]),
            "n_annotations": len(kag["annotations"]),
            "categories": kag["categories"],
            "info_dates": {
                k: v for k, v in kag.get("info", {}).items() if "date" in k
            },
        },
        "public": {
            "n_image_entries": len(pub["images"]),
            "n_annotations": len(pub["annotations"]),
            "categories": pub["categories"],
        },
    }

    # --- 1. isolation: no test files in the Kaggle train JSON ---------------
    kag_files = {file_base(im["file_name"]) for im in kag["images"]}
    assert_clean(kag_files, "gate2: Kaggle train JSON file set")
    out["isolation"] = {
        "kaggle_unique_files": len(kag_files),
        "test_files_in_train_json": 0,  # assert_clean passed
    }
    if args.train_images:
        disk = {file_base(p.name) for p in Path(args.train_images).glob("*.jpeg")}
        out["isolation"]["train_images_on_disk"] = len(disk)
        out["isolation"]["json_files_missing_on_disk"] = sorted(kag_files - disk)
        out["isolation"]["disk_files_missing_in_json"] = sorted(disk - kag_files)

    # --- 2. entry-level diff -------------------------------------------------
    kag_entry_ids = {im["id"] for im in kag["images"]}
    pub_entry_ids = {im["id"] for im in pub["images"]}
    pub_train_entry_ids = {
        im["id"] for im in pub["images"] if file_base(im["file_name"]) in kag_files
    }
    pub_ann_by_entry = Counter(a["image_id"] for a in pub["annotations"])
    dropped = pub_train_entry_ids - kag_entry_ids
    added = kag_entry_ids - pub_entry_ids
    out["entries"] = {
        "public_entries_on_kaggle_train_files": len(pub_train_entry_ids),
        "kaggle_entries": len(kag_entry_ids),
        "public_entries_dropped_in_kaggle": len(dropped),
        "dropped_entry_filament_counts": dict(
            Counter(pub_ann_by_entry.get(e, 0) for e in dropped)
        ),
        "kaggle_entries_not_in_public": sorted(added),
        "kaggle_zero_filament_entries": None,  # filled after geometry()
    }

    # --- 3. annotation-level diff (UUID join) --------------------------------
    kag_by_id = {a["id"]: a for a in kag["annotations"]}
    pub_by_id = {a["id"]: a for a in pub["annotations"]}
    common = set(kag_by_id) & set(pub_by_id)
    kag_only = set(kag_by_id) - set(pub_by_id)
    pub_train_ann_ids = {
        a["id"] for a in pub["annotations"] if a["image_id"] in pub_train_entry_ids
    }
    pub_train_only = pub_train_ann_ids - set(kag_by_id)

    n_seg_eq = n_spine_eq = n_bbox_eq = n_area_eq = n_cat_eq = 0
    cat_transition: Counter = Counter()
    for uid in common:
        ka, pa = kag_by_id[uid], pub_by_id[uid]
        if ka["segmentation"] == pa["segmentation"]:
            n_seg_eq += 1
        if ka["spine"] == pa["spine"]:
            n_spine_eq += 1
        if ka["bbox"] == pa["bbox"]:
            n_bbox_eq += 1
        if ka["area"] == pa["area"]:
            n_area_eq += 1
        if ka["category_id"] == pa["category_id"]:
            n_cat_eq += 1
        cat_transition[(pa["category_id"], ka["category_id"])] += 1

    out["annotations"] = {
        "common_uuids": len(common),
        "kaggle_only_uuids": len(kag_only),
        "public_trainfile_uuids_not_in_kaggle": len(pub_train_only),
        "of_common_segmentation_identical": n_seg_eq,
        "of_common_spine_identical": n_spine_eq,
        "of_common_bbox_identical": n_bbox_eq,
        "of_common_area_identical": n_area_eq,
        "of_common_category_identical": n_cat_eq,
        "category_transition_public_to_kaggle": {
            f"{p}->{k}": v for (p, k), v in sorted(cat_transition.items())
        },
        "kaggle_only_category_counts": dict(
            Counter(kag_by_id[u]["category_id"] for u in kag_only)
        ),
    }

    # --- 4. geometry (F-002/F-003) + endpoint proximity (F-004) --------------
    out["geometry_kaggle"] = geometry(kag)
    out["entries"]["kaggle_zero_filament_entries"] = out["geometry_kaggle"][
        "n_zero_filament_entries"
    ]
    out["endpoint_proximity_kaggle"] = endpoint_proximity(kag)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
