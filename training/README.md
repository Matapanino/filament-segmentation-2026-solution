# Training scripts (as run)

> **These are the scripts as they were run for our submissions. They were not re-run in this repository.**
> They are copies, byte for byte except for a privacy clean-up (machine paths, account names and storage locations
> replaced by placeholders); `SOURCES.md` lists every file, its origin, the SHA-256 of the original and what the
> clean-up changed. They ran on our own machines and on Kaggle, so paths, input names and hash checks follow those
> set-ups. Read them as the exact recipe, and expect to adapt paths and input pins before running them on new data.

The submitted pipeline itself (inference, no training) is in `src/filseg/reproduction/`. The trained models and stored
maps of the final submission are in `weights/final/` (for example `mfmodel_c1536_f<k>.safetensors`,
`unetmodel_f<k>_full.pt`, `unetmodel_f<k>_local.pt`); those of the other later submissions are release files, listed
with size and SHA-256 in `src/filseg/reproduction/data/chain/release_files.json`. No weights or maps are stored in this
folder.

## The four learned stages

| Stage | What it learns | Retrained by | Folder |
| --- | --- | --- | --- |
| Full-disk U-Net and local U-Net | per-pixel filament probability (whole disk at 1024 px; native 1024 px crops with raw, CLAHE and detail channels) | `run_train.py` → `train_cuda.py` | `unet/` |
| Mask2Former | filament instances (Swin-T, one class) on 1536 px crops of the native 2048 px image | `train.py`, then the map exporters | `mask2former/` |
| Instance filter | keep or drop each candidate (StandardScaler + logistic regression on 68 features) | `case/worker.py` | `selector_and_repair/` |
| Mask repair | a per-mask redraw threshold (gradient-boosted regressor, 64 features) | `case/worker.py` (same run) | `selector_and_repair/` |

Each network was trained five times, once per cross-validation fold (four folds in, one held out). The held-out maps
feed the filter and the repair fits; for the test images the five fold models are averaged per pixel. The final map is
0.55 × Mask2Former + 0.225 × full-disk U-Net + 0.225 × local U-Net.

## 1. U-Nets (`unet/`)

Recipe: `segmentation_models_pytorch` U-Net, ImageNet efficientnet-b0 encoder, BCE + soft Dice per annotation set
(averaged over the image's sets), 12 epochs of 1136 sampled images, batch 2 with 4-step accumulation, AdamW
(2×10⁻⁴, weight decay 10⁻⁴), cosine schedule, flips, FP16, seed 42, last epoch kept (no validation-based selection).

As run: one Kaggle GPU notebook per fold, accelerator T4 ×2, both views at once (one per GPU). The notebook wrote
`train_cuda.py`, `prepare_train.py`, `run_train.py`, `views.py`, `protocol.json` and `wave.json` into the working folder,
installed `segmentation-models-pytorch==0.5.0 timm==1.0.22 pycocotools==2.0.10`, and ran

```bash
python -u run_train.py      # runs prepare_train.py (image + mask cache), then train_cuda.py <job>.json per view
```

- Inputs: the competition data and this repository's `src/filseg` attached as a dataset (both found under
  `/kaggle/input`); `protocol.json` (held-out fold, training and validation file lists, fold-table hash; the copy here
  is the fold-0 one) and `wave.json` (the two jobs: `full_raw_1024`, `local_raw_clahe_detail`).
- Outputs per view, in `artifacts/<job id>/`: `last.pt`, `oof_prob.npz` (held-out images), `test_prob.npz` (the 180
  test images), `result.json`.
- Time (fold 0 log): full-disk 41 min, local 72 min, 76 min for the notebook. Fold 2 was trained over several resumed
  sessions; its time is not recorded as one figure.
- Test maps of the final submission: `fold5mean.py` averages the five fold models' `test_prob.npz`
  (`uint8(rint(Σ/5))`; CPU). `test_maps.py infer` makes test maps from checkpoints instead.

## 2. Mask2Former (`mask2former/`)

Recipe (`train_done_c1536_f0.json` holds the full config): `facebook/mask2former-swin-tiny-coco-instance`, new
one-class head; native 2048 px image, random 1536 px crops with rotation, scale 0.8–1.25 and flips, two thirds of the
crops centred on an annotated filament; one randomly drawn annotation set per crop; 4000 optimizer steps of 8 crops;
AdamW (backbone 10⁻⁵, rest 10⁻⁴, weight decay 0.01), 500 warm-up steps then cosine; clipping 1.0; BF16; weight EMA
0.999, the final EMA is the model. Environment: torch 2.5.1, transformers 4.57.1, one RTX 3090 per fold.

As run, for each fold k = 0…4, from a root folder holding this folder as `code/` and the competition data as
`data/MAGFiLO_1.0_Kaggle_2026/` (the evaluator and the exporters look there):

```bash
python code/train.py --run-dir RUN --data-root data/MAGFiLO_1.0_Kaggle_2026 \
    --fold-table code/fold_table_time24h_k5_seed42.json --num-workers 1 --torch-threads 1 \
    --total-steps 4000 --crop 1536 --accum 8 --fold k --seed 42
python code/eval_ckpt.py --run-dir RUN --which final --fold k --export                  # held-out score, plain maps
python code/export_tta.py --run-dir RUN --out-dir RUN_d4 --views 8 --torch-threads 1     # 8 flip/rotation views, 2048 px
python code/export_scale.py --run-dir RUN --out-dir RUN_s125 --size 2560 --views 2 --torch-threads 1   # 1.25x, 2 views
```

- Outputs: `final_ema.safetensors`, `train_done.json`, `train.log`; each exporter writes `MF_f<k>_val.npz`
  (held-out images) and `MF_f<k>_test.npz` (test images) as uint16 maps (value / 510).
- Time (training logs): 4.2–5.9 h per fold on one RTX 3090; the 1.25× export took about 5 min for fold 0.
- The final map averages, per fold, the 8-view archive and the 1.25× archive, then the five folds.
- `train.py`, `model.py` and `geom.py` are our working copies: after these runs they gained opt-in switches that are
  off by default (tests check that the default config, loss and draws are unchanged). Tests: `python -m pytest tests`.

## 3. Instance filter and mask repair (`selector_and_repair/`)

Both are fitted in one CPU run of `case/worker.py`, on the candidates of the held-out maps of all 707 training images:

- **Filter:** one row per (candidate, annotation set), label 1 if IoU > 0.5 with a filament of that set. Cross-fitted
  in 20 random partitions × 5 outer folds (100 fits); inside each, 3 inner folds choose the cutoff. The deployed filter
  is one fit on all 707 images with the cutoff = median of the 100 nested cutoffs (0.35 for the final submission).
- **Repair:** rows = candidates the first, simpler filter keeps; target = the redraw threshold (9 values, 0.20–0.60)
  that best matches the annotation; `HistGradientBoostingRegressor(random_state=42)`; 100 cross-fitted copies for the
  scores, one fit on all 707 for the test images.

As run for the final submission (its arguments are in `cli_ms125.json`; `case/chain_inputs.json` pins the map
archives): `stage.py` collected the held-out map archives, `render_kernel.py` packed the code into one notebook
(`kernel_ms125.py` is its driver, with the packed code removed), and the notebook ran

```bash
pip install numpy==2.1.3 scipy==1.16.3 scikit-learn==1.6.1 pandas==2.3.3 scikit-image==0.25.2 "opencv-python-headless>=4.14.0,<4.14.1" pycocotools
cd case && python -B run_tests.py && python -B -u frozen/evaluate.py && python -B -u worker.py
python -B -u csv_worker.py <package> <test images> <weights> <maps> <inputs> <out/csv> case/out/ship case/chain_inputs.json
```

The CSV step installs its own pinned packages first (listed in `kernel_ms125.py`).

- Outputs: `case/out/ship/w55_selector.json` (the filter, shipped as `selector_ms125.json`),
  `case/out/deploy/w55_regressor.pkl` (the repair regressor, shipped as `regressor_ms125.pkl`), cross-validation
  counts (`counts.json`, `pcand_counts.json`) and the test CSVs.
- Time: 1 h 50 min on 4 cores of an AMD EPYC 7272 server (CPU only). A similar run on a Kaggle CPU notebook with two
  blend weights took 3.7 h.
- Inputs it reads besides the new map archives: the stored held-out U-Net maps, the first filter's saved models and
  records of earlier runs, attached as our private Kaggle datasets (named in `render_kernel.py`). These are not in the
  release files, so a rerun needs them rebuilt from the held-out maps of sections 1 and 2.

## Plugging in new labelled images

- **Format:** the competition's COCO JSON (`MAGFiLO_1.0_Annotations_kaggle2026_train.json` layout): one image entry
  per (image, annotator), polygons in `segmentation`. Each annotator's set is kept separate everywhere.
- **Folds:** observations at most 24 h apart form one group; groups are dealt into 5 folds
  (`case/frozen/eval/make_time_folds.py`: `temporal_groups(files, 24)`, then `assign_groups(groups, k=5, seed=42)`).
  Keep the 180 test images out (`src/filseg/guard.py` refuses them).
- **Pins to update:** the scripts check the inputs they were run on. Mask2Former `train.py` checks the SHA-256 of the
  annotation file; the U-Net `prepare_train.py` checks that the annotation file holds exactly the listed training and
  validation files; the fitting run expects 707 images and the hashes in `case/protocol.json` and
  `case/chain_inputs.json`. Replace these with the new file's values.
- **Order:** retrain both U-Nets and Mask2Former for all five folds; export the held-out and test maps; then refit the
  filter and the regressor on the new held-out maps.

## Evaluating a change

- **Score:** PQ with each annotation set scored separately: one prediction list is matched against each annotator's
  masks (IoU > 0.5); true positives, false positives, false negatives and matched IoU are pooled over all
  (image, annotation set) pairs before PQ is computed. Code: `case/frozen/eval/pq_metric.py` (as run) and
  `src/filseg/evaluate.py` (package).
- **Cross-validation:** the five temporal folds above for the networks; the 20 partitions for the filter and the
  repair, with counts averaged over the partitions (`case/worker.py`, `summarize`).
- **Adoption rule:** a change is adopted only if the paired PQ difference is positive, at least 16 of the 20 partitions
  are positive, and the 95% paired temporal-group bootstrap interval (2000 draws) excludes zero. `finish.py` computes
  these contrasts for one run; `paired_vs_ref.py <run> <reference>` compares two runs image by image.

Comments inside the copied code use our internal experiment names; they are labels only. File-level notes are in
`SOURCES.md`.
