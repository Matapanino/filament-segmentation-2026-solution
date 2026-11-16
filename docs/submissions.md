# Submissions and validation

Every submission we made, how to rebuild each one, and how we decided between them. The [README](../README.md) covers the final submission; this page covers the rest.

## Which variant should I run?

| Variant | `MODE` | CV PQ ↑ (707 images) | Public LB | Runtime and hardware | Needs |
| --- | --- | ---: | ---: | --- | --- |
| **Final submission** | `"ms125"` | 0.4566 | 0.40 | about 9 min on CPU (Kaggle) | competition data only |
| More Mask2Former views | `"ms4"` | 0.4574 | 0.40 | about 10 min on CPU (Kaggle) | competition data + the release dataset (private, not published) |
| Earlier pipeline, every map computed in the notebook | `"pcombo"` | 0.4296 | 0.38 | about 14 min on one T4 + 5 min CPU | competition data only |

- **Final submission** is the one we selected on Kaggle. Its Mask2Former maps come from five models trained on 1536-px crops, each read in 8 flips/rotations and once more at 1.25× scale.
- **More Mask2Former views** (`"ms4"`) reads the 1.25× scale in 8 views and adds a 1.5× reading: +0.0008 PQ over the final in cross-validation, but neither step passed our adoption rule. It needs the private release dataset, so it cannot be rebuilt from this repository alone.
- The **final submission** builds its CSV from probability maps stored in this repository (`weights/final/`, with the five Mask2Former and ten U-Net fold models that made them), so it is exact and needs no GPU. On a T4 the notebook also re-reads two test images with those models and shows how closely the maps match; re-reading all 180 images takes about 3 h on two T4s and gives slightly different maps (0.5–1.7 % of Mask2Former pixels in our checks, about three quarters of them by 1–2 levels of 510) ([reproduction notes](reproduction_notes.md#re-running-the-networks-of-the-chain-modes)).
- The **earlier pipeline** runs every network in the notebook. It is weaker (−0.027 PQ), but it is the simplest path to adapt to new data and the one the report describes stage by stage.
- The other later submissions (such as `"ms4"`) read their stored maps from the **release dataset** (models and maps, 3.8 GB, listed with hashes in `src/filseg/reproduction/data/chain/release_files.json`). That dataset is private and optional: only those other modes use it; the final submission rebuilds from this repository and the competition data.

CV PQ is pooled over all images and annotation sets, with each annotator's labels scored separately ([Validation](#validation)). The `"ms125"` and `"ms4"` values were measured on our own CPU server, the `"pcombo"` value on Kaggle CPU; the two agree to within 0.00001 PQ on the same run. All our submissions are in [Submissions](#submissions).

To build another variant, set `MODE` in the notebook (for example `MODE = "pcombo"`, which needs only the competition data and a T4).

## Submissions

Every `MODE` of the notebook (`--mode` on the command line) rebuilds one of our submissions and checks it against the submitted CSV. The first eight read the maps of the networks in this repository; the last twelve (from `"ms125"`) read stored maps: `"ms125"`, `"ms125m4"` and `"ms125m4r4"` from `weights/final/` in this repository, the other nine from the private release dataset. `"ms125m4"` is the final submission with only the instance filter changed: its keep probability is the plain mean of the final filter's logistic regression and a LightGBM, an XGBoost and a CatBoost model fitted on the same candidates and features (cutoff 0.325). `"ms125m4r4"` keeps the same masks as `"ms125m4"` and changes only the per-mask repair threshold: the plain mean of the final repair regressor's prediction and those of a LightGBM, an XGBoost and a CatBoost regressor fitted on the same rows, snapped to the same nine thresholds. The names are cumulative: "Refit: + context + neighbours" is the filter refitted with the base, context and neighbour features.


| Submission | `MODE` | CV PQ ↑, 425 images | CV PQ ↑, all 707 images | Public LB | Private LB |
| --- | --- | ---: | ---: | ---: | ---: |
| First filter, no repair | `"equivalent"` | 0.418 | 0.419 | 0.36 | 0.33 |
| First filter, one-threshold repair | `"pnat"` | 0.419 | 0.421 | 0.36 | 0.34 |
| First filter, per-object repair | `"pcand"` | 0.421 | 0.422 | 0.37 | 0.34 |
| Refit: + context | `"fresh_ctrl"` | 0.426 | 0.426 | 0.37 | 0.34 |
| Refit: + context + neighbours | `"pneigh"` | 0.426 | 0.427 | 0.37 | 0.33 |
| Refit: + context + Mask2Former shape | `"mfrich"` | 0.427 | 0.429 | 0.38 | 0.35 |
| Refit: all four groups | `"pcombo"` | 0.428 | 0.430 | 0.38 | 0.35 |
| All four groups + 8-view Mask2Former | `"mftta"` | 0.429 | 0.431 | 0.37 | 0.34 |
| **Crop-1536 Mask2Former, 8 views + 1.25× scale; refitted filter and repair; 5-fold U-Nets (final)** | **`"ms125"`** | — | **0.4566** (server) | **0.40** | **0.37** |
| Crop-1536 Mask2Former, 8 views + 1.25× scale; final filter averaged with LightGBM, XGBoost and CatBoost; refitted repair; 5-fold U-Nets | `"ms125m4"` | — | 0.4584 (Kaggle) | 0.40 | 0.37 |
| Crop-1536 Mask2Former, 8 views + 1.25× scale; four-classifier filter; repair threshold averaged over four regressors; 5-fold U-Nets | `"ms125m4r4"` | — | 0.4589 (Kaggle) | 0.40 | 0.37 |
| Crop-1536 Mask2Former, 8 views + 1.25× scale; filter = TabPFN 3.5 classifier alone; repair threshold = TabPFN 3.5 regressor alone; 5-fold U-Nets | not in this repository yet | — | 0.4590 (Kaggle) | 0.40 | 0.36 |
| Crop-1536 Mask2Former, 8 views + 1.25× scale; filter and repair threshold = plain means of five models (the four of `"ms125m4r4"` + TabPFN 3.5); 5-fold U-Nets | not in this repository yet | — | 0.4591 (Kaggle) | 0.40 | 0.37 |
| Crop-1536 Mask2Former, 8 views + 1.25× scale in 8 views; refitted filter and repair; 5-fold U-Nets | `"ms125d4"` | — | 0.4572 (server) | 0.40 | 0.37 |
| Crop-1536 Mask2Former, 8 views + 1.25× (8 views) + 1.5× (4 views); refitted filter and repair; 5-fold U-Nets | `"ms4"` | — | 0.4574 (server) | 0.40 | 0.37 |
| Crop-1536 Mask2Former, 8 views; refitted filter and repair; 5-fold U-Nets | `"f1c15u"` | — | 0.4549 (Kaggle) | 0.40 | 0.36 |
| Crop-768 + crop-1024 Mask2Former, 8 views; refitted filter; 5-fold U-Nets | `"c7c10d4_un5"` | — | 0.4514 (Kaggle) | 0.40 | 0.36 |
| Crop-768 + crop-1024 Mask2Former; refitted filter and repair; 5-fold U-Nets | `"c7c10_refit_un5"` | — | 0.4450 (Kaggle) | 0.40 | 0.36 |
| Crop-768 + crop-1024 Mask2Former; refitted filter; 5-fold U-Nets | `"c7c10_un5"` | — | 0.4441 (Kaggle) | 0.40 | 0.36 |
| Crop-768 + crop-1024 Mask2Former; refitted filter | `"c7c10"` | — | 0.4441 (Kaggle) | 0.40 | 0.36 |
| Crop-768 Mask2Former; refitted filter | `"c768"` | — | 0.4402 (Kaggle) | 0.40 | 0.35 |
| Crop-768 Mask2Former, Mask2Former weight 0.25; refitted filter | `"c768_w25"` | — | 0.4359 (Kaggle) | 0.38 | 0.35 |


The later submissions (from `"ms125"`) were compared on all 707 images, so their 425-image column is left empty; "server" or "Kaggle" names the CPU that measured the value (the two agree to 0.00001 PQ on the same run). Changing only the test-time U-Net maps (the "5-fold U-Nets" rows) leaves the cross-validation unchanged, because validation always reads one out-of-fold model per network. The public leaderboard shows two decimals and does not separate these variants.

The two TabPFN rows were built on Kaggle and have no `MODE` yet: Kaggle submissions 58420651 (CSV `474eda9a9f743679824f67c8ec8eba42899df20c71f5c10256b4b10993468cf5`) and 58420900 (CSV `61e4bff9e650a8f66dc2502ed98fccb9af80307156b2262f12bae2c8dec6006d`). See [Post-processing variants](#post-processing-variants).

**Private leaderboard.** Added after the competition closed; like the public one, it shows two decimals. The final submission scores 0.37, which no submission of ours beats.

`"mftta"` averages each Mask2Former model of `"pcombo"` over 8 flips and right-angle rotations of its input: +0.0012 [−0.0009, +0.0033] PQ over `"pcombo"` on all 707 images (17/20 partitions) and +0.0005 [−0.0018, +0.0029] on the 425 validation images (11/20); both intervals include zero, so our rule did not adopt it.

One submission is not rebuilt here: a 2026-11-13 variant whose filter read 14 extra features from per-fold crop-768 Mask2Former maps (public 0.38); its feature code is not part of this package. Submissions made before this package existed are described in the report.

The CSV hash of every `MODE` and its Kaggle submission number are in the [reproduction notes](reproduction_notes.md#expected-results).

**Outside reference.** A public notebook in this competition that fuses full-resolution U-Nets with several encoders (B2–B5) and a YOLO segmentation model by a pixel-wise AND reported 0.42 on the public leaderboard, against 0.40 for `"ms125"`. This supports the first item of "How to extend" in the report: scale resolution together with encoder capacity.

## Post-processing variants

Both learned steps after the maps are small tabular models: the instance filter (a keep-or-drop classifier; logistic regression in `"ms125"`) and the per-mask repair threshold (a regressor; gradient-boosted trees in `"ms125"`). On the last day we compared alternatives for both with the same nested 20-partition protocol, run on Kaggle (`"ms125"` scores 0.4566 there). ΔPQ is the pooled change on all 707 images, with the 95 % interval over partitions where it was computed; "Up" counts the partitions (of 20) where the change was positive.

| Change | Compared with | ΔPQ | Up | CV PQ (Kaggle) |
| --- | --- | ---: | ---: | ---: |
| Filter = plain mean of logistic regression, LightGBM, XGBoost, CatBoost (`"ms125m4"`) | `"ms125"` | +0.0018 [+0.0008, +0.0030] | 20/20 | 0.4584 |
| \+ repair threshold = plain mean of four regressors (gradient-boosted trees, LightGBM, XGBoost, CatBoost; `"ms125m4r4"`) | `"ms125m4"` | +0.0005 [+0.0002, +0.0008] | 19/20 | 0.4589 |
| TabPFN 3.5 in both steps (classifier and regressor alone) | `"ms125m4r4"` | +0.0001 | 10/20 | 0.4590 |
| Five-model means in both steps (the four models + TabPFN 3.5) | `"ms125m4r4"` | +0.0002 | 15/20 | 0.4591 |
| TabPFN 3.5 classifier alone as the filter (four-regressor repair) | `"ms125m4r4"` | −0.0006 | 3/20 | 0.4584 |
| TabPFN 3.5 regressor instead of the four-regressor mean (both with the TabPFN filter) | the row above | +0.0006 | 19/20 | 0.4590 |
| Ridge regression (linear) as the repair regressor | `"ms125"` | −0.0014 | 0/20 | 0.4552 |
| Logistic-regression filter with logit-transformed features | the logistic-regression filter, same run | −0.0003 | 6/20 | — |
| Logistic-regression filter with rank-Gauss features | the logistic-regression filter, same run | −0.0008 | 3/20 | — |
| Logistic-regression filter with spline features | the logistic-regression filter, same run | −0.0008 | 4/20 | — |
| Filter trained on the IoU each mask earns (ridge) | the logistic-regression filter, same run | −0.0038 | 0/20 | — |
| Filter trained on the IoU each mask earns (gradient-boosted trees) | the logistic-regression filter, same run | −0.0010 | 3/20 | — |

Our comparisons suggest that the filter works well with a linear model, while the repair threshold benefits from a nonlinear one: a ridge regressor keeps only 77 % of the repair gain (0.4506 without repair, 0.4552 with ridge, 0.4566 with gradient-boosted trees), whereas the nonlinear alternatives (four-regressor mean, TabPFN regressor) gain further. For the filter, logistic regression is the simple choice we ship; averaging it with dissimilar classifiers adds the largest step. The gain comes from averaging dissimilar models in both steps.

The four submitted variants (`"ms125m4"`, `"ms125m4r4"`, TabPFN in both steps, five-model means) all score 0.40 on the public leaderboard (two decimals), as does `"ms125"`; that cannot rank them, so the final submission stays `"ms125"`, the simplest. `"ms125m4"` and `"ms125m4r4"` rebuild from this repository (a checkout of `final-ms125`); the TabPFN and five-model submissions are listed in [Submissions](#submissions) but are not reproducible from this repository yet.

## Validation

- **Folds.** Observations at most 24 h apart are chained into one temporal group, so near-duplicate views never fall on both sides of a split. The groups form five folds (142/142/141/141/141 images). The validation set is folds 0–2: 425 images in 343 groups; we also score all 707 images (571 groups). Fold table: `campaigns/20261110_yolo_roi_crossfold/evidence/fold_table.json`.
- **Each annotation set is scored separately.** One prediction list is matched to each annotator's labels (IoU > 0.5), and true positives, false positives, false negatives and matched IoU are pooled over all (image, annotation set) pairs before PQ is computed. We never merge the annotators' labels.
- **Light learned models are cross-fitted.** The instance filter and the threshold regressor are refitted in 20 random group partitions (train on four fifths, score the fifth), and the counts are averaged over the partitions.
- **Adoption rule.** A change is adopted only if the paired PQ difference is positive, at least 16 of the 20 partitions are positive, and the 95% paired temporal-group bootstrap interval (2000 draws) excludes zero.
- **Not the leaderboard.** The public MAGFiLO v1.0 release overlaps the test images, the leaderboard shows two decimals while our decisions turned on thousandths, and its scorer differs from the host's self-evaluation notebook in undisclosed ways. `src/filseg/guard.py` pins the 180 test filenames and refuses any test annotation in training or validation.
- **Validated and deployed versions differ.** Validation uses one out-of-fold model per network; the test pipeline uses the mean of the fold models (final variant: five U-Net and five Mask2Former fold models; no-download variant: the fold-2 U-Nets and three Mask2Former models), with the filter and the regressor fitted once on all 707 training images.

The contribution of each stage, the error analysis and the rejected ideas are in the report.

## Evaluating a change

To evaluate a change the way we did:

- **Score each annotation set separately.** Match one prediction list to each annotator's labels (IoU > 0.5) and pool the counts over all (image, annotation set) pairs before computing PQ: `src/filseg/evaluate.py`.
- **Use temporal folds.** Images taken within 24 h of each other share a group and a fold, so near-duplicate views never sit on both sides of a split: `scripts/make_folds.py`, fold table `configs/` and `campaigns/20261110_yolo_roi_crossfold/evidence/fold_table.json`.
- **Compare paired, over 20 partitions.** Refit the light learned steps (filter, repair) in 20 random group partitions, score the held-out fifth, and adopt a change only if the paired PQ difference is positive in at least 16 of 20 partitions and its 95 % temporal-group bootstrap interval excludes zero ([Validation](#validation); scripts in [`training/`](../training/README.md)).
