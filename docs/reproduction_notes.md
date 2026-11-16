# Reproduction notes

The README's quick start runs everything in one Kaggle notebook. This page holds what you need only to run the pipeline in other ways, to check its outputs, or to see the training recipe and implementation constants that the report leaves out.

## Get the code

```bash
git clone https://github.com/Matapanino/filament-segmentation-2026-solution.git
cd filament-segmentation-2026-solution
```

Every file the final submission (`"ms125"`) and the earlier pipeline read is in the clone (`weights/`, `weights/final/`); nothing else is downloaded. The other later submissions also read the release files: one dataset of models and stored maps, listed with bytes and SHA-256 in `src/filseg/reproduction/data/chain/release_files.json`.

## Expected results

The five networks (two U-Nets, three Mask2Former fold models) give deterministic probability maps. Their map-set SHA-256 (the SHA-256 of the JSON of the archive hashes, sorted keys) is

```
d7aa6a41fafc26b07ce89b306b884f6667ebb68b86289ea4745ee44c0b0d195f
```

Every `MODE` except `"mftta"` reads this map set. `"mftta"` (our second selection) replaces the three Mask2Former maps by their 8-view averages; its map set is

```
44ee26c7d4284b02103f005935611b0911d8f1fe20e1693bf73f0eb21fff9cc2
```

The final submission (`MODE = "ms125"`, the default) has the CSV SHA-256

```
f05ea0d7b6fc8f9345586b193c2866cd654f7fad1b838d609e41af5f46de80ed
```

The no-download variant `"pcombo"` has `35abbeed91108e40ddbd18c094c901d116441e363b10c31aa8092988d8e8f6e6`.

Every `MODE` rebuilds one of our submissions and checks its CSV against the hash below:


| What it builds | `MODE` | CSV SHA-256 | Kaggle submission |
| --- | --- | --- | ---: |
| First filter, no repair | `"equivalent"` | `0b2123de5b614561ddf22c2bbc0f05000164ac6a595a9806104c7358b9f46b70` | 58224327 |
| First filter, one-threshold repair | `"pnat"` | `d54257e7f8a1f815d442037050233e52859b0f9594f128b7b258f368f4d05fbe` | 58224552 |
| First filter, per-object repair | `"pcand"` | `14a5c1b684813081ad129d05e72fbb0c09592f292aeb1b1d84e121100c668c6b` | 58232010 |
| Refit: + context | `"fresh_ctrl"` | `b01866b3f849d7bc43534c188b91545c26f8cd839493175d00ff6d25fb19d085` | 58265972 |
| Refit: + context + neighbours | `"pneigh"` | `c3edf2c8b681d42649dc7ccb6ed10e0400031b37c057d7d2798d6196bed0612c` | 58268408 |
| Refit: + context + Mask2Former shape | `"mfrich"` | `5711532ebb1360bdbc27a2bb04fd1f388b4a6494d0ab911067775aa10c018b85` | 58274749 |
| Refit: all four groups | `"pcombo"` | `35abbeed91108e40ddbd18c094c901d116441e363b10c31aa8092988d8e8f6e6` | 58278257 |
| All four groups + 8-view Mask2Former | `"mftta"` | `3cdd3f06feec44b7b091d1807626c85dfbdcaf44afb285392e440b80432549ac` | 58299400 |
| Crop-1536 Mask2Former, 8 views + 1.25× scale; 5-fold U-Nets **(final)** | `"ms125"` | `f05ea0d7b6fc8f9345586b193c2866cd654f7fad1b838d609e41af5f46de80ed` | 58367893 |
| Crop-1536 Mask2Former, 8 views + 1.25× in 8 views | `"ms125d4"` | `c73318cca158dca7f0b9688de9a4fbc49ef0981e76c0acae140bdf8a589f950c` | 58368642 |
| Crop-1536 Mask2Former, 8 views + 1.25× (8) + 1.5× (4 views) | `"ms4"` | `a3abcb959da41f8db2d895a06e290b8da821147b1b4c26af105cea12cda5ce24` | 58369061 |
| Crop-1536 Mask2Former, 8 views | `"f1c15u"` | `af512e390bb4a365ff5ee0434c0ea2607e529759c0eab79d1c85b595027d395b` | 58360056 |
| Crop-768 + crop-1024 Mask2Former, 8 views; 5-fold U-Nets | `"c7c10d4_un5"` | `573532cd29e5205a4c4e52792fcd4a801da41609d88c4e0a65d24c2f81cb9ad1` | 58338409 |
| Crop-768 + crop-1024; refitted repair; 5-fold U-Nets | `"c7c10_refit_un5"` | `3134fef943bc0d7fd4f9ba7f016837cdf290a10b4f3b98777fa963079ec1bcfb` | 58337103 |
| Crop-768 + crop-1024; 5-fold U-Nets | `"c7c10_un5"` | `6efaef552809d5a89223a767ccba8b5cb97bab981562c1d0f28206b99af1f31a` | 58330231 |
| Crop-768 + crop-1024 Mask2Former | `"c7c10"` | `b11258ecd18d603c67b8db2870bae0d0cf47e24ff246964c0a80f3b0a43a59fe` | 58328986 |
| Crop-768 Mask2Former | `"c768"` | `167f756bdf5eecf4de56e5e208d9110b99ccf6e785b3a6f735ed1d5425bcddf3` | 58327493 |
| Crop-768 Mask2Former, weight 0.25 | `"c768_w25"` | `2dd8156a4aeb06a34d63abda87ecccca87e7904f7cff5040dd98eb8f6567e463` | 58320909 |


## Run time

On one Kaggle Tesla T4: the five networks take 850 s for the 180 test images (4.7 s per image), with peak GPU memory 1.3 GiB allocated (1.9 GiB reserved). The 8-view Mask2Former maps of `"mftta"` take another 2505 s. Building the final CSV on CPU, including the filter, the regressor and the repair, takes 329 s; the filter keeps 1036 of 1428 test candidates. Installing the pinned environments comes on top.

## The later submissions: stored maps

The ten modes from `"ms125"` on are recipes (`src/filseg/reproduction/data/chain/<mode>.json`). Each recipe names:

- **Mask2Former test maps**: one archive per fold model and map set (5 × 1 to 5 × 3 archives), with the model file, the input size and the views that made it. Each archive holds one uint16 map per test image (value / 510 = probability): the per-pixel maximum over queries of sigmoid(mask logit) × class probability at the network output, averaged over the views after undoing each flip or rotation, resized bilinearly to 2048 and rounded. The recipe's Mask2Former map is the level mean of all its archives (integer sum / (510 n), float32).
- **U-Net test maps**: the package's fold-2 maps (`unet_package_*.npz`, identical to the GPU stage's output) or the per-pixel mean of the five fold models' maps (`unet_fold5mean_*.npz`, uint8, rint of the mean).
- **Blend weight** w: p = w · Mask2Former + (1 − w)/2 · full + (1 − w)/2 · local, in the package's float32 arithmetic.
- **Instance filter** (68 features, its own cutoff) and **repair regressor**: refitted on this map's candidates of all 707 training images, stored next to the recipe and hash-checked.

These stored maps were made on an NVIDIA RTX 3090 (PyTorch 2.5.1, native BF16) when the submissions were built. Building the CSV from them is deterministic CPU work: it reproduces every submitted CSV byte for byte. The final submission's maps and models are in the repository (`weights/final/`, plain git, 48 MB parts, hash-checked when reassembled); the other recipes find theirs in the release files by name.

## Re-running the networks of the chain modes

The stored maps of a chain mode can be made again from its models (`filseg.reproduction.chain_maps`, GPU): every Mask2Former archive with its own input size and views, every U-Net fold map, then the 5-fold means, under the release file names. The arithmetic is the exporters' (checked bit for bit against them on a stand-in network for every view kind).

```bash
python -m filseg.reproduction chain-maps --images <test_images dir> --recipe ms125 --output chain_maps            # models from weights/final
python -m filseg.reproduction chain-maps --images <test_images dir> --recipe ms125 --output chain_maps --image-keys 20140420195854Bh 20111122063134Lh   # two images
python -m filseg.reproduction csv --images <test_images dir> --mode ms125 --release chain_maps --maps-from-models --output submission_from_models
```

`chain-maps` writes `chain_maps_compare.json`: for each archive, the images equal to the stored ones, the changed pixels and the largest level difference. On a Kaggle T4 (two GPUs are used when present) the maps are close to, but not bit-identical with, the stored ones, which were made on an RTX 3090 with native BF16 (PyTorch 2.5.1; the T4 environment is PyTorch 2.14.1 without native BF16). Measured on 4 test images with fold model 0:

| Map | Changed pixels | Mean / largest difference (levels of 510) |
| --- | ---: | --- |
| Mask2Former, 8 views at 2048 px | 0.59 % | 0.014 / 33 |
| Mask2Former, 2 views at 2560 px | 0.50 % | 0.011 / 47 |
| Mask2Former, 8 views at 2560 px | 0.59 % | 0.014 / 68 |
| Mask2Former, 4 views at 3072 px | 0.69 % | 0.020 / 45 |
| Full-disk U-Net | 0.023 % | 0.0003 / 9 |
| Local U-Net | 0.012 % | 0.0001 / 2 |

About three quarters of the changed Mask2Former pixels differ by 1 or 2 levels. On the two test images `pipeline.ipynb` re-reads (with all ten Mask2Former archives and ten U-Net fold models of `"ms125"`, from `weights/final/`), 1.3–1.7 % of the Mask2Former pixels and 0.03–0.09 % of the U-Net pixels change; the largest single difference is 242 levels (0.47 in probability) in a 1.25× map, and 72 % of the changed Mask2Former pixels differ by 1 or 2 levels. The whole notebook, this check included, took 16 min on a Kaggle T4 and still printed `'matches_the_expected_csv': True`, because the CSV is built from the stored maps. On a T4, Mask2Former takes about 2.0 s per view at 2048 px, 3.2 s at 2560 px and 4.6 s at 3072 px (peak GPU memory 4.9, 7.6 and 10.8 GB); the U-Nets take 1.2 s (full-disk) and 2.0 s (local) per image. Re-reading all 180 test images for `"ms125"` (`CHAIN_MAPS_ALL = True` in the notebook) is therefore about 6.3 T4-hours (an estimate from these timings), about 3.2 h on two T4s. With `--maps-from-models` the CSV stage records the hashes of the maps it read instead of requiring the stored ones, so its CSV is close to, not equal to, the submitted one.

## Two runtimes instead of one

`pipeline.ipynb` runs everything in one T4 session. To split GPU and CPU work:

1. On a Kaggle T4, run `notebooks/reproduce_maps.ipynb` (the five networks).
2. Only for `"mftta"`: run `notebooks/reproduce_mf_views.ipynb` in the same session, or attach the output of step 1 to it.
3. On a CPU runtime, attach the outputs of steps 1–2 and the competition data, and run `notebooks/reproduce_submission.ipynb` with the chosen `MODE`.

`pipeline.ipynb` itself skips a GPU stage when its completed output is attached. Do not attach any other dataset that contains an `inference_result.json`: the notebook would use those maps instead.

## Command line

Stage 1 runs only on a Kaggle T4; the launcher checks for one and stops otherwise. It installs its two pinned GPU environments itself, into the running interpreter.

```bash
pip install --no-deps -e .            # editable install: the code finds weights/ through the checkout
python -m filseg.reproduction infer --images <test_images dir> --output maps
python -m filseg.reproduction infer-mfviews --images <test_images dir> --output maps --views tta   # only for "mftta"
```

Stage 2 runs on CPU on any machine. Install the CPU pins **after** stage 1 when both run in one interpreter, because stage 1 leaves a different OpenCV version behind.

```bash
pip install -r requirements.txt
python -m filseg.reproduction csv --images <test_images dir> --maps maps --output submission --mode <MODE>
sha256sum submission/submission.csv
```

The later submissions need no stage 1. The final one reads `weights/final/`; for the others, point `--release` at the folder of the release files:

```bash
python -m filseg.reproduction csv --images <test_images dir> --output submission --mode ms125
python -m filseg.reproduction csv --images <test_images dir> --release <release folder> --output submission --mode ms4
```

New images (any folder of 2048×2048 full-disk images, any CUDA GPU, also outside Kaggle): add `--any-images` to `infer` and `csv`. There is no submitted CSV to compare with, so `matches_expected` is `null`.

## Deterministic mode

On by default (`filseg.reproduction.determinism`); `--nondeterministic` turns it off. It fixes the seeds, makes cuDNN deterministic without autotuning, calls `torch.use_deterministic_algorithms(True)` (a nondeterministic CUDA operation raises instead of silently falling back), sets `CUBLAS_WORKSPACE_CONFIG` before CUDA starts, turns TF32 off, and runs a sequential, sorted image loop. Map archives are written with fixed ZIP metadata, so equal maps give equal file bytes. Every run records its settings and runtime (PyTorch, CUDA, cuDNN, driver, GPU) in `inference_result.json`, with hashes of every layer output for the first image, to locate any future divergence. Two independent T4 runs gave byte-identical map archives and CSVs; Mask2Former keeps its BF16 dtype on the T4.

## Training recipe (not needed to reproduce)

The package does not retrain; these settings produced the weights in `weights/`.

- **Losses.** Every loss is computed against each annotation set of an image and averaged over its sets.
- **U-Nets.** `segmentation_models_pytorch` 0.5.0 U-Nets with an ImageNet-initialised efficientnet-b0 encoder. Binary cross-entropy plus soft Dice; 12 epochs of 1136 sampled images; batch 2 with 4-step accumulation; AdamW (learning rate 2×10⁻⁴, weight decay 10⁻⁴); cosine schedule; gradient clipping at 1.0; flips; seed 42; PyTorch 2.11.0+cu130.
- **Local U-Net input.** Three channels: raw, CLAHE (clip limit 2, 16×16 tiles) and a detail channel 0.5 + 2(x − G_σ ∗ x) with σ=8 px. At test time it tiles the disk with stride 768.
- **Mask2Former.** Fine-tuned from `facebook/mask2former-swin-tiny-coco-instance` (transformers 4.57.1). Per draw one annotation set, whose instances are the targets of the Hungarian set loss; random 512 crops of the disk resized to 1024; AdamW (backbone 10⁻⁵, rest 10⁻⁴, weight decay 0.01); batch 8; 12 cosine epochs; clipping at 1.0; EMA of weights with decay 0.99.
- **Checkpoints.** The last epoch of every network, with no checkpoint selection on validation data.
- **Later submissions.** The five fold U-Nets of each view use the recipe above, one per cross-validation fold (trained on the other four folds). The crop-1536 Mask2Former models (final submission) start from the same checkpoint and train on random 1536 px crops of the native 2048 px image (two thirds centred on an instance, scale jitter 0.8–1.25, flips and right-angle rotations), 4000 steps of batch 8, warm-up 500 steps then cosine, AdamW (backbone 10⁻⁵, rest 10⁻⁴, weight decay 0.01), clipping at 1.0, EMA 0.999; one model per fold. The crop-768 and crop-1024 models differ only in the crop size (and in how the batch of 8 is split for gradient accumulation). The scripts as run are in `training/`.

## Implementation constants of the mask repair

- The redraw window is a rectangle aligned with the mask's principal axes, extended at each end by 25% of its length and width (at least 32 px).
- A redrawn pixel must be at least 3 px from every other kept mask.
- The regressor (scikit-learn `HistGradientBoostingRegressor`) predicts the repair threshold τ, snapped to a grid of 9 values from 0.20 to 0.60.
- The one-threshold variant (`"pnat"`) uses τ = J/(1+J) = 0.4017, with J = 0.671 the mean matched IoU.

## Weights

`weights/` is plain git (no LFS). GitHub refuses files over 100 MB, so each Mask2Former file is stored as numbered parts of at most 48 MB in `weights/parts/`. `weights/repo_weights.json` records the size and SHA-256 of every stored file, every part and every reassembled file. On first use, `filseg.reproduction.repo_weights.prepare` checks every part, reassembles the three files next to them (git-ignored; a read-only checkout uses `~/.cache/filseg/weights`) and checks the result. The loader then checks every file against `weights/reproduction_weights.json`, and every tensor against its recorded identity, before strict model loading. The weights root is chosen in this order: `--weights`, the repository's `weights/`, a mounted copy of the weights dataset, a `kagglehub` download.

The selector coefficients (`src/filseg/reproduction/data/*_selector.json`) and the threshold regressor (`weights/regressor.pkl`, scikit-learn 1.6.1) are hash-checked against values pinned in the package before use; a different hash, a missing file or another scikit-learn version raises.

## Test-time checks before each submission

Each submitted CSV passed these checks on the test set, without ground truth, before it was submitted: the number of kept candidates against the first filter, the test medians of the added features against their cross-validation range, the agreement between the test-time filter and the per-fold models, an independent replay of the CSV stage, and the format checks (every row decodes, no empty or overlapping masks, unique ids). The experiments are indexed in `campaigns/INDEX.md` (what was tried and the outcome); their full records are kept outside this repository.

## Diagnostic: the first submission

Our first submission was built before the reproduction package existed, and the package does not regenerate that CSV bit for bit: newly inferred maps differ slightly in every network, although replaying the stored maps reproduces it exactly. The package output, which is deterministic, is therefore the reference; the `"equivalent"` mode is its rebuild of that pipeline. The full record, with per-image matches and stage hashes, is `campaigns/20261111_repro_package/results/reproduction_result.json`; the `finalize` stage (`--references <weights dataset folder>`) reruns the comparison.

## Release tag

Opened outside a checkout, each notebook clones this repository and checks out the tag `v1.1` (`SOURCE_TAG` in its first code cell), the commit these documents describe. Keep `v1.1` on that commit; `v1.0` is the earlier release without the later submissions.
