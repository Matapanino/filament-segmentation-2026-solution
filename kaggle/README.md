# Kaggle kernel helpers

These scripts ran our early training and post-processing experiments as Kaggle kernels. They are kept as a record and are not needed to reproduce the submission (see the [README](../README.md)).

- `build_kernel.py` wraps a Python script into a one-cell notebook with kernel metadata, ready for `kaggle kernels push`.
- `train_m1.py` trains the first U-Net baseline (5 folds) on a Kaggle T4; `kernel-metadata.json` is its kernel metadata.
- `screen_pp_kernel.py` runs the post-processing screen (`scripts/screen_pp.py`) on Kaggle CPU.

They read the package and intermediate maps from private Kaggle datasets that are not published.
