# Kaggle kernel ops (M1+)

## Build & push

```bash
# smoke (fold 0, 2 epochs, separate slug):
python solution/kaggle/build_kernel.py --stage <stage-dir> --smoke
# production (5 folds):
python solution/kaggle/build_kernel.py --stage <stage-dir>

kaggle kernels push -p <stage-dir> --accelerator NvidiaTeslaT4
kaggle kernels status masayakawamata/<slug>
kaggle kernels output masayakawamata/<slug> -p <out-dir>   # after completion
```

## Hard-won environment facts (2026-07-26)

1. **Always pin `--accelerator NvidiaTeslaT4`.** Without it Kaggle may
   allocate a **P100 (sm_60)**, and the current Kaggle image ships a PyTorch
   built for **sm_70+** — every CUDA op dies with
   `no kernel image is available for execution on the device`. The T4 (sm_75)
   works. This is an image/hardware mismatch on Kaggle's side, not our code.
2. **Pushes must be notebook-type with `kernelspec` metadata.**
   `kernel_type: script` errors with "Notebook not found"; a bare
   `nbformat.v4.new_notebook()` without `kernelspec` fails in papermill with
   "No kernel name found". `build_kernel.py` handles both.
3. **A failed push (e.g. the 2-batch-GPU-slot cap) leaves a versionless
   PHANTOM kernel**: hidden from `kernels list`, 404 on `kernels status`, and
   every subsequent push to the slug errors with "Notebook not found". Fix:
   `yes | kaggle kernels delete masayakawamata/<slug>` then push again.
4. **Never hardcode `/kaggle/input/<name>` paths** — competition data mounts
   at `/kaggle/input/competitions/<slug>/...` and datasets have moved roots
   before. The kernel glob-discovers everything (`/kaggle/input/**/...`,
   `recursive=True`).
5. `pip install segmentation_models_pytorch` on the Kaggle image installs
   only the small smp wheel (torch untouched) — no `--no-deps` gymnastics
   needed as of this image.

## Measured timings (smoke, T4)

- cache 707 train images @1024: ~40 s; 1,154 entry masks: ~20 s.
