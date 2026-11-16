"""5-fold-mean test U-Net maps from the as-run fold models' saved test maps (CPU only), plus a distribution shift check.

Inputs (Drive lake, read only): for each fold k = 0..4 and view in (full, local), the as-run run directory of the fold-k
seed-42 U-Net: oof_prob.npz (must be byte-identical (md5) to the OOF map the chain pins for fold k) and test_prob.npz
(the fold model's 180 test maps, uint8 rint(p*255), written by the as-run worker train_cuda.py c241c76c...).
Outputs in <out>: full.npz, local.npz (package determinism.save_npz; member = test stem; uint8 rint(sum_k q_k / 5)),
manifest.json (10 inputs with sha256/md5, outputs with sha256), shift.json.

    PYTHONPATH=<package>/src:<this dir> python -B fold5mean.py --out out/fold5mean
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from audit_fold_models import LAKE, RUNS, VIEW, pins, digest, TEST, run_manifest
from test_maps import level_mean

# The shipped map set d7aa6a41... lives only in our private kernel output filament-pkgfinal-1113-gpu; full.npz, local.npz
# and inference_result.json were pulled read-only into scratch/pkgfinal/maps (kaggle kernels output --file-pattern).
SHIPPED_MAPS = Path(__file__).resolve().parent / 'scratch/pkgfinal/maps'
MAP_SET_SHA = 'd7aa6a41fafc26b07ce89b306b884f6667ebb68b86289ea4745ee44c0b0d195f'
SHIPPED_LABEL = 'test_shipped_fold2'


def now(): return subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S %Z'], text=True).strip()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def hist(maps):
    h = np.zeros(256, np.int64)
    for q in maps:
        if q.dtype != np.uint8 or q.shape != (2048, 2048): raise ValueError('bad map')
        h += np.bincount(q.ravel(), minlength=256)
    return h


def stats(h, n_maps):
    """On p = level / 255: mean over all pixels; p50/p90/p99 of the nonzero pixels; fraction of pixels with p > 0.5."""
    levels = np.arange(256); total = h.sum(); nz = h[1:]; cnz = np.cumsum(nz)
    q = lambda f: float(levels[1:][np.searchsorted(cnz, f * cnz[-1])] / 255)
    return dict(n_maps=n_maps, mean=float((h * levels).sum() / total / 255), nonzero_fraction=float(nz.sum() / total),
                nonzero_p50=q(.5), nonzero_p90=q(.9), nonzero_p99=q(.99), frac_gt_half=float(h[128:].sum() / total))


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a = a.parse_args()
    from filseg.reproduction.determinism import save_npz
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    test_stems = sorted(p.stem for p in TEST.glob('*.jpeg')); assert len(test_stems) == 180
    pin = pins(); inputs = []; shift = {}
    inferred = json.loads((SHIPPED_MAPS / 'inference_result.json').read_text())['networks']
    # the package's mfmaps.map_set_sha256 (its module imports torch): sha256 of the sorted-key JSON of the archive hashes
    names = ('full', 'local', 'MF0', 'MF1', 'MF2')
    assert hashlib.sha256(json.dumps({n: inferred[n]['sha256'] for n in names}, sort_keys=True).encode()).hexdigest() == MAP_SET_SHA
    outputs = {}
    for v, prefix in VIEW.items():
        oof_maps, test_sets = [], []
        for k, rel in RUNS.items():
            d = LAKE / rel / f'{prefix}_f{k}_s42'
            oof, tp, last = d / 'oof_prob.npz', d / 'test_prob.npz', d / 'last.pt'
            row = dict(fold=k, view=v, dir=str(d.relative_to(LAKE)), oof_md5=digest(oof, 'md5'), oof_sha256=digest(oof, 'sha256'),
                       test_prob_sha256=digest(tp, 'sha256'), last_pt_sha256=digest(last, 'sha256'), last_pt_bytes=last.stat().st_size)
            row['oof_equals_chain_pin'] = row['oof_md5'] == pin[k, v]; assert row['oof_equals_chain_pin'], row
            man, row['manifest_sha256'], row['manifest_source'] = run_manifest(d)
            o = load(oof); assert sorted(o) == sorted(man['validation_files']); row['n_oof'] = len(o); oof_maps += list(o.values())
            t = load(tp); assert sorted(t) == test_stems, (k, v, len(t)); row['n_test'] = len(t); test_sets.append(t)
            shift.setdefault(v, {})[f'test_single_fold{k}'] = stats(hist(t.values()), len(t))
            inputs.append(row); print(json.dumps({x: row[x] for x in ('fold', 'view', 'oof_equals_chain_pin', 'n_oof', 'n_test')}), flush=True)
        assert len(oof_maps) == 707
        mean = {s: level_mean([t[s] for t in test_sets]) for s in test_stems}
        path = out / f'{v}.npz'; save_npz(path, mean)
        outputs[v] = dict(file=path.name, sha256=digest(path, 'sha256'), md5=digest(path, 'md5'), bytes=path.stat().st_size, n_images=len(mean),
                          arrays={s: hashlib.sha256(mean[s].tobytes()).hexdigest() for s in test_stems})
        shipped_path = SHIPPED_MAPS / f'{v}.npz'; assert digest(shipped_path, 'sha256') == inferred[v]['sha256']
        shipped = load(shipped_path); assert sorted(shipped) == test_stems
        shift[v].update(oof_stored_707=stats(hist(oof_maps), len(oof_maps)), **{SHIPPED_LABEL: stats(hist(shipped.values()), len(shipped))},
                        test_fold5mean=stats(hist(mean.values()), len(mean)))
        del oof_maps, test_sets, mean, shipped
    rule = 'uint8(rint(sum_k q_k / 5)) over the five fold models (np.rint, half to even); package determinism.save_npz'
    (out / 'manifest.json').write_text(json.dumps(dict(created_at=now(), rule=rule, inputs=inputs, outputs=outputs,
                                                       shipped_maps=dict(source='kernel output <kaggle-user>/filament-pkgfinal-1113-gpu', dir=str(SHIPPED_MAPS), map_set_sha256=MAP_SET_SHA, archives={n: inferred[n]['sha256'] for n in names})), indent=1))
    (out / 'shift.json').write_text(json.dumps(dict(created_at=now(), note='p = level/255; percentiles over nonzero pixels; frac_gt_half = p > 0.5 (level >= 128)',
                                                    views=shift), indent=1))
    print(json.dumps({v: {k: outputs[v][k] for k in ('file', 'sha256')} for v in outputs}))


if __name__ == '__main__':
    main()
