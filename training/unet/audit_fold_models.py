"""UN1 (a): which fold U-Nets made the stored OOF maps, and do their weights / test maps exist on the Drive lake.
Read-only on the lake. Prints JSON. Run with any python with numpy:  python -B audit_fold_models.py"""
import hashlib, json, os, sys, zipfile
from pathlib import Path
import numpy as np

LAKE = Path(os.environ.get('UNET_RUNS_ROOT', 'unet_runs'))  # folder holding the five fold runs' outputs (was a private storage path)
CASE = Path(__file__).resolve().parents[1] / 'selector_and_repair/case'  # the fitting engine's case folder (same evidence files)
PKG = Path(__file__).resolve().parents[2]  # this repository (was an absolute path to the package)
TEST = Path(os.environ.get('TEST_IMAGES', 'MAGFiLO_1.0_Kaggle_2026/test/test_images'))  # competition test images
RUNS = {0: 'detect_20261103/p0_fold0/filament-p0-fold0-gpu-1103/artifacts',
        1: 'relabel_pilot_20261031/filament-relabel-original-gpu-1031/artifacts',
        2: 'detect_20261106/oof_bank5/A/phaseA/artifacts',
        3: 'detect_20261103/w7_scorer/filament-w7s-oof-f3s42-gpu-1104/artifacts',
        4: 'detect_20261103/w7_scorer/filament-w7s-oof-f4s42-gpu-1104/artifacts'}
VIEW = {'full': 'original_full_raw_1024', 'local': 'original_local_raw_clahe_detail'}


def digest(p, algo):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def run_manifest(d):
    """(manifest dict, sha256, source) of a run dir: its manifest.json, else the member of the kernel's diagnostics.tar.gz."""
    import tarfile
    man = d / 'manifest.json'
    if man.exists():
        b = man.read_bytes(); src = 'manifest.json'
    else:
        tar = d.parent.parent / 'diagnostics.tar.gz'; member = f'artifacts/{d.name}/manifest.json'
        with tarfile.open(tar) as t: b = t.extractfile(member).read()
        src = f'{tar.name}:{member}'
    return json.loads(b), hashlib.sha256(b).hexdigest(), src


def pins():
    """md5 of the OOF maps the chain pins, per (fold, view)."""
    out = {}
    mf = json.loads((CASE / 'frozen/evidence/mf_inputs.json').read_text())['original']
    for k in (0, 1):
        for v in VIEW: out[k, v] = mf[f'f{k}_{v}_map']['md5']
    conf = json.loads((CASE / 'support/confirmation_protocol.json').read_text())['original_map_inventory']
    for k in (3, 4):
        for v in VIEW: out[k, v] = conf[f'fold{k}_s42'][v]['map']['md5']
    ev = json.loads((CASE / 'evidence/fold2_map_provenance.json').read_text())
    for v in VIEW: out[2, v] = ev[v]['dataset_map']['md5']
    return out


def main():
    test_names = sorted(p.stem for p in TEST.glob('*.jpeg')) if TEST.is_dir() else []
    shipped = json.loads((PKG / 'weights/reproduction_weights.json').read_text())['networks']
    pin = pins(); rows = []
    for k, rel in RUNS.items():
        for v, prefix in VIEW.items():
            d = LAKE / rel / f'{prefix}_f{k}_s42'
            row = dict(fold=k, view=v, dir=str(d.relative_to(LAKE)))
            oof = d / 'oof_prob.npz'
            row['oof_md5_equals_chain_pin'] = oof.exists() and digest(oof, 'md5') == pin[k, v]
            if True:
                m, _, _ = run_manifest(d)
                row.update(n_train=len(m['train_files']), n_valid=len(m['validation_files']), epochs=m.get('epochs'), seed=m.get('seed'),
                           smoke=m.get('smoke_subset'), fold_table=m.get('fold_table_sha256', '')[:8])
            last = d / 'last.pt'
            row['last_pt_bytes'] = last.stat().st_size if last.exists() else None
            if last.exists(): row['last_pt_sha256'] = digest(last, 'sha256')
            tp = d / 'test_prob.npz'
            if tp.exists():
                with np.load(tp, allow_pickle=False) as z:
                    keys = sorted(z.files); a = z[keys[0]]
                    row.update(test_maps=len(keys), test_dtype=str(a.dtype), test_shape=list(a.shape),
                               test_keys_equal_test_images=keys == test_names)
            if k == 2:
                ck = LAKE / rel.replace('/artifacts', '') / f'{prefix}_f2_s42/checkpoints/epoch_0012.pt'
                row['epoch12_ckpt_sha256_equals_shipped_original'] = ck.exists() and digest(ck, 'sha256') == shipped[v]['original_checkpoint_sha256']
                row['last_pt_sha256_equals_shipped_original'] = row.get('last_pt_sha256') == shipped[v]['original_checkpoint_sha256']
            rows.append(row)
    print(json.dumps(dict(n_test_images=len(test_names), runs=rows), indent=1))


if __name__ == '__main__':
    main()
