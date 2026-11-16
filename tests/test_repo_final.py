"""The final submission from the repository alone: weights/final/ holds, in plain git (files under 48 MB, larger ones as
parts), every file the default mode reads — the five crop-1536 Mask2Former fold models and their config, the ten U-Net
fold checkpoints with their as-run test maps, the stored Mask2Former test maps and the 5-fold-mean U-Net test maps — with the same names and sha256
as the release files. Written before the folder and the code that reads it.
"""
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / 'weights/final'


def needed(name):
    """Release file names a recipe reads: its map archives and U-Net maps (exact CSV) and the models that made them."""
    from filseg.reproduction import chain
    r = chain.load_recipe(name)
    files = {a['file'] for a in r['archives']} | {a['model'] for a in r['archives']} | set(chain.unet_files(r).values())
    files |= {'mf_model_config.json'}
    if r['unet'] == 'fold5mean':   # the fold checkpoints and their as-run test maps (the from-models check compares with them)
        files |= {f'unet{x}_f{k}_{v}{y}' for k in range(5) for v in ('full', 'local') for x, y in (('model', '.pt'), ('', '_test.npz'))}
    return files


def test_the_final_folder_holds_exactly_what_the_default_mode_reads_with_the_release_hashes():
    from filseg.reproduction import chain
    from filseg.reproduction.modes import DEFAULT_MODE
    from filseg.reproduction.repo_weights import LIMIT_BYTES, MANIFEST
    manifest = json.loads((FINAL / MANIFEST).read_text())
    assert set(manifest['files']) == needed(DEFAULT_MODE)
    pins = chain.release_files()
    for name, row in manifest['files'].items():
        assert (row['bytes'], row['sha256']) == (pins[name]['bytes'], pins[name]['sha256']), name
        stored = [p['file'] for p in row['parts']] or [name]
        assert all((FINAL / f).stat().st_size < LIMIT_BYTES for f in stored), name


def test_the_reassembled_files_are_git_ignored_and_the_parts_are_not():
    import subprocess
    ignored = lambda rel: subprocess.run(['git', 'check-ignore', '-q', rel], cwd=REPO).returncode == 0
    assert ignored('weights/final/mfmodel_c1536_f0.safetensors')
    assert not ignored('weights/final/parts/mfmodel_c1536_f0.safetensors.000')
    assert not ignored('weights/final/repo_weights.json')


def test_repo_release_gives_a_checked_folder_for_the_default_mode_and_none_for_a_mode_it_does_not_cover():
    from filseg.reproduction import chain
    root = chain.repo_release('ms125')
    assert root is not None and chain.stored_files(chain.load_recipe('ms125'), [root])
    assert chain.repo_release('c7c10') is None


def test_chain_csv_reads_the_repository_when_no_release_folder_is_given(monkeypatch, tmp_path):
    from filseg.reproduction import chain
    seen = []
    monkeypatch.setattr(chain, 'stored_files', lambda r, roots, strict=True: seen.append(list(roots)) or (_ for _ in ()).throw(RuntimeError('stop')))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr('filseg.reproduction.pipeline.prepare_images', lambda *a, **k: {})
    with pytest.raises(RuntimeError, match='stop'):
        chain.run_chain_csv(images='I', weights=None, release=None, output=tmp_path / 'out', recipe='ms125')
    assert seen and seen[0] == [chain.repo_release('ms125')]


def test_cli_runs_the_default_mode_without_a_release_folder(monkeypatch):
    import sys
    from filseg.reproduction import __main__ as entry, chain
    calls = []
    monkeypatch.setattr(chain, 'run_chain_csv', lambda *a, **k: calls.append(k) or dict(status='complete', csv=dict(sha256='0' * 64)))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'csv', '--images', 'I', '--output', 'O', '--mode', 'ms125'])
    entry.main()
    assert calls and calls[0]['release'] is None


def test_reassembly_does_not_collide_with_another_process_temporary_file(tmp_path):
    """Two processes reassembling the same file at once must not rename each other's temporary file (seen on Kaggle)."""
    from filseg.reproduction.repo_weights import prepare, split
    src = tmp_path / 'src'; src.mkdir(); (src / 'big.bin').write_bytes(os.urandom(250_000))
    split(src, ['big.bin'], tmp_path / 'w', part_bytes=100_000)
    foreign = tmp_path / 'w' / 'big.bin.tmp'; foreign.write_bytes(b'another process')
    prepare(tmp_path / 'w')
    assert foreign.read_bytes() == b'another process'
    assert (tmp_path / 'w' / 'big.bin').read_bytes() == (src / 'big.bin').read_bytes()
