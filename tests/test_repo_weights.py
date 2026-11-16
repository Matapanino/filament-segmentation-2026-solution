"""Model weights stored in the repository: numbered parts, reassembly, hash checks and the weights-root order.

Synthetic files only; no model is loaded. The last tests check the committed weights/ folder when it is present.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_source(tmp_path):
    src = tmp_path / 'source'
    src.mkdir()
    big = bytes(range(256)) * 10 + b'tail'   # 2564 bytes -> 3 parts of at most 1000
    small = b'{"small": true}\n'
    (src / 'big.safetensors').write_bytes(big)
    (src / 'small.json').write_bytes(small)
    return src, big, small


def test_split_writes_parts_under_the_limit_and_a_manifest(tmp_path):
    from filseg.reproduction.repo_weights import MANIFEST, split
    src, big, small = make_source(tmp_path)
    weights = tmp_path / 'weights'
    manifest = split(src, ['big.safetensors', 'small.json'], weights, part_bytes=1000)
    assert (weights / MANIFEST).is_file()
    assert json.loads((weights / MANIFEST).read_text()) == manifest
    row = manifest['files']['big.safetensors']
    assert row['sha256'] == sha(big) and row['bytes'] == len(big)
    assert [p['bytes'] for p in row['parts']] == [1000, 1000, 564]
    for p in row['parts']:
        b = (weights / p['file']).read_bytes()
        assert len(b) <= 1000 and sha(b) == p['sha256']
    assert not (weights / 'big.safetensors').exists(), 'a split file is stored only as parts'
    assert manifest['files']['small.json']['parts'] == []
    assert (weights / 'small.json').read_bytes() == small


def test_parts_reassemble_to_the_expected_hash(tmp_path):
    from filseg.reproduction.repo_weights import prepare, split
    src, big, small = make_source(tmp_path)
    weights = tmp_path / 'weights'
    split(src, ['big.safetensors', 'small.json'], weights, part_bytes=1000)
    root = prepare(weights)
    assert root == weights
    assert sha((root / 'big.safetensors').read_bytes()) == sha(big)
    assert (root / 'small.json').read_bytes() == small
    # A second call reuses the verified file.
    assert prepare(weights) == weights


def test_reassembly_into_another_destination(tmp_path):
    from filseg.reproduction.repo_weights import prepare, split
    src, big, small = make_source(tmp_path)
    weights = tmp_path / 'weights'
    split(src, ['big.safetensors', 'small.json'], weights, part_bytes=1000)
    dest = tmp_path / 'cache'
    root = prepare(weights, dest)
    assert root == dest
    assert (dest / 'big.safetensors').read_bytes() == big and (dest / 'small.json').read_bytes() == small
    assert not (weights / 'big.safetensors').exists()


def test_a_corrupted_part_fails_and_leaves_no_file(tmp_path):
    from filseg.reproduction.repo_weights import prepare, split
    src, big, small = make_source(tmp_path)
    weights = tmp_path / 'weights'
    manifest = split(src, ['big.safetensors', 'small.json'], weights, part_bytes=1000)
    part = weights / manifest['files']['big.safetensors']['parts'][1]['file']
    b = bytearray(part.read_bytes()); b[10] ^= 1; part.write_bytes(bytes(b))
    with pytest.raises(ValueError, match='integrity'):
        prepare(weights)
    assert not (weights / 'big.safetensors').exists()
    assert not any(p.name.endswith('.tmp') for p in weights.iterdir())


def test_a_missing_part_fails(tmp_path):
    from filseg.reproduction.repo_weights import prepare, split
    src, big, small = make_source(tmp_path)
    weights = tmp_path / 'weights'
    manifest = split(src, ['big.safetensors', 'small.json'], weights, part_bytes=1000)
    (weights / manifest['files']['big.safetensors']['parts'][2]['file']).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(weights)


def test_a_corrupted_whole_file_fails(tmp_path):
    from filseg.reproduction.repo_weights import prepare, split
    src, big, small = make_source(tmp_path)
    weights = tmp_path / 'weights'
    split(src, ['big.safetensors', 'small.json'], weights, part_bytes=1000)
    (weights / 'small.json').write_bytes(b'{"small": false}\n')
    with pytest.raises(ValueError, match='integrity'):
        prepare(weights)


def test_a_stale_assembled_file_is_rebuilt_from_the_parts(tmp_path):
    from filseg.reproduction.repo_weights import prepare, split
    src, big, small = make_source(tmp_path)
    weights = tmp_path / 'weights'
    split(src, ['big.safetensors', 'small.json'], weights, part_bytes=1000)
    (weights / 'big.safetensors').write_bytes(b'not the file')
    prepare(weights)
    assert (weights / 'big.safetensors').read_bytes() == big


def fake_weights_root(path, tag):
    path.mkdir(parents=True)
    (path / 'reproduction_weights.json').write_text(json.dumps(dict(tag=tag)))
    return path


def test_weights_root_prefers_the_repository_weights_folder(tmp_path):
    from filseg.reproduction.assets import locate
    from filseg.reproduction.repo_weights import split
    src, _, _ = make_source(tmp_path)
    repo = tmp_path / 'repo'
    split(src, ['big.safetensors', 'small.json'], repo / 'weights', part_bytes=1000)
    kaggle_input = tmp_path / 'kaggle_input'
    fake_weights_root(kaggle_input / 'filament-repro-weights-1111', 'dataset')
    assert locate(None, repo=repo, kaggle_input=kaggle_input) == repo / 'weights'
    assert (repo / 'weights/big.safetensors').is_file(), 'the parts were reassembled'


def test_mounted_dataset_is_only_the_fallback(tmp_path):
    from filseg.reproduction.assets import locate
    repo = tmp_path / 'repo'
    repo.mkdir()
    kaggle_input = tmp_path / 'kaggle_input'
    mounted = fake_weights_root(kaggle_input / 'filament-repro-weights-1111', 'dataset')
    assert locate(None, repo=repo, kaggle_input=kaggle_input) == mounted


def test_an_explicit_root_wins(tmp_path):
    from filseg.reproduction.assets import locate
    from filseg.reproduction.repo_weights import split
    src, _, _ = make_source(tmp_path)
    repo = tmp_path / 'repo'
    split(src, ['big.safetensors', 'small.json'], repo / 'weights', part_bytes=1000)
    explicit = fake_weights_root(tmp_path / 'explicit', 'explicit')
    assert locate(explicit, repo=repo, kaggle_input=tmp_path / 'none') == explicit
    other = tmp_path / 'other_weights'
    split(src, ['big.safetensors', 'small.json'], other, part_bytes=1000)
    assert locate(other, repo=repo, kaggle_input=tmp_path / 'none') == other
    assert (other / 'big.safetensors').is_file(), 'an explicit repository-style folder is reassembled too'


def test_the_package_default_repo_is_the_checkout():
    from filseg.reproduction import assets
    assert assets.REPO == REPO


COMMITTED = REPO / 'weights'


@pytest.mark.skipif(not (COMMITTED / 'repo_weights.json').is_file(), reason='weights/ not committed in this checkout')
def test_committed_weights_folder_is_consistent():
    from filseg.reproduction.repo_weights import LIMIT_BYTES
    manifest = json.loads((COMMITTED / 'repo_weights.json').read_text())
    pinned = json.loads((REPO / 'src/filseg/reproduction/data/weights_manifest_identity.json').read_text())
    release = manifest['files']['reproduction_weights.json']
    assert release['sha256'] == pinned['sha256'] and release['bytes'] == pinned['bytes']
    listed = json.loads((COMMITTED / 'reproduction_weights.json').read_text())
    needed = {r['file'] for r in listed['networks'].values()} | set(listed['files']) | {'reproduction_weights.json', 'regressor.pkl'}
    assert set(manifest['files']) == needed
    from filseg.reproduction.pcand import REGRESSOR_SHA256
    assert manifest['files']['regressor.pkl']['sha256'] == REGRESSOR_SHA256
    for name, row in manifest['files'].items():
        files = [p['file'] for p in row['parts']] or [name]
        assert sum((COMMITTED / f).stat().st_size for f in files) == row['bytes']
        assert all((COMMITTED / f).stat().st_size < LIMIT_BYTES for f in files)
        if row['parts']:
            assert sum(p['bytes'] for p in row['parts']) == row['bytes']
    ignored = (REPO / '.gitignore').read_text().split()
    for name, row in manifest['files'].items():
        if row['parts']:
            assert '/weights/' + name in ignored, 'the reassembled file must never be committed'
