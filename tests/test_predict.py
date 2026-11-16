"""New images: the pipeline on any folder of 2048x2048 full-disk H-alpha images (not only the 180 competition test images),
written before the change.

image_set='any' lists every *.jpeg / *.jpg / *.png under the folder (keys = file stems, which must be unique) instead of
the frozen, hash-pinned test set; maps and CSV then cover exactly those images. Absent (None) = the frozen test set, as
before. Synthetic images and maps; the learned files are the repository's.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_pnat import synthetic_maps


def write_images(folder, names, size=2048):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        img = np.full((size, size), 120, np.uint8); img[200:220, 200:300] = 40
        cv2.imwrite(str(folder / n), img)
    return folder


def test_prepare_images_lists_any_folder_with_image_set_any(tmp_path):
    from filseg.reproduction.pipeline import prepare_images
    src = write_images(tmp_path / 'new', ['a.jpeg', 'b.jpg', 'c.png'])
    files = prepare_images(src, tmp_path / 'unused', image_set='any')
    assert sorted(files) == ['a', 'b', 'c'] and all(Path(p).is_file() for p in files.values())


def test_prepare_images_refuses_duplicate_stems_and_an_empty_folder(tmp_path):
    from filseg.reproduction.pipeline import prepare_images
    src = write_images(tmp_path / 'dup', ['a.jpeg', 'a.png'])
    with pytest.raises(ValueError, match='unique'):
        prepare_images(src, tmp_path / 'u', image_set='any')
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError):
        prepare_images(tmp_path / 'empty', tmp_path / 'u', image_set='any')


def test_prepare_images_default_still_requires_the_frozen_test_set(tmp_path):
    from filseg.reproduction.pipeline import prepare_images
    src = write_images(tmp_path / 'new', ['a.jpeg'])
    with pytest.raises(ValueError, match='frozen'):
        prepare_images(src, tmp_path / 'u')


def fake_maps_root(tmp_path, keys):
    """A completed maps folder (as run_maps writes it) for the given image keys, from synthetic maps."""
    from filseg.reproduction.audit import identity
    from filseg.reproduction.determinism import save_npz
    root = tmp_path / 'maps'; root.mkdir(); base = synthetic_maps(); networks = {}
    for name in ('full', 'local', 'MF0', 'MF1', 'MF2'):
        save_npz(root / f'{name}.npz', {k: base[name] for k in keys}); networks[name] = dict(file=f'{name}.npz', **identity(root / f'{name}.npz'))
    (root / 'inference_result.json').write_text(json.dumps(dict(status='complete', networks=networks, image_set='any')))
    return root


def test_run_csv_builds_a_competition_format_csv_for_new_images(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    from filseg.reproduction.csv_io import read_rows
    from filseg.reproduction.pipeline import run_csv
    src = write_images(tmp_path / 'new', ['day1.jpeg', 'day2.jpeg'])
    result = run_csv(src, None, fake_maps_root(tmp_path, ['day1', 'day2']), tmp_path / 'out', pcand=True, selector='pcombo', image_set='any')
    assert result['status'] == 'complete' and result['images'] == 2 and result['image_set'] == 'any'
    ids = [i for i, _ in read_rows(tmp_path / 'out' / 'submission.csv')]
    assert all(i.rsplit('_', 1)[0] in ('day1', 'day2') for i in ids)


def test_run_csv_refuses_maps_that_do_not_cover_the_new_images(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    from filseg.reproduction.pipeline import run_csv
    src = write_images(tmp_path / 'new', ['day1.jpeg', 'day2.jpeg'])
    with pytest.raises(ValueError, match='coverage'):
        run_csv(src, None, fake_maps_root(tmp_path, ['day1']), tmp_path / 'out', pcand=True, selector='pcombo', image_set='any')


def test_cli_passes_any_images_to_the_csv_stage(monkeypatch):
    import sys
    from filseg.reproduction import __main__ as entry, pipeline
    calls = []
    monkeypatch.setattr(pipeline, 'run_csv', lambda *a, **k: calls.append((a, k)) or dict(status='complete', csv=dict(sha256='0' * 64)))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'csv', '--images', 'I', '--maps', 'M', '--output', 'O', '--mode', 'pcombo', '--any-images'])
    entry.main()
    assert calls[0][1]['image_set'] == 'any'


# ---------------------------------------------------------------- the final submission on new images
def test_chain_maps_cli_passes_any_images_to_the_launcher(monkeypatch):
    import sys
    from filseg.reproduction import __main__ as entry, chain_maps
    calls = []
    monkeypatch.setattr(chain_maps, 'launch', lambda **k: calls.append(k) or dict(status='complete'))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'chain-maps', '--images', 'I', '--output', 'O', '--recipe', 'ms125', '--any-images'])
    entry.main()
    assert calls[0]['image_set'] == 'any' and calls[0]['release_root']


def test_chain_maps_child_reads_any_images(monkeypatch):
    from filseg.reproduction import chain_maps
    code = chain_maps.CHILD.format(images='I', release='R', output='O', recipe=['ms125'], names=['x'], keys=None, deadline=1.0, deterministic=True,
                                   failure='F', stage='s', image_set='any')
    assert "image_set='any'" in code


def test_new_images_skip_the_comparison_with_the_stored_test_maps():
    from filseg.reproduction import chain_maps
    code = chain_maps.FINISH.format(output='O', release='R', recipe=['ms125'], keys=None, names=None, failure='F', image_set='any')
    assert "if 'any' is None:" in code and code.index("if 'any' is None:") < code.index('compare_maps(')
