"""Chain modes: our later submissions (2026-11-13 to 2026-11-15), rebuilt from stored test maps with one parameterised
CSV stage, written before the package module.

Each submission is a recipe (data/chain/<name>.json): the Mask2Former test-map archives (5 fold models x n map sets,
uint16 levels / 510), the U-Net test maps (the package's fold-2 maps or the 5-fold means), the blend weight w, the
selector fitted for that arm (the 68 P-combo features, its own cutoff) and the P-cand regressor (the package's or the
arm's refit). The arithmetic is the chain CSV worker's, pinned as a verbatim fixture (tests/data/chain_csv_reference.py):
MF = the level mean over every archive, blend = w MF + (1 - w)/2 full + (1 - w)/2 local in the package's insertion order.
Synthetic data, except the opt-in real-data test (FILSEG_RELEASE + FILSEG_TEST_IMAGES), which rebuilds three images of
every recipe and compares their rows with the submitted CSV.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from test_pnat import synthetic_maps

DATA = Path(__file__).parent / 'data'
CHAIN = Path(__file__).resolve().parents[1] / 'src/filseg/reproduction/data/chain'
# name: (CSV sha256, rows, weight, cutoff, map sets, U-Net test maps, Kaggle submission ref), in documentation order
EXPECTED = {
    'ms125': ('f05ea0d7b6fc8f9345586b193c2866cd654f7fad1b838d609e41af5f46de80ed', 1134, 0.55, 0.35, 2, 'fold5mean', 58367893),
    'ms125m4': ('219def27e1081c23131ae56ee193683c48369434e7c2e0b9cd68effbc3cfda26', 1121, 0.55, 0.325, 2, 'fold5mean', 58411930),
    'ms125m4r4': ('79638003f7aa0f62c10e2948a09fe085a1594f53f099ab375811214156dd689f', 1121, 0.55, 0.325, 2, 'fold5mean', 58414228),
    'ms125d4': ('c73318cca158dca7f0b9688de9a4fbc49ef0981e76c0acae140bdf8a589f950c', 1139, 0.55, 0.35, 2, 'fold5mean', 58368642),
    'ms4': ('a3abcb959da41f8db2d895a06e290b8da821147b1b4c26af105cea12cda5ce24', 1143, 0.55, 0.35, 3, 'fold5mean', 58369061),
    'f1c15u': ('af512e390bb4a365ff5ee0434c0ea2607e529759c0eab79d1c85b595027d395b', 1141, 0.55, 0.325, 1, 'fold5mean', 58360056),
    'c7c10d4_un5': ('573532cd29e5205a4c4e52792fcd4a801da41609d88c4e0a65d24c2f81cb9ad1', 1100, 0.55, 0.35, 2, 'fold5mean', 58338409),
    'c7c10_refit_un5': ('3134fef943bc0d7fd4f9ba7f016837cdf290a10b4f3b98777fa963079ec1bcfb', 1128, 0.55, 0.35, 2, 'fold5mean', 58337103),
    'c7c10_un5': ('6efaef552809d5a89223a767ccba8b5cb97bab981562c1d0f28206b99af1f31a', 1128, 0.55, 0.35, 2, 'fold5mean', 58330231),
    'c7c10': ('b11258ecd18d603c67b8db2870bae0d0cf47e24ff246964c0a80f3b0a43a59fe', 1094, 0.55, 0.35, 2, 'package', 58328986),
    'c768': ('167f756bdf5eecf4de56e5e208d9110b99ccf6e785b3a6f735ed1d5425bcddf3', 1085, 0.55, 0.35, 1, 'package', 58327493),
    'c768_w25': ('2dd8156a4aeb06a34d63abda87ecccca87e7904f7cff5040dd98eb8f6567e463', 1044, 0.25, 0.35, 1, 'package', 58320909),
}
FINAL = 'ms125'


def reference():
    spec = importlib.util.spec_from_file_location('chain_csv_reference', DATA / 'chain_csv_reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def rng_maps(seed, n_mf, shape=(64, 64)):
    r = np.random.default_rng(seed)
    maps = dict(full=r.integers(0, 256, shape, dtype=np.uint8), local=r.integers(0, 256, shape, dtype=np.uint8))
    for i in range(n_mf): maps[f'NEW{i // 3}_{i % 3}'] = r.integers(0, 511, shape).astype(np.uint16)
    return maps


# ---------------------------------------------------------------- the mode table
def test_chain_modes_are_the_ten_recipes_in_documentation_order_with_their_hashes():
    from filseg.reproduction.modes import CHAIN_MODES
    assert list(CHAIN_MODES) == list(EXPECTED)
    for name, (sha, rows, *_ , ref) in EXPECTED.items():
        row = CHAIN_MODES[name]
        assert row['sha256'] == sha and row['ref'] == ref and row['kwargs'] == dict(recipe=name), name
        assert row['label'] and isinstance(row['label'], str)


def test_chain_modes_are_runnable_after_the_earlier_modes_and_resolve():
    from filseg.reproduction import modes
    table = modes.runnable()
    assert list(table)[-len(EXPECTED):] == list(EXPECTED)
    assert modes.resolve(FINAL) == (FINAL, modes.CHAIN_MODES[FINAL])
    assert modes.CHAIN_MODES[FINAL]['label'].endswith('(final)')


# ---------------------------------------------------------------- the recipes
@pytest.mark.parametrize('name', list(EXPECTED))
def test_recipe_pins_the_submission_its_maps_and_its_learned_files(name):
    from filseg.reproduction import chain, refit
    sha, rows, w, cutoff, sets, unet, ref = EXPECTED[name]
    r = chain.load_recipe(name)
    assert (r['csv']['sha256'], r['csv']['rows'], r['weight'], r['cutoff'], r['sets'], r['unet'], r['submission']['ref']) == EXPECTED[name]
    assert [a['name'] for a in r['archives']] == reference().test_sources(dict(sets=sets))
    assert len({a['sha256'] for a in r['archives']}) == 5 * sets
    selector = chain.selector(r)
    assert selector['feature_order'] == refit.PCOMBO_FEATURES and selector['threshold'] == cutoff
    assert selector['role'] == f'deployment_refit_chain2_w{round(100 * w):02d}_full707_mf_mixture'
    path, digest = chain.regressor(r, weights_root=Path('/nonexistent')) if r['regressor'] != 'package' else (None, None)
    if path is not None: assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest == r['regressor']['sha256']


def test_recipe_loader_refuses_a_changed_selector_file(tmp_path, monkeypatch):
    from filseg.reproduction import chain
    for p in CHAIN.iterdir(): (tmp_path / p.name).write_bytes(p.read_bytes())
    data = json.loads((tmp_path / 'selector_ms125.json').read_text()); data['intercept'] = [data['intercept'][0] + 1e-9]
    (tmp_path / 'selector_ms125.json').write_text(json.dumps(data))
    monkeypatch.setattr(chain, 'CHAIN', tmp_path)
    with pytest.raises(ValueError):
        chain.selector(chain.load_recipe('ms125'))


def test_every_file_a_recipe_reads_is_listed_in_the_release_files_without_paths():
    from filseg.reproduction import chain
    files = chain.release_files()
    for name in EXPECTED:
        r = chain.load_recipe(name)
        for a in r['archives']:
            assert files[a['file']] == dict(bytes=a['bytes'], sha256=a['sha256']) and a['model'] in files
        for f in chain.unet_files(r).values(): assert f in files
    text = json.dumps(files)
    assert '/Users/' not in text and 'masayakawamata' not in text and 'lake' not in text


def test_chain_refit_role_is_accepted_only_with_the_p_combo_order_and_a_grid_cutoff():
    from filseg.reproduction import chain, refit
    sel = dict(chain.selector(chain.load_recipe('ms125')))
    refit.check(sel)
    for bad in (dict(threshold=0.36), dict(feature_order=sel['feature_order'][::-1]), dict(role='deployment_refit_chain2_wxx_full707_mf_mixture')):
        with pytest.raises(ValueError):
            refit.check({**sel, **bad})


# ---------------------------------------------------------------- the arithmetic, against the chain worker
@pytest.mark.parametrize('w', [0.25, 0.55, 0.7])
def test_weighted_blend_is_the_chain_workers_bit_for_bit(w):
    from filseg.reproduction import arithmetic, chain
    ref = reference(); m = rng_maps(1, 3)
    mf = arithmetic.mf_levels_mean(m['NEW0_0'], m['NEW0_1'], m['NEW0_2'])
    got = chain.weighted_blend(m['full'], m['local'], mf, w)
    assert got.dtype == np.float32 and np.array_equal(got, ref.weighted_blend(w)(m['full'], m['local'], mf))
    if w == 0.25: assert np.array_equal(got, arithmetic.blend_with_mf(m['full'], m['local'], mf))


@pytest.mark.parametrize('n', [5, 10, 15])
def test_chain_mf_is_the_level_mean_over_every_archive(n):
    from filseg.reproduction import chain
    ref = reference(); m = rng_maps(2, n); names = [k for k in m if k.startswith('NEW')]
    got = chain.chain_mf(m, names)
    assert np.array_equal(got, ref.chain_mf(m, names))
    assert np.array_equal(got, sum(m[k].astype(np.int64) for k in names).astype(np.float32) / np.float32(510 * n))


def test_image_stages_chain_path_equals_the_workers_run_time_swap():
    """The worker taught the package a variant MF map and a blend weight at run time; the chain argument must give the
    same stages and rows without touching the package's own functions."""
    from filseg.reproduction import arithmetic, mfmaps, pipeline
    from test_pnat import keep_all_model
    ref = reference(); raw, model = keep_all_model(); base = synthetic_maps()
    maps = dict(full=base['full'], local=base['local'], **{f'NEW{k}_0': base['MF0'] for k in range(5)})
    names = [f'NEW{k}_0' for k in range(5)]
    got = pipeline.image_stages(raw, maps, model, chain=dict(names=names, weight=0.55))
    old_v, old_b = mfmaps.variant_mf, arithmetic.blend_with_mf
    try:
        mfmaps.variant_mf = lambda maps_, variant: ref.chain_mf(maps_, names)
        arithmetic.blend_with_mf = ref.weighted_blend(0.55)
        want = pipeline.image_stages(raw, maps, model, mf_variant='chain2')
    finally:
        mfmaps.variant_mf, arithmetic.blend_with_mf = old_v, old_b
    assert got['stages'] == want['stages'] and got['rles'] == want['rles'] and got['candidates'] == want['candidates'] > 0


def test_image_stages_without_chain_is_unchanged():
    from filseg.reproduction import pipeline
    from test_pnat import keep_all_model
    raw, model = keep_all_model(); maps = synthetic_maps()
    a = pipeline.image_stages(raw, maps, model); b = pipeline.image_stages(raw, maps, model, chain=None)
    assert a['stages'] == b['stages'] and a['rles'] == b['rles']


# ---------------------------------------------------------------- stored maps and the CLI
def fake_release(tmp_path, name):
    """A release folder holding the files a recipe reads, with the pinned names; content from synthetic maps."""
    from filseg.reproduction import chain
    r = chain.load_recipe(name); root = tmp_path / 'release'; root.mkdir()
    for a in r['archives']: (root / a['file']).write_bytes(b'x')
    for f in chain.unet_files(r).values(): (root / f).write_bytes(b'x')
    return r, root


def test_stored_maps_are_found_by_name_and_refused_on_a_hash_mismatch(tmp_path):
    from filseg.reproduction import chain
    r, root = fake_release(tmp_path, 'ms125')
    with pytest.raises(ValueError, match='hash'):
        chain.stored_files(r, [root])
    with pytest.raises(FileNotFoundError):
        chain.stored_files(r, [tmp_path / 'empty'])


def test_cli_chain_mode_runs_the_chain_csv_and_reports_the_match(monkeypatch, capsys):
    from filseg.reproduction import __main__ as entry, chain
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs)); return dict(status='complete', csv=dict(sha256=EXPECTED['ms125'][0]))

    monkeypatch.setattr(chain, 'run_chain_csv', recorder)
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'csv', '--images', 'I', '--release', 'REL', '--output', 'O', '--mode', 'ms125'])
    entry.main()
    assert len(calls) == 1 and calls[0][1]['recipe'] == 'ms125' and calls[0][1]['release'] == 'REL'
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out['matches_expected'] is True and out['mode'] == 'ms125'


def test_cli_refuses_release_with_an_earlier_mode(monkeypatch):
    from filseg.reproduction import __main__ as entry
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'csv', '--images', 'I', '--release', 'REL', '--output', 'O', '--mode', 'pcombo', '--maps', 'M'])
    with pytest.raises(SystemExit):
        entry.main()


# ---------------------------------------------------------------- opt-in: real stored maps, three images per recipe
REAL = os.environ.get('FILSEG_RELEASE'), os.environ.get('FILSEG_TEST_IMAGES')


@pytest.mark.skipif(not all(REAL), reason='set FILSEG_RELEASE (release folder) and FILSEG_TEST_IMAGES (test JPEG folder)')
@pytest.mark.parametrize('name', list(EXPECTED))
def test_real_stored_maps_rebuild_three_images_of_the_submitted_csv(name, tmp_path):
    from filseg.reproduction import chain
    from filseg.reproduction.csv_io import read_rows
    release, images = (Path(p) for p in REAL)
    want = {}
    for ident, rle in read_rows(release / f'submission_{name}.csv'): want.setdefault(ident.rsplit('_', 1)[0], []).append(rle)
    keys = sorted(want)[:3]
    got = chain.image_rows(name, images, [release], keys)
    assert {k: got[k] for k in keys} == {k: want[k] for k in keys}


# ---------------------------------------------------------------- notebooks, paths and figures
def test_find_release_returns_the_folder_of_the_release_manifest_and_refuses_two(tmp_path):
    from filseg.reproduction.paths import find_release
    assert find_release([tmp_path]) is None
    (tmp_path / 'a').mkdir(); (tmp_path / 'a' / 'release_manifest.json').write_text('{}')
    assert find_release([tmp_path / 'none', tmp_path]) == tmp_path / 'a'
    (tmp_path / 'b').mkdir(); (tmp_path / 'b' / 'release_manifest.json').write_text('{}')
    with pytest.raises(ValueError):
        find_release([tmp_path])


def test_chain_modes_need_no_gpu_stage_and_run_the_from_models_check_only_with_a_gpu(tmp_path):
    from filseg.reproduction.paths import gpu_stages
    assert gpu_stages('ms125', roots=[tmp_path], gpu=False) == ('ms125', [])
    assert gpu_stages('ms125', roots=[tmp_path], gpu=True) == ('ms125', ['reproduce_chain_maps.ipynb'])
    assert gpu_stages('pcombo', roots=[tmp_path], gpu=False) == ('pcombo', ['reproduce_maps.ipynb'])


def test_chain_maps_notebook_rereads_two_images_from_the_repository_models_and_prints_the_comparison():
    from test_notebooks_final import cells
    code = '\n'.join(cells('reproduce_chain_maps.ipynb', 'code'))
    assert 'SOURCE_TAG = "v1.1"' in code and '"chain-maps"' in code and '"--image-keys"' in code
    assert 'CHAIN_MAPS_ALL' in code and 'chain_maps_compare.json' in code
    assert '"--release"' not in code, 'the default reads the models in weights/final'
    assert 'returncode' in code and 'check=True)\nCHAIN' not in code, 'an optional check: its failure is printed, the CSV stage still runs'
    pipe = '\n'.join(cells('pipeline.ipynb', 'code'))
    assert 'if "reproduce_chain_maps.ipynb" in GPU_STAGES' in pipe


def test_submission_notebook_builds_a_chain_mode_from_the_release_files():
    from test_notebooks_final import cells
    code = [c for c in cells('reproduce_submission.ipynb', 'code') if '"--mode", MODE' in c][0]
    assert 'find_release()' in code and '["--release", str(RELEASE)]' in code and 'if "recipe" in MODE_ROW["kwargs"]:' in code
    assert 'repo_release(MODE) or find_release()' in code, 'the final submission reads weights/final first (no extra input)'
    assert code.index('find_release()') < code.index('subprocess.run(command')


def test_illustrate_refuses_a_csv_run_of_another_recipe(tmp_path):
    from filseg.reproduction.figures import illustrate
    (tmp_path / 'csv_result.json').write_text(json.dumps(dict(recipe='ms4', per_image={})))
    with pytest.raises(ValueError, match='recipe'):
        illustrate('k', tmp_path, tmp_path, tmp_path, 'ms125')
