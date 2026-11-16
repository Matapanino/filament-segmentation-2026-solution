"""Chain maps re-inferred from the models (GPU) and their difference from the stored release maps (CPU), written before
the package module.

A chain recipe (data/chain/<name>.json) names, for every Mask2Former archive, the release model file, the input size
(2048, 2560 or 3072) and the views. The as-run exporters computed each archive as
    rint(bilinear_2048(mean over the views of inverse(max_q(sigmoid(mask_q) * score_q))) * 510) -> uint16
at the network output, BF16 autocast (a1native/export_tta.py at 2048; ms1/export_scale.py and export_scale_v.py for the
scaled input, the native 2048 view resized by torch bilinear). The U-Net fold models give unet_f{k}_{view}_test.npz with
the package's unet_map, and the 5-fold mean is uint8(rint(sum / 5)) (unet_refit/test_maps.py, fold5mean.py). Output
files carry the release names, so the chain CSV stage reads them in place of the stored ones.
Synthetic data and a stand-in network only; no real model.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

KEYS = ['k0', 'k1', 'k2']


def pool4(x):
    """A local, exactly D4-equivariant stand-in for the network output: 4x4 average pooling of [1,1,H,W] -> [H/4, W/4]."""
    return torch.nn.functional.avg_pool2d(x, 4)[0, 0]


class StandIn(torch.nn.Module):
    """Two queries: mask logits a * pool4(x) + b (one per query), fixed class logits; the Mask2Former output fields."""

    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.tensor([6.0, -3.0]))
        self.register_buffer('b', torch.tensor([-2.0, 1.0]))
        self.register_buffer('cls', torch.tensor([[1.5, -0.5], [0.2, 0.3]]))

    def forward(self, x):
        p = pool4(x.float())
        masks = self.a[:, None, None] * p[None] + self.b[:, None, None]
        return SimpleNamespace(class_queries_logits=self.cls[None], masks_queries_logits=masks[None])


def as_run_fg(model, x, size, names):
    """The exporters' fg_lowres, re-typed: scaled input, per view forward, fg at the output, inverse, sum, / n (n > 1)."""
    from filseg.reproduction import chain_maps as M
    xs = x if size == x.shape[-1] else torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    total = None
    for t in M.transforms(names):
        xv = xs if t[0] == 'id' else M.forward(xs, t).contiguous()
        o = model(xv)
        s = o.class_queries_logits[0].float().softmax(-1)[:, 0]
        fg = M.inverse((o.masks_queries_logits[0].float().sigmoid() * s[:, None, None]).amax(0), t)
        total = fg if total is None else total + fg
    return total if len(names) == 1 else total / len(names)


def image(seed, n=2048):
    return np.random.default_rng(seed).integers(0, 256, (n, n), dtype=np.uint8)


# ---------------------------------------------------------------- views, scaled input, the level map
def test_view_names_are_the_as_run_transforms():
    from filseg.reproduction import chain_maps as M, mf_d4
    assert [M.TRANSFORMS[t[0]] for t in mf_d4.TRANSFORMS] == mf_d4.TRANSFORMS
    assert M.TRANSFORMS['flipH'] == ('flipH', True, 2)          # export_scale_v VIEW_SETS[4][3]: flip_W + rot180
    x = torch.rand(1, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(M.forward(x, M.TRANSFORMS['flipH']), torch.flip(x, dims=(-2,)))
    with pytest.raises(ValueError, match='view'):
        M.transforms(['id', 'rot45'])


def test_every_recipe_view_set_is_known_and_in_the_exporters_order():
    from filseg.reproduction import chain, chain_maps as M
    d4 = ['id', 'rot90', 'rot180', 'rot270', 'flipW', 'flipW_rot90', 'flipW_rot180', 'flipW_rot270']
    allowed = {1: [['id']], 2: [['id', 'flipW']], 4: [['id', 'flipW', 'rot180', 'flipH']], 8: [d4]}
    for name in chain.names():
        for a in chain.load_recipe(name)['archives']:
            assert a['views'] in allowed[len(a['views'])], (name, a['file'])
            assert a['size'] in (2048, 2560, 3072) and a['size'] % 32 == 0
            M.transforms(a['views'])


@pytest.mark.parametrize('size', [2048, 2560, 3072])
def test_inverse_realigns_the_network_output_grid_at_every_input_size(size):
    from filseg.reproduction import chain_maps as M
    # integer values: every 4x4 block sum is exact in float32 whatever the summation order of the rotated block
    x = torch.randint(0, 16, (1, 1, size // 8, size // 8), generator=torch.Generator().manual_seed(size)).float()
    ref = pool4(x)
    for t in M.TRANSFORMS.values():
        assert torch.equal(M.inverse(M.forward(x, t), t), x)
        assert torch.equal(M.inverse(pool4(M.forward(x, t).contiguous()), t), ref), t[0]


def test_scaled_input_is_torch_bilinear_and_untouched_at_2048():
    from filseg.reproduction import chain_maps as M
    x = torch.rand(1, 1, 2048, 2048, generator=torch.Generator().manual_seed(0))
    assert M.scaled_input(x, 2048) is x
    y = M.scaled_input(x, 2560)
    assert y.shape == (1, 1, 2560, 2560)
    assert torch.equal(y, torch.nn.functional.interpolate(x, size=(2560, 2560), mode='bilinear', align_corners=False))
    with pytest.raises(ValueError, match='multiple of 32'):
        M.scaled_input(x, 2050)


def test_model_input_is_the_native_view():
    from filseg.reproduction import chain_maps as M
    raw = image(3, 64)
    x = M.model_input(raw, 'cpu')
    assert x.shape == (1, 1, 64, 64) and x.dtype == torch.float32
    assert torch.equal(x[0, 0], torch.from_numpy(raw.astype(np.float32) / 255))


@pytest.mark.parametrize('size,names', [(2048, ['id']),
                                        (2048, ['id', 'rot90', 'rot180', 'rot270', 'flipW', 'flipW_rot90', 'flipW_rot180', 'flipW_rot270']),
                                        (2560, ['id', 'flipW']), (3072, ['id', 'flipW', 'rot180', 'flipH'])])
def test_fg_lowres_is_the_exporters_view_mean_bit_for_bit(size, names):
    from filseg.reproduction import chain_maps as M
    model = StandIn()
    x = M.model_input(image(size), 'cpu')
    timer = {}
    got = M.fg_lowres(model, x, size, names, timer=timer)
    assert got.shape == (size // 4, size // 4) and got.dtype == torch.float32
    assert torch.equal(got, as_run_fg(model, x, size, names))
    assert sorted(timer) == sorted(names) and all(v >= 0 for v in timer.values())


def test_level_map_is_rint_of_the_bilinear_2048_map_times_510():
    from filseg.reproduction import chain_maps as M
    fg = torch.rand(640, 640, generator=torch.Generator().manual_seed(2))
    lv = M.level_map(fg)
    up = torch.nn.functional.interpolate(fg[None, None], size=(2048, 2048), mode='bilinear', align_corners=False)[0, 0].numpy()
    assert lv.dtype == np.int64 and lv.shape == (2048, 2048)
    assert np.array_equal(lv, np.rint(up * 510).astype(np.int64))
    assert 0 <= lv.min() and lv.max() <= 510


def test_map_writer_bytes_equal_the_package_save_npz(tmp_path):
    from filseg.reproduction import chain_maps as M
    from filseg.reproduction.determinism import save_npz
    arrays = {k: np.random.default_rng(i).integers(0, 511, (2048, 2048)).astype(np.uint16) for i, k in enumerate(KEYS)}
    w = M.MapWriter(tmp_path / 'a.npz')
    for k in KEYS: w.write(k, arrays[k])
    w.close()
    save_npz(tmp_path / 'b.npz', arrays)
    assert (tmp_path / 'a.npz').read_bytes() == (tmp_path / 'b.npz').read_bytes()
    w = M.MapWriter(tmp_path / 'c.npz'); w.write('k1', arrays['k1'])
    with pytest.raises(AssertionError):
        w.write('k0', arrays['k0'])


def test_unet_level_mean_is_rint_of_the_sum_over_n():
    from filseg.reproduction import chain_maps as M
    q = [np.full((2048, 2048), v, np.uint8) for v in (1, 2, 2, 2, 5)]       # 12 / 5 = 2.4 -> 2
    assert M.unet_level_mean(q).dtype == np.uint8 and int(M.unet_level_mean(q)[0, 0]) == 2
    q = [np.full((2048, 2048), v, np.uint8) for v in (0, 0, 0, 5, 5)]       # 10 / 5 = 2
    assert int(M.unet_level_mean(q)[0, 0]) == 2
    q = [np.full((2048, 2048), v, np.uint8) for v in (255, 255, 255, 255, 254)]
    assert int(M.unet_level_mean(q)[0, 0]) == 255                          # 254.8 -> 255, no overflow
    assert np.array_equal(M.unet_level_mean(q[:1]), q[0])


# ---------------------------------------------------------------- the plan of a recipe and the two-GPU split
def test_plan_of_ms125_is_its_ten_archives_and_ten_unet_fold_models():
    from filseg.reproduction import chain, chain_maps as M
    r = chain.load_recipe('ms125'); pins = chain.release_files()
    units = M.plan('ms125')
    mf = [u for u in units if u['kind'] == 'mf']; un = [u for u in units if u['kind'] == 'unet']
    assert [u['file'] for u in mf] == [a['file'] for a in r['archives']]
    assert [(u['model'], u['size'], u['views']) for u in mf] == [(a['model'], a['size'], a['views']) for a in r['archives']]
    assert [u['file'] for u in un] == [f'unet_f{k}_{v}_test.npz' for k in range(5) for v in ('full', 'local')]
    assert [u['model'] for u in un] == [f'unetmodel_f{k}_{v}.pt' for k in range(5) for v in ('full', 'local')]
    assert all(u['file'] in pins and u['model'] in pins for u in units)
    assert M.derived('ms125') == ['unet_fold5mean_full.npz', 'unet_fold5mean_local.npz']


def test_plan_of_several_recipes_is_deduplicated_and_filtered_by_file_name():
    from filseg.reproduction import chain_maps as M
    names = ['mf_c1536_f0_d4_test.npz', 'mf_c1536s125_f0_test.npz', 'mf_c1536s125d4_f0_test.npz', 'mf_c1536s150h4_f0_test.npz', 'unet_f0_full_test.npz']
    units = M.plan(['ms4', 'ms125'], names=names)
    assert sorted(u['file'] for u in units) == sorted(names)
    assert {u['file']: (u['size'], len(u['views'])) for u in units if u['kind'] == 'mf'} == {
        'mf_c1536_f0_d4_test.npz': (2048, 8), 'mf_c1536s125_f0_test.npz': (2560, 2),
        'mf_c1536s125d4_f0_test.npz': (2560, 8), 'mf_c1536s150h4_f0_test.npz': (3072, 4)}
    with pytest.raises(ValueError, match='not in'):
        M.plan('ms125', names=['mf_c768_f0_test.npz'])


def test_plan_of_a_package_unet_recipe_has_no_unet_units():
    from filseg.reproduction import chain_maps as M
    assert {u['kind'] for u in M.plan('c768')} == {'mf'} and M.derived('c768') == []


def test_split_puts_every_unit_on_one_gpu_and_balances_the_cost():
    from filseg.reproduction import chain_maps as M
    units = [u for u in M.plan('ms125') if u['kind'] == 'mf']
    parts = M.split(units, 2)
    assert sorted(u['file'] for p in parts for u in p) == sorted(u['file'] for u in units)
    costs = [sum(u['cost'] for u in p) for p in parts]
    assert max(costs) <= 0.55 * sum(costs)
    assert M.split(units[:1], 2)[1] == []


# ---------------------------------------------------------------- the GPU loop on CPU with stand-ins: layout, resume
def stand_in_loaders(calls=None):
    def mf(unit):
        if calls is not None: calls.append(unit['file'])
        return StandIn()

    def unet(unit):
        return dict(fold=unit['fold'], view=unit['view']), None

    def unet_map(model, raw, spec):
        return ((raw.astype(np.int32) + 7 * model['fold'] + (model['view'] == 'local')) % 256).astype(np.uint8)
    return dict(mf=mf, unet=unet, unet_map=unet_map)


def read(key):
    return image(KEYS.index(key))


def test_run_units_writes_release_named_archives_with_receipts_and_resumes(tmp_path):
    from filseg.reproduction import chain_maps as M
    from filseg.reproduction.determinism import save_npz
    names = ['mf_c1536s125_f0_test.npz', 'unet_f0_full_test.npz']
    units = M.plan('ms125', names=names); calls = []
    rows = M.run_units(units, KEYS, read, tmp_path, stand_in_loaders(calls), device='cpu')
    assert [r['file'] for r in rows] == names and all(r['status'] == 'complete' for r in rows)
    mf = rows[0]
    assert mf['images'] == 3 and mf['size'] == 2560 and mf['views'] == ['id', 'flipW'] and mf['seconds'] >= 0
    assert set(mf['view_seconds']) == {'id', 'flipW'} and mf['seconds_per_view'] >= 0
    assert mf['equal_to_release'] is False and len(mf['sha256']) == 64
    with np.load(tmp_path / names[0]) as z:
        assert sorted(z.files) == KEYS and z['k0'].dtype == np.uint16 and z['k0'].shape == (2048, 2048)
        expect = M.level_map(as_run_fg(StandIn(), M.model_input(read('k1'), 'cpu'), 2560, ['id', 'flipW'])).astype(np.uint16)
        assert np.array_equal(z['k1'], expect)
        arrays = {k: z[k] for k in z.files}
    save_npz(tmp_path / 'same.npz', arrays)
    assert (tmp_path / 'same.npz').read_bytes() == (tmp_path / names[0]).read_bytes()
    with np.load(tmp_path / names[1]) as z:
        assert z['k2'].dtype == np.uint8 and np.array_equal(z['k2'], stand_in_loaders()['unet_map'](dict(fold=0, view='full'), read('k2'), None))
    assert not list(tmp_path.glob('*.partial'))
    again = M.run_units(units, KEYS, read, tmp_path, stand_in_loaders(calls), device='cpu')
    assert [r['status'] for r in again] == ['skipped_existing'] * 2 and calls == [names[0]]
    result = json.loads((tmp_path / M.RESULT).read_text())
    assert sorted(result['archives']) == sorted(names) and result['archives'][names[0]]['status'] == 'complete'


def test_run_units_stops_at_the_deadline_without_leaving_a_file(tmp_path):
    from filseg.reproduction import chain_maps as M
    units = M.plan('ms125', names=['mf_c1536s125_f0_test.npz', 'mf_c1536s125_f1_test.npz'])
    rows = M.run_units(units, KEYS, read, tmp_path, stand_in_loaders(), device='cpu', deadline=0)
    assert [r['status'] for r in rows] == ['stopped_at_deadline', 'not_started']
    assert not (tmp_path / 'mf_c1536s125_f0_test.npz').exists() and not list(tmp_path.glob('*.partial'))


def test_fold5mean_is_written_once_every_fold_model_of_a_view_is_there(tmp_path):
    from filseg.reproduction import chain_maps as M
    from filseg.reproduction.determinism import save_npz
    full = [f'unet_f{k}_full_test.npz' for k in range(5)]
    M.run_units(M.plan('ms125', names=full + ['unet_f0_local_test.npz']), KEYS, read, tmp_path, stand_in_loaders(), device='cpu')
    made = M.write_fold5mean(tmp_path, 'ms125')
    assert sorted(made) == ['unet_fold5mean_full.npz'] and not (tmp_path / 'unet_fold5mean_local.npz').exists()
    folds = [np.load(tmp_path / f) for f in full]
    expect = {k: M.unet_level_mean([z[k] for z in folds]) for k in KEYS}
    save_npz(tmp_path / 'expect.npz', expect)
    assert (tmp_path / 'unet_fold5mean_full.npz').read_bytes() == (tmp_path / 'expect.npz').read_bytes()
    result = json.loads((tmp_path / M.RESULT).read_text())
    assert result['archives']['unet_fold5mean_full.npz']['inputs'] == full


# ---------------------------------------------------------------- comparison with the stored maps
def write_archive(path, arrays):
    from filseg.reproduction.determinism import save_npz
    save_npz(path, arrays)


def test_compare_maps_counts_pixels_levels_and_names_the_first_differing_artefact(tmp_path):
    from filseg.reproduction import chain, chain_maps as M
    r = chain.load_recipe('ms125'); rel = tmp_path / 'release'; new = tmp_path / 'new'; rel.mkdir(); new.mkdir()
    a0, a1 = r['archives'][0]['file'], r['archives'][1]['file']
    base = {k: np.random.default_rng(i).integers(0, 511, (2048, 2048)).astype(np.uint16) for i, k in enumerate(KEYS)}
    base['k1'][0, :3] = 100; base['k2'][5, 5] = 100
    write_archive(rel / a0, base); write_archive(new / a0, base)
    changed = {k: v.copy() for k, v in base.items()}
    changed['k1'][0, :3] = 102; changed['k2'][5, 5] = 99
    write_archive(rel / a1, base); write_archive(new / a1, changed)
    u = {k: np.full((2048, 2048), 9, np.uint8) for k in KEYS}
    write_archive(rel / 'unet_f0_full_test.npz', u); write_archive(new / 'unet_f0_full_test.npz', u)
    out = M.compare_maps(new, rel, 'ms125')
    rows = {x['file']: x for x in out['artefacts']}
    assert rows[a0]['images_equal'] == 3 and rows[a0]['changed_pixels'] == 0 and rows[a0]['file_sha256_equal'] is True
    assert rows[a1]['images_equal'] == 1 and rows[a1]['changed_pixels'] == 4 and rows[a1]['max_abs_level_diff'] == 2
    assert rows[a1]['first_differing_image'] == 'k1' and rows[a1]['level_diff_histogram'] == {'1': 1, '2': 3}
    assert rows[a1]['mean_abs_level_diff'] == pytest.approx(7 / (3 * 2048 * 2048))
    assert out['first_differing']['file'] == a1 and out['all_equal'] is False
    assert rows['unet_f0_full_test.npz']['changed_pixels'] == 0
    assert rows[r['archives'][2]['file']]['present'] is False
    assert out['compared'] == 3


def test_compare_maps_reads_only_the_images_both_archives_hold(tmp_path):
    from filseg.reproduction import chain, chain_maps as M
    a0 = chain.load_recipe('ms125')['archives'][0]['file']; rel = tmp_path / 'r'; new = tmp_path / 'n'; rel.mkdir(); new.mkdir()
    base = {k: np.full((2048, 2048), i, np.uint16) for i, k in enumerate(KEYS)}
    write_archive(rel / a0, base); write_archive(new / a0, {'k2': base['k2']})
    row = [x for x in M.compare_maps(new, rel, 'ms125', names=[a0])['artefacts']][0]
    assert row['keys_compared'] == 1 and row['images_equal'] == 1 and row['file_sha256_equal'] is False


# ---------------------------------------------------------------- the CSV stage finds the re-inferred files first
def test_reinferred_files_shadow_the_release_files_in_the_chain_csv_lookup(tmp_path):
    from filseg.reproduction import chain, chain_maps as M
    r = chain.load_recipe('ms125'); rel = tmp_path / 'release'; new = tmp_path / 'new'; rel.mkdir(); new.mkdir()
    wanted = [a['file'] for a in r['archives']] + list(chain.unet_files(r).values())
    for f in wanted: (rel / f).write_bytes(b'stored')
    produced = [u['file'] for u in M.plan('ms125')] + M.derived('ms125')
    assert set(chain.unet_files(r).values()) <= set(produced) and {a['file'] for a in r['archives']} <= set(produced)
    for f in produced[:2] + M.derived('ms125'): (new / f).write_bytes(b'new')
    paths, seen = chain.stored_files(r, [new, rel], strict=False)
    assert paths['NEW0_0'].parent == new and paths['NEW0_1'].parent == new and paths['NEW1_0'].parent == rel
    assert paths['full'].parent == new and paths['local'].parent == new
    assert seen['NEW0_0']['equal_to_release'] is False


# ---------------------------------------------------------------- entry points
def test_gpu_entry_points_refuse_outside_a_kaggle_t4(tmp_path):
    from filseg.reproduction import chain_maps as M
    with pytest.raises(RuntimeError):
        M.infer_recipe_maps(tmp_path, tmp_path, tmp_path / 'o', 'ms125')
    with pytest.raises(RuntimeError, match='Kaggle'):
        M.launch(tmp_path, tmp_path, tmp_path / 'o', 'ms125')


def test_cli_chain_maps_stage_calls_the_launcher(monkeypatch, capsys):
    from filseg.reproduction import __main__ as entry, chain_maps as M
    calls = []
    monkeypatch.setattr(M, 'launch', lambda *a, **k: calls.append((a, k)) or dict(status='complete', elapsed_seconds=1.0))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'chain-maps', '--images', 'I', '--release', 'REL', '--output', 'O',
                                      '--recipe', 'ms4', 'ms125', '--names', 'mf_c1536_f0_d4_test.npz', '--image-keys', 'a', 'b', '--budget-seconds', '900'])
    entry.main()
    (a, k), = calls
    assert k == dict(images='I', release_root='REL', output='O', recipe=['ms4', 'ms125'], names=['mf_c1536_f0_d4_test.npz'],
                     keys=['a', 'b'], budget_seconds=900, deterministic=True)
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['status'] == 'complete'


def test_cli_chain_maps_compare_stage(monkeypatch, capsys):
    from filseg.reproduction import __main__ as entry, chain_maps as M
    calls = []
    monkeypatch.setattr(M, 'compare_maps', lambda *a, **k: calls.append((a, k)) or dict(all_equal=False, first_differing=dict(file='x')))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'chain-maps-compare', '--images', '-', '--maps', 'NEW', '--release', 'REL',
                                      '--output', 'O', '--recipe', 'ms125'])
    entry.main()
    assert calls == [(('NEW', 'REL', ['ms125']), dict(names=None, output='O'))]


# ---------------------------------------------------------------- one new image (any 2048 x 2048 grey image, not only the test set)
def stand_in_models(recipe):
    from filseg.reproduction import chain
    r = chain.load_recipe(recipe)
    return dict(device='cpu', mf={a['model']: StandIn() for a in r['archives']},
                unet={(k, v): (dict(fold=k, view=v), None) for k in range(5) for v in ('full', 'local')}, unet_map=stand_in_loaders()['unet_map'])


def test_maps_for_image_is_the_archive_writers_arithmetic_for_any_image(tmp_path):
    from filseg.reproduction import chain, chain_maps as M
    r = chain.load_recipe('ms125'); raw = read('k1')
    got = M.maps_for_image(raw, 'ms125', stand_in_models('ms125'))
    assert sorted(got) == sorted([a['name'] for a in r['archives']] + ['full', 'local'])
    assert all(got[a['name']].dtype == np.uint16 and got[a['name']].shape == (2048, 2048) for a in r['archives'])
    assert got['full'].dtype == np.uint8 and got['local'].dtype == np.uint8
    files = ['mf_c1536_f0_d4_test.npz', 'mf_c1536s125_f3_test.npz'] + [f'unet_f{k}_{v}_test.npz' for k in range(5) for v in ('full', 'local')]
    M.run_units(M.plan('ms125', names=files), ['k1'], read, tmp_path, stand_in_loaders(), device='cpu'); M.write_fold5mean(tmp_path, 'ms125', ['k1'])
    assert np.array_equal(got['NEW0_0'], np.load(tmp_path / files[0])['k1'])
    assert np.array_equal(got['NEW3_1'], np.load(tmp_path / files[1])['k1'])
    for v in ('full', 'local'):
        assert np.array_equal(got[v], np.load(tmp_path / f'unet_fold5mean_{v}.npz')['k1'])


def test_maps_for_image_takes_an_image_outside_the_test_set_and_refuses_a_wrong_one():
    from filseg.reproduction import chain_maps as M
    models = stand_in_models('c768')
    got = M.maps_for_image(image(99), 'c768', dict(models, unet={v: (dict(fold=2, view=v), None) for v in ('full', 'local')}))
    assert sorted(got) == sorted([f'NEW{k}_0' for k in range(5)] + ['full', 'local'])
    with pytest.raises(ValueError, match='2048'):
        M.maps_for_image(image(1, 1024), 'c768', models)
    with pytest.raises(ValueError, match='uint8'):
        M.maps_for_image(image(1).astype(np.float32), 'c768', models)
    with pytest.raises(KeyError, match='mfmodel_c768_f0'):
        M.maps_for_image(image(1), 'c768', dict(models, mf={}))


def test_load_recipe_models_loads_each_distinct_model_once(monkeypatch):
    from filseg.reproduction import chain_maps as M
    calls = []
    monkeypatch.setattr(M, 'load_mf', lambda root, u, device: calls.append(u['model']) or ('mf', u['model']))
    monkeypatch.setattr(M, 'load_unet', lambda root, u, device: calls.append(u['model']) or (('unet', u['model']), 'spec'))
    models = M.load_recipe_models('REL', 'ms125', 'cpu')
    assert sorted(models['mf']) == [f'mfmodel_c1536_f{k}.safetensors' for k in range(5)]
    assert sorted(models['unet']) == [(k, v) for k in range(5) for v in ('full', 'local')]
    assert len(calls) == len(set(calls)) == 15 and models['device'] == 'cpu'
