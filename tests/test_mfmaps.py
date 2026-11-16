"""MF-map variants (D4 TTA, tiled native-scale reading and their combination), written before the package module.

The GPU code reuses the CV lanes' inference code: mf_d4.py is detect_20261113/mftta/d4.py and mf_tile.py is
detect_20261113/mfhires/tile.py, byte for byte. The new test map sets are additions: the five archives of the map set
d7aa6a41... are untouched, the variant archives are MF{k}_tta and MF{k}_tiled (uint16, value / 510, the stored format),
and the CSV stage reads the variant MF map only when a variant is named. Synthetic data only; no model, no fitting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pnat import keep_all_model, synthetic_maps

D4_SHA256 = '70510616c1e390149513041e50658cea1b9d92aaf3882e592026156199701e6b'     # detect_20261113/mftta/d4.py
TILE_SHA256 = 'bd9af95699650e3fec2f6eb9e359c801af2ac67467a8cfd0dba227aea9ef6e47'   # detect_20261113/mfhires/tile.py
# Archive sha256 of the package's deterministic T4 maps (pkg_final hash_check.json) and of the lanes' T4 test outputs
# (mftta test_tta_MF{k}.npz, mfhires test_tiled_MF{k}.npz).
ARCHIVES = {
    'full': '832e7ff901f2f7d6f0c973d08b2e0f644f977e790954993d3dbff5c38ff242dd',
    'local': 'd0a007fe6473737ac612bef8e62cab753fea78f15389d67088b706ffed1063f2',
    'MF0': 'f385e1bd162a4f5c821b13f1180a4a3243c9c6f8bc5ed670a449cd03f395fab2',
    'MF1': 'f0550a6fa48e67a3af9ff55d1417c45bfae99d934fb6db5d6d8aa664ff3e268c',
    'MF2': '3ef39ea36a39b315ae5d7672b704a0a95a1a62f58276f25a85777caa404c9dbd',
    'MF0_tta': '2a123ff3b6cfee00dd99f76ba7fb6da6bec659a4278964c406add2188beb3a72',
    'MF1_tta': 'b9c9ef3751b70ee5cf14aafc2e1220fcada84bd79c8433810cacfdf321b81541',
    'MF2_tta': '021ad5c97eaa4e746d1c9985b00ed1b78d36730c338ac02328698812762dfc6f',
    'MF0_tiled': 'b8df4fe3e1a593a730cbffaf09dffdfcca5f2330ac02d9cc871f0f381719b0e2',
    'MF1_tiled': '2e24c2b8072d2083ba2ca849d7aac65f9af10a67afa8cb0013677748c37d8346',
    'MF2_tiled': 'c6b5f1e84151db3d204053c2ed1e0fd227b2290d87e895af191ce2a5c69b4592',
}
PACKAGE_MAP_SET = 'd7aa6a41fafc26b07ce89b306b884f6667ebb68b86289ea4745ee44c0b0d195f'


def pool4(x):
    """A local, exactly D4-equivariant stand-in for the network: 4x4 average pooling of [1,1,H,W] -> [H/4, W/4]."""
    return torch.nn.functional.avg_pool2d(x, 4)[0, 0]


def variant_maps(base=None, tta_shift=0, tiled_shift=0):
    maps = base or synthetic_maps()
    for k in range(3):
        m = maps[f'MF{k}'].astype(np.int32)
        maps[f'MF{k}_tta'] = np.clip(m + tta_shift, 0, 510).astype(np.uint16)
        maps[f'MF{k}_tiled'] = np.clip(m + tiled_shift, 0, 510).astype(np.uint16)
    return maps


# ---------------------------------------------------------------- the lanes' inference code, byte for byte
def test_vendored_view_modules_are_the_cv_sources_byte_for_byte():
    from filseg.reproduction import mf_d4, mf_tile
    assert hashlib.sha256(Path(mf_d4.__file__).read_bytes()).hexdigest() == D4_SHA256
    assert hashlib.sha256(Path(mf_tile.__file__).read_bytes()).hexdigest() == TILE_SHA256


def test_source_manifest_records_the_vendored_modules():
    import filseg.reproduction as package
    manifest = json.loads((Path(package.__file__).parent / 'source_manifest.json').read_text())
    assert manifest['mf_d4.py'] == dict(source='detect_20261113/mftta/d4.py', sha256=D4_SHA256, byte_identical=True)
    assert manifest['mf_tile.py'] == dict(source='detect_20261113/mfhires/tile.py', sha256=TILE_SHA256, byte_identical=True)


def test_d4_transforms_invert_exactly():
    from filseg.reproduction import mf_d4 as D4
    x = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0))
    outs = [D4.forward(x, t) for t in D4.TRANSFORMS]
    assert len(D4.TRANSFORMS) == 8 and D4.TRANSFORMS[0] == ('id', False, 0)
    assert all(not torch.equal(outs[i], outs[j]) for i in range(8) for j in range(i + 1, 8))
    for t, y in zip(D4.TRANSFORMS, outs):
        assert torch.equal(D4.inverse(y, t), x)
        assert torch.equal(D4.inverse(D4.forward(x[0, 0], t), t), x[0, 0])


def test_tiling_round_trip_and_partition_of_unity():
    from filseg.reproduction import mf_tile as T
    assert T.offsets() == [0, 512, 1024] and len(T.tiles()) == 9
    total = torch.zeros(T.CANVAS, T.CANVAS, dtype=torch.float64)
    for (oy, ox), w in T.weights():
        total[oy // 4:oy // 4 + T.TILE_OUT, ox // 4:ox // 4 + T.TILE_OUT] += w.double()
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)
    x = torch.rand(1, 1, 2048, 2048, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(T.tiled(pool4, x), pool4(x), atol=1e-5)
    assert not torch.allclose(T.tiled(lambda t: pool4(torch.roll(t, 4, dims=-1)), x), pool4(x), atol=1e-3)


# ---------------------------------------------------------------- one image: identity, TTA and tiled maps
def raw_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (2048, 2048), dtype=np.uint8)


def test_image_views_are_the_lanes_formulas():
    from filseg.reproduction import mfmaps, mf_d4 as D4, mf_tile as T
    raw = raw_image()
    out = mfmaps.image_views(pool4, raw, device='cpu')
    assert set(out) == {'identity', 'tta', 'tiled'}
    assert all(v.dtype == np.uint16 and v.shape == (2048, 2048) and int(v.max()) <= 510 for v in out.values())
    x = mfmaps.model_input(raw, device='cpu')
    assert x.shape == (1, 1, 1024, 1024) and x.dtype == torch.float32
    ident, mean = D4.tta(pool4, x)
    assert np.array_equal(out['identity'], D4.quantise(D4.upsample(ident)))
    assert np.array_equal(out['tta'], D4.quantise(D4.upsample(mean)))
    assert np.array_equal(out['tiled'], T.quantise(T.upsample(T.tiled(pool4, T.native_input(raw, device='cpu')))))
    # an equivariant stand-in: the 8-view mean equals the identity view up to float rounding
    assert int(np.abs(out['tta'].astype(int) - out['identity'].astype(int)).max()) <= 1


def test_model_input_is_the_package_mf_view():
    import cv2
    from filseg.reproduction import mfmaps
    raw = raw_image()
    expected = cv2.resize(raw, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    assert torch.equal(mfmaps.model_input(raw, device='cpu')[0, 0], torch.from_numpy(expected))


# ---------------------------------------------------------------- map sets
def test_map_set_hash_reproduces_the_package_map_set():
    from filseg.reproduction import mfmaps
    assert mfmaps.map_set_sha256({n: ARCHIVES[n] for n in mfmaps.MAP_SETS['package']}) == PACKAGE_MAP_SET
    assert mfmaps.MAP_SETS['package'] == ('full', 'local', 'MF0', 'MF1', 'MF2')


def test_variant_map_sets_and_their_expected_hashes():
    from filseg.reproduction import mfmaps
    tta = ('MF0_tta', 'MF1_tta', 'MF2_tta'); tiled = ('MF0_tiled', 'MF1_tiled', 'MF2_tiled')
    assert mfmaps.MF_ARCHIVES == dict(mftta=tta, mfhires=('MF0', 'MF1', 'MF2') + tiled, mfcomb=tta + tiled)
    for v in ('mftta', 'mfhires', 'mfcomb'):
        assert mfmaps.MAP_SETS[v] == ('full', 'local') + mfmaps.MF_ARCHIVES[v]
        assert mfmaps.EXPECTED_MAP_SETS[v] == mfmaps.map_set_sha256({n: ARCHIVES[n] for n in mfmaps.MAP_SETS[v]})
    assert mfmaps.EXPECTED_MAP_SETS['package'] == PACKAGE_MAP_SET


# ---------------------------------------------------------------- arithmetic
def test_level_mean_of_three_is_mf_mean_bit_for_bit_and_blend_is_candidate_blend():
    from filseg.reproduction.arithmetic import mf_mean, mf_levels_mean, candidate_blend, blend_with_mf
    rng = np.random.default_rng(1)
    a, b, c = (rng.integers(0, 511, (64, 64)).astype(np.uint16) for _ in range(3))
    full, local = (rng.integers(0, 256, (64, 64)).astype(np.uint8) for _ in range(2))
    assert mf_levels_mean(a, b, c).dtype == np.float32
    assert np.array_equal(mf_levels_mean(a, b, c), mf_mean(a, b, c))
    assert np.array_equal(blend_with_mf(full, local, mf_mean(a, b, c)), candidate_blend(full, local, a, b, c))
    with pytest.raises(ValueError):
        mf_levels_mean(a, b, c.astype(np.int32))
    with pytest.raises(ValueError):
        mf_levels_mean(a, b, np.full_like(c, 511))


def test_variant_mf_maps():
    from filseg.reproduction import mfmaps
    from filseg.reproduction.arithmetic import mf_mean
    rng = np.random.default_rng(2)
    maps = {n: rng.integers(0, 511, (32, 32)).astype(np.uint16) for n in ('MF0', 'MF1', 'MF2', 'MF0_tta', 'MF1_tta', 'MF2_tta', 'MF0_tiled', 'MF1_tiled', 'MF2_tiled')}
    assert np.array_equal(mfmaps.variant_mf(maps, 'mftta'), mf_mean(maps['MF0_tta'], maps['MF1_tta'], maps['MF2_tta']))
    six = lambda names: sum(maps[n].astype(np.int64) for n in names).astype(np.float32) / np.float32(3060)
    hires = mfmaps.variant_mf(maps, 'mfhires'); comb = mfmaps.variant_mf(maps, 'mfcomb')
    assert hires.dtype == comb.dtype == np.float32
    assert np.array_equal(hires, six(mfmaps.MF_ARCHIVES['mfhires'])) and np.array_equal(comb, six(mfmaps.MF_ARCHIVES['mfcomb']))
    # the CV lanes' per-image map is (a + b) / 1020; at test the three folds are averaged: the same value up to rounding
    mix = lambda a, b: (a.astype(np.float32) + b.astype(np.float32)) / np.float32(1020)
    lane = sum(mix(maps[f'MF{k}'], maps[f'MF{k}_tiled']) for k in range(3)) / np.float32(3)
    assert np.allclose(hires, lane, atol=1e-6)
    with pytest.raises(ValueError):
        mfmaps.variant_mf(maps, 'package')


def test_hires_with_tiled_equal_to_the_stored_map_is_the_old_mf_mean_bit_for_bit():
    from filseg.reproduction import mfmaps
    from filseg.reproduction.arithmetic import mf_mean
    rng = np.random.default_rng(3)
    maps = {f'MF{k}': rng.integers(0, 511, (32, 32)).astype(np.uint16) for k in range(3)}
    maps.update({f'MF{k}_tiled': maps[f'MF{k}'].copy() for k in range(3)})
    assert np.array_equal(mfmaps.variant_mf(maps, 'mfhires'), mf_mean(maps['MF0'], maps['MF1'], maps['MF2']))


# ---------------------------------------------------------------- the CSV stage reads the variant map only when named
def test_image_stages_default_is_the_old_path_and_an_identical_variant_gives_the_same_stages():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.pcand import uniform
    raw, model = keep_all_model()
    old = image_stages(raw, synthetic_maps(), model, pcand=uniform(.4016711682033202))
    for v in ('mftta', 'mfhires', 'mfcomb'):
        maps = variant_maps()
        new = image_stages(raw, maps, model, pcand=uniform(.4016711682033202), mf_variant=v)
        assert new['stages'] == old['stages'] and new['rles'] == old['rles'], v


def test_image_stages_variant_reads_the_variant_mf_map():
    from filseg.reproduction.pipeline import image_stages
    from filseg.reproduction.audit import array_sha
    from filseg.reproduction import mfmaps
    from filseg.reproduction.arithmetic import blend_with_mf
    raw, model = keep_all_model()
    maps = variant_maps(tta_shift=-150, tiled_shift=60)
    old = image_stages(raw, maps, model)
    for v in ('mftta', 'mfhires', 'mfcomb'):
        out = image_stages(raw, maps, model, mf_variant=v)
        mf = mfmaps.variant_mf(maps, v)
        assert out['stages']['MF_mean'] == array_sha(mf) != old['stages']['MF_mean']
        assert out['stages']['candidate_blend'] == array_sha(blend_with_mf(maps['full'], maps['local'], mf))
    with pytest.raises(ValueError):
        image_stages(raw, maps, model, mf_variant='nope')


def test_variant_archives_are_loaded_hash_checked(tmp_path):
    from filseg.reproduction import mfmaps
    from filseg.reproduction.determinism import save_npz
    from filseg.reproduction.audit import identity
    keys = ['a', 'b']; networks = {}
    for k in range(3):
        for kind in ('tta', 'tiled'):
            name = f'MF{k}_{kind}'; p = tmp_path / (name + '.npz')
            save_npz(p, {key: np.full((4, 4), k, np.uint16) for key in keys}); networks[name] = dict(file=p.name, **identity(p))
    (tmp_path / mfmaps.RESULT).write_text(json.dumps(dict(status='complete', networks=networks)))
    got = mfmaps.load_variant_archives(tmp_path, 'mfcomb', keys)
    assert sorted(got) == sorted(mfmaps.MF_ARCHIVES['mfcomb']) and got['MF2_tiled']['b'][0, 0] == 2
    assert sorted(mfmaps.load_variant_archives(tmp_path, 'mfhires', keys)) == ['MF0_tiled', 'MF1_tiled', 'MF2_tiled']
    with pytest.raises(ValueError):
        mfmaps.load_variant_archives(tmp_path, 'mftta', keys + ['c'])
    save_npz(tmp_path / 'MF1_tta.npz', {key: np.full((4, 4), 9, np.uint16) for key in keys})
    with pytest.raises(ValueError):
        mfmaps.load_variant_archives(tmp_path, 'mftta', keys)


def test_run_csv_has_the_variant_switch_off_by_default():
    import inspect
    from filseg.reproduction import pipeline
    assert inspect.signature(pipeline.image_stages).parameters['mf_variant'].default is None
    assert inspect.signature(pipeline.run_csv).parameters['mf_variant'].default is None
    assert inspect.signature(pipeline.run_csv).parameters['variant_maps'].default is None


# ---------------------------------------------------------------- selector roles and named modes
def selector68(role, threshold):
    from filseg.reproduction import refit
    n = len(refit.PCOMBO_FEATURES)
    return dict(role=role, feature_order=list(refit.PCOMBO_FEATURES), classes=[0, 1], scaler_mean=[0.] * n, scaler_scale=[1.] * n,
                coef=[[0.] * n], intercept=[0.], threshold=threshold)


def test_variant_selector_roles_read_pcombo_features_and_need_a_pinned_cutoff(monkeypatch):
    from filseg.reproduction import refit
    assert set(refit.MFMAP_ROLES) == {'mftta', 'mfhires', 'mfcomb'}
    assert len(set(refit.MFMAP_ROLES.values()) | {refit.ROLE, refit.PNEIGH_ROLE, refit.MFRICH_ROLE, refit.PCOMBO_ROLE, refit.LR39_ROLE}) == 8
    for v, role in refit.MFMAP_ROLES.items():
        if v not in refit.MFMAP_THRESHOLDS:
            with pytest.raises(ValueError):
                refit.check(selector68(role, .35))
        monkeypatch.setitem(refit.MFMAP_THRESHOLDS, v, .35)
        assert refit.check(selector68(role, .35))['role'] == role
        for bad in (selector68(role, .375), dict(selector68(role, .35), feature_order=list(refit.MFRICH_FEATURES))):
            with pytest.raises(ValueError):
                refit.check(bad)


def test_mftta_cutoff_is_pinned_before_the_ship_fit():
    """Stage 2 of mfmap_ship: the mftta cutoff is the median of the CV lane's 100 nested cutoffs (0.35; sorted values 50 and 51
    both 0.35), pinned before the deployment fit; the variants that were not adopted stay refused."""
    from filseg.reproduction import refit
    assert refit.MFMAP_THRESHOLDS == {'mftta': 0.35}
    role = refit.MFMAP_ROLES['mftta']
    assert role == 'deployment_refit_mftta_full707_mf_mixture'
    assert refit.check(selector68(role, .35))['threshold'] == .35
    with pytest.raises(ValueError):
        refit.check(selector68(role, .375))
    for v in ('mfhires', 'mfcomb'):
        with pytest.raises(ValueError):
            refit.check(selector68(refit.MFMAP_ROLES[v], .35))


# The deployment mftta selector fitted on Kaggle CPU (mfmap_ship Stage 2, two fits gave these exact bytes; the 100 saved
# 'mftta' CV selectors were refitted exactly from the same bank) and the CSV the ship kernel built with it.
MFTTA_SELECTOR_SHA256 = '89a0c469c78c4a6b0095880560bd18a15f107441ce4169c7c5df4a528ea7a4cf'
MFTTA_CSV_SHA256 = '3cdd3f06feec44b7b091d1807626c85dfbdcaf44afb285392e440b80432549ac'


def test_pinned_mftta_selector_is_package_data():
    import hashlib
    from pathlib import Path
    from filseg.reproduction import refit
    from filseg.reproduction.modes import MFMAP_MODES
    path, digest = refit.deployed('mftta')
    assert digest == MFTTA_SELECTOR_SHA256 and hashlib.sha256(Path(path).read_bytes()).hexdigest() == MFTTA_SELECTOR_SHA256
    assert Path(path) == Path(refit.__file__).parent / 'data/mftta_selector.json'
    sel = refit.load_selector(path, digest)
    assert sel['role'] == refit.MFMAP_ROLES['mftta'] and sel['threshold'] == .35 and sel['feature_order'] == refit.PCOMBO_FEATURES
    assert sel['training']['images'] == 707 and sel['training']['candidates'] == 5482
    assert MFMAP_MODES['mftta']['sha256'] == MFTTA_CSV_SHA256 and MFMAP_MODES['mftta']['ref'] == 58299400   # submitted once by mfmap_ship
    for v in ('mfhires', 'mfcomb'):
        assert v not in refit.PINNED and MFMAP_MODES[v]['sha256'] is None


def test_mfmap_modes_table():
    from filseg.reproduction import mfmaps
    from filseg.reproduction.modes import MODES, MFMAP_MODES
    assert list(MFMAP_MODES)[:3] == ['mftta', 'mfhires', 'mfcomb'] and not set(MFMAP_MODES) & set(MODES)   # then mfvar (test_mfvar.py)
    for v in ('mftta', 'mfhires', 'mfcomb'):
        row = MFMAP_MODES[v]
        assert row['kwargs'] == dict(pnat=False, pcand=True, selector=v, mf_variant=v)
        assert row['map_set'] == mfmaps.EXPECTED_MAP_SETS[v] and row['label']


def test_cli_mfmap_mode_reaches_run_csv_with_the_variant(monkeypatch):
    from test_modes import cli
    from filseg.reproduction.modes import MFMAP_MODES
    for v in MFMAP_MODES:
        args, kwargs = cli(monkeypatch, '--mode', v)
        assert args == ('I', None, 'M', 'O') and kwargs == MFMAP_MODES[v]['kwargs']
    args, kwargs = cli(monkeypatch, '--mode', 'mftta', '--variant-maps', 'V')
    assert kwargs == dict(MFMAP_MODES['mftta']['kwargs'], variant_maps='V')
    with pytest.raises(SystemExit):
        cli(monkeypatch, '--mode', 'pcombo', '--variant-maps', 'V')


def test_cli_infer_mfviews_reaches_the_launcher(monkeypatch):
    import sys
    from filseg.reproduction import __main__ as entry, launch
    calls = []
    monkeypatch.setattr(launch, 'run_gpu_mf_views', lambda *a, **k: calls.append((a, k)) or dict(status='complete'))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'infer-mfviews', '--images', 'I', '--output', 'O'])
    entry.main()
    assert calls == [(('I', None, 'O', 'bf16', 6600), dict(deterministic=True, views=('tta',)))], 'the default GPU stage is the D4 TTA only'
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'infer-mfviews', '--images', 'I', '--output', 'O', '--views', 'tta', 'tiled'])
    entry.main()
    assert calls[-1] == (('I', None, 'O', 'bf16', 6600), dict(deterministic=True, views=('tta', 'tiled')))
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'infer-mfviews', '--images', 'I', '--output', 'O', '--views', 'tiled'])
    with pytest.raises(SystemExit):
        entry.main()


def test_image_views_tta_only_skips_the_tiled_reading():
    from filseg.reproduction import mfmaps
    seen = []
    def fg(x):
        seen.append(tuple(x.shape)); return pool4(x)
    out = mfmaps.image_views(fg, raw_image(), device='cpu', views=('tta',))
    assert set(out) == {'identity', 'tta'} and len(seen) == 8
    seen.clear(); mfmaps.image_views(fg, raw_image(), device='cpu')
    assert len(seen) == 17, 'the default of image_views reads both (8 D4 views + 9 tiles)'


def test_mf_views_notebook_runs_the_cli_stage():
    import filseg.reproduction as package
    nb = json.loads((Path(package.__file__).parents[3] / 'notebooks/reproduce_mf_views.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert '"infer-mfviews"' in code and 'find_test_images' in code and '"--views", "tta"' in code
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    from filseg.reproduction import mfmaps
    assert mfmaps.EXPECTED_MAP_SETS['mftta'] in text
