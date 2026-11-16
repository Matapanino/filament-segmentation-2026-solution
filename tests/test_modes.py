"""Named output modes: one table (mode -> run_csv arguments -> expected CSV sha256), the CLI --mode switch, and the default.

Synthetic only; run_csv is replaced by a recorder, so each test checks which code path a mode reaches, not the CSV.
"""
import json
import sys

import pytest

EXPECTED = {
    'pcombo': ('35abbeed91108e40ddbd18c094c901d116441e363b10c31aa8092988d8e8f6e6', dict(pnat=False, pcand=True, selector='pcombo')),
    'mfrich': ('5711532ebb1360bdbc27a2bb04fd1f388b4a6494d0ab911067775aa10c018b85', dict(pnat=False, pcand=True, selector='mfrich')),
    'pneigh': ('c3edf2c8b681d42649dc7ccb6ed10e0400031b37c057d7d2798d6196bed0612c', dict(pnat=False, pcand=True, selector='pneigh')),
    'fresh_ctrl': ('b01866b3f849d7bc43534c188b91545c26f8cd839493175d00ff6d25fb19d085', dict(pnat=False, pcand=True, selector='fresh_ctrl')),
    'pcand': ('14a5c1b684813081ad129d05e72fbb0c09592f292aeb1b1d84e121100c668c6b', dict(pnat=False, pcand=True)),
    'pnat': ('d54257e7f8a1f815d442037050233e52859b0f9594f128b7b258f368f4d05fbe', dict(pnat=True, pcand=None)),
    'equivalent': ('0b2123de5b614561ddf22c2bbc0f05000164ac6a595a9806104c7358b9f46b70', dict(pnat=False, pcand=None)),
}


def cli(monkeypatch, *flags, sha='0' * 64):
    from filseg.reproduction import __main__ as entry, pipeline
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(status='complete', csv=dict(sha256=sha))

    monkeypatch.setattr(pipeline, 'run_csv', recorder)
    monkeypatch.setattr(sys, 'argv', ['filseg.reproduction', 'csv', '--images', 'I', '--maps', 'M', '--output', 'O', *flags])
    entry.main()
    assert len(calls) == 1
    return calls[0]


def test_mode_table_lists_every_reproducible_csv_with_its_hash_and_arguments():
    from filseg.reproduction.modes import MODES
    assert list(MODES) == list(EXPECTED), 'the table order is the documentation order: default first'
    for name, (sha, kwargs) in EXPECTED.items():
        assert MODES[name]['sha256'] == sha, name
        assert MODES[name]['kwargs'] == kwargs, name
        assert MODES[name]['label'] and isinstance(MODES[name]['label'], str)


FINAL_PAIR = ('ms125', 'pcombo')   # the final submission and the no-extra-download mode; DEFAULT_MODE (one line in modes.py) names the one the notebooks build


def test_default_mode_is_one_of_the_final_pair_and_runnable():
    from filseg.reproduction.modes import DEFAULT_MODE, runnable
    assert DEFAULT_MODE in FINAL_PAIR and runnable()[DEFAULT_MODE]['sha256']


def test_runnable_modes_are_the_seven_then_the_pinned_mf_map_mode_then_the_chain_modes():
    from filseg.reproduction.modes import CHAIN_MODES, MFMAP_MODES, MODES, runnable
    table = runnable()
    assert list(table) == list(EXPECTED) + ['mftta'] + list(CHAIN_MODES)
    assert all(table[n] is MODES[n] for n in EXPECTED) and table['mftta'] is MFMAP_MODES['mftta']
    assert all(row['sha256'] and row['ref'] for row in table.values())


def test_resolve_names_a_runnable_mode_and_refuses_the_rest():
    from filseg.reproduction import modes
    assert modes.resolve('mftta') == ('mftta', modes.MFMAP_MODES['mftta'])
    assert modes.resolve('pnat') == ('pnat', modes.MODES['pnat'])
    assert modes.resolve(None) == (modes.DEFAULT_MODE, modes.runnable()[modes.DEFAULT_MODE])
    for bad in ('mfhires', 'mfcomb', 'nope', ''):
        with pytest.raises(ValueError):
            modes.resolve(bad)


def map_layout(tmp_path, maps=True, views=None):
    """A fake /kaggle/working/maps and /kaggle/input: package maps attached, and MF-view receipts recording a map set."""
    from filseg.reproduction import mfmaps
    working, attached = tmp_path / 'working_maps', tmp_path / 'input'
    working.mkdir(); attached.mkdir()
    if maps:
        (attached / 'pkg' / 'maps').mkdir(parents=True)
        (attached / 'pkg' / 'maps' / 'inference_result.json').write_text('{"status": "complete"}')
    if views is not None:
        (attached / 'views' / 'maps').mkdir(parents=True)
        sha = mfmaps.EXPECTED_MAP_SETS['mftta'] if views == 'expected' else '0' * 64
        (attached / 'views' / 'maps' / mfmaps.RESULT).write_text(json.dumps(dict(status='complete', map_sets=dict(mftta=dict(sha256=sha)))))
    return (working, attached)


@pytest.mark.parametrize('maps,views,mode,expected', [
    (False, None, 'pcombo', ['reproduce_maps.ipynb']),
    (True, None, 'pcombo', []),
    (True, 'expected', 'pcombo', []),
    (True, None, 'mftta', ['reproduce_mf_views.ipynb']),
    (True, 'other', 'mftta', ['reproduce_mf_views.ipynb']),
    (True, 'expected', 'mftta', []),
    (False, None, 'mftta', ['reproduce_maps.ipynb', 'reproduce_mf_views.ipynb']),
])
def test_gpu_stages_run_only_what_is_missing_for_the_mode(tmp_path, maps, views, mode, expected):
    from filseg.reproduction.paths import gpu_stages
    assert gpu_stages(mode, roots=map_layout(tmp_path, maps, views)) == (mode, expected)


def test_find_variant_maps_prefers_this_session_and_refuses_two_candidates(tmp_path):
    from filseg.reproduction import mfmaps
    from filseg.reproduction.paths import find_variant_maps
    roots = map_layout(tmp_path, True, 'expected')
    assert find_variant_maps('mftta', roots) == roots[1] / 'views' / 'maps'
    (roots[0] / mfmaps.RESULT).write_text((roots[1] / 'views' / 'maps' / mfmaps.RESULT).read_text())
    assert find_variant_maps('mftta', roots) == roots[0]
    (roots[0] / mfmaps.RESULT).unlink()
    (roots[1] / 'copy').mkdir(); (roots[1] / 'copy' / mfmaps.RESULT).write_text((roots[1] / 'views' / 'maps' / mfmaps.RESULT).read_text())
    with pytest.raises(ValueError):
        find_variant_maps('mftta', roots)
    assert find_variant_maps('mftta', (tmp_path / 'missing',)) is None


@pytest.mark.parametrize('default', FINAL_PAIR)
def test_switching_the_default_is_one_line_and_reaches_every_entry_point(monkeypatch, tmp_path, default):
    """Setting modes.DEFAULT_MODE alone switches the mode the notebooks resolve, the GPU stages pipeline.ipynb runs and the CSV
    the submission notebook checks; nothing else names the default (tests/test_notebooks_final.py)."""
    from filseg.reproduction import modes
    from filseg.reproduction.paths import gpu_stages
    monkeypatch.setattr(modes, 'DEFAULT_MODE', default)
    name, row = modes.resolve(None)
    assert name == default and row['sha256'] == modes.runnable()[default]['sha256']
    stages = gpu_stages(None, roots=map_layout(tmp_path, True, None))[1]
    assert stages == (['reproduce_mf_views.ipynb'] if default == 'mftta' else [])


def test_submission_refs_are_the_scored_submissions():
    from filseg.reproduction.modes import MODES
    refs = {n: m['ref'] for n, m in MODES.items()}
    assert refs == dict(pcombo=58278257, mfrich=58274749, pneigh=58268408, fresh_ctrl=58265972, pcand=58232010, pnat=58224552, equivalent=58224327)


@pytest.mark.parametrize('name', list(EXPECTED))
def test_cli_mode_reaches_run_csv_with_the_table_arguments(monkeypatch, name):
    args, kwargs = cli(monkeypatch, '--mode', name)
    assert args == ('I', None, 'M', 'O') and kwargs == EXPECTED[name][1]


@pytest.mark.parametrize('name', ['pcombo', 'mfrich', 'pneigh', 'fresh_ctrl'])
def test_selector_modes_resolve_to_the_pinned_selector_files(name):
    from filseg.reproduction import refit
    from filseg.reproduction.modes import MODES
    path, sha = refit.deployed(MODES[name]['kwargs']['selector'])
    assert refit.load_selector(path, sha)['role'] == {'pcombo': refit.PCOMBO_ROLE, 'mfrich': refit.MFRICH_ROLE, 'pneigh': refit.PNEIGH_ROLE, 'fresh_ctrl': refit.ROLE}[name]


def test_fresh_ctrl_name_is_the_same_selector_as_selector_true():
    from filseg.reproduction import refit
    assert refit.deployed('fresh_ctrl') == refit.deployed()


@pytest.mark.parametrize('other', ['--pnat', '--pcand'])
def test_cli_rejects_mode_together_with_the_old_flags(monkeypatch, other):
    with pytest.raises(SystemExit):
        cli(monkeypatch, '--mode', 'pcombo', other)


def test_cli_rejects_an_unknown_mode(monkeypatch):
    with pytest.raises(SystemExit):
        cli(monkeypatch, '--mode', 'nope')


def test_cli_mode_prints_the_expected_hash_and_whether_it_matches(monkeypatch, capsys):
    cli(monkeypatch, '--mode', 'pcombo', sha=EXPECTED['pcombo'][0])
    out = capsys.readouterr().out
    assert '"mode": "pcombo"' in out and '"expected_csv_sha256": "' + EXPECTED['pcombo'][0] + '"' in out and '"matches_expected": true' in out
    cli(monkeypatch, '--mode', 'mfrich', sha='1' * 64)
    assert '"matches_expected": false' in capsys.readouterr().out


def test_cli_without_flags_still_builds_the_equivalent_csv(monkeypatch):
    # The README documents "no flag" as the incumbent-equivalent CSV; the default path for a judge is the notebook.
    assert cli(monkeypatch)[1] == dict(pnat=False, pcand=None)
