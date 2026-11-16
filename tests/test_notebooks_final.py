"""The notebooks' default (modes.DEFAULT_MODE, named nowhere else), the mode switch, the mode -> hash tables, the MF-view
stage of an MF-map mode (mftta) and the pipeline figures."""
import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def cells(name, kind):
    nb = json.loads((REPO / 'notebooks' / name).read_text())
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == kind]


def assigns(source):
    return {t.id: n.value for n in ast.parse(source).body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}


def mode_table_rows(markdown):
    rows = {}
    for line in markdown.splitlines():
        m = re.match(r'\|\s*`"?(\w+)"?`\s*\|.*`([0-9a-f]{64})`', line.strip())
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


# ---------------------------------------------------------------- reproduce_submission.ipynb
def test_submission_notebook_default_is_the_package_default_and_the_pipeline_may_set_mode():
    code = [c for c in cells('reproduce_submission.ipynb', 'code') if re.search(r'^MODE = ', c, re.M)]
    assert len(code) == 1
    value = assigns(code[0])['MODE']
    assert ast.unparse(value) == "globals().get('MODE') or DEFAULT_MODE"
    assert 'from filseg.reproduction.modes import DEFAULT_MODE, resolve' in code[0]
    assert code[0].index('MODE = ') < code[0].index('MODE, MODE_ROW = resolve(MODE)')


@pytest.mark.parametrize('name', ['reproduce_submission.ipynb', 'pipeline.ipynb', 'reproduce_mf_views.ipynb', 'reproduce_maps.ipynb'])
def test_no_notebook_names_the_default_mode(name):
    # switching the default is the one line modes.DEFAULT_MODE (tests/test_modes.py)
    code = '\n'.join(cells(name, 'code'))
    assert '"pcombo"' not in code and "'pcombo'" not in code and '"mftta"' not in code and "'mftta'" not in code
    assert '"ms125"' not in code and "'ms125'" not in code


def test_submission_notebook_passes_the_mode_to_the_cli_and_checks_the_table_hash():
    code = '\n'.join(cells('reproduce_submission.ipynb', 'code'))
    assert '"--mode", MODE' in code
    assert 'MODE_ROW["sha256"]' in code and 'MODES[MODE]' not in code
    assert 'FLAGS' not in code and '"--pcand"' not in code, 'one table of modes, in the package'


def test_submission_notebook_reads_the_mf_view_maps_of_an_mf_map_mode():
    code = [c for c in cells('reproduce_submission.ipynb', 'code') if '"--mode", MODE' in c][0]
    assert 'VARIANT_MAPS = None' in code and 'find_variant_maps(VARIANT)' in code
    assert '["--variant-maps", str(VARIANT_MAPS)]' in code
    assert code.index('VARIANT_MAPS = None') < code.index('find_variant_maps(VARIANT)') < code.index('subprocess.run(command')


def test_submission_notebook_finds_the_images_with_the_package_helper():
    code = '\n'.join(cells('reproduce_submission.ipynb', 'code'))
    assert 'find_test_images()' in code and 'rglob("test_images")' not in code


def test_mode_hash_table_in_the_notebook_equals_the_package_table():
    from filseg.reproduction.modes import runnable
    rows = {}
    for md in cells('reproduce_submission.ipynb', 'markdown'):
        rows.update(mode_table_rows(md))
    assert rows == {n: m['sha256'] for n, m in runnable().items()}


def test_pipeline_mode_table_lists_every_runnable_mode_and_marks_the_default_final():
    # pipeline.ipynb shows plain names and MODE only; the hashes are in reproduce_submission.ipynb and docs/reproduction_notes.md
    from filseg.reproduction.modes import DEFAULT_MODE, runnable
    rows = [line for md in cells('pipeline.ipynb', 'markdown') for line in md.splitlines() if re.search(r'\|\s*`"\w+"`\s*\|\s*$', line)]
    names = [re.search(r'`"(\w+)"`\s*\|\s*$', line).group(1) for line in rows]
    assert sorted(names) == sorted(runnable()) and len(names) == len(runnable())
    final = [n for n, line in zip(names, rows) if '(final)' in line]
    assert final == [DEFAULT_MODE]


# ---------------------------------------------------------------- reproduce_maps.ipynb
def test_maps_notebook_uses_the_known_paths_helper():
    code = '\n'.join(cells('reproduce_maps.ipynb', 'code'))
    assert 'find_test_images()' in code
    assert '/kaggle/input/filament-segmentation-2026/test/test_images' not in code and 'rglob("test_images")' not in code
    assert code.index('sys.path.insert(0, str(REPO / "src"))') < code.index('from filseg.reproduction.paths import find_test_images')


PUBLISHED_REPOSITORY = 'https://github.com/Matapanino/filament-segmentation-2026-solution.git'
RELEASE_TAG = 'v1.1'   # the tag of the published repository's release commit (a commit cannot name its own sha)


@pytest.mark.parametrize('name', ['reproduce_maps.ipynb', 'reproduce_submission.ipynb', 'pipeline.ipynb', 'reproduce_mf_views.ipynb'])
def test_clone_fallback_checks_out_the_release_tag_of_the_published_repository(name):
    code = '\n'.join(cells(name, 'code'))
    setup = [c for c in cells(name, 'code') if 'SOURCE_TAG =' in c]
    assert len(setup) == 1 and ast.literal_eval(assigns(setup[0])['SOURCE_TAG']) == RELEASE_TAG
    assert '"--detach", SOURCE_TAG' in code and 'SOURCE_COMMIT' not in code
    assert f'"git", "clone", "--no-checkout", "{PUBLISHED_REPOSITORY}"' in code
    assert 'solar-filament-seg-2026' not in json.dumps(json.loads((REPO / 'notebooks' / name).read_text()))


# ---------------------------------------------------------------- pipeline.ipynb
def test_pipeline_sets_the_mode_before_stage_one_and_runs_the_gpu_stages_the_mode_needs():
    code = cells('pipeline.ipynb', 'code')
    joined = '\n'.join(code)
    mode_cells = [c for c in code if re.search(r'^MODE = ', c, re.M)]
    assert len(mode_cells) == 1 and ast.literal_eval(assigns(mode_cells[0])['MODE']) is None   # None: modes.DEFAULT_MODE
    order = ['MODE = None', 'MODE, GPU_STAGES = gpu_stages(MODE)', '"reproduce_maps.ipynb"', '"reproduce_mf_views.ipynb"', '"reproduce_submission.ipynb"']
    assert [joined.index(x) for x in order] == sorted(joined.index(x) for x in order)
    assert 'if "reproduce_maps.ipynb" in GPU_STAGES' in joined and 'if "reproduce_mf_views.ipynb" in GPU_STAGES' in joined


def test_pipeline_draws_the_figures_after_stage_two_from_two_test_images():
    code = '\n'.join(cells('pipeline.ipynb', 'code'))
    keys = set((REPO / 'src/filseg/data/test_filenames.txt').read_text().split())
    figure = [c for c in cells('pipeline.ipynb', 'code') if 'FIGURE_IMAGES =' in c]
    assert len(figure) == 1
    chosen = ast.literal_eval(assigns(figure[0])['FIGURE_IMAGES'])
    assert 1 <= len(chosen) <= 2 and set(chosen) <= keys
    assert 'illustrate(' in figure[0] and 'variant_maps=VARIANT_MAPS' in figure[0]
    assert code.index('"reproduce_submission.ipynb"') < code.index('FIGURE_IMAGES =')


def test_pipeline_explains_each_stage_in_markdown_with_the_plain_names_of_the_report():
    text = '\n'.join(cells('pipeline.ipynb', 'markdown')).lower()
    for stage in ('probability maps', 'mask2former', 'averaged map', 'candidates', 'instance filter', 'mask repair', 'run-length'):
        assert stage in text, stage
    for internal in ('ship_r3', 'p-cand', 'p-combo', 'lr39', 'mf-tta', 'selector'):
        assert internal not in text, internal


def test_pipeline_ends_with_one_takeaway_per_figure_panel():
    from filseg.reproduction.figures import PANELS
    nb = json.loads((REPO / 'notebooks/pipeline.ipynb').read_text())
    last = nb['cells'][-1]
    assert last['cell_type'] == 'markdown' and 'FIGURE_IMAGES =' in ''.join(nb['cells'][-2]['source'])
    items = re.findall(r'^(\d)\. \*\*', ''.join(last['source']), re.M)
    assert items == [str(i) for i in range(1, len(PANELS) + 1)]


def test_pipeline_notebook_stays_small():
    assert (REPO / 'notebooks/pipeline.ipynb').stat().st_size < 5_000_000


def test_pipeline_still_does_not_duplicate_stage_logic():
    code = '\n'.join(cells('pipeline.ipynb', 'code'))
    for stage_call in ('run_csv', 'run_maps', 'run_gpu_pipeline', '-m", "filseg.reproduction', 'image_stages'):
        assert stage_call not in code


# ---------------------------------------------------------------- %run of a notebook shares the caller's namespace
def test_run_of_a_notebook_sees_the_mode_set_by_the_caller(tmp_path):
    shell_module = pytest.importorskip('IPython.core.interactiveshell')
    from filseg.reproduction.modes import DEFAULT_MODE
    cell = [c for c in cells('reproduce_submission.ipynb', 'code') if re.search(r'^MODE = ', c, re.M)][0]
    lines = [l for l in cell.splitlines() if l.startswith('from filseg.reproduction.modes import') or l.startswith('MODE = ') or l.startswith('MODE, MODE_ROW = ')]
    assert len(lines) == 3
    nb = dict(cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=[l + '\n' for l in lines] + ['seen = MODE\n'])],
              metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')), nbformat=4, nbformat_minor=5)
    path = tmp_path / 'stage.ipynb'
    path.write_text(json.dumps(nb))
    shell = shell_module.InteractiveShell.instance()
    try:
        shell.user_ns.pop('MODE', None)
        shell.run_line_magic('run', str(path))
        assert shell.user_ns['seen'] == DEFAULT_MODE
        for mode in ('mfrich', 'mftta'):
            shell.user_ns['MODE'] = mode
            shell.run_line_magic('run', str(path))
            assert shell.user_ns['seen'] == mode
        shell.user_ns['MODE'] = None   # pipeline.ipynb's MODE = None
        shell.run_line_magic('run', str(path))
        assert shell.user_ns['seen'] == DEFAULT_MODE
    finally:
        shell_module.InteractiveShell.clear_instance()
