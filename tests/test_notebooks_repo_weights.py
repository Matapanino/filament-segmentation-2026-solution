"""The notebooks load the weights from the repository checkout and find their inputs under /kaggle/input."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def notebook_code(name):
    nb = json.loads((REPO / 'notebooks' / name).read_text())
    return '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')


def test_maps_notebook_finds_the_test_images_instead_of_a_fixed_mount():
    code = notebook_code('reproduce_maps.ipynb')
    assert '/kaggle/input/filament-segmentation-2026/test/test_images' not in code
    assert 'find_test_images()' in code  # known mount layouts in order, then a loud failure (filseg.reproduction.paths)


@pytest.mark.parametrize('name', ['reproduce_maps.ipynb', 'reproduce_submission.ipynb'])
def test_notebooks_do_not_require_the_weights_dataset(name):
    code = notebook_code(name)
    assert '--weights' not in code
    assert 'filament-repro-weights-1111' not in code


def test_submission_notebook_uses_maps_from_this_session_first():
    code = notebook_code('reproduce_submission.ipynb')
    assert code.index('/kaggle/working/maps') < code.index('rglob("inference_result.json")')


def test_pipeline_notebook_runs_the_two_stage_notebooks_in_order():
    code = notebook_code('pipeline.ipynb')
    assert code.index('reproduce_maps.ipynb') < code.index('reproduce_submission.ipynb')
    for stage_call in ('run_csv', 'run_maps', 'run_gpu_pipeline', '-m", "filseg.reproduction'):
        assert stage_call not in code, 'the pipeline notebook must not duplicate stage logic'
