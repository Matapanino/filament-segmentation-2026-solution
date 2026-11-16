"""Where the inputs are on Kaggle: the competition's test images (known layouts in order, then one unambiguous match, else a
loud failure), and the GPU-stage outputs a mode needs (this session's maps first, then attached outputs)."""
import json
from pathlib import Path

INPUT_ROOT = Path('/kaggle/input')
WORKING_MAPS = Path('/kaggle/working/maps')   # where reproduce_maps.ipynb and reproduce_mf_views.ipynb write in this session
MAP_ROOTS = (WORKING_MAPS, INPUT_ROOT)
KNOWN_TEST_IMAGE_DIRS = (
    'competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images',  # the mount seen on Kaggle, 2026-11
    'filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images',               # the same data, older mount root
    'competitions/filament-segmentation-2026/test/test_images',
    'filament-segmentation-2026/test/test_images',                                        # the path the notebook once hard-coded
)


def find_test_images(root=INPUT_ROOT):
    """The test JPEG directory under root. The image set itself is checked later against the pinned 180 names and hashes."""
    root = Path(root)
    for rel in KNOWN_TEST_IMAGE_DIRS:
        if (root / rel).is_dir():
            return root / rel
    found = sorted(p for p in root.rglob('test_images') if p.is_dir()) if root.is_dir() else []
    if len(found) == 1:
        return found[0]
    top = sorted(str(p) for p in root.glob('*/*')) if root.is_dir() else []
    raise FileNotFoundError(f'No unique test image directory under {root}. Tried, in order: {list(KNOWN_TEST_IMAGE_DIRS)}. '
                            f'test_images directories found: {[str(p) for p in found]}. Top of {root}: {top[:50]}. '
                            'Attach the competition data (filament-segmentation-2026), or set IMAGE_ROOT to the test JPEG directory.')


def _found(root, name):
    root = Path(root)
    return sorted(root.rglob(name)) if root.is_dir() else []


def find_variant_maps(variant, roots=MAP_ROOTS):
    """The directory of a completed MF-view run (reproduce_mf_views.ipynb) whose recorded map set of `variant` is the
    expected one, or None. Roots are tried in order; two such directories under one root are refused. The CSV stage
    still checks every archive against the receipts, and the CSV against its pinned sha256."""
    from .mfmaps import EXPECTED_MAP_SETS, RESULT
    expected = EXPECTED_MAP_SETS[variant]
    for root in roots:
        found = []
        for path in _found(root, RESULT):
            result = json.loads(path.read_text())
            if result.get('status') == 'complete' and ((result.get('map_sets') or {}).get(variant) or {}).get('sha256') == expected:
                found.append(path.parent)
        if len(found) > 1:raise ValueError(f'More than one MF-view output of {variant} under {root}: {[str(p) for p in found]}')
        if found:return found[0]
    return None


RELEASE_MANIFEST = 'release_manifest.json'
WORKING_CHAIN_MAPS = Path('/kaggle/working/chain_maps')   # maps re-inferred from the release models in this session


def find_release(roots=(WORKING_CHAIN_MAPS, INPUT_ROOT)):
    """The folder of the release files (the stored maps and models of the chain modes; chain.release_files() lists them),
    found by its release_manifest.json: the first root that has one, or None. Two under one root are refused."""
    for root in roots:
        found = [p.parent for p in _found(root, RELEASE_MANIFEST)]
        if len(found) > 1:raise ValueError(f'More than one release folder under {root}: {[str(p) for p in found]}')
        if found:return found[0]
    return None


def has_gpu():
    """A CUDA GPU is visible (nvidia-smi lists one)."""
    import shutil, subprocess
    if not shutil.which('nvidia-smi'):return False
    try:return bool(subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=60).stdout.strip())
    except (OSError, subprocess.SubprocessError):return False


def gpu_stages(mode=None, roots=MAP_ROOTS, gpu=None):
    """(mode, the GPU stage notebooks still to run for it, in order): reproduce_maps.ipynb unless package maps
    (inference_result.json) are present, and for an MF-map mode reproduce_mf_views.ipynb unless its map set is present.
    mode None means modes.DEFAULT_MODE."""
    from .modes import resolve
    name, row = resolve(mode)
    if 'recipe' in row['kwargs']:
        # A chain mode builds its CSV from stored maps (CPU). With a GPU, reproduce_chain_maps.ipynb first re-reads two test
        # images with the models stored in the repository and compares those maps with the stored ones.
        return name, ['reproduce_chain_maps.ipynb'] if (has_gpu() if gpu is None else gpu) else []
    stages = [] if any(_found(root, 'inference_result.json') for root in roots) else ['reproduce_maps.ipynb']
    variant = row['kwargs'].get('mf_variant')
    if variant and find_variant_maps(variant, roots) is None:stages.append('reproduce_mf_views.ipynb')
    return name, stages
