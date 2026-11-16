"""requirements.txt lists every package the reproduction imports, with the versions the verified runs used."""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# import name -> distribution name
DIST = {'numpy': 'numpy', 'scipy': 'scipy', 'pandas': 'pandas', 'cv2': 'opencv-python-headless', 'pycocotools': 'pycocotools',
        'skimage': 'scikit-image', 'sklearn': 'scikit-learn', 'kagglehub': 'kagglehub', 'torch': 'torch',
        'safetensors': 'safetensors', 'segmentation_models_pytorch': 'segmentation-models-pytorch', 'transformers': 'transformers',
        'matplotlib': 'matplotlib', 'lightgbm': 'lightgbm', 'xgboost': 'xgboost', 'catboost': 'catboost'}
# Imported by the notebooks' own code cells; provided by the Jupyter runtime that executes them (recorded, not pinned).
RUNTIME = {'IPython'}


def pins(text, commented=False):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if commented:
            if not line.startswith('#'):
                continue
            line = line.lstrip('#').strip()
        elif line.startswith('#'):
            continue
        out += [w for w in line.split() if re.fullmatch(r'[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+', w)]
    return out


def read(name):
    return (REPO / name).read_text()


def test_requirements_txt_installs_exactly_the_cpu_environment():
    assert pins(read('requirements.txt')) == pins(read('requirements-reproduce-cpu.txt'))
    assert not any(l.strip().startswith('-r') for l in read('requirements.txt').splitlines()), 'every package is listed in the file itself'


def test_gpu_requirement_files_match_what_the_launcher_installs():
    from filseg.reproduction.launch import MF, UNET
    launcher = read('src/filseg/reproduction/launch.py')
    cudnn = re.findall(r"nvidia-cudnn-cu13==[0-9.]+", launcher)
    assert cudnn == ['nvidia-cudnn-cu13==9.27.0.42']
    assert pins(read('requirements-reproduce-unet.txt')) == UNET + cudnn
    assert pins(read('requirements-reproduce-gpu.txt')) == MF


def test_requirements_txt_documents_both_gpu_environments():
    listed = pins(read('requirements.txt'), commented=True)
    for name in ('requirements-reproduce-unet.txt', 'requirements-reproduce-gpu.txt'):
        for pin in pins(read(name)):
            assert pin in listed, pin


def test_every_imported_package_is_pinned():
    pinned = {p.split('==')[0].lower() for f in ('requirements.txt', 'requirements-reproduce-unet.txt', 'requirements-reproduce-gpu.txt') for p in pins(read(f))}
    imported = set()
    for path in (REPO / 'src/filseg').rglob('*.py'):
        if 'reproduction' not in path.parts and path.name not in ('instancing.py', 'postproc.py', '__init__.py'):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split('.')[0])
    third_party = {m for m in imported if m in DIST}
    unknown = {m for m in imported if m not in DIST and m not in __import__('sys').stdlib_module_names and m != 'filseg'}
    assert not unknown, unknown
    for module in third_party:
        assert DIST[module].lower() in pinned, module


def notebook_imports():
    import json
    imported = set()
    for path in (REPO / 'notebooks').glob('*.ipynb'):
        for cell in json.loads(path.read_text())['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = '\n'.join(l for l in ''.join(cell['source']).splitlines() if not l.lstrip().startswith(('%', '!')))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split('.')[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split('.')[0])
    return imported


def test_every_package_the_notebooks_import_is_pinned_or_the_jupyter_runtime():
    pinned = {p.split('==')[0].lower() for p in pins(read('requirements.txt'))}
    imported = notebook_imports()
    unknown = {m for m in imported if m not in DIST and m not in RUNTIME and m not in __import__('sys').stdlib_module_names and m != 'filseg'}
    assert not unknown, unknown
    for module in imported & set(DIST):
        assert DIST[module].lower() in pinned, module


def test_matplotlib_is_pinned_to_the_kaggle_image_version():
    assert 'matplotlib==3.10.0' in pins(read('requirements-reproduce-cpu.txt'))
