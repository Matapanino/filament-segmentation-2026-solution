"""Hash-checked model weights: the repository's weights/ folder by default, a Kaggle dataset only as a fallback."""
import json
from pathlib import Path
from .audit import verify
DATASET='masayakawamata/filament-repro-weights-1111'
REPO=Path(__file__).resolve().parents[3]


def locate(root=None,repo=None,kaggle_input=Path('/kaggle/input')):
    """Weights root, in this order: an explicit root, the repository's weights/ folder, a mounted Kaggle dataset, a kagglehub download."""
    from .repo_weights import MANIFEST,prepare
    if root is not None:
        root=Path(root);return prepare(root) if (root/MANIFEST).is_file() else root
    weights=(REPO if repo is None else Path(repo))/'weights'
    if (weights/MANIFEST).is_file():return prepare(weights)
    matches=list(Path(kaggle_input).rglob('reproduction_weights.json')) if Path(kaggle_input).is_dir() else []
    if len(matches)==1:return matches[0].parent
    if len(matches)>1:raise ValueError('Specify the exact weights root')
    import kagglehub
    return Path(kagglehub.dataset_download(DATASET,force_download=False))


def resolve_assets(root=None):
    root=locate(root);manifest=json.loads((root/'reproduction_weights.json').read_text())
    # The repository pins the manifest itself after remote export.
    pin=Path(__file__).parent/'data/weights_manifest_identity.json'
    if not pin.exists():raise RuntimeError('Release manifest pin is not yet installed')
    verify(root/'reproduction_weights.json',json.loads(pin.read_text()))
    for name,row in manifest['files'].items():verify(root/name,row)
    for row in manifest['networks'].values():verify(root/row['file'],row)
    return root,manifest
