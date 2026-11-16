"""Model weights stored in the repository's weights/ folder, in plain git (no LFS).

GitHub refuses files over 100 MB and warns above 50 MiB, so any file larger than PART_BYTES is stored as numbered parts.
weights/repo_weights.json records the size and sha256 of every stored file, of every part and of every reassembled file.
prepare() reassembles the parts, checks every hash and raises on any mismatch; nothing is loaded from unchecked bytes.
"""
import hashlib, json, os, shutil
from pathlib import Path

MANIFEST = 'repo_weights.json'
PART_BYTES = 48_000_000          # under GitHub's 50 MiB warning, so a push never warns
LIMIT_BYTES = 95_000_000         # no stored file may reach this size
CACHE = Path.home() / '.cache/filseg/weights'


def _identity(path):
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            sha.update(b); md5.update(b); n += len(b)
    return dict(bytes=n, sha256=sha.hexdigest(), md5=md5.hexdigest())


def _check(path, expected):
    if not Path(path).is_file():
        raise FileNotFoundError('Missing weights file: ' + str(path))
    got = _identity(path)
    for k in ('bytes', 'sha256', 'md5'):
        if k in expected and got[k] != expected[k]:
            raise ValueError('Artifact integrity failure: ' + str(path) + ' ' + k)
    return got


def split(source_dir, names, weights_dir, part_bytes=PART_BYTES, provenance=None):
    """Store each named file of source_dir under weights_dir: whole if it fits in part_bytes, otherwise as parts/<name>.NNN."""
    if part_bytes >= LIMIT_BYTES:
        raise ValueError('Parts must stay under ' + str(LIMIT_BYTES) + ' bytes')
    source_dir, weights_dir = Path(source_dir), Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in sorted(names):
        src = source_dir / name
        row = dict(_identity(src), parts=[])
        if row['bytes'] <= part_bytes:
            shutil.copyfile(src, weights_dir / name)
        else:
            (weights_dir / 'parts').mkdir(exist_ok=True)
            with src.open('rb') as f:
                for i, b in enumerate(iter(lambda: f.read(part_bytes), b'')):
                    rel = 'parts/%s.%03d' % (name, i)
                    (weights_dir / rel).write_bytes(b)
                    row['parts'].append(dict(file=rel, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()))
        files[name] = row
    manifest = dict(schema='filseg-repo-weights-v1', part_bytes=part_bytes, files=files, **(dict(provenance=provenance) if provenance else {}))
    (weights_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def _assemble(weights_dir, name, row, target):
    for p in row['parts']:
        _check(weights_dir / p['file'], p)
    tmp = target.with_name(f'{target.name}.{os.getpid()}.tmp')   # per process: parallel first runs must not rename each other's file
    sha, n = hashlib.sha256(), 0
    try:
        with tmp.open('wb') as out:
            for p in row['parts']:
                with (weights_dir / p['file']).open('rb') as f:
                    for b in iter(lambda: f.read(1 << 22), b''):
                        out.write(b); sha.update(b); n += len(b)
        if n != row['bytes'] or sha.hexdigest() != row['sha256']:
            raise ValueError('Artifact integrity failure: reassembled ' + name)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def prepare(weights_dir, destination=None):
    """Return a folder holding every file of the manifest, reassembled and hash-checked.

    The default destination is weights_dir itself (the reassembled files are git-ignored); a read-only checkout uses
    ~/.cache/filseg/weights. An existing reassembled file is reused only if its hash is right; otherwise it is rebuilt.
    """
    weights_dir = Path(weights_dir)
    manifest = json.loads((weights_dir / MANIFEST).read_text())
    if destination is None:
        destination = weights_dir if os.access(weights_dir, os.W_OK) else CACHE
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, row in sorted(manifest['files'].items()):
        target = destination / name
        if not row['parts']:
            _check(weights_dir / name, row)
            if destination != weights_dir:
                shutil.copyfile(weights_dir / name, target)
                _check(target, row)
            continue
        if target.is_file() and target.stat().st_size == row['bytes'] and _identity(target)['sha256'] == row['sha256']:
            continue
        _assemble(weights_dir, name, row, target)
    return destination
