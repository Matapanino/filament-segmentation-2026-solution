"""What a reader of the published repository sees: documents with working links, no private identifiers, and package
data without e-mail addresses or private storage paths."""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO / 'notebooks').glob('*.ipynb'))
DOCS = [REPO / 'README.md', REPO / 'docs/reproduction_notes.md', REPO / 'campaigns/INDEX.md', REPO / 'kaggle/README.md']
PRIVATE = {
    'private Kaggle slug': re.compile(r'masayakawamata/[a-z0-9][a-z0-9-]*'),
    'e-mail address': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}'),
    'local path': re.compile(r'/Users/'),
    'Drive remote': re.compile(r'gdrive:|kaggle-harness-lake'),
    'old repository': re.compile(r'solar-filament-seg-2026|repro-package-20261111'),
    'Japanese text': re.compile(r'[぀-ヿ一-鿿]'),
}


def markdown_of(notebook):
    nb = json.loads(notebook.read_text())
    return '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown')


def documents():
    return [(p.name, p.read_text()) for p in DOCS] + [(p.name + ' (markdown)', markdown_of(p)) for p in NOTEBOOKS]


@pytest.mark.parametrize('kind', sorted(PRIVATE))
def test_documents_hold_no_private_identifier(kind):
    hits = [(name, m.group(0)) for name, text in documents() for m in PRIVATE[kind].finditer(text)]
    assert hits == []


def test_documents_have_no_export_or_slot_markers():
    for name, text in documents():
        assert '<!--slot:' not in text and 'EXPORT:' not in text and '**TBD**' not in text, name


@pytest.mark.parametrize('doc', ['README.md', 'docs/reproduction_notes.md'])
def test_every_relative_link_and_image_of_a_document_exists(doc):
    path = REPO / doc
    targets = re.findall(r'\]\(([^)\s]+)\)', path.read_text())
    local = [t.split('#')[0] for t in targets if not re.match(r'[a-z]+:', t) and not t.startswith('#')]
    assert local or doc != 'README.md', 'expected relative links in the README'
    missing = [t for t in local if t and not (path.parent / t).exists()]
    assert missing == []


def test_readme_shows_the_report_and_the_two_figures():
    for name in ('docs/technical_report.pdf', 'docs/architecture.png', 'docs/example_segmentations.png'):
        assert (REPO / name).is_file() and (REPO / name).stat().st_size > 10_000, name


def test_reference_manifest_holds_no_e_mail_drive_path_or_private_slug():
    def strings(value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield k
                yield from strings(v)
        elif isinstance(value, list):
            for v in value:
                yield from strings(v)
        elif isinstance(value, str):
            yield value
    manifest = json.loads((REPO / 'src/filseg/reproduction/data/reference_manifest.json').read_text())
    for text in strings(manifest):
        for kind in ('private Kaggle slug', 'e-mail address', 'local path', 'Drive remote', 'old repository'):
            assert not PRIVATE[kind].search(text), (kind, text)
    text = json.dumps(manifest)
    for key in ('owner', 'remote', 'source_remote', 'file_id'):
        assert '"' + key + '":' not in text, key


def test_reference_manifest_keeps_what_the_finalize_stage_reads():
    manifest = json.loads((REPO / 'src/filseg/reproduction/data/reference_manifest.json').read_text())
    for name in ('full', 'local', 'MF0', 'MF1', 'MF2'):
        row = manifest['maps'][name]
        assert {'filename', 'bytes', 'sha256'} <= set(row), name
    for name in ('test_features.csv', 'test_scores.csv'):
        assert {'bytes', 'sha256'} <= set(manifest['historical_aux_outputs'][name]), name
