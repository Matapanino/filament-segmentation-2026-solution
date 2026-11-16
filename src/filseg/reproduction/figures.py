"""Figures for pipeline.ipynb: one test image through every stage, drawn with matplotlib (Agg, no pyplot state).

illustrate() reruns the submission stage on one image with the same mode, checks that every stage hash equals the one the
CSV run recorded in csv_result.json (so the figure shows exactly what the CSV contains), and renders eight panels:
the image, the four probability maps, the candidates kept or dropped by the selector, the repaired masks, and the
pixels the repair added or removed. It never writes or changes the CSV. For an MF-map mode (mftta) the Mask2Former
panel and the blend are the variant's map, read from the MF-view archives the CSV run read.
"""
import io
import json
import textwrap
from pathlib import Path

import numpy as np

OVERLAY_FACTOR = 8   # overlays at 256 x 256 (block max), about the size a panel is drawn, so thin filaments stay visible
PANELS = ('Test image', 'Full-disk U-Net', 'Local U-Net', 'Mask2Former (mean of 3 fold models)', 'Averaged map',
          'Instance filter: kept (yellow) / dropped (grey)', 'Submitted masks',
          'Mask repair: added (sky blue) / removed (reddish purple)')
MF_PANEL = {None: PANELS[3], 'mftta': 'Mask2Former (8 views, mean of 3 fold models)'}   # the Mask2Former panel per MF map (mf_variant)


TITLE_WIDTH = 44   # characters per title line: longer titles are cut at the panel edge at fontsize 11


def panel_title(name):
    return textwrap.fill(name, TITLE_WIDTH)


def _rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))


# Overlay colours of report Fig. 1 (colour-blind safe), except that dropped candidates are grey, not faded.
KEPT, DROPPED, ADDED, REMOVED = _rgb('#F0E442'), _rgb('#9A9A9A'), _rgb('#56B4E9'), _rgb('#CC79A7')


def downsample(image, factor=4):
    """Block mean (2048 -> 512 for factor 4), float32."""
    a = np.asarray(image, np.float32)
    h, w = a.shape[0] // factor, a.shape[1] // factor
    return a[:h * factor, :w * factor].reshape(h, factor, w, factor).mean(axis=(1, 3))


def mask_preview(mask, factor=4):
    """Block max, so that one-pixel-wide filaments stay visible."""
    m = np.asarray(mask, bool)
    h, w = m.shape[0] // factor, m.shape[1] // factor
    return m[:h * factor, :w * factor].reshape(h, factor, w, factor).any(axis=(1, 3))


def _paint(base, mask, colour, alpha=.65):
    m = mask_preview(mask, OVERLAY_FACTOR)
    base[m] = (1 - alpha) * base[m] + alpha * np.asarray(colour, np.float32)
    return base


def render(raw, full, local, mf, blend, candidates, keep, finals, title='', dpi=72, mf_title=None):
    """PNG bytes of the eight panels. full/local/mf/blend are probability maps in [0, 1]; candidates are the
    unrepaired masks, keep one bool per candidate, finals the submitted (repaired) masks of the kept candidates.
    mf_title names the Mask2Former panel (default PANELS[3], the package MF mean)."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    keep = np.asarray(keep, bool)
    if len(keep) != len(candidates):raise ValueError('One keep flag per candidate')
    kept = [m for m, k in zip(candidates, keep) if k]
    if len(finals) != len(kept):raise ValueError('One submitted mask per kept candidate')
    gray = downsample(raw) / 255.
    dim = np.repeat((.55 * downsample(raw, OVERLAY_FACTOR) / 255.)[..., None], 3, axis=2)
    selection = dim.copy()
    for m, k in zip(candidates, keep):
        _paint(selection, m, KEPT if k else DROPPED)
    submitted = dim.copy()
    palette = colormaps['tab10']
    for i, m in enumerate(finals):
        _paint(submitted, m, palette(i % 10)[:3])
    change = dim.copy()
    for before, after in zip(kept, finals):
        _paint(change, np.asarray(after, bool) & ~np.asarray(before, bool), ADDED, .9)
        _paint(change, np.asarray(before, bool) & ~np.asarray(after, bool), REMOVED, .9)
    images = [(gray, 'gray'), (downsample(full), 'magma'), (downsample(local), 'magma'), (downsample(mf), 'magma'), (downsample(blend), 'magma'),
              (selection, None), (submitted, None), (change, None)]
    fig = Figure(figsize=(16, 8.8), dpi=dpi, layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4)
    names = PANELS[:3] + (mf_title or PANELS[3],) + PANELS[4:]
    for ax, (image, cmap), name in zip(axes.ravel(), images, names):
        if cmap is None:ax.imshow(np.clip(image, 0, 1), interpolation='nearest')
        else:ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation='antialiased')
        ax.set_title(panel_title(name), fontsize=11)
        ax.set_axis_off()
    if title:fig.suptitle(title, fontsize=13)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, metadata={'Software': None})
    return buffer.getvalue()


def figure_title(key, candidates, kept):
    """The figure title: the test image and its candidate counts, without internal mode or selector names."""
    return f'{key}: {candidates} candidates, {kept} kept'


def check_stages(recomputed, recorded):
    """The rerun must give the stage hashes the CSV run recorded for this image, all of them."""
    if set(recomputed) != set(recorded):raise ValueError('Stage sets differ: '+repr(sorted(set(recomputed) ^ set(recorded))))
    differ = sorted(k for k in recorded if recomputed[k] != recorded[k])
    if differ:raise ValueError('The figure run differs from the CSV run at: '+', '.join(differ))


def illustrate(key, images, maps_root, output, mode, weights=None, variant_maps=None):
    """Rerun one test image in `mode`, check it against the CSV run in `output`, write and return the figure.
    variant_maps: for an MF-map mode, the directory of the MF-view archives (default maps_root, as run_csv)."""
    from .modes import resolve
    mode, row = resolve(mode)
    kwargs = row['kwargs'];variant = kwargs.get('mf_variant');output = Path(output);maps_root = Path(maps_root)
    run = json.loads((output / 'csv_result.json').read_text())
    if 'recipe' in kwargs:return _illustrate_chain(key, images, maps_root, output, mode, kwargs['recipe'], run, weights)
    if (run.get('mf_variant') or {}).get('name') != variant:
        raise ValueError(f'The CSV run in {output} read another MF map than mode {mode} ({variant or "the package MF mean"})')
    recorded = run['per_image']
    if key not in recorded:raise ValueError('Not a test image of this run: '+repr(key))
    from pycocotools import mask as mu
    from . import pcand as P
    from . import refit
    from .arithmetic import blend_with_mf, candidate_blend, mf_mean
    from .assets import resolve_assets
    from .audit import verify
    from .instancing import SHIP_R3, iter_instances
    from .pipeline import DATA, image_stages, read_image
    from .quality import load_quality
    import cv2
    cv2.setNumThreads(1)  # as run_csv
    root, _ = resolve_assets(weights)
    model = load_quality(root / 'quality_model.json')
    selector = refit.load_selector(*refit.deployed(kwargs['selector'])) if kwargs.get('selector') else None
    predictor = P.load_regressor(*P.deployed(root)) if kwargs['pcand'] else None
    paths = sorted(Path(images).rglob(key + '.jpeg'))
    if len(paths) != 1:raise FileNotFoundError('Expected one image file for '+key+', found '+repr([str(p) for p in paths]))
    verify(paths[0], json.loads((DATA / 'test_image_manifest.json').read_text())[key])
    raw = read_image(paths[0])
    maps = {}
    for name in ('full', 'local', 'MF0', 'MF1', 'MF2'):
        with np.load(maps_root / (name + '.npz'), allow_pickle=False) as z:maps[name] = z[key]
    if variant:
        from .mfmaps import load_variant_archives
        from .pipeline import test_keys
        for name, z in load_variant_archives(variant_maps or maps_root, variant, test_keys()).items():
            with z:maps[name] = z[key]
    captured = []

    def instances(probability):
        masks = [np.asarray(m, bool) for m in iter_instances(probability, SHIP_R3)]
        captured.append([m.copy() for m in masks])
        return masks

    value = image_stages(raw, maps, model, instance_stage=instances, pnat=kwargs['pnat'], pcand=predictor, selector=selector, mf_variant=variant)
    check_stages(value['stages'], recorded[key]['stages'])
    keep = np.asarray(value['keep'], bool)
    if int(keep.sum()) != recorded[key]['kept']:raise ValueError('Kept count differs from the CSV run')
    candidates = captured[0]
    finals = [mu.decode(dict(size=[raw.shape[0], raw.shape[1]], counts=value['rles'][j].encode())).astype(bool) for j in np.flatnonzero(keep)]
    if variant:
        from .mfmaps import variant_mf
        mf = variant_mf(maps, variant)
        blend = blend_with_mf(maps['full'], maps['local'], mf)
    else:
        mf = mf_mean(maps['MF0'], maps['MF1'], maps['MF2'])
        blend = candidate_blend(*(maps[n] for n in ('full', 'local', 'MF0', 'MF1', 'MF2')))
    title = figure_title(key, len(candidates), int(keep.sum()))
    png = render(raw, maps['full'].astype(np.float32) / 255, maps['local'].astype(np.float32) / 255, mf, blend, candidates, keep, finals, title,
                 mf_title=MF_PANEL[variant])
    (output / ('figure_' + key + '.png')).write_bytes(png)
    kept = [m for m, k in zip(candidates, keep) if k]
    summary = dict(image=key, mode=mode, candidates=len(candidates), kept=int(keep.sum()), dropped=int((~keep).sum()),
                   stage_hashes_equal_the_csv_run=True, png_bytes=len(png),
                   repair_pixels_added=int(sum((f & ~c).sum() for c, f in zip(kept, finals))),
                   repair_pixels_removed=int(sum((c & ~f).sum() for c, f in zip(kept, finals))))
    if selector is not None:summary['lr39_would_keep'] = int(sum(value['selector']['lr39_keep']))
    return png, summary


def _illustrate_chain(key, images, release, output, mode, recipe, run, weights=None):
    """illustrate() for a chain mode: the recipe's stored maps under release (chain.py), the same checks and panels."""
    if run.get('recipe') != recipe:raise ValueError(f'The CSV run in {output} built recipe {run.get("recipe")!r}, not {recipe!r}')
    recorded = run['per_image']
    if key not in recorded:raise ValueError('Not a test image of this run: '+repr(key))
    from pycocotools import mask as mu
    from . import chain
    from .audit import verify
    from .instancing import SHIP_R3, iter_instances
    from .pipeline import DATA, image_stages, read_image, test_keys
    r, model, selector, reg, P = chain._setup(recipe, weights)
    paths = sorted(Path(images).rglob(key + '.jpeg'))
    if len(paths) != 1:raise FileNotFoundError('Expected one image file for '+key+', found '+repr([str(p) for p in paths]))
    verify(paths[0], json.loads((DATA / 'test_image_manifest.json').read_text())[key])
    raw = read_image(paths[0])
    maps = {n: z[key] for n, z in chain._open(chain.stored_files(r, [release]), test_keys()).items()}
    names = [a['name'] for a in r['archives']]; captured = []

    def instances(probability):
        masks = [np.asarray(m, bool) for m in iter_instances(probability, SHIP_R3)]
        captured.append([m.copy() for m in masks])
        return masks

    value = image_stages(raw, maps, model, instance_stage=instances, pcand=P.load_regressor(*reg), selector=selector, chain=dict(names=names, weight=r['weight']))
    check_stages(value['stages'], recorded[key]['stages'])
    keep = np.asarray(value['keep'], bool)
    if int(keep.sum()) != recorded[key]['kept']:raise ValueError('Kept count differs from the CSV run')
    candidates = captured[0]
    finals = [mu.decode(dict(size=[raw.shape[0], raw.shape[1]], counts=value['rles'][j].encode())).astype(bool) for j in np.flatnonzero(keep)]
    mf = chain.chain_mf(maps, names); blend = chain.weighted_blend(maps['full'], maps['local'], mf, r['weight'])
    title = figure_title(key, len(candidates), int(keep.sum()))
    png = render(raw, maps['full'].astype(np.float32) / 255, maps['local'].astype(np.float32) / 255, mf, blend, candidates, keep, finals, title,
                 mf_title=f'Mask2Former (mean of {len(names)} maps)')
    (output / ('figure_' + key + '.png')).write_bytes(png)
    kept = [m for m, k in zip(candidates, keep) if k]
    summary = dict(image=key, mode=mode, candidates=len(candidates), kept=int(keep.sum()), dropped=int((~keep).sum()),
                   stage_hashes_equal_the_csv_run=True, png_bytes=len(png),
                   repair_pixels_added=int(sum((f & ~c).sum() for c, f in zip(kept, finals))),
                   repair_pixels_removed=int(sum((c & ~f).sum() for c, f in zip(kept, finals))))
    summary['lr39_would_keep'] = int(sum(value['selector']['lr39_keep']))
    return png, summary
