"""Pipeline figures for the notebook: synthetic inputs only, no model. Matplotlib only, small PNGs."""
import struct

import numpy as np
import pytest

pytest.importorskip('matplotlib')


def png_size(data):
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', data[16:24])


def synthetic(n=2048):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, (n, n), dtype=np.uint8)
    yy, xx = np.mgrid[:n, :n]
    blob = np.exp(-(((yy - n / 3) / 60.) ** 2 + ((xx - n / 2) / 300.) ** 2)).astype(np.float32)
    full = local = blob
    mf = np.roll(blob, 40, axis=0)
    blend = .25 * mf + .375 * full + .375 * local
    candidates = [blend > .5, np.zeros((n, n), bool), np.zeros((n, n), bool)]
    candidates[1][1500, 200:900] = True          # a one-pixel line must stay visible after downsampling
    candidates[2][1200:1300, 1200:1300] = True
    keep = np.array([True, False, True])
    finals = [candidates[0] | np.roll(candidates[0], 3, axis=1), candidates[2]]
    return raw, full, local, mf, blend, candidates, keep, finals


def test_mask_preview_keeps_one_pixel_lines():
    from filseg.reproduction.figures import mask_preview
    m = np.zeros((2048, 2048), bool)
    m[1001, 10:20] = True
    p = mask_preview(m)
    assert p.shape == (512, 512) and p.dtype == bool and p[250, 3:5].all()


def test_downsample_is_a_block_mean():
    from filseg.reproduction.figures import downsample
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(downsample(a, 2), [[2.5, 4.5], [10.5, 12.5]])
    assert downsample(np.zeros((2048, 2048), np.uint8)).shape == (512, 512)


def test_render_gives_a_small_png_with_eight_panels():
    from filseg.reproduction.figures import PANELS, render
    data = render(*synthetic(), title='synthetic')
    width, height = png_size(data)
    assert len(PANELS) == 8 and width > height > 300
    assert len(data) < 1_500_000


def test_render_is_byte_deterministic():
    from filseg.reproduction.figures import render
    args = synthetic()
    assert render(*args, title='a') == render(*args, title='a')


def test_render_rejects_a_keep_vector_of_the_wrong_length():
    from filseg.reproduction.figures import render
    raw, full, local, mf, blend, candidates, keep, finals = synthetic()
    with pytest.raises(ValueError):
        render(raw, full, local, mf, blend, candidates, keep[:2], finals, title='x')
    with pytest.raises(ValueError):
        render(raw, full, local, mf, blend, candidates, keep, finals[:1], title='x')


def test_stage_check_requires_the_csv_run_stage_hashes():
    from filseg.reproduction.figures import check_stages
    recorded = dict(RLE='a', keep='b', SHIP_R3='c')
    check_stages(dict(recorded), recorded)
    with pytest.raises(ValueError):
        check_stages(dict(recorded, RLE='z'), recorded)
    with pytest.raises(ValueError):
        check_stages(dict(RLE='a', keep='b'), recorded)


def test_illustrate_rejects_an_unknown_mode(tmp_path):
    from filseg.reproduction.figures import illustrate
    with pytest.raises(ValueError):
        illustrate('k', tmp_path, tmp_path, tmp_path, 'nope')


@pytest.mark.parametrize('mode', ['mfhires', 'mfcomb', ''])
def test_illustrate_accepts_only_the_runnable_modes(tmp_path, mode):
    from filseg.reproduction.figures import illustrate
    with pytest.raises(ValueError):
        illustrate('k', tmp_path, tmp_path, tmp_path, mode)


def test_illustrate_refuses_a_csv_run_that_read_another_mf_map(tmp_path):
    # the figure must read the MF map the CSV run read: the package MF mean for the 7 modes, the variant's map for mftta
    import json
    from filseg.reproduction.figures import illustrate
    result = tmp_path / 'csv_result.json'
    result.write_text(json.dumps(dict(per_image={'k': {}})))
    with pytest.raises(ValueError, match='MF map'):
        illustrate('k', tmp_path, tmp_path, tmp_path, 'mftta')
    result.write_text(json.dumps(dict(per_image={'k': {}}, mf_variant=dict(name='mftta'))))
    with pytest.raises(ValueError, match='MF map'):
        illustrate('k', tmp_path, tmp_path, tmp_path, 'pcombo')


def test_render_names_the_mf_map_it_shows():
    from filseg.reproduction.figures import MF_PANEL, PANELS, render
    assert MF_PANEL == {None: PANELS[3], 'mftta': 'Mask2Former (8 views, mean of 3 fold models)'}
    args = synthetic()
    tta = render(*args, title='a', mf_title=MF_PANEL['mftta'])
    assert tta == render(*args, title='a', mf_title=MF_PANEL['mftta']) != render(*args, title='a')


def test_panel_titles_use_the_plain_stage_names_of_the_report():
    from filseg.reproduction.figures import PANELS
    assert PANELS == ('Test image', 'Full-disk U-Net', 'Local U-Net', 'Mask2Former (mean of 3 fold models)', 'Averaged map',
                      'Instance filter: kept (yellow) / dropped (grey)', 'Submitted masks',
                      'Mask repair: added (sky blue) / removed (reddish purple)')


def test_overlay_colours_are_those_of_report_figure_1():
    from filseg.reproduction.figures import ADDED, DROPPED, KEPT, REMOVED
    rgb = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    assert (KEPT, DROPPED, ADDED, REMOVED) == (rgb('#F0E442'), rgb('#9A9A9A'), rgb('#56B4E9'), rgb('#CC79A7'))


def test_kept_and_dropped_candidates_are_painted_in_their_colours():
    from filseg.reproduction import figures
    raw, full, local, mf, blend, candidates, keep, finals = synthetic()
    seen = []
    original = figures._paint
    try:
        figures._paint = lambda base, mask, colour, alpha=.65: seen.append(tuple(np.round(colour, 6))) or base
        figures.render(raw, full, local, mf, blend, candidates, keep, finals, title='x')
    finally:
        figures._paint = original
    assert seen[:3] == [tuple(np.round(c, 6)) for c in (figures.KEPT, figures.DROPPED, figures.KEPT)]
    assert tuple(np.round(figures.ADDED, 6)) in seen and tuple(np.round(figures.REMOVED, 6)) in seen


def test_figure_title_names_no_internal_mode_or_selector():
    from filseg.reproduction.figures import figure_title
    assert figure_title('20111122063134Lh', 18, 10) == '20111122063134Lh: 18 candidates, 10 kept'


def test_long_panel_titles_wrap_instead_of_being_cut_at_the_panel_edge():
    # 'Mask repair: added (sky blue) / removed (reddish purple)' was cut at about 50 characters in the 16-inch figure
    from filseg.reproduction.figures import MF_PANEL, PANELS, TITLE_WIDTH, panel_title
    for name in PANELS + (MF_PANEL['mftta'],):
        wrapped = panel_title(name)
        assert wrapped.replace('\n', ' ') == name and all(len(l) <= TITLE_WIDTH for l in wrapped.splitlines())
    assert TITLE_WIDTH <= 48 and panel_title(PANELS[7]).count('\n') == 1
