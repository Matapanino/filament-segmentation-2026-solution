"""label_mode='consensus_mix': consensus label sets (majority of annotators), default 'random' unchanged."""
import hashlib
import json

import numpy as np
import pytest
from pycocotools import mask as mu

import geom
import train

# sha256 of the 50-draw trace below, recorded with the pre-change train.py/geom.py (md5 e8d45ddb... / fe69b8ad...).
GOLDEN_RANDOM_TRACE = '54cc86aa923cf649d95774adcb29a8c7538efb78a6cac9edf5f3ef351afb712d'


def rect(x0, y0, w, h):
    return [[x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]]


def ann(seg, tag=None):
    return dict(segmentation=seg, tag=tag)


def fixture():
    a, b, c, d = rect(600, 600, 200, 120), rect(1200, 700, 160, 220), rect(800, 1300, 300, 90), rect(1400, 1400, 100, 100)
    groups = {
        'one': [[dict(segmentation=a), dict(segmentation=b)]],
        'two': [[dict(segmentation=a), dict(segmentation=c)], [dict(segmentation=a)]],
        'three': [[dict(segmentation=a), dict(segmentation=b), dict(segmentation=d)],
                  [dict(segmentation=a), dict(segmentation=c)], [dict(segmentation=b), dict(segmentation=c)]],
        'empty': [[], [dict(segmentation=d)]],
    }
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8)
    return {f: img for f in groups}, groups, sorted(groups)


def trace(ds, n=50):
    h = hashlib.sha256()
    for j in range(n):
        img, masks, meta = ds.draw(j)
        h.update(img.tobytes()); h.update(np.asarray(masks.shape).tobytes()); h.update(masks.tobytes())
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()


def three_set_image():
    """A: sets 0,1 (shifted). B: sets 0,2 (set 2 larger). C: set 1 only. D: all three. E: set 2 only.
    F: set 0 + set 1, but IoU ~0.2 (< 0.3: not matched -> two singletons, both dropped)."""
    s0 = [ann(rect(100, 100, 200, 100), 'A0'), ann(rect(600, 600, 100, 100), 'B0'), ann(rect(1500, 300, 150, 150), 'D0'),
          ann(rect(300, 1600, 100, 100), 'F0')]
    s1 = [ann(rect(110, 105, 200, 100), 'A1'), ann(rect(1000, 1000, 120, 80), 'C1'), ann(rect(1505, 305, 150, 150), 'D1'),
          ann(rect(367, 1600, 100, 100), 'F1')]
    s2 = [ann(rect(595, 595, 115, 115), 'B2'), ann(rect(1495, 295, 150, 150), 'D2'), ann(rect(1700, 1700, 60, 60), 'E2')]
    return [s0, s1, s2]


def tags(anns):
    return sorted(a['tag'] for a in anns)


def test_defaults_and_cli_flags():
    assert train.DEFAULTS['label_mode'] == 'random'
    assert train.DEFAULTS['consensus_p'] == 0.5
    assert train.DEFAULTS['consensus_iou'] == 0.3


def test_consensus_keeps_exactly_the_two_of_three_objects():
    sets = three_set_image()
    iou_f = mu.iou(mu.frPyObjects(sets[0][3]['segmentation'], 2048, 2048),
                   mu.frPyObjects(sets[1][3]['segmentation'], 2048, 2048), [0])[0][0]
    assert 0.1 < iou_f < 0.3
    cons = geom.consensus_sets(sets, iou_thr=0.3)
    assert len(cons) == 3
    # kept: A (2/3), B (2/3), D (3/3); dropped: C, E (1/3), F (unmatched, 1/3 each)
    assert tags(cons[0]) == ['A0', 'B0', 'D0']        # drawn set's own instance when it is in the cluster
    assert tags(cons[1]) == ['A1', 'B2', 'D1']        # B not in set 1 -> the cluster's largest member (B2)
    assert tags(cons[2]) == ['A0', 'B2', 'D2']        # A: A0 and A1 have equal area -> first member (set 0)
    # the consensus lists hold the original annotation dicts (no copies, polygons untouched)
    assert any(x is sets[0][0] for x in cons[0])


def test_consensus_two_sets_threshold_is_one_set():
    # ceil(2/2) = 1: with two sets every cluster is kept (the union, one instance per matched pair)
    sets = three_set_image()[:2]
    cons = geom.consensus_sets(sets, iou_thr=0.3)
    assert tags(cons[0]) == ['A0', 'B0', 'C1', 'D0', 'F0', 'F1']
    assert tags(cons[1]) == ['A1', 'B0', 'C1', 'D1', 'F0', 'F1']


def test_greedy_match_one_instance_per_set():
    # two set-0 instances overlap one set-1 instance: only the higher-IoU one joins its cluster
    s0 = [ann(rect(100, 100, 100, 100), 'big'), ann(rect(100, 100, 100, 60), 'part')]
    s1 = [ann(rect(100, 100, 100, 95), 'x')]
    s2 = []
    cons = geom.consensus_sets([s0, s1, s2], iou_thr=0.3)
    assert tags(cons[0]) == ['big']
    assert tags(cons[1]) == ['x']
    assert tags(cons[2]) == ['big']


def ds_for(**kw):
    images, groups, files = fixture()
    groups = dict(groups, three=three_set_image())
    cfg = dict(train.DEFAULTS, **kw)
    return train.CropBatches(images, groups, files, cfg), images, groups


def test_random_mode_trace_is_byte_identical_to_pre_change_code():
    images, groups, files = fixture()
    assert trace(train.CropBatches(images, groups, files, dict(train.DEFAULTS))) == GOLDEN_RANDOM_TRACE
    explicit = dict(train.DEFAULTS, label_mode='random', consensus_p=1.0)
    assert trace(train.CropBatches(images, groups, files, explicit)) == GOLDEN_RANDOM_TRACE


def test_single_set_images_and_p0_unchanged():
    rnd, _, _ = ds_for(label_mode='random')
    mix1, _, _ = ds_for(label_mode='consensus_mix', consensus_p=1.0)
    mix0, _, _ = ds_for(label_mode='consensus_mix', consensus_p=0.0)
    seen_single = 0
    for j in range(80):
        r, m0 = rnd.draw(j), mix0.draw(j)
        assert np.array_equal(r[0], m0[0]) and np.array_equal(r[1], m0[1])
        assert m0[2]['label'] == 'set' and {k: v for k, v in m0[2].items() if k not in ('label', 'set')} == r[2]
        if len(rnd.groups[r[2]['file']]) == 1:
            seen_single += 1
            m1 = mix1.draw(j)
            assert np.array_equal(r[0], m1[0]) and np.array_equal(r[1], m1[1]) and m1[2]['label'] == 'set'
    assert seen_single >= 10


def test_consensus_draws_use_cached_sets_and_stay_aligned(monkeypatch):
    ds, images, groups = ds_for(label_mode='consensus_mix', consensus_p=1.0)
    assert set(ds.consensus) == {'two', 'three', 'empty'}
    monkeypatch.setattr(geom, 'consensus_sets', lambda *a, **k: (_ for _ in ()).throw(AssertionError('not cached')))
    n_cons = 0
    for j in range(120):
        img, masks, meta = ds.draw(j)
        f = meta['file']
        if len(groups[f]) < 2:
            continue
        n_cons += 1
        assert meta['label'] == 'consensus'
        anns = ds.consensus[f][meta['set']]
        m = geom.continuous_matrix(meta, ds.cfg['crop'])
        assert np.array_equal(img, geom.warp_image(images[f], m, ds.cfg['crop']))
        assert np.array_equal(masks, geom.warp_instances(anns, m, ds.cfg['crop']))
        # geometry: the full-resolution consensus raster warped like the image equals the union of the crop masks
        if anns:
            full = geom.warp_instances(anns, np.array([[1.0, 0, 0], [0, 1.0, 0]]), 2048).max(0) * 255
            warped = geom.warp_image(full, m, ds.cfg['crop']) > 127
            union = masks.max(0).astype(bool) if len(masks) else np.zeros_like(warped)
            if warped.sum() > 500:
                assert (warped & union).sum() / (warped | union).sum() > 0.95
    assert n_cons >= 40


def test_mix_probability_and_bad_mode():
    ds, _, groups = ds_for(label_mode='consensus_mix', consensus_p=0.5)
    labels = [ds.draw(j)[2]['label'] for j in range(400) if len(groups[ds.file_for_draw(j)]) >= 2]
    frac = labels.count('consensus') / len(labels)
    assert 0.4 < frac < 0.6
    with pytest.raises(ValueError):
        ds_for(label_mode='majority')


def test_consensus_rule_default_and_strict_majority():
    assert train.DEFAULTS['consensus_rule'] == 'ceil_half'
    sets2 = three_set_image()[:2]
    # strict majority on 2 sets = floor(2/2)+1 = 2: both sets must agree (A, D matched; B, C, F singletons dropped)
    cons = geom.consensus_sets(sets2, iou_thr=0.3, rule='strict_majority')
    assert tags(cons[0]) == ['A0', 'D0'] and tags(cons[1]) == ['A1', 'D1']
    assert geom.consensus_sets(sets2, iou_thr=0.3, rule='ceil_half') == geom.consensus_sets(sets2, iou_thr=0.3)
    # 3 sets: both rules need 2 sets
    sets3 = three_set_image()
    assert geom.consensus_sets(sets3, 0.3, rule='strict_majority') == geom.consensus_sets(sets3, 0.3)
    with pytest.raises(ValueError):
        geom.consensus_sets(sets3, 0.3, rule='half')


def test_consensus_rule_reaches_dataset():
    ds, _, groups = ds_for(label_mode='consensus_mix', consensus_rule='strict_majority')
    assert ds.consensus['empty'] == [[], []]  # [[], [d]]: d is in 1 of 2 sets -> dropped
    assert tags(ds.consensus['three'][0]) == ['A0', 'B0', 'D0']
    dflt, _, _ = ds_for(label_mode='consensus_mix')
    assert len(dflt.consensus['empty'][0]) == 1  # ceil_half keeps it
    with pytest.raises(ValueError):
        ds_for(label_mode='consensus_mix', consensus_rule='half')
