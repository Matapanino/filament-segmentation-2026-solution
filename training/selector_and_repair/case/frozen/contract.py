"""Frozen arithmetic and fail-closed admission rules. No compute launch methods."""
import numpy as np

DENOMINATORS = dict(full=255, local=255, m2f_public=510, yolo_roi=510)
WEIGHTS = {
    'incumbent': dict(m2f_public=.25, full=.375, local=.375),
    'yolo25': dict(m2f_public=.1875, full=.28125, local=.28125, yolo_roi=.25),
}


def decode_maps(raw):
    assert set(raw) == set(DENOMINATORS)
    assert len({v.shape for v in raw.values()}) == 1
    result = {}
    for name, denominator in DENOMINATORS.items():
        a = raw[name]
        assert a.dtype == (np.uint8 if denominator == 255 else np.uint16)
        assert a.size and a.max() <= denominator
        result[name] = a.astype(np.float32) / denominator
    return result


def mix(maps, arm):
    # Deliberately flat, insertion-ordered sum: nested .75 * incumbent differs.
    return sum(np.float32(w) * maps[k] for k, w in WEIGHTS[arm].items())


def assert_fold_disjoint(train_files, entire_validation_fold, component):
    assert train_files is not None, (component, 'unknown training provenance')
    assert entire_validation_fold, (component, 'missing validation fold')
    overlap = sorted(set(train_files) & set(entire_validation_fold))
    assert not overlap, (component, 'validation-fold training overlap', len(overlap), overlap[:3])


def replay_pq(counts):
    counts = np.asarray(counts, dtype=np.float64)
    assert counts.ndim == 3 and counts.shape[2] == 6
    assert np.isfinite(counts).all() and (counts >= 0).all()
    c = counts.mean(axis=0).sum(axis=0)
    denominator = c[1] + .5 * (c[2] + c[3])
    return float(c[0] / denominator) if denominator else 0.


def gate(result):
    try:
        if result['leakage_passed'] is not True or result['full_pipeline_replay_passed'] is not True:
            return False
        if sorted(result['roi_maps_complete']) != [0, 1]:
            return False
        subsets = result['subsets']
        expected = dict(new284=284, combined425=425, fold0=142, fold1=142, fold2=141)
        for name, n in expected.items():
            r = subsets[name]
            if r['images'] != n or len(r['partition_deltas']) != 20:
                return False
            scalars = [r['delta'], r['mean_paired_partition_delta'], *r['paired_group_CI95'], *r['partition_deltas']]
            if not np.isfinite(scalars).all():
                return False
        for name in ('new284', 'combined425'):
            r = subsets[name]
            if not (r['delta'] > 0 and r['mean_paired_partition_delta'] > 0 and
                    sum(d > 0 for d in r['partition_deltas']) >= 16 and
                    r['paired_group_CI95'][0] > 0):
                return False
        return all(subsets[f'fold{k}']['delta'] > 0 and
                   subsets[f'fold{k}']['mean_paired_partition_delta'] > 0 for k in (0, 1))
    except (KeyError, TypeError, ValueError):
        return False
