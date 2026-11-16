"""Score instance files (jsonl.gz) on the 141 fold-2 images with the host-parity PQ (`pqscore.py`). CPU only.

  python score.py check  --ann ANN.json --fold-table FT.json --instances SAVED.json.gz --reference CPU_result.json --out OUT.json
  python score.py arm    --ann ANN.json --fold-table FT.json --arm-dir DIR --out OUT.json
  python score.py compare --t T.json --c C.json --out result.json      (gate + bootstrap)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pqscore  # noqa: E402

GATE = dict(rule='PQ(T) >= 0.40 and the 95% CI of the paired per-image bootstrap of PQ(T) - PQ(C) is above 0',
            pq_min=0.40, decoder='D', threshold=0.5)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def fold_files(fold_table, fold=2):
    folds = json.loads(Path(fold_table).read_text())
    return sorted(f for f, k in folds.items() if k == fold)


def load_entries(ann, files):
    from train import annotation_sets

    class S:
        def __init__(self, a):
            self.annotations = a
    groups = annotation_sets(json.loads(Path(ann).read_text()))
    return pqscore.gt_entries({f: [S(a) for a in groups[f]] for f in files}, files)


def score_instances(entries, files, path):
    preds = pqscore.read_instances(path)
    assert sorted(preds) == files, f'{path}: image coverage differs'
    per, met, bins = pqscore.score_set(entries, preds, files)
    n_inst = int(sum(len(preds[f]) for f in files))
    return dict(file=Path(path).name, metric=met, area_bins=bins, n_instances=n_inst,
                per_file={f: per[i].tolist() for i, f in enumerate(files)})


def cmd_check(a):
    files = fold_files(a.fold_table)
    entries = load_entries(a.ann, files)
    got = score_instances(entries, files, a.instances)
    ref = json.loads(Path(a.reference).read_text())
    ref_metric = ref['scores']['m2f_instances']
    ref_per = {f: ref['per_file'][f]['m2f_instances']['counts'] for f in files}
    per_equal = sum(np.allclose(got['per_file'][f], ref_per[f], rtol=0, atol=1e-9) for f in files)
    res = dict(time=now(), what='scorer reproduces the recorded as-run m2f_instances score on the 141 fold-2 images',
               instances=str(a.instances), reference=str(a.reference), images=len(files),
               recorded_pq=ref_metric['pq'], scored_pq=got['metric']['pq'],
               abs_diff=abs(ref_metric['pq'] - got['metric']['pq']), per_file_counts_equal=int(per_equal),
               recorded_counts={k: ref_metric[k] for k in pqscore.FIELDS}, scored=got['metric'])
    res['passed'] = bool(res['abs_diff'] < 1e-12 and per_equal == len(files))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('recorded_pq', 'scored_pq', 'per_file_counts_equal', 'passed')}))
    return 0 if res['passed'] else 1


def cmd_arm(a):
    files = fold_files(a.fold_table, a.fold)
    entries = load_entries(a.ann, files)
    arm_dir = Path(a.arm_dir)
    meta = json.loads((arm_dir / 'infer_meta.json').read_text())
    out = dict(time=now(), arm=meta['arm'], holdout_fold=a.fold, images=len(files), infer_meta=meta, decodes={})
    for tag in meta['decodes']:
        out['decodes'][tag] = score_instances(entries, files, arm_dir / f'instances_{tag}.jsonl.gz')
        m = out['decodes'][tag]['metric']
        print(tag, json.dumps({k: m[k] for k in ('pq', 'sq', 'rq', 'tp', 'fp', 'fn')}), flush=True)
    out['primary'] = dict(decode='D_t0.50', **out['decodes']['D_t0.50']['metric'])
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_compare(a):
    t = json.loads(Path(a.t).read_text())
    c = json.loads(Path(a.c).read_text())
    files = sorted(t['decodes']['D_t0.50']['per_file'])
    assert files == sorted(c['decodes']['D_t0.50']['per_file']) and len(files) == 141

    def arr(r, tag):
        return np.array([r['decodes'][tag]['per_file'][f] for f in files])
    primary = pqscore.paired_bootstrap(arr(t, 'D_t0.50'), arr(c, 'D_t0.50'), draws=10000, seed=0)
    pq_t = t['decodes']['D_t0.50']['metric']['pq']
    real = bool(pq_t >= GATE['pq_min'] and primary['ci95'][0] > 0)
    grid = {}
    for tag in sorted(set(t['decodes']) & set(c['decodes'])):
        grid[tag] = dict(T=t['decodes'][tag]['metric'], C=c['decodes'][tag]['metric'],
                         T_minus_C_pq=t['decodes'][tag]['metric']['pq'] - c['decodes'][tag]['metric']['pq'])
    res = dict(time=now(), gate=GATE, verdict='real' if real else 'not confirmed', images=len(files),
               T=t['decodes']['D_t0.50']['metric'], C=c['decodes']['D_t0.50']['metric'], T_minus_C=primary,
               area_bins=dict(T=t['decodes']['D_t0.50']['area_bins'], C=c['decodes']['D_t0.50']['area_bins']),
               descriptive_grid=grid, t_infer=t['infer_meta'], c_infer=c['infer_meta'])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(verdict=res['verdict'], pq_T=pq_t, pq_C=res['C']['pq'], delta=primary['delta'],
                          ci95=primary['ci95'])))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('check', 'arm'):
        p = sp.add_parser(name)
        p.add_argument('--ann', required=True)
        p.add_argument('--fold-table', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--fold', type=int, default=2)
        if name == 'check':
            p.add_argument('--instances', required=True)
            p.add_argument('--reference', required=True)
        else:
            p.add_argument('--arm-dir', required=True)
    p = sp.add_parser('compare')
    p.add_argument('--t', required=True)
    p.add_argument('--c', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    return dict(check=cmd_check, arm=cmd_arm, compare=cmd_compare)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
