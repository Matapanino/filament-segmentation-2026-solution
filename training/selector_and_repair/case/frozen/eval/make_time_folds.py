"""Group temporally adjacent observations before crop generation.

Groups are connected components along observation time: any consecutive gap
<= the requested window stays together, across all observatories. This is a
provisional leakage guard, not a physical filament-event identification.
"""
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def temporal_groups(files, gap_hours):
    ordered = sorted((datetime.strptime(f[:14], '%Y%m%d%H%M%S'), f) for f in files)
    groups = []
    for time, filename in ordered:
        if not groups or (time - groups[-1][-1][0]).total_seconds() > gap_hours * 3600:
            groups.append([])
        groups[-1].append((time, filename))
    return [[f for _, f in group] for group in groups]


def assign_groups(groups, k=5, seed=42):
    groups = list(groups)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    sizes = [0] * k
    years = [Counter() for _ in range(k)]
    sites = [Counter() for _ in range(k)]
    assignment = {}
    for group in groups:
        # First balance sample count, then distribute matching sites and years.
        fold = min(range(k), key=lambda j: (sizes[j],
                   sum(years[j][f[:4]] + sites[j][f[-2]] for f in group), j))
        for f in group:
            assignment[f] = fold
            years[fold][f[:4]] += 1
            sites[fold][f[-2]] += 1
        sizes[fold] += len(group)
    return assignment


def main():
    legacy = json.loads((HERE.parents[1] / 'solution/configs/cv_folds_k5_seed0.json').read_text())
    days = defaultdict(list)
    for f in legacy:
        days[f[:8]].append(f)
    report = {'n_images': len(legacy), 'unique_days': len(days),
              'multi_image_days': sum(len(v) > 1 for v in days.values()),
              'legacy_crossfold_days': sum(len({legacy[f] for f in v}) > 1 for v in days.values()),
              'warning': 'Temporal grouping is a proxy; no pixel/event-identity audit performed.', 'splits': {}}
    out = HERE / 'configs'
    out.mkdir(exist_ok=True)
    for hours in [24, 168]:
        groups = temporal_groups(legacy, hours)
        assignment = assign_groups(groups)
        assert all(len({assignment[f] for f in group}) == 1 for group in groups)
        (out / f'time{hours}h_k5_seed42.json').write_text(json.dumps(assignment, indent=2, sort_keys=True))
        report['splits'][str(hours)] = {'groups': len(groups), 'max_group': max(map(len, groups)),
            'fold_sizes': dict(Counter(assignment.values())),
            'legacy_crossfold_groups': sum(len({legacy[f] for f in g}) > 1 for g in groups),
            'groups_by_file': {f: min(g) for g in groups for f in g},
            'site_counts': {k: dict(Counter(f[-2] for f, v in assignment.items() if v == k)) for k in range(5)},
            'year_counts': {k: dict(Counter(f[:4] for f, v in assignment.items() if v == k)) for k in range(5)}}
    (HERE / 'cv_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({**{k: v for k, v in report.items() if k != 'splits'},
        'splits': {k: {a: b for a, b in v.items() if a in ['groups', 'max_group', 'fold_sizes', 'legacy_crossfold_groups']}
                   for k, v in report['splits'].items()}}, indent=2))

if __name__ == '__main__':
    main()
