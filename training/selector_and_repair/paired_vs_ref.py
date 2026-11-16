"""HC1: paired cross-call contrasts of a host call vs the host replay hrep55 (host vs host only; orchestrator 09:0x), with
finish.py's own statistics (_vs_reference: delta, CI95 by paired 24 h-group bootstrap, positives/20, fold means).
Arms: w55 vs ref w55 (counts.json) and w55_refit vs ref w55_refit, w55_frozen vs ref w55_frozen (pcand_counts.json).
usage: paired_vs_ref.py <label> [ref=hrep55]  -> host/out/<label>/paired_vs_<ref>.json and a short table on stdout"""
import json, sys
from pathlib import Path
C = Path(__file__).resolve().parent; HC = C / 'host'  # the chain folder (was an absolute path)
sys.path.insert(0, str(C))
import finish as F
lab = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else 'hrep55'
cfg = F.read(F.CASE / 'protocol.json'); folds = F.read(F.CASE / 'frozen/evidence/fold_table.json'); files = sorted(folds); groups = F.temporal_groups(files, 24)
out = {}
for fname, pairs in (('counts.json', [('w55', 'w55'), ('pcombo', 'pcombo')]), ('pcand_counts.json', [('w55_frozen', 'w55_frozen'), ('w55_refit', 'w55_refit')])):
    if not (HC / 'out' / lab / 'case/out' / fname).is_file(): print('pending', fname); continue
    per = F.read(HC / 'out' / lab / 'case/out' / fname)['per_partition_counts']
    tmp = HC / 'out' / f'.{ref}_{fname}'                     # _vs_reference reads <root>/out/<ref>/case/out/counts.json
    root = HC / 'state' / 'pairref' / fname.split('.')[0]; (root / 'out' / ref / 'case/out').mkdir(parents=True, exist_ok=True)
    src = F.read(HC / 'out' / ref / 'case/out' / fname); src.setdefault('files', files)
    (root / 'out' / ref / 'case/out/counts.json').write_text(json.dumps(src))
    out.update(F._vs_reference(ref, per, files, folds, groups, cfg, pairs, chain2_root=root))
(HC / 'out' / lab / f'paired_vs_{ref}.json').write_text(json.dumps(out, indent=1, default=float))
for n, v in out.items():
    a = v['all707']; w = v['without_fold2']
    print(f"{n}: all707 {a['delta']:+.6f} CI95 [{a['paired_group_CI95'][0]:+.6f}, {a['paired_group_CI95'][1]:+.6f}] pos {a['positive_of20']}/20 (score {a['score']:.6f} vs {a['reference_score']:.6f}); without_fold2 {w['delta']:+.6f} pos {w['positive_of20']}/20; folds " +
          ' '.join(f"{v[f'fold{k}']['delta']:+.4f}" for k in range(5)))
