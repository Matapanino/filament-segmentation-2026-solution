"""Stage one chainF call (order INT1; local; reads arrays, fits nothing): chain2m's staging (orders CH2, PP1, MT1) plus
chain2u's U-Net sets (order CH3/CH4, --full/--local), --pcand-refit (order PC1) and a --combined stub (order ST1, not ported).
CLI: stage.py <label> <weights> (--from <chain2 label> | <tok0..tok4>) [--full s] [--local s] [--rule r] [--levels l]
     [--pcand-refit w55,w70] [--combined <archive> --thr t]. Flags absent = chain2m's staging exactly.

Below: chain2's original header.


usage: stage.py <label> <weights> <tok0> <tok1> <tok2> <tok3> <tok4>
  weights: comma-separated MF blend weights, 0.25 required (e.g. 0.25,0.40,0.55)
  fold token k: 'runA' or 'runA+runB+...' — a1native artifacts run ids whose export holds MF_f<k>_val.npz (the held-out
  fold-k images) and MF_f<k>_test.npz (the 180 test images); every fold has the same number of sets (set j = the j-th run
  id of each token). 'stored' is not supported here (chain/ has it).

Adapter = checks, not conversion: each a1native export must already be the package MF format (verified here per file):
uint16 [2048, 2048] arrays keyed by image stem, levels <= 510 (rint(p x 510)), one npz per fold and split; the export
manifest must name the fold, denominator 510 and the file sha256. Keys must equal the fold-k images of the 24 h k=5 seed-42
fold table (case/frozen/evidence/fold_table.json) and the package's 180 test names.
Writes stage/<label>/chain_inputs.json and stage/<label>/data/ (byte copies of the export files, never links, renamed
chain2-<label>-oof_f<k>_s<j>.npz / chain2-<label>-test_f<k>_s<j>.npz, plus dataset-metadata.json for the private dataset).
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
ART = H.parent / 'a1native/artifacts'
FOLDS = H / 'case/frozen/evidence/fold_table.json'
TEST_NAMES = H.parent / 'pkg/src/filseg/data/test_filenames.txt'
STAGE_ROOT = H / 'stage'
SETS = H.parent / 'chain2u' / 'sets'           # chain2u's U-Net set manifests (read only)
UNET_SHAPE = (2048, 2048)
VIEWS = ('full', 'local')
DEN = 510
sys.path[:0] = [str(H), str(H / 'case/frozen'), str(H / 'case/frozen/eval'), str(H / 'case')]
from render_kernel import LABEL, dataset_slug, combined_dataset_slug   # noqa: E402
import mt as M                                   # noqa: E402


def stored_pins():
    """sha256 of the U-Net OOF maps the chain pins, per view and fold (the CV's stored full/local maps)."""
    case = H / 'case'; out = {v: {} for v in VIEWS}
    mf = json.loads((case / 'frozen/evidence/mf_inputs.json').read_text())['original']
    conf = json.loads((case / 'support/confirmation_protocol.json').read_text())['original_map_inventory']
    ev = json.loads((case / 'evidence/fold2_map_provenance.json').read_text())
    for v in VIEWS:
        for k in (0, 1): out[v][k] = mf[f'f{k}_{v}_map']['sha256']
        out[v][2] = ev[v]['dataset_map']['sha256']
        for k in (3, 4): out[v][k] = conf[f'fold{k}_s42'][v]['map']['sha256']
    return out


STORED_SHA256 = stored_pins()
# the package's shipped U-Net test maps (map set d7aa6a41..., inference_result.json of kernel filament-pkgfinal-1113-gpu)
PACKAGE_SHA256 = dict(full='832e7ff901f2f7d6f0c973d08b2e0f644f977e790954993d3dbff5c38ff242dd', local='d0a007fe6473737ac612bef8e62cab753fea78f15389d67088b706ffed1063f2')


def digest(p):
    h = {n: hashlib.new(n) for n in ('sha256', 'md5')}
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            for x in h.values(): x.update(b)
    return dict(bytes=Path(p).stat().st_size, sha256=h['sha256'].hexdigest(), md5=h['md5'].hexdigest())


def check_archive(path, keys):
    """The package MF format, per array. Returns small statistics (no arrays)."""
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(keys): raise ValueError(f'{path.name}: keys differ from the expected image set ({len(z.files)} vs {len(keys)})')
        hi = 0; lo = DEN; means = []
        for k in sorted(keys):
            a = z[k]
            if a.dtype != np.uint16 or a.shape != (2048, 2048): raise ValueError(f'{path.name}:{k}: {a.dtype} {a.shape}, expected uint16 (2048, 2048)')
            m = int(a.max())
            if m > DEN: raise ValueError(f'{path.name}:{k}: level {m} > {DEN}')
            hi = max(hi, m); lo = min(lo, int(a.min())); means.append(float(a.mean()) / DEN)
    return dict(arrays=len(keys), max_level=hi, min_level=lo, mean_probability=float(np.mean(means)))


def check_unet_archive(path, keys):
    """The package U-Net format, per array: uint8 [2048, 2048], keys = exactly the expected stems. Small statistics."""
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(keys): raise ValueError(f'{path}: keys differ from the expected image set ({len(z.files)} vs {len(keys)})')
        means = []; hi = 0
        for k in sorted(keys):
            a = z[k]
            if a.dtype != np.uint8 or a.shape != UNET_SHAPE: raise ValueError(f'{path}:{k}: {a.dtype} {a.shape}, expected uint8 {UNET_SHAPE}')
            hi = max(hi, int(a.max())); means.append(float(a.mean()) / 255)
    return dict(arrays=len(keys), max_level=hi, mean_probability=float(np.mean(means)))


def export_files(run, k, d=None):
    """d: the export directory (default the a1native artifacts run directory; chain2s passes <archive dir>/f<k>)."""
    d = ART / run if d is None else Path(d); man = json.loads((d / 'export_manifest.json').read_text())
    if int(man['fold']) != k: raise ValueError(f'{run}: export fold {man["fold"]} used for fold {k}')
    fmt = man['map_format']
    if fmt.get('dtype') != 'uint16' or fmt.get('denominator') != DEN or list(fmt.get('shape')) != [2048, 2048] or fmt.get('keys') != 'image stem':
        raise ValueError(f'{run}: export map_format is not the package MF format: {fmt}')
    out = {}
    for split in ('val', 'test'):
        name = f'MF_f{k}_{split}.npz'; p = d / name
        if not p.is_file(): raise FileNotFoundError(p)
        idt = digest(p); want = man['files'][name]
        if idt['sha256'] != want['sha256'] or idt['bytes'] != want['bytes']: raise ValueError(f'{run}/{name}: file differs from its export manifest')
        out[split] = (p, idt)
    return man, out


def read_set(name):
    p = SETS / f'{name}.json'
    if not p.is_file(): raise FileNotFoundError(f'U-Net set {name!r}: {p} (write it with make_set.py)')
    s = json.loads(p.read_text())
    if s.get('name') != name or sorted(s.get('views', {})) != sorted(VIEWS): raise ValueError(f'{p}: not a U-Net set manifest')
    return s, digest(p)['sha256']


def check_combined(src, thr=None):
    """chain2s: every check of a combiner's 5 fold archives before anything is written. thr: the combined arm's SHIP_R3 thr
    (None = archive.json's); archive.json, when present, must agree (thr and every file's sha256 / bytes). Raises ValueError."""
    src = Path(src); folds = json.loads(FOLDS.read_text()); test = sorted(TEST_NAMES.read_text().split())
    aj = json.loads((src / 'archive.json').read_text()) if (src / 'archive.json').is_file() else None
    if aj is not None and 'thr' in aj:
        if thr is not None and float(thr) != float(aj['thr']): raise ValueError(f'combined thr {thr} differs from archive.json thr {aj["thr"]}')
        thr = aj['thr']
    if thr is None: raise ValueError('combined: no thr (give --combined-thr or archive.json thr)')
    thr = float(thr)
    if not 0 < thr < 1: raise ValueError(f'combined thr {thr}: must be in (0, 1)')
    files = {}; checks = {}; exports = {}
    for k in range(5):
        d = src / f'f{k}'
        if not (d / 'export_manifest.json').is_file(): raise ValueError(f'{d}: no export_manifest.json')
        try: man, got = export_files(src.name, k, d)
        except (KeyError, FileNotFoundError) as e: raise ValueError(f'{d}: {e!r}') from None
        valid = sorted(f for f, v in folds.items() if v == k)
        for split, keys, kind in (('val', valid, 'oof'), ('test', test, 'test')):
            p, idt = got[split]
            if aj is not None:
                want = aj.get('folds', {}).get(str(k), {}).get(p.name)
                if not want or want.get('sha256') != idt['sha256'] or want.get('bytes', idt['bytes']) != idt['bytes']: raise ValueError(f'{p}: differs from archive.json')
            checks[f'{kind}_f{k}'] = check_archive(p, keys); files[f'{kind}_f{k}'] = (p, idt)
        exports[str(k)] = dict(time=man.get('time'), note=man.get('note'), formula=man['map_format'].get('formula'), view=man['map_format'].get('view'))
    name = (aj or {}).get('name') or src.name
    return dict(name=name, thr=thr, files=files, checks=checks, exports=exports,
                archive_json_sha256=digest(src / 'archive.json')['sha256'] if aj is not None else None, kind=(aj or {}).get('kind'))


def stage_combined(label, root, cmb, combined):
    """chain2s's staging of the combiner archives (its stage_from block; chainF file names), shared by both chainF modes:
    byte copies into stage/<label>/cmb_data/ (the call's second private dataset); returns pins['combined']."""
    ds = combined_dataset_slug(label); data = root / 'cmb_data'
    c = dict(arm=M.COMBINED_ARM, name=cmb['name'], kind=cmb['kind'], thr=cmb['thr'], source=str(Path(combined)), archive_json_sha256=cmb['archive_json_sha256'],
             dataset=ds, dataset_scope=ds.split('/')[-1], oof={}, test={}, checks=cmb['checks'], exports=cmb['exports'])
    staged = {}
    for k in range(5):
        for kind in ('oof', 'test'):
            p, idt = cmb['files'][f'{kind}_f{k}']; name = f'chainf-{label}-cmb-{kind}_f{k}.npz'
            c[kind][str(k)] = [dict(file=name, source=str(p), **idt)]; staged[name] = (p, idt)
    data.mkdir(parents=True)
    for name, (p, idt) in staged.items():
        shutil.copyfile(p, data / name)
        if digest(data / name) != idt: raise ValueError(f'copy of {p} differs')
    (data / 'dataset-metadata.json').write_text(json.dumps(dict(title=c['dataset_scope'], id=ds, licenses=[dict(name='other')]), indent=1))
    return c


def stage_unet(label, unet, folds, test, staged, checks):
    """pins['unet'] for the swapped views; staged files added to `staged`. Every check before anything is written."""
    out = {}
    for view in VIEWS:
        if view not in unet: continue
        sets = list(unet[view]); u = dict(sets=sets, oof={}, test={}, manifests=[])
        for j, name in enumerate(sets):
            s, msha = read_set(name); u['manifests'].append(dict(set=name, sha256=msha, description=s.get('description')))
            for k in range(5):
                row = s['views'][view][str(k)]; valid = sorted(f for f, v in folds.items() if v == k)
                for kind, keys in (('oof', valid), ('test', test)):
                    p = Path(row[kind])
                    if not p.is_absolute() or not p.is_file(): raise FileNotFoundError(f'set {name} {view} fold {k} {kind}: {p}')
                    checks[f'unet_{view}_{kind}_f{k}_s{j}'] = check_unet_archive(p, keys); idt = digest(p)
                    fname = f'chainf-{label}-unet_{view}_{kind}_f{k}_s{j}.npz'
                    u[kind].setdefault(str(k), []).append(dict(file=fname, set=name, fold=k, model=row.get('model'), source=str(p), **idt)); staged[fname] = (p, idt)
        u['stored_equivalent'] = bool(len(sets) == 1 and all(u['oof'][str(k)][0]['sha256'] == STORED_SHA256[view][k] for k in range(5))
                                      and all(u['test'][str(k)][0]['sha256'] == PACKAGE_SHA256[view] for k in range(5)))
        out[view] = u
    return out


def stage(label, weights, tokens, rule='', unet=None, extra=None, no_w25=False, combined='', combined_thr=None):
    if not LABEL.match(label): raise ValueError('label: lowercase letters, digits, hyphens; at most 22 characters')
    ws = M.parse_weights(weights, not no_w25); src = M.fold_sources(tokens); n = M.n_sets(src)
    import rulespec; rspec = rulespec.rule_spec(rulespec.parse_rule(rule)) if rule and rule.strip() else None   # chain2r: checked before anything is written
    folds = json.loads(FOLDS.read_text()); test = sorted(TEST_NAMES.read_text().split()); assert len(test) == 180
    root = STAGE_ROOT / label; data = root / 'data'
    if (root / 'chain_inputs.json').exists(): raise FileExistsError(f'{root} is already staged; use a new label')
    pins = dict(label=label, folds=list(tokens), weights=ws, sets=n, dataset=dataset_slug(label), dataset_scope=dataset_slug(label).split('/')[-1],
                oof={}, test={}, exports={}, checks={})
    staged = {}
    for k in range(5):                                  # every check before anything is written
        valid = sorted(f for f, v in folds.items() if v == k)
        for kind in ('oof', 'test'): pins[kind][str(k)] = []
        pins['exports'][str(k)] = []
        for j, run in enumerate(src[k]):
            man, files = export_files(run, k)
            for split, keys, kind in (('val', valid, 'oof'), ('test', test, 'test')):
                p, idt = files[split]; pins['checks'][f'{kind}_f{k}_s{j}'] = check_archive(p, keys)
                name = f'chain2-{label}-{kind}_f{k}_s{j}.npz'; pins[kind][str(k)].append(dict(file=name, run_id=run, set=j, source=str(p), **idt)); staged[name] = (p, idt)
            pins['exports'][str(k)].append(dict(run_id=run, set=j, step=man.get('step'), time=man.get('time'), cfg_sha256=man.get('cfg_sha256'),
                                                view=man['map_format'].get('view'), formula=man['map_format'].get('formula'), metrics=man.get('metrics')))
    if unet: pins['unet'] = stage_unet(label, unet, folds, test, staged, pins['checks'])   # chain2u (CH3): checked before anything is written
    cmb = None
    if combined:                                         # chain2s (ported after F0): checked before anything is written
        if rspec is not None: raise ValueError('--combined excludes --rule')
        cmb = check_combined(combined, combined_thr); M.layout(ws, None, dict(thr=cmb['thr']), require_base=not no_w25)   # w55 required
    data.mkdir(parents=True)
    for name, (p, idt) in staged.items():
        shutil.copyfile(p, data / name)
        if digest(data / name)['sha256'] != idt['sha256']: raise ValueError(f'copy of {p} differs')
    (data / 'dataset-metadata.json').write_text(json.dumps(dict(title=pins['dataset_scope'], id=pins['dataset'], licenses=[dict(name='other')]), indent=1))
    if rspec is not None: pins['rule'] = rspec   # chain2r: only when given
    pins.update(extra or {})                       # chainF: pcand_refit (only when given)
    if no_w25: pins['no_w25'] = True               # chainF: only when --no-w25 was given
    if cmb is not None: pins['combined'] = stage_combined(label, root, cmb, combined)   # chain2s
    (root / 'chain_inputs.json').write_text(json.dumps(pins, indent=1))
    return pins


CHAIN2_STAGE = H.parent / 'chain2' / 'stage'      # read only


def stage_from(label, weights, ref, rule='', stage_root=STAGE_ROOT, ref_root=CHAIN2_STAGE, levels='', extra=None, no_w25=False, combined='', combined_thr=None):
    """chain2r (order PP1): a call on the map sets of an already staged and uploaded chain2 call `ref` (its private dataset is
    reused as is: same files, same identities; nothing is copied or uploaded). New label and weights; 'rule' (canonical text
    + fields) only when a rule is given — without one the pins are chain2's and the call is chain2 exactly."""
    import rulespec
    if not LABEL.match(label): raise ValueError('label: lowercase letters, digits, hyphens; at most 22 characters')
    ws = M.parse_weights(weights, not no_w25)
    r = rulespec.parse_rule(rule) if rule and rule.strip() else None
    import levels as LV                                  # chain2m (order MT1): --levels '0.5;0.5,0.4'
    lv = LV.parse_levelsets(levels) if levels and levels.strip() else None
    if lv is not None and r is not None: raise ValueError('--levels and --rule are exclusive')
    if lv is not None: M.layout(ws, lv, require_base=not no_w25)                 # the arm layout must be valid
    cmb = None
    if combined:                                         # chain2s: --combined <archive dir> [--thr t]
        if r is not None or lv is not None: raise ValueError('--combined excludes --rule and --levels')
        cmb = check_combined(combined, combined_thr); M.layout(ws, None, dict(thr=cmb['thr']), require_base=not no_w25)   # w55 required
    src = ref_root / ref / 'chain_inputs.json'
    if not src.is_file(): raise FileNotFoundError(src)
    root = stage_root / label
    if (root / 'chain_inputs.json').exists(): raise FileExistsError(f'{root} is already staged; use a new label')
    pins = json.loads(src.read_text())
    pins.update(label=label, weights=ws, reuses_chain2_stage=ref)
    if r is not None: pins['rule'] = rulespec.rule_spec(r)
    if lv is not None: pins['levels'] = lv
    pins.update(extra or {})                             # chainF: pcand_refit (only when given)
    if no_w25: pins['no_w25'] = True                     # chainF: only when --no-w25 was given
    if cmb is not None: root.mkdir(parents=True, exist_ok=True); pins['combined'] = stage_combined(label, root, cmb, combined)   # chain2s
    root.mkdir(parents=True, exist_ok=True); (root / 'chain_inputs.json').write_text(json.dumps(pins, indent=1))
    return pins


# ---------------------------------------------------------------- chainF (order INT1)
DEPLOYED = dict(quality_model_sha256='dbb530cf270b69e049380d7b9ff1690ce5a57be6c3ae04f1ea1675fc08740606', dataset='<kaggle-user>/filament-repro-weights-1111',
                role='deployment_full707_original_only', threshold=0.35)      # pcand/prepare.py DEPLOYED (pcand_ship's keep rule), verbatim
FLAGS = ('--from', '--full', '--local', '--rule', '--levels', '--pcand-refit', '--combined', '--thr', '--test-unet')
SWITCHES = ('--no-w25',)                           # flags without a value


def parse_cli(argv):
    """<label> <weights> (--from <ref> | <tok0..tok4>) [flags] -> dict(label, weights, ref, tokens, unet, rule, levels, pcand, combined)."""
    argv = list(argv)
    if len(argv) < 3: raise ValueError('usage: <label> <weights> (--from <chain2 label> | <tok0..tok4>) [flags]')
    out = dict(label=argv[0], weights=argv[1], ref=None, tokens=[], unet={}, rule='', levels='', pcand=None, combined=None, thr=None, no_w25=False, test_unet=None)
    rest = argv[2:]
    while rest and not rest[0].startswith('--'): out['tokens'].append(rest.pop(0))
    seen = set()
    while rest:
        flag = rest.pop(0)
        if flag in SWITCHES:
            if flag in seen: raise ValueError(f'{flag} given twice')
            seen.add(flag); out['no_w25'] = True; continue
        if flag not in FLAGS or not rest: raise ValueError(f'bad flag or missing value: {flag!r}')
        if flag in seen: raise ValueError(f'{flag} given twice')
        seen.add(flag); val = rest.pop(0)
        if flag == '--from': out['ref'] = val
        elif flag in ('--full', '--local'): out['unet'][flag[2:]] = M.parse_set_spec(val)
        elif flag == '--rule': out['rule'] = val
        elif flag == '--levels': out['levels'] = val
        elif flag == '--pcand-refit': out['pcand'] = [a.strip() for a in val.split(',') if a.strip()]
        elif flag == '--combined': out['combined'] = val
        elif flag == '--thr': out['thr'] = val
        elif flag == '--test-unet':
            if val != 'fold5mean': raise ValueError(f'--test-unet: only fold5mean is known, not {val!r}')
            out['test_unet'] = val
    if (out['ref'] is None) == (len(out['tokens']) != 5): raise ValueError('give either --from <chain2 label> or five fold tokens')
    if out['ref'] is not None and out['tokens']: raise ValueError('--from and fold tokens are exclusive')
    if out['unet'] and out['ref'] is not None: raise ValueError('--full/--local need fold tokens (the U-Net archives are uploaded with the MF maps); --from reuses a chain2 dataset as is')
    if out['levels'] and out['ref'] is None: raise ValueError('--levels needs --from (chain2m)')
    if out['thr'] is not None and out['combined'] is None: raise ValueError('--thr needs --combined')
    if out['combined'] is not None and out['levels']: raise ValueError('--combined excludes --levels')
    return out


CHAIN3 = H.parent / 'chain3'                       # read only: the call whose private dataset already holds the fold5mean files
FOLD5MEAN_STAGE = 'un5_c7c10d4_w55'
FOLD5MEAN_SHA256 = dict(full='ff0c7a809993de4ac99681ff83ea0175118ec274c4dae75cf065469e42695fb9', local='c62d6e2aa7273c4e8c57a994e11e47737c13f9e0a2a2a4c9696f479e479cfccf')


def test_unet_pins(chain3_root=CHAIN3):
    """pins['test_unet'] (--test-unet fold5mean): the 5-fold-mean U-Net test maps as chain3's validated un5_c7c10d4_w55 call
    staged them (file name, bytes, sha256; its private dataset is attached, nothing uploaded). The source files must still have
    those bytes and the fold5mean sha256 (pcand ship's rule)."""
    p3 = json.loads((chain3_root / 'stage' / FOLD5MEAN_STAGE / 'chain3_inputs.json').read_text()); out = dict(name='fold5mean', dataset=p3['dataset'])
    for v in ('full', 'local'):
        rows = p3['unet'][v]
        if len(rows) != 1 or rows[0]['kind'] != 'file': raise ValueError(f'chain3 {FOLD5MEAN_STAGE} {v}: not one staged file')
        r = rows[0]
        if r['sha256'] != FOLD5MEAN_SHA256[v] or digest(r['source'])['sha256'] != r['sha256']: raise ValueError(f'fold5mean {v}: sha256 differs')
        out[v] = dict(file=r['file'], bytes=r['bytes'], sha256=r['sha256'], md5=r['md5'], source=r['source'])
    return out


def pcand_pins(arms, weights, levels, no_w25=False):
    """pins['pcand_refit']: the arms (weight arms of this call, not multi-level) and the deployed LR39 pin."""
    import levels as LV
    lay = M.layout(M.parse_weights(weights, not no_w25), LV.parse_levelsets(levels) if levels and levels.strip() else None, require_base=not no_w25)
    if not arms or len(set(arms)) != len(arms): raise ValueError(f'--pcand-refit: bad arm list {arms}')
    for a in arms:
        if a not in lay['NEW_VARIANTS']: raise ValueError(f'--pcand-refit: {a} is not a weight arm of this call ({lay["NEW_VARIANTS"]})')
        if lay['LEVELS'].get(a) is not None and len(lay['LEVELS'][a]) > 1: raise ValueError(f'--pcand-refit: {a} is a multi-level arm (refused, order INT1 decision 4)')
    return dict(arms=list(arms), deployed_selector=DEPLOYED)


def stage_cli(argv):
    c = parse_cli(argv)
    extra = dict(pcand_refit=pcand_pins(c['pcand'], c['weights'], c['levels'], c['no_w25'])) if c['pcand'] is not None else {}
    if c['test_unet']: extra['test_unet'] = test_unet_pins()                   # chainF: --test-unet fold5mean (after F0)
    cmb = dict(combined=str(Path(c['combined']).resolve()) if c['combined'] else '', combined_thr=float(c['thr']) if c['thr'] is not None else None)
    if c['ref'] is not None: return stage_from(c['label'], c['weights'], c['ref'], c['rule'], levels=c['levels'], extra=extra, no_w25=c['no_w25'], **cmb)
    return stage(c['label'], c['weights'], c['tokens'], c['rule'], unet=c['unet'], extra=extra, no_w25=c['no_w25'], **cmb)


if __name__ == '__main__':
    p = stage_cli(sys.argv[1:])
    print(json.dumps(dict(label=p['label'], weights=p['weights'], sets=p['sets'], folds=p['folds'], reuses=p.get('reuses_chain2_stage'), dataset=p['dataset'],
                          rule=p.get('rule'), levels=p.get('levels'), pcand_refit=p.get('pcand_refit'), no_w25=bool(p.get('no_w25')),
                          unet={v: dict(sets=u['sets'], stored_equivalent=u['stored_equivalent']) for v, u in p.get('unet', {}).items()})))
