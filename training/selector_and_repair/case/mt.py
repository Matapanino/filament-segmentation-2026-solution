"""Chain2 (order CH2): pure helpers (weight arms, fold tokens, the multi-set MF map average, the weighted blend, size bins,
contrasts, feature names, the descriptive 707 gate rule). No fit, no transport. Adapted from chain/case/mt.py (CH1).

Arms: inc / inc_pnat / inc_pcand_v1 / pcombo read the stored OOF maps (pcombo's exact replay). Weight arms w<NN>, one per
blend weight w given on the command line (0.25 required): the MF map is the per-image average of one or more new
Mask2Former map sets (fold token 'runA_fk+runB_fk'; each map from the fold model that did not train on the image):
level sum / (510 n) in float32, no requantisation; the mixture is w * MF + (1 - w) / 2 * full + (1 - w) / 2 * local in the
shipped flat insertion order (w = 0.25 gives .25 / .375 / .375, the shipped mixture bit for bit). Everything downstream is
chain CH1 arm new: each weight arm refits pcombo's 68-feature selector on its own candidates and uses pcombo's 100 outer v1
regressors unchanged.

chain2s (ported into chainF after F0, order INT1): an optional combined-map arm 'cmb' (weight 1.0, chain_inputs.json
'combined'): MF map = a combiner's OOF probability map (one archive per base fold, map / 510), mixture = MF * float32(1.0)
(no U-Net terms), instancing SHIP_R3 with its own thr. Without it the layout is chain2m's (and chainF F0's) exactly.
"""
import math
import re
import numpy as np
import mr as R
import pc as PC

FIXED_ARMS = ['inc', 'inc_pnat', 'inc_pcand_v1', 'pcombo']
REPLAY_CONTRASTS = ['pcombo_minus_inc_pcand_v1', 'inc_pcand_v1_minus_inc_pnat', 'inc_pcand_v1_minus_inc', 'inc_pnat_minus_inc']   # also in pcombo's science_result
BASE_WEIGHT = .25
BASE_ARM = 'w25'
DEFAULT_WEIGHTS = (.25, .40, .55)
GATE_CONTRAST = 'w25_minus_pcombo'
DESCRIPTIVE_CONTRASTS = []
BLOCKING = ('replay_passed', 'integrity_passed', 'inputs_passed')
DENOMINATOR = 510
SIZE_BINS = ('lt256', '256to1024', 'gt1024')
RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
COMBINED_ARM = 'cmb'             # chain2s: the combined-map arm
COMBINED_WEIGHT = 1.0
COMBINED_REF = 'w55'             # chain2s: the reference arm (replayed exactly in the same kernel; cmb's cross-call contrast)
COMBINED = None


def arm_name(w): return f'w{int(round(w * 100)):02d}'


def parse_weights(spec, require_base=True):
    """'0.25,0.40,0.55' (or a list of floats) -> sorted floats; each in (0, 1) on the 0.01 grid, unique, 0.25 included."""
    items = [x.strip() for x in spec.split(',')] if isinstance(spec, str) else list(spec)
    if not items or any(x == '' for x in items): raise ValueError('weights: a comma-separated list such as 0.25,0.40,0.55')
    try: ws = [float(x) for x in items]
    except (TypeError, ValueError): raise ValueError(f'weights: not numbers: {spec!r}')
    for w in ws:
        if not 0 < w < 1 or abs(w * 100 - round(w * 100)) > 1e-9: raise ValueError(f'weight {w}: must be in (0, 1) on the 0.01 grid')
    if len({arm_name(w) for w in ws}) != len(ws): raise ValueError('weights: duplicates')
    if require_base and BASE_WEIGHT not in ws: raise ValueError('weights: 0.25 (arm w25, the chain CH1 arm new) is required')   # chainF: unless no_w25
    return sorted(ws)


def parse_combined(spec):
    """chain2s: None, or a dict with 'thr' (SHIP_R3's thr for the combined arm, in (0, 1)) and optionally 'name' ->
    dict(arm='cmb', thr, name)."""
    if spec is None: return None
    if not isinstance(spec, dict) or 'thr' not in spec: raise ValueError('combined: a dict with thr')
    t = spec['thr']
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 < t < 1: raise ValueError(f'combined thr {t!r}: must be in (0, 1)')
    return dict(arm=COMBINED_ARM, thr=float(t), name=spec.get('name'))


def layout(weights, levelsets=None, combined=None, *, require_base=True):
    """chain2m (order MT1): levelsets None = chain2's layout. Otherwise w25 stays the plain SHIP_R3 arm (no levels) and every
    other weight w gets one arm per level set, arm_name(w) + levels.level_tag(set) ('' for [0.5]: that arm runs the level code
    with one level and must equal chain2's arm); contrasts as chain2, plus <arm>_minus_<plain arm of w> when both are in the call."""
    import levels as LV
    ws = parse_weights(weights, require_base); cmb = parse_combined(combined)      # chainF: require_base=False = the opt-in no-w25 layout
    if cmb is not None and levelsets is not None: raise ValueError('combined and levels are exclusive')
    if cmb is not None and COMBINED_REF not in [arm_name(w) for w in ws]: raise ValueError(f'combined: the reference arm {COMBINED_REF} must be in the call')
    if levelsets is None: spec = [(arm_name(w), w, None) for w in ws]
    else:
        sets = [list(x) for x in levelsets]
        assert sets and all(s == sorted(s, reverse=True) and len(set(s)) == len(s) for s in sets) and len({tuple(s) for s in sets}) == len(sets), ('LEVELSETS', sets)
        spec = [(arm_name(w), w, None) if w == BASE_WEIGHT else (arm_name(w) + LV.level_tag(s), w, s) for w in ws for s in ([None] if w == BASE_WEIGHT else sets)]
    if cmb is not None: spec = spec + [(COMBINED_ARM, COMBINED_WEIGHT, None)]
    arms = [a for a, _, _ in spec]; assert len(set(arms)) == len(arms) and all('_minus_' not in a for a in arms)
    base_of = {a: arm_name(w) for a, w, _ in spec}
    arm_contrasts = [f'{a}_minus_pcombo' for a in arms] + [f'{a}_minus_{BASE_ARM}' for a in arms if a != BASE_ARM and BASE_ARM in arms]   # chainF: no w25 arm, no w25 contrasts
    arm_contrasts += [f'{a}_minus_{base_of[a]}' for a in arms if base_of[a] != a and base_of[a] in arms and base_of[a] != BASE_ARM]
    out = dict(WEIGHTS=ws, ARMS=FIXED_ARMS + arms, PASS2_ARMS=['inc_pcand_v1', 'pcombo'] + arms, SELECTOR_ARMS=['pcombo'] + arms, NEW_VARIANTS=arms,
                VARIANT=dict({a: 'stored' for a in FIXED_ARMS}, **{a: a for a in arms}), WEIGHT={a: w for a, w, _ in spec}, ARM_CONTRASTS=arm_contrasts,
                CONTRASTS=[(n, *n.split('_minus_')) for n in arm_contrasts + REPLAY_CONTRASTS], LEVELS={a: s for a, _, s in spec}, BASE_OF=base_of,
                LEVELSETS=None if levelsets is None else [list(x) for x in levelsets])
    if cmb is not None: out['COMBINED'] = cmb       # chain2s: only with the option (chain2m's layout dict otherwise)
    return out


def configure(weights, levelsets=None, combined=None, *, require_base=True):
    """Set this module's arm layout (ARMS, PASS2_ARMS, SELECTOR_ARMS, NEW_VARIANTS, VARIANT, WEIGHT, CONTRASTS, LEVELS, ...) for one call."""
    x = layout(weights, levelsets, combined, require_base=require_base); globals().update(x); globals()['COMBINED'] = x.get('COMBINED')   # chain2s: COMBINED or None


def multi_level(arm):
    """chain2m: True when the arm's candidates come from more than one level (it then has the feature 'level')."""
    s = LEVELS.get(arm) if arm in NEW_VARIANTS else None
    return s is not None and len(s) > 1


def arm_feature_names(names68, arm):
    """The selector columns of an arm: the 68 of pcombo, plus 'level' for a multi-level arm."""
    return list(names68) + (['level'] if multi_level(arm) else [])


configure(DEFAULT_WEIGHTS)


def parse_token(token):
    """One fold token: 'runA' or 'runA+runB+...' (a1native artifacts run ids, one per map set). 'stored' is not supported."""
    if not isinstance(token, str) or not token: raise ValueError('empty fold token')
    runs = token.split('+')
    for r in runs:
        if not RUN_ID.match(r) or r == 'stored' or '..' in r: raise ValueError(f'fold token {token!r}: bad run id {r!r}')
    if len(set(runs)) != len(runs): raise ValueError(f'fold token {token!r}: repeated run id')
    return runs


def fold_sources(tokens):
    """Five fold tokens -> {k: [run ids]}; every fold has the same number of map sets (set j = the j-th run id of each token)."""
    tokens = list(tokens)
    if len(tokens) != 5: raise ValueError('five fold tokens required')
    src = {k: parse_token(t) for k, t in enumerate(tokens)}
    if len({len(v) for v in src.values()}) != 1: raise ValueError('every fold needs the same number of map sets')
    return src


def n_sets(src): return len(src[0])


def mean_levels(maps):
    """The MF probability (float32) of n uint16 maps (levels <= 510) of one image: the integer level sum / (510 n), one float32
    division, no requantisation. n = 1 gives map / 510 bit for bit (the stored maps' rule and chain CH1 arm new)."""
    assert len(maps) >= 1
    a = [np.asarray(m) for m in maps]
    for m in a: assert m.dtype == np.uint16 and m.shape == a[0].shape and int(m.max(initial=0)) <= DENOMINATOR
    total = a[0].astype(np.uint32)
    for m in a[1:]: total = total + m
    return total.astype(np.float32) / np.float32(DENOMINATOR * len(a))


UNET_DENOMINATOR = 255
UNET_VIEWS = ('full', 'local')
SET_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def parse_set_spec(spec):
    """chain2u (order CH3): a U-Net set spec 'a' or 'a+b+...' (set names, chain2u/sets/<name>.json); no repeats."""
    if not isinstance(spec, str) or not spec: raise ValueError('empty U-Net set spec')
    names = spec.split('+')
    for n in names:
        if not SET_NAME.match(n) or '..' in n: raise ValueError(f'U-Net set spec {spec!r}: bad set name {n!r}')
    if len(set(names)) != len(names): raise ValueError(f'U-Net set spec {spec!r}: repeated set')
    return names


def mean_unet(maps):
    """chain2u (order CH3): the U-Net probability (float32) of n uint8 maps (levels rint(p 255)) of one image: integer level
    sum / (255 n), one float32 division, no requantisation (as mean_levels for MF). n = 1 gives map / 255 bit for bit (the
    stored maps' rule in worker.branch_maps)."""
    assert len(maps) >= 1
    a = [np.asarray(m) for m in maps]
    for m in a: assert m.dtype == np.uint8 and m.shape == a[0].shape
    total = a[0].astype(np.uint32)
    for m in a[1:]: total = total + m
    return total.astype(np.float32) / np.float32(UNET_DENOMINATOR * len(a))


def variant_mf(variant, z):
    """The weight arm's MF map (the same for every weight arm): z = dict(new=[uint16 maps of the image, one per set])."""
    if variant not in NEW_VARIANTS: raise KeyError(variant)
    return mean_levels(z['new'])


def mix_w(maps, w):
    """w * MF + (1 - w) / 2 * full + (1 - w) / 2 * local: contract.mix's flat insertion-ordered float32 sum with the weights
    (w, (1 - w) / 2, (1 - w) / 2); w = 0.25 gives contract.mix(maps, 'incumbent') bit for bit."""
    if not 0 < w < 1: raise ValueError(f'weight {w}')
    u = (1 - w) / 2
    return sum(np.float32(x) * maps[k] for k, x in (('m2f_public', w), ('full', u), ('local', u)))


def mixture(maps, variant):
    """The candidate/repair mixture of a map variant: 'stored' -> the shipped mixture (contract.mix 'incumbent')."""
    if variant == 'stored':
        from contract import mix
        return mix(maps, 'incumbent')
    if COMBINED is not None and variant == COMBINED_ARM: return maps['m2f_public'] * np.float32(COMBINED_WEIGHT)   # chain2s: the MF map
    return mix_w(maps, WEIGHT[variant])


def size_bin(area):
    return 'lt256' if area < 256 else ('256to1024' if area <= 1024 else 'gt1024')


def size_counts(qs, gt_areas, pred_areas):
    """qs: per annotation set, an IoU matrix [n_gt, n_pred]; match = IoU > 0.5 (PQ). Returns, summed over sets, TP and FN
    of GT instances by GT area bin and FP (predictions matched by no GT of the set) by predicted area bin."""
    out = {c: {b: 0 for b in SIZE_BINS} for c in ('tp', 'fn', 'fp')}
    for q, ga in zip(qs, gt_areas):
        q = np.asarray(q, float).reshape(len(ga), len(pred_areas)); hit = q > .5
        for i, a in enumerate(ga): out['tp' if hit[i].any() else 'fn'][size_bin(a)] += 1
        for j, a in enumerate(pred_areas):
            if not hit[:, j].any(): out['fp'][size_bin(a)] += 1
    return out


def feature_names(order39, ctx):
    """pcombo's 68 columns: fresh_ctrl's 54, then the 6 neighbour features, then the 8 MF features."""
    return PC.feature_names(order39, ctx, 'pcombo')


def gate707(result, contrast=GATE_CONTRAST):
    """Descriptive (decides nothing here): the sibling lanes' gate rule. Blocking checks passed; all707 paired group CI95
    lower > 0; >= 16/20 positive; each base-fold mean paired delta > 0."""
    try:
        if any(result[k] is not True for k in BLOCKING): return False
        S = result['subsets']
        if S['all707']['images'] != 707 or not R._contrast_ok(S['all707']['contrasts'][contrast]): return False
        for k in range(5):
            d = S[f'fold{k}']['contrasts'][contrast]['mean_paired_partition_delta']
            if not np.isfinite(d) or d <= 0: return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def final_verdict(gate, all_checks_passed):
    """A verdict only if every blocking check passed in the kernel and in the local audit (the exact pcombo replay)."""
    if not all_checks_passed: return 'no verdict: replay, reproduction, input or integrity failure'
    return 'confirmed' if gate else 'Improvement could not be confirmed'
