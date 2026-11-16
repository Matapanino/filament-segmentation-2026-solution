"""Refit selector: keep/drop from the 39 quality + 15 context features of the unrepaired candidates.

The cross-validated fresh_ctrl arm (detect_20261112/presel): one StandardScaler + LogisticRegression(C=1, lbfgs,
max_iter=2000, random_state=42) fitted on the MF-mixture SHIP_R3 candidates of all 707 training images, labels by the
per-annotation-set IoU > 0.5 rule, cutoff 0.375 (the median of the 100 nested CV cutoffs). It replaces the shipped
LR39 keep; the repair stage that follows (P-nat or P-cand) is unchanged.

The probability uses the shipped apply_quality arithmetic, so the shipped LR39 (lr39) run through this stage gives
the old keep bit for bit.

A second refit role, the cross-validated P-neigh arm (detect_20261112/pneigh), reads the same 54 features plus the 6
neighbour-context features of neighbour.py (60), with the same recipe, training images and cutoff 0.375 (the median of
its 100 nested CV cutoffs).

A third refit role, the cross-validated MF-rich arm (detect_20261112/mfrich), reads the same 54 features plus the 8
MF-derived features of mfrich_features.py (62) on the three-fold MF mean, with the same recipe, training images and cutoff
0.375 (the median of its 100 nested CV cutoffs).

A fourth refit role, the cross-validated P-combo arm (detect_20261112/pcombo), reads the 54 features, then the 6
neighbour features, then the 8 MF-rich features (68), with the same recipe and training images and its own cutoff 0.35
(the median of its 100 nested CV cutoffs).

Pinned selectors are addressed by name through deployed(): 'fresh_ctrl' (the default), 'pneigh', 'mfrich' and 'pcombo'.

The MF-var role (detect_20261113/mfvar) reads the P-combo 68 features on the MF-TTA map, then the 5 view-disagreement
features of view_features.py (73), with the same recipe and training images and its own cutoff 0.3 (the median of mfvar's
100 nested cutoffs; sorted values 50 and 51 are both 0.3). At test the 5 features are the mean over MF0, MF1 and MF2 of
each model's features from its own 8 D4 views (the preregistered rule).
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .context_features import FEATURES as CONTEXT
from .mfrich_features import FEATURES as MFRICH
from .neighbour import FEATURES as NEIGHBOUR
from .pcand import FEATURES as _PCAND_FEATURES
from .view_features import FEATURES as VIEWS

ROLE = 'deployment_refit_fresh_ctrl_full707_mf_mixture'
LR39_ROLE = 'deployment_full707_original_only'
THRESHOLD = 0.375
FEATURES = list(_PCAND_FEATURES[:54])          # 39 quality features, then the 15 context features
assert FEATURES[39:] == list(CONTEXT)
PNEIGH_ROLE = 'deployment_refit_pneigh_full707_mf_mixture'
PNEIGH_FEATURES = FEATURES + list(NEIGHBOUR)   # the 54, then the 6 neighbour features
MFRICH_ROLE = 'deployment_refit_mfrich_full707_mf_mixture'
MFRICH_FEATURES = FEATURES + list(MFRICH)      # the 54, then the 8 MF-rich features
PCOMBO_ROLE = 'deployment_refit_pcombo_full707_mf_mixture'
PCOMBO_FEATURES = FEATURES + list(NEIGHBOUR) + list(MFRICH)   # the 54, then the 6 neighbour, then the 8 MF-rich features
PCOMBO_THRESHOLD = 0.35                        # the median of pcombo's 100 nested CV cutoffs (the other refit roles: 0.375)
# The deployment selector fitted on Kaggle CPU (detect_20261112/refit_ship): two fits gave these exact bytes.
SELECTOR_FILE = Path(__file__).parent / 'data/refit_selector.json'
SELECTOR_SHA256 = '05759a2561e702f1fa4c044c36e32a1e867be15a1f6122451f090edabd2fd2fe'
# The deployment selectors of the later ship lanes, fitted on Kaggle CPU the same way (two fits, identical bytes).
PINNED = {'fresh_ctrl': (SELECTOR_FILE, SELECTOR_SHA256),
          'pneigh': (Path(__file__).parent / 'data/pneigh_selector.json', '12237dd42263df43ae81c894ef585af9d8f82b6910d99dd228349759afe7408c'),
          'mfrich': (Path(__file__).parent / 'data/mfrich_selector.json', 'bf499652dafb7bd2533c63fa2449761454ccd5ee137548d2945b9d8501d78556'),
          'pcombo': (Path(__file__).parent / 'data/pcombo_selector.json', 'fd25d491d828aaac1bdd8c07d120edbe32ac24d8f5980599d00190f7573b389c'),
          # mfmap_ship Stage 2: the P-combo recipe refitted on the candidates of the MF D4-TTA out-of-fold maps (cutoff MFMAP_THRESHOLDS['mftta'])
          'mftta': (Path(__file__).parent / 'data/mftta_selector.json', '89a0c469c78c4a6b0095880560bd18a15f107441ce4169c7c5df4a528ea7a4cf'),
          # mfvar_ship Stage 1a: the P-combo recipe on the MF-TTA candidates with the 5 view features (73), cutoff MFVAR_THRESHOLD
          'mfvar': (Path(__file__).parent / 'data/mfvar_selector.json', '53f4db37e99351451047d2c1c7f9be1377fe1d18332edc864048770aed06cc27')}

# MF-map variant roles (mfmaps.py): the P-combo recipe (68 features in PCOMBO_FEATURES order) refitted on the candidates of
# the variant's out-of-fold MF maps, each with its own cutoff (the median of its CV lane's 100 nested cutoffs). A role is
# usable only once its cutoff and selector file are pinned (MFMAP_THRESHOLDS, PINNED).
MFMAP_ROLES = {v: f'deployment_refit_{v}_full707_mf_mixture' for v in ('mftta', 'mfhires', 'mfcomb')}
MFMAP_THRESHOLDS = {'mftta': 0.35}             # mftta: the median of detect_20261113/mftta's 100 nested cutoffs (values 50 and 51: 0.35)
MFVAR_ROLE = 'deployment_refit_mfvar_full707_mf_mixture'
MFVAR_FEATURES = PCOMBO_FEATURES + list(VIEWS)   # the 68 P-combo features (on the MF-TTA map), then the 5 view features
MFVAR_THRESHOLD = 0.3                          # the median of detect_20261113/mfvar's 100 nested cutoffs (values 50 and 51: 0.3)


# Chain roles (chain.py): the P-combo recipe (68 features in PCOMBO_FEATURES order) refitted, in the same call, on the
# candidates of one chain map (blend weight w); its cutoff is that call's median nested cutoff, a value of the CV lanes'
# threshold grid. The recipe pins the selector file (sha256) and the cutoff; chain.selector checks both.
CHAIN_ROLE = re.compile(r'deployment_refit_chain2_w[0-9]{2}_full707_mf_mixture')
CHAIN_CUTOFF_GRID = tuple(round(i * .025, 6) for i in range(41))


def check(model):
    """Validate a selector state; raises ValueError on any departure from the deployable recipes (fresh_ctrl, P-neigh, MF-rich, P-combo, LR39)."""
    order = list(model.get('feature_order', []))
    if model.get('role') == ROLE:
        if order != FEATURES or model.get('threshold') != THRESHOLD:raise ValueError('Wrong refit selector recipe')
    elif model.get('role') == PNEIGH_ROLE:
        if order != PNEIGH_FEATURES or model.get('threshold') != THRESHOLD:raise ValueError('Wrong P-neigh selector recipe')
    elif model.get('role') == MFRICH_ROLE:
        if order != MFRICH_FEATURES or model.get('threshold') != THRESHOLD:raise ValueError('Wrong MF-rich selector recipe')
    elif model.get('role') == PCOMBO_ROLE:
        if order != PCOMBO_FEATURES or model.get('threshold') != PCOMBO_THRESHOLD:raise ValueError('Wrong P-combo selector recipe')
    elif model.get('role') in MFMAP_ROLES.values():
        variant = next(v for v, r in MFMAP_ROLES.items() if r == model.get('role'))
        if variant not in MFMAP_THRESHOLDS:raise ValueError('The '+variant+' selector cutoff is not pinned')
        if order != PCOMBO_FEATURES or model.get('threshold') != MFMAP_THRESHOLDS[variant]:raise ValueError('Wrong '+variant+' selector recipe')
    elif model.get('role') == MFVAR_ROLE:
        if order != MFVAR_FEATURES or model.get('threshold') != MFVAR_THRESHOLD:raise ValueError('Wrong MF-var selector recipe')
    elif CHAIN_ROLE.fullmatch(str(model.get('role'))):
        if order != PCOMBO_FEATURES or model.get('threshold') not in CHAIN_CUTOFF_GRID:raise ValueError('Wrong chain selector recipe')
    elif model.get('role') == LR39_ROLE:
        if len(order) != 39 or model.get('threshold') != .35:raise ValueError('Wrong LR39 recipe')
    else:
        raise ValueError('Unknown selector role')
    if model.get('classes') != [0, 1]:raise ValueError('Selector classes must be [0, 1]')
    n = len(order)
    if len(model['scaler_mean']) != n or len(model['scaler_scale']) != n or np.shape(model['coef']) != (1, n) or len(model['intercept']) != 1:
        raise ValueError('Selector shapes do not match its feature order')
    return model


def lr39(quality_model):
    """The shipped LR39 deployment model as a selector state (switching back gives the old path)."""
    return dict(quality_model, role=LR39_ROLE, classes=quality_model.get('classes', [0, 1]))


def load_selector(path, sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:raise ValueError('Selector file hash differs from the pinned value')
    return check(json.loads(data))


def deployed(name='fresh_ctrl'):
    """(path, sha256) of a pinned deployment selector ('fresh_ctrl', the default, 'pneigh', 'mfrich', 'pcombo', 'mftta' or 'mfvar'), for load_selector."""
    if name not in PINNED:raise ValueError('Unknown pinned selector: '+repr(name))
    return PINNED[name]


def needs_context(model):
    return any(n in CONTEXT for n in model['feature_order'])


def needs_neighbour(model):
    return any(n in NEIGHBOUR for n in model['feature_order'])


def needs_mfrich(model):
    return any(n in MFRICH for n in model['feature_order'])


def needs_views(model):
    return any(n in VIEWS for n in model['feature_order'])


def apply_selector(rows, context_rows, model, neighbour_rows=None, mfrich_rows=None, view_rows=None):
    """rows: the 39 quality-feature dicts of the candidates; context_rows: their 15 context-feature dicts (or None if
    the selector reads no context feature); neighbour_rows: their 6 neighbour-feature dicts, mfrich_rows: their 8
    MF-rich feature dicts and view_rows: their 5 view-disagreement feature dicts (each None if the selector reads none;
    rows a selector does not read are ignored). Returns features, probability and keep (probability >= threshold)."""
    order = model['feature_order']
    if context_rows is None:
        if needs_context(model):raise ValueError('Context features are required by this selector')
        merged = rows
    else:
        if len(context_rows) != len(rows):raise ValueError('One context row per candidate')
        merged = [{**r, **c} for r, c in zip(rows, context_rows)]
    if neighbour_rows is None:
        if needs_neighbour(model):raise ValueError('Neighbour features are required by this selector')
    else:
        if len(neighbour_rows) != len(rows):raise ValueError('One neighbour row per candidate')
        merged = [{**r, **g} for r, g in zip(merged, neighbour_rows)]
    if mfrich_rows is None:
        if needs_mfrich(model):raise ValueError('MF-rich features are required by this selector')
    else:
        if len(mfrich_rows) != len(rows):raise ValueError('One MF-rich row per candidate')
        merged = [{**r, **g} for r, g in zip(merged, mfrich_rows)]
    if needs_views(model):
        if view_rows is None:raise ValueError('View features are required by this selector')
        if len(view_rows) != len(rows):raise ValueError('One view row per candidate')
        merged = [{**r, **g} for r, g in zip(merged, view_rows)]
    # Built exactly as apply_quality builds it (a pandas frame, column-major): the memory layout changes the BLAS path
    # and therefore the last bits of the probability.
    x = pd.DataFrame(merged)[order].to_numpy(float) if merged else np.empty((0, len(order)))
    if not np.isfinite(x).all():raise ValueError('Nonfinite selector features')
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (x - np.asarray(model['scaler_mean'])) / np.asarray(model['scaler_scale'])
        probability = expit(z @ np.asarray(model['coef']).T + np.asarray(model['intercept'])).ravel()
    if not np.isfinite(probability).all():raise ValueError('Nonfinite selector probability')
    return x, probability, probability >= model['threshold']
