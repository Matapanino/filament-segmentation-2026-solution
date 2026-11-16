"""Saved full707 original-only LR39 deployment state, without sklearn/refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from .features import features_for_mask


def load_quality(path):
    obj=json.loads(Path(path).read_text())
    models=obj.get('models',[obj]);selected=[m for m in models if m.get('role')=='deployment_full707_original_only']
    if len(selected)!=1:raise ValueError('Exactly one deployment quality model is required')
    m=selected[0]
    if m['classes']!=[0,1] or len(m['feature_order'])!=39 or m['threshold']!=.35:raise ValueError('Wrong deployment quality recipe')
    return m


def apply_quality(masks,full,local,raw,model):
    probs={'full':full.astype(np.float32)/255,'local':local.astype(np.float32)/255}
    probs['blend']=(probs['full']+probs['local'])*.5
    rows=[features_for_mask(mask,probs,raw) for mask in masks]
    x=pd.DataFrame(rows)[model['feature_order']].to_numpy(float) if rows else np.empty((0,39))
    if not np.isfinite(x).all():raise ValueError('Nonfinite quality features')
    z=(x-np.asarray(model['scaler_mean']))/np.asarray(model['scaler_scale'])
    probability=expit(z@np.asarray(model['coef']).T+np.asarray(model['intercept'])).ravel()
    if not np.isfinite(probability).all():raise ValueError('Nonfinite quality probability')
    return rows,x,probability,probability>=model['threshold']
