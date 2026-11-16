"""PC1 (refit the P-cand v1 repair per MF-weight arm): pure helpers. No fit, no transport.

For one chain2 label and some of its weight arms v (w<NN>), four arms are scored on the arm's own candidates and the arm's
own pass-2 keeps (chain2's 68-feature selector of arm v, read from chain2's kernel output):
  <v>_raw     the kept candidates, unrepaired (descriptive)
  <v>_frozen  P-cand v1 with chain2's taus: pcombo's 100 outer v1 regressors fitted on the stored-map bank (= chain2 arm v; replay)
  <v>_refit   P-cand v1 with 100 outer v1 regressors refit on arm v's own bank (same outer-training images, row rule, 64
              features, grid, tie rule, HistGradientBoostingRegressor(random_state=42) defaults)
  <v>_gtau    one global tau per outer fit (s, k): the grid tau with the best pooled PQ of the uniform repair of the saved C5
              (s, k) keeps on the outer-training images of arm v's bank (the v1 row rule); every kept candidate gets that tau
and pcombo (chain2's pcombo counts, read). Contrasts in chain2's format (delta, CI95, positives/20, fold means), reported on
all707 and on 'no_fold2' (the 566 images outside base fold 2).
"""
import numpy as np
from pycocotools import mask as mu
import pcand as C
import pnat
import rules
import mr as S

GRID = C.GRID
VARIANTS = ('raw', 'frozen', 'refit', 'gtau')


def encode(ms): return [mu.encode(np.asfortranarray(m, dtype=np.uint8)) for m in ms]


def ious(gts, pred):
    return [np.asarray(mu.iou(g, pred, [0] * len(pred))).reshape(len(g), len(pred)) if pred else np.zeros((len(g), 0)) for g in gts]


def total_counts(qs): return sum((rules.counts(q) for q in qs), start=np.zeros(6))


# ---------------------------------------------------------------- the 9-tau table of one kept set
def tau_table(seeds, p, gts, prep=None):
    """Uniform whole-kept-set repair (pcand.repair_taus, every candidate the same tau) at each grid tau, scored against the
    nonempty annotation sets gts (lists of RLEs): per candidate the v1 target score (pcand.candidate_scores; rows of the
    v1 regressor) and, per tau, the six PQ counts of the repaired set (the global-tau chooser). An empty kept set gives no
    scores and, per tau, the counts of an empty prediction (every GT unmatched). tau_path_equal: the TAU column equals
    pnat.repair (the unchanged P-nat path)."""
    if not seeds:
        c = total_counts(ious(gts, [])).tolist()
        return dict(scores=[], counts=[list(c) for _ in GRID], tau_path_equal=True)
    pr = C.prepare(seeds, p) if prep is None else prep; Sc = np.zeros((len(seeds), len(GRID))); counts = []; equal = True
    for ti, t in enumerate(GRID):
        out, _ = C.repair_taus(seeds, p, [t] * len(seeds), pr)
        q = ious(gts, encode(out)); Sc[:, ti] = C.candidate_scores(q, len(seeds)); counts.append(total_counts(q).tolist())
        if t == C.TAU:
            ref, _ = pnat.repair(seeds, p); equal = all(np.array_equal(a, b) for a, b in zip(out, ref))
    return dict(scores=Sc.tolist(), counts=counts, tau_path_equal=bool(equal))


# ---------------------------------------------------------------- the global tau
def pool_counts(table, pattern_of, key, train):
    """table[f][i]: per grid tau the six counts of image f's keep pattern i; pattern_of[f][key]: the pattern of fit key's saved
    C5 selector on f. Sum over the training images only (images with nothing kept add their FN)."""
    out = [np.zeros(6) for _ in GRID]
    for f in train:
        rows = table[f][pattern_of[f][key]]; assert len(rows) == len(GRID)
        for j, c in enumerate(rows): out[j] = out[j] + np.asarray(c, float)
    return out


def choose_global(pooled):
    """The grid tau with the highest pooled PQ; ties as pcand.choose_target (TAU first, then nearer to TAU, then lower)."""
    assert len(pooled) == len(GRID)
    v = [rules.pq(c) for c in pooled]; top = max(v)
    return float(min((t for t, x in zip(GRID, v) if x == top), key=C._order))


# ---------------------------------------------------------------- v1 rows (tv.py / pcand_tau's rule on a bank)
def training_rows(A, train, key, names):
    """Rows of fit key: the candidates of each training image's keep pattern A[f]['pattern_of'][key]; target = the best grid tau
    of the uniform whole-kept-set repair (pcand.choose_target); never-matched rows excluded."""
    X, y, excluded = [], [], 0
    for f in train:
        i = A[f]['pattern_of'][key]; pattern = A[f]['patterns'][i]
        for r, j in enumerate(pattern):
            t = C.choose_target(A[f]['scores'][i][r])
            if t is None: excluded += 1; continue
            X.append([A[f]['features'][j][n] for n in names]); y.append(t)
    return np.asarray(X, float).reshape(len(X), len(names)), np.asarray(y, float), excluded


def all_rows(A, f, names):
    return np.asarray([[r[n] for n in names] for r in A[f]['features']], float).reshape(A[f]['n'], len(names))


def keep_of(m, x):
    """The package's quality.apply_quality arithmetic (tv.keep_of / pcand_ship fit_worker.keep_of, unchanged): keep, probability."""
    from scipy.special import expit
    z = (np.asarray(x, float) - np.asarray(m['scaler_mean'])) / np.asarray(m['scaler_scale'])
    probability = expit(z @ np.asarray(m['coef']).T + np.asarray(m['intercept'])).ravel()
    assert np.isfinite(probability).all()
    return probability >= m['threshold'], probability


# ---------------------------------------------------------------- arms, contrasts, cohorts
def arm(v, variant): return f'{v}_{variant}'


def arms_of(vs): return ['pcombo'] + [arm(v, x) for v in vs for x in VARIANTS]


def contrasts_of(vs):
    """Per arm v: the two asked contrasts first (refit and global tau vs frozen v1), then each vs pcombo, frozen vs pcombo
    (= chain2's v_minus_pcombo, a replay check), refit vs global tau, and the three repair gains over the unrepaired keeps."""
    out = []
    for v in vs:
        a = {x: arm(v, x) for x in VARIANTS}
        pairs = [(a['refit'], a['frozen']), (a['gtau'], a['frozen']), (a['refit'], 'pcombo'), (a['gtau'], 'pcombo'), (a['frozen'], 'pcombo'),
                 (a['refit'], a['gtau']), (a['frozen'], a['raw']), (a['refit'], a['raw']), (a['gtau'], a['raw'])]
        out += [(f'{x}_minus_{y}', x, y) for x, y in pairs]
    return out


def cohorts(files, folds, query425):
    """chain2's nine cohorts plus no_fold2 (every image outside base fold 2: the fold the recipes were screened on)."""
    c = S.cohorts(files, folds, query425); c['no_fold2'] = [f for f in files if folds[f] != 2]
    return c


def summarize(per, query, files, groups, cfg, arms, contrasts):
    """chain2's worker.summarize with arms and contrasts as arguments: one group draw per bootstrap step shared by every arm,
    so a contrast's statistics do not depend on which other arms are summarized."""
    ix = [files.index(f) for f in query]; seeds = cfg['partition_seeds']
    values = {s: {a: np.asarray(per[s][a])[ix] for a in arms} for s in seeds}
    means = {a: np.mean([values[s][a] for s in seeds], axis=0) for a in arms}
    gid = {f: min(g) for g in groups for f in g}; gids = sorted({gid[f] for f in query}); indices = [[i for i, f in enumerate(query) if gid[f] == g] for g in gids]
    mats = {a: np.stack([c[ix].sum(0) for ix in indices]) for a, c in means.items()}; rng = np.random.default_rng(cfg['bootstrap']['seed']); boots = {a: [] for a in arms}
    for _ in range(cfg['bootstrap']['draws']):
        ids = rng.integers(len(gids), size=len(gids))
        for a in arms: boots[a].append(rules.pq(mats[a][ids].sum(0)))
    scores = {a: rules.pq(c.sum(0)) for a, c in means.items()}; partition_pq = {a: [rules.pq(values[s][a].sum(0)) for s in seeds] for a in arms}; out = {}
    for name, a, b in contrasts:
        ds = np.asarray(partition_pq[a]) - partition_pq[b]
        out[name] = dict(delta=scores[a] - scores[b], mean_paired_partition_delta=float(ds.mean()), positive_of20=int((ds > 0).sum()), partition_deltas=ds.tolist(),
                         **S.intervals(np.asarray(boots[a]) - boots[b]))
    return dict(images=len(query), groups=len(gids), scores=scores, mean_counts={a: c.sum(0).tolist() for a, c in means.items()}, partition_pq=partition_pq, contrasts=out)


def report_row(subsets, name):
    """chain2's report format for one contrast, on all707 and on no_fold2: delta, CI95, positives/20, fold means."""
    out = {}
    for cohort, ks in (('all707', range(5)), ('no_fold2', (0, 1, 3, 4))):
        x = subsets[cohort]['contrasts'][name]
        out[cohort] = dict(delta=x['delta'], CI95=x['paired_group_CI95'], positive_of20=x['positive_of20'], mean_paired_partition_delta=x['mean_paired_partition_delta'],
                           fold_means={f'fold{k}': subsets[f'fold{k}']['contrasts'][name]['mean_paired_partition_delta'] for k in ks})
    return out
