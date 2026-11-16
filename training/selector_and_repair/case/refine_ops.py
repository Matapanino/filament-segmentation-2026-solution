"""w6_refine (R6-B) crop geometry, targets, paste-back, P-cal and calibration ops (numpy + cv2 only; no fitting of networks).

Binding text: protocol.json (instances_and_crops, paste_back, members_read_per_fold_seed, amendment_review_2026-11-03).

Coordinates: x = column, y = row, pixel centres at integers (cv2 convention). A kept instance c (bool 2048 x 2048) gets
  mu = centroid, e1 = largest eigenvector of the coordinate covariance (sign: e1.x > 0, or e1.y > 0 if e1.x == 0),
  e2 = e1 rotated by +90 deg (-e1.y, e1.x), u = (x - mu).e1, v = (x - mu).e2, l / w = extents of u / v,
  m = mu + mid(u) e1 + mid(v) e2, Lc = l + 2 max(0.25 l, 32), Hc = w + 2 max(0.25 w, 32), s = min(1, 1024 / Lc, 256 / Hc),
  canvas = s R(-theta)(x - m) + (512, 128), i.e. A = [[s e1.x, s e1.y, 512 - s m.e1], [s e2.x, s e2.y, 128 - s m.e2]].
Native window W = {x : |(x - m).e1| <= Lc / 2 and |(x - m).e2| <= Hc / 2}. Valid canvas pixel = preimage inside the image
[0, 2047]^2 and inside the window.
"""
import cv2
import numpy as np

CH, CW = 256, 1024
CX, CY = 512., 128.
N = 2048
R_SEP = 3


# ------------------------------------------------------------------ geometry
def geometry(mask):
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(np.float64)
    mu = P.mean(0); Q = P - mu
    C = Q.T @ Q / len(P)
    w, V = np.linalg.eigh(C); e1 = V[:, int(np.argmax(w))].copy()
    if e1[0] < 0 or (e1[0] == 0 and e1[1] < 0): e1 = -e1
    e2 = np.array([-e1[1], e1[0]])
    u = Q @ e1; v = Q @ e2
    l = float(u.max() - u.min()); wd = float(v.max() - v.min())
    m = mu + (u.max() + u.min()) / 2 * e1 + (v.max() + v.min()) / 2 * e2
    Lc = l + 2 * max(.25 * l, 32.); Hc = wd + 2 * max(.25 * wd, 32.)
    s = min(1., CW / Lc, CH / Hc)
    A = np.array([[s * e1[0], s * e1[1], CX - s * (m @ e1)], [s * e2[0], s * e2[1], CY - s * (m @ e2)]], np.float64)
    return dict(m=m, e1=e1, e2=e2, l=l, w=wd, Lc=Lc, Hc=Hc, s=s, A=A)


def geom_vector(g):
    """(9,) float64: m.x, m.y, e1.x, e1.y, Lc, Hc, s, l, w."""
    return np.array([g['m'][0], g['m'][1], g['e1'][0], g['e1'][1], g['Lc'], g['Hc'], g['s'], g['l'], g['w']], np.float64)


def geom_from_vector(v):
    m = np.array(v[0:2]); e1 = np.array(v[2:4]); e2 = np.array([-e1[1], e1[0]]); Lc, Hc, s = float(v[4]), float(v[5]), float(v[6])
    A = np.array([[s * e1[0], s * e1[1], CX - s * (m @ e1)], [s * e2[0], s * e2[1], CY - s * (m @ e2)]], np.float64)
    return dict(m=m, e1=e1, e2=e2, Lc=Lc, Hc=Hc, s=s, l=float(v[7]), w=float(v[8]), A=A)


def compose_error(g, pts=None):
    """max |A^-1(A x) - x| over test points (G_geometry)."""
    A = g['A']; Ai = cv2.invertAffineTransform(A)
    if pts is None:
        pts = np.array([[0, 0], [2047, 0], [0, 2047], [2047, 2047], [1023.5, 777.25], g['m']], np.float64)
    y = pts @ A[:, :2].T + A[:, 2]; x = y @ Ai[:, :2].T + Ai[:, 2]
    return float(np.abs(x - pts).max())


def window_bbox(g):
    """Native bbox (y0, y1, x0, x1), half-open, of the window clipped to the image."""
    m, e1, e2 = g['m'], g['e1'], g['e2']; a, b = g['Lc'] / 2, g['Hc'] / 2
    c = np.array([m + sa * a * e1 + sb * b * e2 for sa in (-1, 1) for sb in (-1, 1)])
    x0 = max(0, int(np.floor(c[:, 0].min())) - 1); x1 = min(N, int(np.ceil(c[:, 0].max())) + 2)
    y0 = max(0, int(np.floor(c[:, 1].min())) - 1); y1 = min(N, int(np.ceil(c[:, 1].max())) + 2)
    return y0, y1, x0, x1


def window_native(g, bbox):
    y0, y1, x0, x1 = bbox
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dx = xx - g['m'][0]; dy = yy - g['m'][1]
    u = dx * g['e1'][0] + dy * g['e1'][1]; v = dx * g['e2'][0] + dy * g['e2'][1]
    return (np.abs(u) <= g['Lc'] / 2) & (np.abs(v) <= g['Hc'] / 2)


def canvas_valid(g):
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float64)
    a = (xx - CX) / g['s']; b = (yy - CY) / g['s']
    px = g['m'][0] + a * g['e1'][0] + b * g['e2'][0]; py = g['m'][1] + a * g['e1'][1] + b * g['e2'][1]
    inside = (px >= 0) & (px <= N - 1) & (py >= 0) & (py <= N - 1)
    win = (np.abs(xx - CX) <= g['s'] * g['Lc'] / 2) & (np.abs(yy - CY) <= g['s'] * g['Hc'] / 2)
    return inside & win


def to_canvas(native, g):
    """Native float array (2048 x 2048) -> canvas (256 x 1024), bilinear, border 0."""
    return cv2.warpAffine(np.ascontiguousarray(native, np.float32), g['A'], (CW, CH), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def from_canvas(canvas, g, bbox):
    """Canvas array -> native values on the bbox (bilinear, WARP_INVERSE_MAP, border 0)."""
    y0, y1, x0, x1 = bbox; A = g['A']
    Mb = np.concatenate([A[:, :2], (A[:, :2] @ np.array([x0, y0], np.float64) + A[:, 2])[:, None]], 1)
    return cv2.warpAffine(np.ascontiguousarray(canvas, np.float32), Mb, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------------------ crops of one image
def image_crops(masks, keep_local, img_u8, p):
    """Crops of the kept instances of one image. masks: list of bool 2048 masks (all candidates of the image, PC order);
    keep_local: bool per candidate. Returns list of dict(local, geom, X (4, H, W) float16, valid bool)."""
    kept = [j for j in range(len(masks)) if keep_local[j]]
    union = np.zeros((N, N), bool)
    for j in kept: union |= masks[j]
    img = img_u8.astype(np.float32) / 255
    out = []
    for j in kept:
        g = geometry(masks[j]); other = union & ~masks[j]
        X = np.stack([to_canvas(img, g), to_canvas(p, g), to_canvas(masks[j].astype(np.float32), g), to_canvas(other.astype(np.float32), g)]).astype(np.float16)
        out.append(dict(local=j, geom=g, X=X, valid=canvas_valid(g)))
    return out


def target_native(gt_masks):
    """T(x) = mean over the drawing sets of 1[x in g_a]."""
    return np.mean(np.stack([m.astype(np.float32) for m in gt_masks]), 0)


def drawn_by(entries, local, thr=.2):
    """D(c) from the control's IoU records of one file: list of (set index in entries, gt row) with IoU(c, g_a) >= thr.
    entries = records[f]['annotations'] (sets with at least one GT, PC candidate layout)."""
    out = []
    for a, e in enumerate(entries):
        if e['n_gt'] == 0: continue
        iou = np.asarray(e['iou'], float).reshape(e['n_gt'], -1)
        col = iou[:, local]; g = int(np.argmax(col))
        if col[g] >= thr: out.append((a, g))
    return out


# ------------------------------------------------------------------ paste-back
def separation(others_bbox_padded, pad, shape):
    """B in bbox coords: other kept pixels and every pixel within Euclidean distance R_SEP of them."""
    if not others_bbox_padded.any(): return np.zeros(shape, bool)
    d = cv2.distanceTransform((~others_bbox_padded).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    B = d <= R_SEP
    py0, px0 = pad
    return B[py0:py0 + shape[0], px0:px0 + shape[1]]


def image_context(seeds_full, bboxes):
    """Per kept instance: seed in bbox, B in bbox (from the other kept instances), window-independent."""
    union = np.zeros((N, N), bool)
    for s_ in seeds_full: union |= s_
    ctx = []
    for i, (s_, bb) in enumerate(zip(seeds_full, bboxes)):
        y0, y1, x0, x1 = bb; Y0, Y1, X0, X1 = max(0, y0 - 4), min(N, y1 + 4), max(0, x0 - 4), min(N, x1 + 4)
        oth = (union & ~s_)[Y0:Y1, X0:X1]
        B = separation(oth, (y0 - Y0, x0 - X0), (y1 - y0, x1 - x0))
        ctx.append(dict(seed=s_[y0:y1, x0:x1], B=B, bbox=bb))
    return ctx


def components_touching(R, seed):
    if not R.any(): return R
    k, lab = cv2.connectedComponents(R.astype(np.uint8), connectivity=8)
    hit = np.unique(lab[seed & R]); hit = hit[hit > 0]
    return np.isin(lab, hit) & R


def paste_back(ctx, R_list, grow=False):
    """R_list[i] = bool bbox array {x in W_i : q_i(x) >= tau}. grow=False: R_i = (R_i & c_i) | (R_i - c_i - B_i);
    grow=True (P-grow): R_i = c_i | (R_i - B_i). Contested non-seed pixels are removed from both; M_i = components of R_i
    that intersect c_i; empty -> c_i. Returns (list of bbox-local M_i, n_fallback)."""
    cnt = np.zeros((N, N), np.uint8); new = []
    for c, R in zip(ctx, R_list):
        nw = R & ~c['seed'] & ~c['B']; new.append(nw)
        y0, y1, x0, x1 = c['bbox']; cnt[y0:y1, x0:x1] += nw.astype(np.uint8)
    out = []; fb = 0
    for c, R, nw in zip(ctx, R_list, new):
        y0, y1, x0, x1 = c['bbox']; contested = cnt[y0:y1, x0:x1] >= 2
        nw = nw & ~contested
        Ri = (c['seed'] | nw) if grow else ((R & c['seed']) | nw)
        M = components_touching(Ri, c['seed'])
        if not M.any(): M = c['seed'].copy(); fb += 1
        out.append(M)
    return out, fb


def full(M, bbox):
    y0, y1, x0, x1 = bbox; F = np.zeros((N, N), bool); F[y0:y1, x0:x1] = M; return F


def check_member(ctx, Ms, Q=None, tau=None, W=None, identity=False, strict=True):
    """Protocol invariants on one image member. Q/W bbox-local native q and window (for the new-pixel rule)."""
    ok = dict(count=len(Ms) == len(ctx), nonempty=all(M.any() for M in Ms))
    occ = np.zeros((N, N), np.uint8); seeds = np.zeros((N, N), np.uint8)
    for c, M in zip(ctx, Ms):
        y0, y1, x0, x1 = c['bbox']; occ[y0:y1, x0:x1] += M.astype(np.uint8); seeds[y0:y1, x0:x1] += c['seed'].astype(np.uint8)
    ok['disjoint'] = bool(occ.max() <= 1)
    others_hit = True; newrule = True; comp = True; ident = True
    for i, (c, M) in enumerate(zip(ctx, Ms)):
        y0, y1, x0, x1 = c['bbox']
        sl = seeds[y0:y1, x0:x1].astype(bool) & ~c['seed']   # other kept seeds inside this bbox
        others_hit &= not (M & sl).any()
        if strict:
            nw = M & ~c['seed']
            if nw.any() and Q is not None:
                newrule &= bool((Q[i][nw] >= tau).all() and W[i][nw].all() and not (c['B'] & nw).any())
            k, lab = cv2.connectedComponents(M.astype(np.uint8), connectivity=8)
            comp &= all((lab == q)[c['seed']].any() for q in range(1, k))
        if identity: ident &= bool(np.array_equal(M, c['seed']))
    ok.update(no_other_seed=bool(others_hit), new_pixel_rule=bool(newrule), components_touch_seed=bool(comp), identity=bool(ident))
    return ok


# ------------------------------------------------------------------ P-cal (4-parameter seed-aware logistic map of p)
def logit_p(p):
    q = np.clip(np.asarray(p, np.float64), 1e-3, 1 - 1e-3); return np.log(q / (1 - q))


def pcal_sample(crops, seed):
    """crops: iterable (in key order) of (p_canvas, seed_channel, T, lossmask, nD) for trainable crops.
    Returns l, s, T, w arrays (k = min(4096, n_loss) pixels per crop, weight nD / k)."""
    rng = np.random.default_rng(seed); L, S, Tt, Wt = [], [], [], []
    for p, sd, T, lm, nD in crops:
        idx = np.flatnonzero(lm.ravel()); n = len(idx)
        if n == 0: continue
        k = min(4096, n); pick = idx[rng.choice(n, k, replace=False)]
        L.append(logit_p(p.ravel()[pick].astype(np.float64))); S.append((sd.ravel()[pick] >= .5).astype(np.float64))
        Tt.append(T.ravel()[pick].astype(np.float64)); Wt.append(np.full(k, nD / k))
    return np.concatenate(L), np.concatenate(S), np.concatenate(Tt), np.concatenate(Wt)


def pcal_loss(theta, l, s, T, w):
    a, b, c, d = theta; z = a + b * l + c * s + d * s * l
    sp = np.logaddexp(0, z); loss = (w * (sp - T * z)).sum() / w.sum()
    r = w * (1 / (1 + np.exp(-z)) - T) / w.sum()
    return loss, np.array([r.sum(), (r * l).sum(), (r * s).sum(), (r * s * l).sum()])


def pcal_fit(l, s, T, w):
    from scipy.optimize import minimize
    r = minimize(pcal_loss, np.array([0., 1., 0., 0.]), args=(l, s, T, w), jac=True, method='L-BFGS-B')
    return dict(theta=r.x.tolist(), success=bool(r.success), message=str(r.message), nit=int(r.nit),
                bce_p=float(pcal_loss(np.array([0., 1., 0., 0.]), l, s, T, w)[0]), bce_pcal=float(r.fun), n_pixels=int(len(l)))


def pcal_apply(theta, p_canvas, seed_canvas):
    a, b, c, d = theta; l = logit_p(p_canvas); s = (np.asarray(seed_canvas) >= .5).astype(np.float64)
    return (1 / (1 + np.exp(-(a + b * l + c * s + d * s * l)))).astype(np.float32)


# ------------------------------------------------------------------ calibration
def calib_accumulate(acc, q, T):
    """Accumulate 10 equal-width bins of q: count, sum q, sum T; plus sum (q - T)^2."""
    q = np.asarray(q, np.float64).ravel(); T = np.asarray(T, np.float64).ravel()
    b = np.minimum((q * 10).astype(int), 9)
    acc['n'] += np.bincount(b, minlength=10); acc['sq'] += np.bincount(b, q, 10); acc['sT'] += np.bincount(b, T, 10)
    acc['brier'] += float(((q - T) ** 2).sum()); return acc


def calib_new(): return dict(n=np.zeros(10), sq=np.zeros(10), sT=np.zeros(10), brier=0.)


def calib_report(acc):
    n = acc['n']; tot = n.sum(); mq = np.where(n > 0, acc['sq'] / np.maximum(n, 1), np.nan); mT = np.where(n > 0, acc['sT'] / np.maximum(n, 1), np.nan)
    ece = float(np.nansum(n / max(tot, 1) * np.abs(mq - mT)))
    return dict(bins=[dict(lo=k / 10, hi=(k + 1) / 10, n=int(n[k]), mean_q=None if n[k] == 0 else float(mq[k]), mean_T=None if n[k] == 0 else float(mT[k]))
                      for k in range(10)], ece=ece, brier=float(acc['brier'] / max(tot, 1)), pixels=int(tot))
