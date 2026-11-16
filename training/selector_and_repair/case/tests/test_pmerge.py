"""Synthetic specification tests for P-merge (pmerge.py), written before pmerge.py.

P-merge: for each pair of neighbouring final INC + P-cand v1 objects of an image (minimum pixel distance <= 48 px), a
classifier predicts from prediction-only pair features whether joining them raises PQ; pairs are joined greedily when the
predicted probability is >= 0.5. A join = the union of the two masks plus a 5-px bridge along the straight segment between
their closest points, restricted to pixels with mixture probability >= min(tau_a, tau_b), minus pixels of any other object.
"""
import itertools, sys, unittest
from pathlib import Path
import numpy as np
import cv2
from pycocotools import mask as mu
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen/eval'), str(H)]
import refine_ops as R
import pcand2 as C
import spatial as S
import pmerge as P
from rules import counts as count6, pq
from filseg import postproc
from filseg.instancing import SHIP_R3

N = 2048
TAU = C.TAU


def bar(y0, y1, x0, x1):
    m = np.zeros((N, N), bool); m[y0:y1, x0:x1] = True; return m


def line_mask(p0, p1, th=7):
    m = np.zeros((N, N), np.uint8); cv2.line(m, p0, p1, 1, th); return m.astype(bool)


def obj(full): return P.obj_from_full(full)


def brute_closest(a, b):
    ya, xa = np.nonzero(a); yb, xb = np.nonzero(b)
    d2 = (ya[:, None] - yb[None, :]) ** 2 + (xa[:, None] - xb[None, :]) ** 2
    m = int(d2.min()); ia, ib = np.nonzero(d2 == m)
    best = min((int(ya[i]), int(xa[i]), int(yb[j]), int(xb[j])) for i, j in zip(ia, ib))
    return m, (best[0], best[1]), (best[2], best[3])


def gts_of(sets): return S.decode_gts([[mu.encode(np.asfortranarray(g.astype(np.uint8))) for g in s] for s in sets])


def full_counts(masks, sets):
    pred = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]; tot = np.zeros(6)
    for s in sets:
        g = [mu.encode(np.asfortranarray(x.astype(np.uint8))) for x in s]
        q = np.asarray(mu.iou(g, pred, [0] * len(pred))).reshape(len(g), len(pred)) if pred else np.zeros((len(g), 0))
        tot += count6(q)
    return tot


class Constants(unittest.TestCase):
    def test_fixed_constants(self):
        self.assertEqual(P.MAX_GAP, 48); self.assertEqual(P.BRIDGE_THICKNESS, 5); self.assertEqual(P.MERGE_THRESHOLD, 0.5)
        self.assertEqual(P.PRIMARY, 'inc_pmerge_minus_inc_pcand_v1')
        self.assertEqual(len(P.FEATURES), 18); self.assertEqual(len(set(P.FEATURES)), 18)
        # the linker constants are the shipped R3 rule's and link_collinear's defaults
        self.assertEqual((P.LINKER_GAP, P.LINKER_ANGLE), (SHIP_R3.link_gap, SHIP_R3.link_angle))
        self.assertEqual((P.LINKER_WINDOW, P.LINKER_MIN_AREA), (SHIP_R3.link_window, SHIP_R3.link_min_area))
        self.assertEqual(P.LINKER_MIN_ELONG, 1.0)


class Geometry(unittest.TestCase):
    def test_obj_from_full_is_tight(self):
        m = bar(10, 20, 30, 45); M, bb = obj(m)
        self.assertEqual(bb, (10, 20, 30, 45)); self.assertTrue(M.all())
        self.assertTrue(np.array_equal(R.full(M, bb), m))

    def test_closest_pair_equals_brute_force_with_lexicographic_ties(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            a = np.zeros((N, N), bool); b = np.zeros((N, N), bool)
            for m, cy, cx in ((a, 500, 500), (b, 500 + rng.integers(-40, 40), 540 + rng.integers(0, 30))):
                ys = cy + rng.integers(-12, 12, 40); xs = cx + rng.integers(-12, 12, 40)
                for y, x in zip(ys, xs): m[y - 2:y + 3, x - 2:x + 3] = True
            b &= ~a
            if not b.any(): continue
            exp = brute_closest(a, b)
            if exp[0] > 48 ** 2: continue
            (Ma, ba), (Mb, bb) = obj(a), obj(b)
            got = P.closest_pair(P.boundary(Ma, ba), P.boundary(Mb, bb))
            self.assertEqual(got, exp)

    def test_closest_pair_tie_order_and_cap(self):
        a = bar(100, 110, 100, 110); b = bar(100, 110, 120, 130)          # many pairs at distance 10: lexicographic first
        got = P.closest_pair(P.boundary(*obj(a)), P.boundary(*obj(b)))
        self.assertEqual(got, (121, (100, 109), (100, 120)))
        c = bar(100, 110, 157, 170)                                          # exactly 48 px: a pair
        self.assertEqual(P.closest_pair(P.boundary(*obj(bar(100, 110, 100, 110))), P.boundary(*obj(c)))[0], 48 ** 2)
        d = bar(100, 110, 158, 170)                                          # 49 px: not a pair
        self.assertIsNone(P.closest_pair(P.boundary(*obj(bar(100, 110, 100, 110))), P.boundary(*obj(d))))

    def test_candidate_pairs_match_brute_force(self):
        ms = [bar(100, 110, 100, 200), bar(100, 110, 230, 300), bar(150, 160, 100, 200), bar(300, 310, 100, 200), bar(100, 140, 348, 352)]
        objs = [obj(m) for m in ms]
        got = {(r['a'], r['b']): r['d2'] for r in P.candidate_pairs(objs)}
        exp = {}
        for i, j in itertools.combinations(range(len(ms)), 2):
            d = brute_closest(ms[i], ms[j])[0]
            if d <= 48 ** 2: exp[i, j] = d
        self.assertEqual(got, exp)
        self.assertTrue(all(r['a'] < r['b'] for r in P.candidate_pairs(objs)))

    def test_band_is_crop_invariant_and_the_linker_stroke(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            pa = (int(rng.integers(200, 300)), int(rng.integers(200, 300))); pb = (int(pa[0] + rng.integers(-48, 48)), int(pa[1] + rng.integers(-48, 48)))
            ref = np.zeros((N, N), np.uint8); cv2.line(ref, (pa[1], pa[0]), (pb[1], pb[0]), 1, 5)
            crop = (min(pa[0], pb[0]) - 4, max(pa[0], pb[0]) + 5, min(pa[1], pb[1]) - 4, max(pa[1], pb[1]) + 5)
            got = P.band(pa, pb, crop); self.assertTrue(np.array_equal(R.full(got, crop), ref.astype(bool)))
            line = np.zeros((N, N), np.uint8); cv2.line(line, (pa[1], pa[0]), (pb[1], pb[0]), 1, 1)
            self.assertTrue(np.array_equal(R.full(P.centre_line(pa, pb, crop), crop), line.astype(bool)))


class Merge(unittest.TestCase):
    def scene(self):
        p = np.zeros((N, N), np.float32)
        a = bar(500, 510, 400, 500); b = bar(500, 510, 530, 600); c = bar(490, 499, 505, 525)   # c reaches into the bridge band
        p[500:510, 400:500] = .6; p[500:510, 530:600] = .6; p[497:504, 500:530] = .45          # the gap's upper rows are faint (.45)
        return p, [a, b, c]

    def test_bridge_rules(self):
        p, (a, b, c) = self.scene(); objs = [obj(m) for m in (a, b, c)]
        pr = [r for r in P.candidate_pairs(objs) if (r['a'], r['b']) == (0, 1)][0]
        self.assertEqual(pr['d2'], 31 ** 2); self.assertEqual((pr['pa'], pr['pb']), ((500, 499), (500, 530)))
        (M, bb), info = P.merge_objs(objs, 0, 1, pr['pa'], pr['pb'], 0.45, p)
        F = R.full(M, bb); cr = P.crop_of([objs[0][1], objs[1][1]]); band = R.full(P.band(pr['pa'], pr['pb'], cr), cr)
        self.assertTrue((F & a).sum() == a.sum() and (F & b).sum() == b.sum())        # union kept
        new = F & ~a & ~b
        self.assertTrue(new.any()); self.assertTrue((new <= band).all()); self.assertTrue((p[new] >= np.float32(.45)).all())
        self.assertFalse((F & c).any())                                                 # never another object's pixel
        self.assertTrue(info['connected']); self.assertEqual(info['bridge_pixels'], int(new.sum()))
        # threshold above the gap: the merged object is the plain union, one instance, not connected
        (M2, bb2), info2 = P.merge_objs(objs, 0, 1, pr['pa'], pr['pb'], 0.5, p)
        self.assertTrue(np.array_equal(R.full(M2, bb2), a | b)); self.assertFalse(info2['connected']); self.assertEqual(info2['bridge_pixels'], 0)

    def test_full_path_equals_local_path(self):
        p, ms = self.scene(); objs = [obj(m) for m in ms]
        pr = [r for r in P.candidate_pairs(objs) if (r['a'], r['b']) == (0, 1)][0]
        (M, bb), _ = P.merge_objs(objs, 0, 1, pr['pa'], pr['pb'], 0.45, p)
        out, infos = P.apply_merges(ms, [(0, 1, pr['pa'], pr['pb'], 0.45)], p)
        self.assertEqual(len(out), 2); self.assertTrue(np.array_equal(out[0], R.full(M, bb))); self.assertTrue(np.array_equal(out[1], ms[2]))
        self.assertEqual(P.overlap_pixels(out), 0)

    def test_no_merges_is_identity(self):
        p, ms = self.scene(); out, infos = P.apply_merges(ms, [], p)
        self.assertEqual(len(out), 3); self.assertTrue(all(o is m or np.array_equal(o, m) for o, m in zip(out, ms))); self.assertEqual(infos, [])

    def test_crossing_bridges_never_overlap(self):
        p = np.full((N, N), .9, np.float32)
        a = bar(500, 504, 400, 500); b = bar(530, 534, 530, 630)        # bridge a-b runs diagonally down-right
        c = bar(530, 534, 400, 500); d = bar(500, 504, 530, 630)        # bridge c-d crosses it
        ms = [a, b, c, d]; objs = [obj(m) for m in ms]
        prs = {(r['a'], r['b']): r for r in P.candidate_pairs(objs)}
        r1, r2 = prs[0, 1], prs[2, 3]
        out, infos = P.apply_merges(ms, [(0, 1, r1['pa'], r1['pb'], .5), (2, 3, r2['pa'], r2['pb'], .5)], p)
        self.assertEqual(len(out), 2); self.assertEqual(P.overlap_pixels(out), 0)
        self.assertTrue(infos[0]['bridge_pixels'] > 0 and infos[1]['bridge_pixels'] > 0)

    def test_select_merges_greedy(self):
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        # .95 (1,2) first; .9 ties (0,1) and (0,4): (0,1) uses 1 -> skipped; (0,4) taken; (2,3) at .5 uses 2 -> skipped; .49 below
        self.assertEqual(P.select_merges(pairs, [.9, .95, .5, .49, .9]), [1, 4])
        self.assertEqual(P.select_merges([(0, 1), (2, 3)], [.5, .5]), [0, 1])          # threshold inclusive, tie by (a, b)
        self.assertEqual(P.select_merges([(2, 3), (0, 1)], [.7, .7]), [1, 0])
        self.assertEqual(P.select_merges([], []), [])

    def test_oracle_merges_take_positive_gains_only(self):
        self.assertEqual(P.oracle_merges([(0, 1), (1, 2), (2, 3)], [0.0, .3, .2]), [1])
        self.assertEqual(P.oracle_merges([(0, 1), (2, 3)], [.1, .2]), [1, 0])


class Linker(unittest.TestCase):
    def test_piece_equals_postproc_pieces(self):
        m = line_mask((300, 300), (500, 380))
        got = P.linker_piece(*obj(m)); ref = postproc._pieces(m.astype(np.uint8), 32, 30)[0]
        self.assertEqual(got['tips'], ref.tips); self.assertEqual(got['tip_axes'], ref.tip_axes); self.assertEqual(got['axis'], ref.axis)
        self.assertAlmostEqual(got['elong'], ref.elong, places=12); self.assertEqual(got['area'], int(m.sum()))

    def test_joins_agree_with_link_collinear(self):
        cases = [(line_mask((300, 300), (500, 300)), line_mask((540, 300), (700, 300))),         # collinear, gap 40
                 (line_mask((300, 300), (500, 300)), line_mask((520, 260), (520, 100))),         # perpendicular
                 (line_mask((300, 300), (500, 300)), line_mask((575, 300), (700, 300))),         # gap 75 > 60
                 (line_mask((300, 300), (500, 300)), bar(296, 300, 520, 527)),                    # small piece (< 32 px)
                 (line_mask((300, 300), (500, 300)), line_mask((540, 330), (700, 400)))]          # 37 deg off the axis
        for a, b in cases:
            a = a & ~b; binm = (a | b).astype(np.uint8)
            n_before = cv2.connectedComponents(binm, connectivity=8)[0]
            linked = postproc.link_collinear(binm, 60, 45.0, min_area=32, tip_window=30)
            joined = cv2.connectedComponents(linked, connectivity=8)[0] < n_before
            self.assertEqual(P.linker_joins(P.linker_piece(*obj(a)), P.linker_piece(*obj(b))), joined)


class Features(unittest.TestCase):
    def test_features_are_prediction_only_and_ordered(self):
        p = np.zeros((N, N), np.float32); p[500:510, 400:500] = .6; p[500:510, 530:600] = .6
        maps = dict(full=np.full((N, N), .2, np.float32), local=np.full((N, N), .3, np.float32), m2f_public=np.full((N, N), .4, np.float32))
        a = bar(500, 510, 400, 500); b = bar(500, 510, 530, 600); objs = [obj(a), obj(b)]
        r = P.candidate_pairs(objs)[0]
        f = P.pair_features(objs, r, taus=[.3, .5], selector=[.9, .7], p=p, maps=maps)
        self.assertEqual(list(f), P.FEATURES)
        self.assertEqual(f['gap_px'], 31.0); self.assertEqual(f['axis_angle_deg'], 0.0)
        self.assertEqual((f['gap_axis_angle_a_deg'], f['gap_axis_angle_b_deg']), (0.0, 0.0))
        self.assertEqual((f['tau_a'], f['tau_b'], f['selector_a'], f['selector_b']), (.3, .5, .9, .7))
        self.assertEqual((f['area_a'], f['area_b']), (1000., 700.))
        self.assertAlmostEqual(f['bridge_p_min'], 0.0); self.assertGreater(f['bridge_p_mean'], 0.0)   # the line crosses a dark gap
        self.assertAlmostEqual(f['bridge_full_mean'], .2, places=6); self.assertAlmostEqual(f['bridge_mf_mean'], .4, places=6)
        self.assertEqual(f['linker_join'], 1.0)
        self.assertTrue(np.isfinite(list(f.values())).all())
        # the signature takes no annotation; the same inputs give the same features
        self.assertEqual(f, P.pair_features(objs, r, taus=[.3, .5], selector=[.9, .7], p=p, maps=maps))


class Labels(unittest.TestCase):
    def test_pair_delta_and_gain_are_exact_on_the_whole_image(self):
        p = np.full((N, N), .9, np.float32)
        a = bar(500, 510, 400, 500); b = bar(500, 510, 520, 600); c = bar(900, 910, 400, 500); d = bar(900, 910, 520, 600)
        e = bar(1300, 1310, 400, 450)
        ms = [a, b, c, d, e]; objs = [obj(m) for m in ms]
        g_ab = bar(500, 510, 400, 600)                                   # one annotated filament split into a and b
        sets = [[g_ab, bar(1300, 1310, 400, 450)], [g_ab]]                 # two annotation sets; c and d are unmatched FPs
        gts = gts_of(sets); prs = {(r['a'], r['b']): r for r in P.candidate_pairs(objs)}
        base = full_counts(ms, sets)
        for (i, j), tau in [((0, 1), .5), ((2, 3), .5)]:
            r = prs[i, j]; (M, bb), _ = P.merge_objs(objs, i, j, r['pa'], r['pb'], tau, p)
            ca, cb, cm = P.iou_cols(*objs[i], gts), P.iou_cols(*objs[j], gts), P.iou_cols(M, bb, gts)
            delta = P.pair_delta(ca, cb, cm)
            out, _ = P.apply_merges(ms, [(i, j, r['pa'], r['pb'], tau)], p)
            ref = (full_counts(out, sets) - base)[:4]
            self.assertTrue(np.array_equal(delta[1:], ref[1:])); self.assertAlmostEqual(delta[0], ref[0], places=12)
        r = prs[2, 3]; (M, bb), _ = P.merge_objs(objs, 2, 3, r['pa'], r['pb'], .5, p)
        d23 = P.pair_delta(P.iou_cols(*objs[2], gts), P.iou_cols(*objs[3], gts), P.iou_cols(M, bb, gts))
        self.assertEqual(d23.tolist(), [0., 0., -2., 0.])                  # two FPs become one, in both sets
        self.assertAlmostEqual(P.gain(d23, .4), .4)
        self.assertEqual(P.label(P.gain(d23, .4)), 1); self.assertEqual(P.label(0.0), 0); self.assertEqual(P.label(-1e-9), 0)

    def test_gain_formula(self):
        self.assertAlmostEqual(P.gain(np.array([.7, 1., -2., -1.]), .5), .7 - .5 * (1. - 1. - .5))


class BaseSetPath(unittest.TestCase):
    def test_cluster_path_equals_full_repair(self):
        p, seeds = scene_repair()
        bs = P.BaseSet(seeds, p)
        for taus in ([TAU] * 3, [.2, .6, .35], [.6, .2, .5], [.3, .3, .3]):
            ref, _ = C.repair_taus(seeds, p, taus)
            got = bs.masks(taus)
            self.assertEqual(len(got), 3)
            for (M, bb, dg), r in zip(got, ref):
                self.assertTrue(np.array_equal(R.full(M, bb), r)); self.assertEqual(dg, S.digest(r))
        self.assertEqual(P.BaseSet([], p).masks([]), [])


def scene_repair():
    p = np.zeros((N, N), np.float32)
    a = bar(397, 404, 300, 700); b = bar(437, 444, 320, 680); d = bar(1300, 1307, 1200, 1500)
    p[392:409, 300:500] = .9; p[392:409, 500:700] = .5; p[394:407, 700:780] = .33; p[420:432, 320:680] = .45
    p[432:449, 320:680] = .8; p[1295:1312, 1150:1550] = .55
    return p, [a, b, d]


class Classifier(unittest.TestCase):
    def test_single_class_gives_a_constant(self):
        m = P.fit_classifier(np.zeros((3, 2)), np.array([0, 0, 0]), make=None)
        self.assertEqual(m.prob(np.zeros((4, 2))).tolist(), [0.] * 4)
        m = P.fit_classifier(np.zeros((0, 2)), np.array([], int), make=None)
        self.assertEqual(m.prob(np.zeros((2, 2))).tolist(), [0.] * 2)

    def test_two_classes_use_the_probability_of_class_one(self):
        class Fake:
            classes_ = np.array([0, 1])
            def fit(self, X, y): self.n_iter_ = 3; return self
            def predict_proba(self, X): return np.column_stack([1 - np.asarray(X)[:, 0], np.asarray(X)[:, 0]])
        m = P.fit_classifier(np.array([[.2], [.8]]), np.array([0, 1]), make=Fake)
        self.assertEqual(m.prob(np.array([[.3], [.6]])).tolist(), [.3, .6]); self.assertEqual(m.n_iter, 3)

    @unittest.skipUnless(Path('/kaggle/input').is_dir(), 'the real classifier is fitted only on Kaggle (no model fitting on the local Mac)')
    def test_real_classifier_recipe(self):
        import worker as W
        rng = np.random.default_rng(0); X = rng.random((200, 18)); y = (X[:, 0] > .5).astype(int)
        m = P.fit_classifier(X, y, W.make_classifier)
        self.assertIsInstance(m, P.Fitted); self.assertEqual(m.m.random_state, 42); self.assertGreater(m.n_iter, 0)
        pr = m.prob(np.array([[.9] + [.5] * 17, [.1] + [.5] * 17])); self.assertGreater(pr[0], .5); self.assertLess(pr[1], .5)
        self.assertEqual(m.prob(np.zeros((0, 18))).shape, (0,))


if __name__ == '__main__': unittest.main()
