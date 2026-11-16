"""Synthetic specification tests for neigh.py (the 6 neighbour-context features), written before neigh.py (red log: tdd_red.log).

The features are computed per image on the unrepaired SHIP_R3 candidates (pixel-disjoint full-size masks) and the
mixture map p only. Distances are minimum Euclidean distances between pixel centres (pmerge.closest_pair), axes are the
R3 linker's principal axes (pmerge.linker_piece), the bridge is pmerge's 1-px centre line between the closest points.
"""
import inspect, itertools, sys, unittest
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen/eval'), str(H)]
import neigh as NB
import pmerge as PM

N = 2048


def bar(y0, y1, x0, x1):
    m = np.zeros((N, N), bool); m[y0:y1, x0:x1] = True; return m


def pmap(masks, inside=.9, outside=.2):
    p = np.full((N, N), outside, np.float32)
    for m in masks: p[m] = inside
    return p


class Names(unittest.TestCase):
    def test_names_and_constants(self):
        self.assertEqual(NB.FEATURES, ['neigh_count_64', 'neigh_count_128', 'neigh_gap_nearest', 'neigh_axis_abs_cos_nearest',
                                       'neigh_area_share_aligned_128', 'neigh_bridge_p_min_nearest'])
        self.assertEqual((NB.R_NEAR, NB.R_FAR, NB.GAP_CAP, NB.ALIGN_COS), (64, 128, 512, .7))
        self.assertEqual((NB.NO_NEIGHBOUR_COS, NB.NO_NEIGHBOUR_BRIDGE), (0.0, 1.0))
        self.assertGreaterEqual(NB.NO_CAP, int(np.ceil(np.hypot(N - 1, N - 1))))

    def test_signature_reads_no_annotation(self):
        self.assertEqual(list(inspect.signature(NB.neighbour_features).parameters), ['masks', 'p'])


class Features(unittest.TestCase):
    def test_collinear_pair(self):
        a = bar(500, 506, 100, 200); b = bar(500, 506, 230, 330); p = pmap([a, b]); p[502, 215] = .05
        fa, fb = NB.neighbour_features([a, b], p)
        for f in (fa, fb): self.assertEqual(list(f), NB.FEATURES)
        self.assertEqual((fa['neigh_count_64'], fa['neigh_count_128']), (1., 1.))
        self.assertEqual(fa['neigh_gap_nearest'], 31.); self.assertEqual(fb['neigh_gap_nearest'], 31.)
        self.assertAlmostEqual(fa['neigh_axis_abs_cos_nearest'], 1., places=12)
        self.assertAlmostEqual(fa['neigh_area_share_aligned_128'], .5); self.assertAlmostEqual(fb['neigh_area_share_aligned_128'], .5)
        # the centre line runs along row 500 (closest points: lexicographic tie rule), so the dark pixel at row 502 is off it
        self.assertAlmostEqual(fa['neigh_bridge_p_min_nearest'], .2, places=6); self.assertEqual(fa['neigh_bridge_p_min_nearest'], fb['neigh_bridge_p_min_nearest'])
        p[500, 215] = .05
        self.assertAlmostEqual(NB.neighbour_features([a, b], p)[0]['neigh_bridge_p_min_nearest'], .05, places=6)

    def test_bridge_equals_pmerge_pair_feature(self):
        a = bar(300, 310, 300, 420); b = bar(340, 352, 450, 600); rng = np.random.default_rng(0)
        p = rng.random((N, N)).astype(np.float32)
        objs = [PM.obj_from_full(a), PM.obj_from_full(b)]; pair = PM.candidate_pairs(objs, NB.NO_CAP)[0]
        maps = dict(full=p, local=p, m2f_public=p)
        ref = PM.pair_features(objs, pair, [.4, .4], [.5, .5], p, maps)['bridge_p_min']
        fa, fb = NB.neighbour_features([a, b], p)
        self.assertEqual(fa['neigh_bridge_p_min_nearest'], ref); self.assertEqual(fb['neigh_bridge_p_min_nearest'], ref)
        self.assertEqual(fa['neigh_gap_nearest'], min(512., float(np.sqrt(pair['d2']))))

    def test_distance_thresholds_inclusive(self):
        a = bar(1000, 1004, 100, 200)
        for gap, c64, c128 in ((64, 1, 1), (65, 0, 1), (128, 0, 1), (129, 0, 0)):
            b = bar(1000, 1004, 199 + gap, 260 + gap)
            f = NB.neighbour_features([a, b], pmap([a, b]))[0]
            self.assertEqual(f['neigh_gap_nearest'], float(gap)); self.assertEqual((f['neigh_count_64'], f['neigh_count_128']), (c64, c128))

    def test_counts_and_aligned_area_share(self):
        a = bar(800, 806, 800, 1000)                     # area 1200, horizontal
        b = bar(800, 806, 1040, 1140)                    # collinear, gap 41, area 600
        c = bar(700, 760, 900, 905)                      # perpendicular, gap 41 (rows 759 -> 800), area 300
        d = bar(800, 806, 1200, 1300)                    # collinear, gap 201 from a: outside 128
        e = bar(1500, 1506, 100, 200)                    # far away
        masks = [a, b, c, d, e]; f = NB.neighbour_features(masks, pmap(masks))
        self.assertEqual((f[0]['neigh_count_64'], f[0]['neigh_count_128']), (2., 2.))
        self.assertAlmostEqual(f[0]['neigh_area_share_aligned_128'], 1200 / (1200 + 600))
        self.assertAlmostEqual(f[0]['neigh_axis_abs_cos_nearest'], 1., places=12)   # b and c tie at 41: lower index b
        self.assertEqual(f[0]['neigh_gap_nearest'], 41.)
        # c: perpendicular to everything near it -> share 1; nearest is a (gap 41), |cos| ~ 0
        self.assertAlmostEqual(f[2]['neigh_area_share_aligned_128'], 1.); self.assertAlmostEqual(f[2]['neigh_axis_abs_cos_nearest'], 0., places=9)
        # d: b is 60 px away (within 64), a is 201 away
        self.assertEqual((f[3]['neigh_count_64'], f[3]['neigh_count_128']), (1., 1.)); self.assertAlmostEqual(f[3]['neigh_area_share_aligned_128'], 600 / 1200)
        # e: nothing within 512 -> gap capped at 512, counts 0, share 1; its nearest neighbour still defines cos and bridge
        self.assertEqual(f[4]['neigh_gap_nearest'], 512.); self.assertEqual((f[4]['neigh_count_64'], f[4]['neigh_count_128']), (0., 0.))
        self.assertEqual(f[4]['neigh_area_share_aligned_128'], 1.); self.assertAlmostEqual(f[4]['neigh_axis_abs_cos_nearest'], 1., places=12)
        self.assertAlmostEqual(f[4]['neigh_bridge_p_min_nearest'], .2, places=6)

    def test_single_candidate_and_empty(self):
        a = bar(10, 20, 10, 300)
        f = NB.neighbour_features([a], pmap([a]))
        self.assertEqual(f, [dict(neigh_count_64=0., neigh_count_128=0., neigh_gap_nearest=512., neigh_axis_abs_cos_nearest=0.,
                                  neigh_area_share_aligned_128=1., neigh_bridge_p_min_nearest=1.)])
        self.assertEqual(NB.neighbour_features([], np.zeros((N, N), np.float32)), [])

    def test_min_distance_equals_brute_force(self):
        rng = np.random.default_rng(5)
        for trial in range(6):
            masks = []; occupied = np.zeros((N, N), bool)
            while len(masks) < 4:
                y, x = rng.integers(200, 400, 2); h, w = rng.integers(3, 12, 2)
                m = bar(y, y + h, x, x + w); m[y + rng.integers(h), x:x + w] = True
                grown = np.zeros_like(occupied); grown[y - 1:y + h + 1, x - 1:x + w + 1] = True
                if (grown & occupied).any(): continue
                masks.append(m); occupied |= m
            f = NB.neighbour_features(masks, pmap(masks))
            pts = [np.argwhere(m) for m in masks]
            for i in range(4):
                d = min(np.sqrt(((pts[i][:, None, :] - pts[j][None, :, :]) ** 2).sum(-1)).min() for j in range(4) if j != i)
                self.assertAlmostEqual(f[i]['neigh_gap_nearest'], min(512., d), places=9)
                c64 = sum(np.sqrt(((pts[i][:, None, :] - pts[j][None, :, :]) ** 2).sum(-1)).min() <= 64 for j in range(4) if j != i)
                self.assertEqual(f[i]['neigh_count_64'], float(c64))

    def test_permutation_equivariant_without_ties(self):
        a = bar(800, 806, 800, 1000); b = bar(810, 816, 1050, 1160); c = bar(600, 700, 820, 826); d = bar(1300, 1310, 300, 330)
        masks = [a, b, c, d]; p = pmap(masks); f = NB.neighbour_features(masks, p)
        for perm in itertools.permutations(range(4)):
            g = NB.neighbour_features([masks[i] for i in perm], p)
            for k, i in enumerate(perm):
                for n in NB.FEATURES: self.assertAlmostEqual(g[k][n], f[i][n], places=12, msg=(perm, n))

    def test_values_finite_and_bounded(self):
        rng = np.random.default_rng(9); masks = [bar(100 * i, 100 * i + 5, 50 * i, 50 * i + 90) for i in range(1, 8)]
        p = rng.random((N, N)).astype(np.float32)
        for f in NB.neighbour_features(masks, p):
            self.assertTrue(np.isfinite(list(f.values())).all())
            self.assertTrue(0 <= f['neigh_axis_abs_cos_nearest'] <= 1 + 1e-12 and 0 < f['neigh_area_share_aligned_128'] <= 1)
            self.assertTrue(0 <= f['neigh_bridge_p_min_nearest'] <= 1 and 0 < f['neigh_gap_nearest'] <= 512)


if __name__ == '__main__':
    unittest.main()
