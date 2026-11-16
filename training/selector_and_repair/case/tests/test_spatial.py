"""Synthetic specification tests for P-spatial (spatial.py), written before spatial.py.

P-spatial keeps P-cand v1's per-candidate tau and modulates the repair threshold inside each kept candidate's P-nat window
by the along-axis brightness of its seed: tau(x) = clip(tau_v1 * (peak(bin(x)) / m) ** gamma, 0.05, 0.80).
"""
import hashlib, itertools, sys, unittest
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H))
import refine_ops as R
import pnat
import pcand2 as C
import spatial as S

TAU = pnat.TAU
N = 2048


def line_seed(x0, x1, y, half=3, slope=0.0):
    m = np.zeros((N, N), bool)
    for x in range(x0, x1):
        yc = int(round(y + slope * (x - x0))); m[yc - half:yc + half + 1, x] = True
    return m


def scene():
    """Three kept seeds: A (horizontal, a bright core and a faint right half), B (parallel, close below A: windows overlap),
    D (isolated, diagonal). p is float32 like the real mixture."""
    p = np.zeros((N, N), np.float32)
    a = line_seed(300, 700, 400); b = line_seed(320, 680, 440); d = line_seed(1200, 1600, 1500, slope=.3)
    p[392:409, 300:500] = .9; p[392:409, 500:700] = .5                # A: bright left, faint right (seed itself)
    p[394:407, 700:780] = .33                                           # A: faint tail beyond the seed end (in the window)
    p[389:392, 300:700] = .42                                           # A: margin above, just over 0.40
    p[432:449, 320:680] = .8; p[420:432, 320:680] = .45                 # B and a contested strip between A and B
    for x in range(1200, 1600):
        yc = int(round(1500 + .3 * (x - 1200))); p[yc - 6:yc + 7, x] = .7 if x < 1400 else .35
    return p, [a, b, d]


def gt_rles(masks): return [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]


class Constants(unittest.TestCase):
    def test_fixed_grid_and_constants(self):
        self.assertEqual(S.GAMMAS, (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(S.BIN_PX, 16); self.assertEqual((S.TAU_MIN, S.TAU_MAX), (0.05, 0.80))
        self.assertEqual(S.PRIMARY, 'inc_pspatial_minus_inc_pcand_v1'); self.assertEqual(S.SECONDARY, ['inc_pspatial_minus_inc_pnat'])


class Profile(unittest.TestCase):
    def test_bins_follow_the_principal_axis_and_window(self):
        p, (a, b, d) = scene(); pr = S.profile(a, p)
        g = C._geom(a); bb = R.window_bbox(g)
        self.assertEqual(pr['bbox'], bb); self.assertEqual(pr['bins'].shape, (bb[1] - bb[0], bb[3] - bb[2]))
        self.assertEqual(pr['nb'], int(np.ceil(g['Lc'] / 16))); self.assertEqual(pr['bins'].min(), 0); self.assertEqual(pr['bins'].max(), pr['nb'] - 1)
        # horizontal seed: bins are columns of width 16 starting at the window's left edge (u = -Lc/2)
        y0, y1, x0, x1 = bb; xs = np.arange(x0, x1, dtype=np.float64); u = (xs - g['m'][0]) * g['e1'][0]
        exp = np.clip(np.floor((u + g['Lc'] / 2) / 16).astype(int), 0, pr['nb'] - 1)
        self.assertTrue(np.array_equal(pr['bins'][(y1 - y0) // 2], exp))

    def test_peaks_median_and_copied_bins(self):
        p, (a, b, d) = scene(); pr = S.profile(a, p)
        bins, has = pr['bins'], pr['has']; y0, y1, x0, x1 = pr['bbox']; sd = a[y0:y1, x0:x1]
        for k in np.flatnonzero(has):
            self.assertEqual(pr['peak'][k], float(p[y0:y1, x0:x1][sd & (bins == k)].max()))
        self.assertTrue(set(np.round(pr['peak'][has], 6)) <= {0.9, 0.5})
        self.assertEqual(pr['m'], float(np.median(pr['peak'][has])))
        idx = np.flatnonzero(has)
        for k in np.flatnonzero(~has):   # empty bins copy the nearest seeded bin, ties to the lower index
            dist = np.abs(idx - k); self.assertEqual(pr['peak'][k], pr['peak'][idx[np.flatnonzero(dist == dist.min())[0]]])
        self.assertTrue(np.allclose(pr['ratio'], pr['peak'] / pr['m']))
        self.assertEqual(pr['copied'], int((~has).sum())); self.assertEqual(pr['copied_tail'] + pr['copied_interior'], pr['copied'])
        self.assertGreater(pr['copied_tail'], 0)   # the window margin beyond each seed end never holds seed pixels

    def test_nearest_fill_tie_goes_lower(self):
        peak = np.array([np.nan, 1., np.nan, 3., np.nan, np.nan]); has = ~np.isnan(peak)
        self.assertEqual(S.fill_nearest(peak, has).tolist(), [1., 1., 1., 3., 3., 3.])   # bin 2 is equidistant to 1 and 3 -> 1

    def test_profile_reads_only_seed_and_p(self):
        p, (a, b, d) = scene(); pr1 = S.profile(a, p); p2 = p.copy(); p2[1700:1800, 100:200] = .99   # outside A's window
        pr2 = S.profile(a, p2)
        self.assertTrue(np.array_equal(pr1['ratio'], pr2['ratio'])); self.assertTrue(np.array_equal(pr1['bins'], pr2['bins']))


class TauMap(unittest.TestCase):
    def test_gamma_zero_is_constant_tau(self):
        p, (a, b, d) = scene(); pr = S.profile(a, p)
        for t in C.GRID_V1:
            tm = S.tau_map(pr, t, 0.0, np.float32)
            self.assertEqual(tm.dtype, np.float32); self.assertTrue((tm == np.float32(t)).all())

    def test_formula_and_clip(self):
        p, (a, b, d) = scene(); pr = S.profile(a, p)
        for g in S.GAMMAS:
            tm = S.tau_map(pr, 0.45, g, np.float64)
            exp = np.clip(0.45 * pr['ratio'] ** g, 0.05, 0.80)[pr['bins']]
            self.assertTrue(np.array_equal(tm, exp))
        faint = pr['bins'] == pr['bins'].max(); bright = pr['bins'] == 0
        tm = S.tau_map(pr, 0.45, 1.0, np.float64); self.assertLess(tm[faint].max(), 0.45); self.assertGreater(tm[bright].min(), 0.45)
        tm = S.tau_map(pr, 0.45, -0.5, np.float64); self.assertGreater(tm[faint].min(), 0.45)
        pr0 = dict(pr, ratio=np.full(pr['nb'], 100.)); self.assertTrue((S.tau_map(pr0, .6, 1., np.float64) == .80).all())
        pr0 = dict(pr, ratio=np.full(pr['nb'], 1e-4)); self.assertTrue((S.tau_map(pr0, .2, 1., np.float64) == .05).all())

    def test_zero_peak_is_deterministic(self):
        pr = dict(bins=np.zeros((2, 2), int), ratio=np.array([0.]), nb=1)
        self.assertEqual(S.tau_map(pr, .4, -.5, np.float64)[0, 0], .80); self.assertEqual(S.tau_map(pr, .4, .5, np.float64)[0, 0], .05)
        self.assertEqual(S.tau_map(pr, .4, 0., np.float64)[0, 0], .4)


class Repair(unittest.TestCase):
    def test_gamma_zero_byte_identical_to_scalar_and_pnat(self):
        p, seeds = scene(); profs = [S.profile(s, p) for s in seeds]
        ref, st = pnat.repair(seeds, p)
        out, st2 = S.repair_maps(seeds, p, [TAU] * 3, [0.] * 3, profs)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(ref, out))); self.assertEqual(st, st2)
        for taus in itertools.product((0.30, 0.45, 0.60), repeat=3):
            r1, s1 = C.repair_taus(seeds, p, taus); r2, s2 = S.repair_maps(seeds, p, taus, [0.] * 3, profs)
            self.assertTrue(all(np.array_equal(x, y) for x, y in zip(r1, r2))); self.assertEqual(s1, s2)

    def test_constant_map_equals_scalar_for_any_gamma(self):
        p, seeds = scene(); profs = [dict(S.profile(s, p)) for s in seeds]
        for pr in profs: pr['ratio'] = np.ones(pr['nb'])
        for g in S.GAMMAS:
            r1, s1 = C.repair_taus(seeds, p, [0.45, 0.35, 0.55]); r2, s2 = S.repair_maps(seeds, p, [0.45, 0.35, 0.55], [g] * 3, profs)
            self.assertTrue(all(np.array_equal(x, y) for x, y in zip(r1, r2))); self.assertEqual(s1, s2)

    def test_float32_boundary_comparison_matches_scalar(self):
        p, seeds = scene()
        for t in (0.45, TAU, 0.35, 0.55):   # float32(t) below and above t
            f = np.float32(t); p2 = p.copy(); p2[389:392, 300:700] = f; p2[386:389, 300:700] = np.nextafter(f, np.float32(0))
            profs = [S.profile(s, p2) for s in seeds]
            r1, _ = C.repair_taus(seeds, p2, [t] * 3); r2, _ = S.repair_maps(seeds, p2, [t] * 3, [0.] * 3, profs)
            self.assertTrue(all(np.array_equal(x, y) for x, y in zip(r1, r2)))
            self.assertTrue(r1[0][389:392, 300:700].any())       # p == float32(t) passes the comparison as in P-nat

    def test_positive_gamma_extends_the_faint_tail(self):
        p, seeds = scene(); profs = [S.profile(s, p) for s in seeds]
        r0, _ = S.repair_maps(seeds, p, [0.45] * 3, [0.] * 3, profs); r1, _ = S.repair_maps(seeds, p, [0.45] * 3, [1.] * 3, profs)
        self.assertFalse(r0[0][394:407, 700:780].any())         # p = .33 < .45
        self.assertTrue(r1[0][394:407, 700:780].any())          # faint end and copied tail bins: m = (.9 + .5) / 2 (26 seeded bins), tau = .45 * .5 / .7 = .32 < .33
        rn, _ = S.repair_maps(seeds, p, [0.45] * 3, [-.5] * 3, profs)
        self.assertLessEqual(int(rn[0].sum()), int(r0[0].sum()))

    def test_invariants_hold_for_mixed_gammas(self):
        p, seeds = scene(); profs = [S.profile(s, p) for s in seeds]
        for gs in itertools.product(S.GAMMAS, repeat=2):
            out, st = S.repair_maps(seeds, p, [0.45, 0.30, 0.50], [gs[0], gs[1], 0.75], profs)
            self.assertEqual(len(out), 3); self.assertEqual(st['kept'], 3)
            occ = np.sum([o.astype(np.uint8) for o in out], 0); self.assertLessEqual(int(occ.max()), 1)

    def test_empty(self):
        p, _ = scene(); out, st = S.repair_maps([], p, [], [], []); self.assertEqual(out, []); self.assertEqual(st['kept'], 0)


class FastPath(unittest.TestCase):
    def setUp(self):
        self.p, self.seeds = scene()
        g1 = line_seed(300, 760, 400, half=5); g2 = line_seed(320, 690, 440, half=6); g3 = line_seed(1200, 1500, 1500, half=6, slope=.3)
        self.gts = [gt_rles([g1, g2, g3]), gt_rles([line_seed(300, 700, 401, half=4)])]

    def full_counts(self, taus, gammas):
        profs = [S.profile(s, self.p) for s in self.seeds]
        out, _ = S.repair_maps(self.seeds, self.p, taus, gammas, profs)
        pred = [mu.encode(np.asfortranarray(m, dtype=np.uint8)) for m in out]
        qs = [np.asarray(mu.iou(g, pred, [0] * len(pred))).reshape(len(g), len(pred)) for g in self.gts]
        return qs, out

    def test_clusters(self):
        pr = C.prepare(self.seeds, self.p); cl = S.clusters(pr)
        self.assertEqual(sorted(map(sorted, cl)), [[0, 1], [2]])

    def test_fast_equals_full_bitwise(self):
        fi = S.FastImage(self.seeds, self.p, S.decode_gts(self.gts))
        rng = np.random.default_rng(3)
        for _ in range(12):
            taus = [float(x) for x in rng.choice(C.GRID_V1, 3)]; g = float(rng.choice(S.GAMMAS))
            qs, out = self.full_counts(taus, [g] * 3)
            fq = fi.iou_matrices(taus, g)
            for a, b in zip(qs, fq): self.assertEqual(a.shape, b.shape); self.assertTrue(np.array_equal(a, b))
            c_full = sum((S.count6(q) for q in qs), start=np.zeros(6)); c_fast, sc, dig = fi.evaluate(taus, g)
            self.assertTrue(np.array_equal(c_full, c_fast)); self.assertEqual(sc.tolist(), C.candidate_scores(qs, 3).tolist())
            self.assertEqual(dig, [S.digest(m) for m in out])

    def test_cache_reuses_clusters(self):
        fi = S.FastImage(self.seeds, self.p, S.decode_gts(self.gts))
        fi.evaluate([0.45, 0.45, 0.45], 0.5); n = fi.computed
        fi.evaluate([0.45, 0.45, 0.30], 0.5); self.assertEqual(fi.computed, n + 1)    # only D's cluster is new
        fi.evaluate([0.45, 0.45, 0.30], 0.5); self.assertEqual(fi.computed, n + 1)

    def test_empty_and_no_gt(self):
        fi = S.FastImage([], self.p, S.decode_gts(self.gts)); c, sc, dig = fi.evaluate([], 0.5)
        self.assertEqual(c.tolist(), [0., 0., 0., 4., 0., 0.]); self.assertEqual(sc.tolist(), [])
        fi = S.FastImage(self.seeds, self.p, []); c, sc, _ = fi.evaluate([.45] * 3, 0.)
        self.assertEqual(c.tolist(), [0.] * 6); self.assertEqual(sc.tolist(), [0.] * 3)


class Choice(unittest.TestCase):
    def test_choose_gamma_ties(self):
        G = S.GAMMAS
        self.assertEqual(S.choose_gamma([.1, .2, .3, .2, .1, 0, 0]), 0.0)
        self.assertEqual(S.choose_gamma([.1, .2, .3, .2, .4, 0, 0]), 0.5)
        self.assertEqual(S.choose_gamma([.4, .2, .4, .2, .4, 0, 0]), 0.0)            # tie with 0 -> 0
        self.assertEqual(S.choose_gamma([.4, .2, .3, .2, .4, 0, .4]), -0.5)           # |-.5| == |.5|: lower
        self.assertEqual(S.choose_gamma([.1, .5, .3, .5, .4, 0, .5]), -0.25)
        with self.assertRaises(AssertionError): S.choose_gamma([.1] * 6)
        self.assertEqual(len(G), 7)

    def test_oracle_gammas(self):
        M = np.array([[0, 0, 0, 0, 0, 0, 0], [.6, .6, .6, .6, .6, .6, .6], [0, 0, .5, .7, .7, 0, 0], [.8, 0, .5, 0, .8, 0, 0]], float)
        self.assertEqual(S.oracle_gammas(M), [0.0, 0.0, 0.25, -0.5])


class InnerFolds(unittest.TestCase):
    def test_inner_assignment_is_the_selector_machinery(self):
        sys.path.insert(0, str(H / 'frozen/eval'))
        from make_time_folds import temporal_groups, assign_groups
        files = sorted(f'2011{m:02d}{d:02d}120000{s}h' for m in range(1, 13) for d, s in [(3, 'B'), (17, 'C'), (25, 'M')])
        groups = temporal_groups(files, 24); s = 2026110607; outer = assign_groups(groups, 5, s)
        for k in range(5):
            train = sorted(f for f in files if outer[f] != k)
            inner = S.inner_assignment(groups, train, s, k)
            self.assertEqual(inner, assign_groups([g for g in groups if g[0] in train], k=3, seed=s + 10000 + k))
            self.assertEqual(set(inner), set(train)); self.assertEqual(set(inner.values()), {0, 1, 2})
            for g in groups: self.assertLessEqual(len({inner.get(f) for f in g}), 1)
        split = [['20110101000000Bh', '20110101010000Ch'], ['20110301000000Bh']]
        with self.assertRaises(AssertionError): S.inner_assignment(split, ['20110101000000Bh', '20110301000000Bh'], s, 0)   # splits a group


class Gate(unittest.TestCase):
    def test_gate_on_new_contrast(self):
        good = dict(images=425, contrasts={S.PRIMARY: dict(partition_deltas=[.001] * 20, paired_group_CI95=[.0001, .002])})
        r = dict(replay_passed=True, integrity_passed=True, subsets=dict(combined425=good, **{f'fold{k}': dict(contrasts={S.PRIMARY: dict(mean_paired_partition_delta=.001)}) for k in range(3)}))
        self.assertTrue(C.gate(r, S.PRIMARY))
        r['integrity_passed'] = False; self.assertFalse(C.gate(r, S.PRIMARY))


if __name__ == '__main__':
    unittest.main()
