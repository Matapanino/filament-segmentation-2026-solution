"""chain2m (order MT1): multi-threshold candidates in worker.py (synthetic images of test_worker_synthetic; stand-in fits).

- an arm with levels [0.5] runs the level code and equals chain2's arm exactly (candidates, features, then counts, keeps,
  selectors and ship selector in the end-to-end plumbing);
- an arm with levels [0.5, 0.4]: candidates = SHIP_R3 at 0.5, then SHIP_R3 with thr 0.4; per-level neighbour/MF features;
  the feature 'level'; overlapping pairs and nesting recorded; 69 selector columns; kept candidates never overlap (greedy
  rule after the cutoff); the nested sweep scores the greedy survivors only.
"""
import json, sys, tempfile, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / 'tests'))
import numpy as np
import test_worker_synthetic as T
W = T.W; M = T.M; P = T.P; NB = T.NB; MF = T.MF
from filseg.instancing import iter_instances, SHIP_R3
import levels as LV


class LevelImages(unittest.TestCase):
    setUp = T.Images.setUp

    def tearDown(self):
        W.configure(M.DEFAULT_WEIGHTS); T.Images.tearDown(self)

    def test_one_level_equals_chain2(self):
        f = self.files[1]
        W.configure([.25, .55]); m0, p0, r0 = W.candidate_rows(f, 'w55'); _, x0 = W.image_pass_variant('w55', f)
        W.configure([.25, .55], [[.5]]); self.assertEqual(M.LEVELS['w55'], [.5])
        m1, p1, r1 = W.candidate_rows(f, 'w55'); _, x1 = W.image_pass_variant('w55', f)
        self.assertTrue(np.array_equal(p0, p1)); self.assertEqual(r0, r1); self.assertEqual([W.packed(m) for m in m0], [W.packed(m) for m in m1])
        for k in x0: self.assertEqual(x0[k], x1[k], k)
        self.assertEqual(x1['overlap'], []); self.assertEqual(x1['level'], [.5] * x1['n']); self.assertNotIn('level', r1[0])

    def test_two_levels(self):
        f = self.files[1]; W.configure([.25, .55], [[.5, .4]]); arm = 'w55l50l40'
        masks, p, rows = W.candidate_rows(f, arm)
        a = [np.asarray(m, bool) for m in iter_instances(p, SHIP_R3)]; b = [np.asarray(m, bool) for m in iter_instances(p, SHIP_R3.with_(thr=.4))]
        self.assertTrue(len(a) and len(b))
        self.assertEqual([W.packed(m) for m in masks], [W.packed(m) for m in a + b])
        self.assertEqual([r['level'] for r in rows], [.5] * len(a) + [.4] * len(b))
        for blk, ms in ((rows[:len(a)], a), (rows[len(a):], b)):             # neighbour / MF features per level
            for r, g in zip(blk, NB.neighbour_features(ms, p)): self.assertEqual({n: r[n] for n in NB.FEATURES}, g)
            for r, g in zip(blk, MF.image_features(ms, W.branch_maps(f, arm)['m2f_public'])): self.assertEqual({n: r[n] for n in MF.FEATURES}, g)
        self.assertEqual(len(rows[0]), 79)
        _, x = W.image_pass_variant(arm, f)
        self.assertEqual(x['overlap'], [list(t) for t in LV.overlap_pairs(masks)]); self.assertTrue(x['overlap'])
        for i in range(len(a)): self.assertGreaterEqual(x['nesting']['parent'][i], len(a))   # every 0.5 piece sits in a 0.4 component
        self.assertIn('levels 0.5,0.4', W.bank_text(arm)); self.assertEqual(W.bank_text('w25'), W.bank_text_chain2('w25'))
        self.assertEqual(M.arm_feature_names(['x'], arm), ['x', 'level'])


class LevelPlumbing(T.Plumbing):
    LEVELSETS = None

    def fake_variant(self, v, f):
        f_, d = super().fake_variant(v, f); levels = M.LEVELS.get(v)
        if levels is None: return f_, d
        n = d['n']; lev = [levels[i % len(levels)] for i in range(n)]
        if len(levels) > 1:
            for r, t in zip(d['features'], lev): r['level'] = float(t)
        pairs = [[i, i + 1] for i in range(0, n - 1, 2)] if len(levels) > 1 else []
        d.update(level=lev, overlap=pairs, nesting=dict(parent=[-1] * n, children=[[] for _ in range(n)], identical_to_parent=[False] * n), areas=[10] * n)
        return f_, d

    def run_levels(self, t, weights, levelsets):
        orig = W.configure
        W.configure = lambda w, l=None: orig(w, levelsets)
        try: return self.run_main(t, weights=weights)
        finally: W.configure = orig; W.configure(M.DEFAULT_WEIGHTS)

    def test_main_end_to_end(self): pass                      # inherited tests run in test_worker_synthetic
    def test_variant_v1_taus_come_from_the_stored_bank_regressors(self): pass
    def test_selector_fits_exclude_query_groups(self): pass

    def test_one_level_arm_equals_chain2_arm(self):
        with tempfile.TemporaryDirectory() as t0, tempfile.TemporaryDirectory() as t1:
            a = self.run_levels(Path(t0), [.25, .55], None); b = self.run_levels(Path(t1), [.25, .55], [[.5]])
            for s in a['counts.json']['per_partition_counts']:
                for arm in ('w25', 'w55', 'pcombo'): self.assertEqual(a['counts.json']['per_partition_counts'][s][arm], b['counts.json']['per_partition_counts'][s][arm])
            self.assertEqual(a['keeps.json']['keeps'], b['keeps.json']['keeps'])
            strip = lambda o: [{k: v for k, v in x.items()} for x in o['selectors.json']['models']]
            self.assertEqual(strip(a), strip(b))
            for arm in ('w25', 'w55'):
                self.assertEqual((Path(t0) / f'out/ship/{arm}_selector.json').read_bytes(), (Path(t1) / f'out/ship/{arm}_selector.json').read_bytes())
            ra, rb = a['science_result.json'], b['science_result.json']
            self.assertEqual(ra['subsets']['all707']['scores'], rb['subsets']['all707']['scores']); self.assertEqual(rb['levels'], dict(w25=None, w55=[.5]))

    def test_two_level_arm(self):
        seen = []; orig = LV.resolve_probabilities
        def spy(prob, records, files, overlap): seen.append(sum(len(overlap[f]) for f in files)); return orig(prob, records, files, overlap)
        with tempfile.TemporaryDirectory() as t:
            LV.resolve_probabilities = spy
            try: o = self.run_levels(Path(t), [.25, .55], [[.5], [.5, .4]])
            finally: LV.resolve_probabilities = orig
            r = o['science_result.json']; arm = 'w55l50l40'
            self.assertEqual(o['counts.json']['arms'], ['inc', 'inc_pnat', 'inc_pcand_v1', 'pcombo', 'w25', 'w55', arm])
            self.assertIn(f'{arm}_minus_w55', r['subsets']['all707']['contrasts']); self.assertTrue(r['integrity_passed'])
            self.assertEqual(len(seen), 100 * 2 + 2)                    # 100 nested sweeps of w55 ([0.5]) and of the 2-level arm, + their ship keeps
            self.assertTrue(any(seen))
            sel = [x for x in o['selectors.json']['models'] if x['arm'] == arm]; self.assertEqual(len(sel), 100); self.assertEqual(len(sel[0]['coef'][0]), 69)
            self.assertTrue(all(len(x['coef'][0]) == 68 for x in o['selectors.json']['models'] if x['arm'] != arm))
            rec = json.loads((Path(t) / f'out/ship/{arm}_selector.json').read_text()); self.assertEqual(rec['feature_order'][-1], 'level'); self.assertEqual(len(rec['feature_order']), 69)
            self.assertEqual(json.loads((Path(t) / 'out/ship/w55_selector.json').read_text())['feature_order'], M.feature_names(T.FAKE39, T.CTX))
            M.configure([.25, .55], [[.5], [.5, .4]])
            for f in self.files:
                pairs = [tuple(x) for x in self.fake_variant(arm, f)[1]['overlap']]
                for s, kp in o['keeps.json']['keeps'][f].items():
                    k = set(kp[arm]); self.assertFalse(any(i in k and j in k for i, j in pairs))
            lv = r['descriptive']['levels'][arm]['all707']; self.assertEqual(set(lv['candidates']), {'0.5', '0.4'})
            self.assertEqual(sum(lv['candidates'].values()), sum(self.fake_variant(arm, f)[1]['n'] for f in self.files))
            self.assertIn('w55', r['descriptive']['levels']); self.assertNotIn('w25', r['descriptive']['levels'])
            self.assertTrue(any(self.fake_variant(arm, f)[1]['overlap'] for f in self.files)); M.configure(M.DEFAULT_WEIGHTS)


if __name__ == '__main__':
    unittest.main()
