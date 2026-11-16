"""chain2s: the combined-map arm `cmb` (synthetic, no fitting).

Layout: the arm exists only with the combined option; its weight is 1.0; its mixture is the MF map bit for bit (no U-Net
terms); without the option the layout equals chain2m's (and so chain2's). Worker: the cmb candidates are
iter_instances(MF, SHIP_R3.with_(thr=t)) of the combiner map; the replay arms w25 / w55 keep SHIP_R3 and are unchanged by
the extra arm; the 8 MF-rich features read the combiner map, the neighbour and v1 window features read the arm's blend
(= the combiner map). End to end (stand-in fits): one more selector arm with 68 columns, the contrasts cmb_minus_pcombo and
cmb_minus_w25, and the replay arms' counts, keeps, selectors and ship selectors equal those of the call without the option.
"""
import importlib.util, json, sys, tempfile, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / 'tests'))
import numpy as np
import test_worker_synthetic as T
W = T.W; M = T.M; NB = T.NB; MF = T.MF; C = T.C
from filseg.instancing import iter_instances, SHIP_R3

CHAIN2M_CASE = H.parent.parent / 'chain2m' / 'case'     # read only: the layout reference


def chain2m_mt():
    spec = importlib.util.spec_from_file_location('mt_chain2m_reference', CHAIN2M_CASE / 'mt.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Layout(unittest.TestCase):
    def tearDown(self): M.configure(M.DEFAULT_WEIGHTS)

    def test_without_option_equals_chain2m(self):
        if not (CHAIN2M_CASE / 'mt.py').is_file(): self.skipTest('chain2m tree not present (the Kaggle kernel); checked locally')
        ref = chain2m_mt()
        for ws, lv in (([.25, .55], None), ([.25, .4, .55], None), ([.25, .55], [[.5], [.5, .4]]), ([.25, .55], [[.5, .4, .3]])):
            self.assertEqual(M.layout(ws, lv), ref.layout(ws, lv)); self.assertEqual(M.layout(ws, lv, None), ref.layout(ws, lv))
        M.configure([.25, .55]); self.assertIsNone(M.COMBINED); self.assertNotIn('cmb', M.NEW_VARIANTS)

    def test_combined_arm(self):
        x = M.layout([.25, .55], None, dict(thr=.4, name='st1ci'))
        self.assertEqual(x['NEW_VARIANTS'], ['w25', 'w55', 'cmb']); self.assertEqual(x['WEIGHT']['cmb'], 1.0)
        self.assertEqual(x['SELECTOR_ARMS'], ['pcombo', 'w25', 'w55', 'cmb']); self.assertEqual(x['PASS2_ARMS'][-1], 'cmb')
        self.assertEqual(x['VARIANT']['cmb'], 'cmb'); self.assertIsNone(x['LEVELS']['cmb'])
        self.assertEqual(x['ARM_CONTRASTS'], ['w25_minus_pcombo', 'w55_minus_pcombo', 'cmb_minus_pcombo', 'w55_minus_w25', 'cmb_minus_w25'])
        self.assertEqual(x['COMBINED'], dict(arm='cmb', thr=.4, name='st1ci'))
        y = M.layout([.25, .55])                                    # the replay arms' part of the layout is unchanged
        self.assertEqual([c for c in x['CONTRASTS'] if 'cmb' not in c[0]], y['CONTRASTS'])
        M.configure([.25, .55], None, dict(thr=.4)); self.assertEqual(M.COMBINED['thr'], .4); self.assertEqual(M.arm_feature_names(['a'], 'cmb'), ['a'])
        self.assertFalse(M.multi_level('cmb'))
        M.configure([.25, .55]); self.assertIsNone(M.COMBINED); self.assertNotIn('cmb', M.WEIGHT)

    def test_rejects(self):
        with self.assertRaises(ValueError): M.layout([.25, .55], [[.5, .4]], dict(thr=.4))      # combined and levels are exclusive
        with self.assertRaises(ValueError): M.layout([.25, .4], None, dict(thr=.4))             # the reference arm w55 is required
        for t in (0, 1, 1.2, -.1, float('nan'), 'x', None):
            with self.assertRaises(ValueError): M.layout([.25, .55], None, dict(thr=t))
        with self.assertRaises(ValueError): M.layout([.25, .55], None, dict(name='x'))
        with self.assertRaises(ValueError): M.parse_weights('0.25,1.0')                         # the weight check is unchanged
        with self.assertRaises(ValueError): M.mix_w({}, 1.0)

    def test_mixture_is_the_mf_map(self):
        M.configure([.25, .55], None, dict(thr=.4))
        rng = np.random.default_rng(3); mf = (rng.integers(0, 511, (64, 64)).astype(np.float32) / np.float32(510))
        maps = dict(full=rng.random((64, 64), dtype=np.float32), local=rng.random((64, 64), dtype=np.float32), m2f_public=mf)
        p = M.mixture(maps, 'cmb')
        self.assertEqual(p.dtype, np.float32); self.assertTrue(np.array_equal(p.view(np.uint32), mf.view(np.uint32)))
        self.assertIsNot(p, mf)
        self.assertTrue(np.array_equal(M.mixture(maps, 'w55'), M.mix_w(maps, .55)))


class CombinedImages(unittest.TestCase):
    def setUp(self):
        T.Images.setUp(self)
        cmb = np.zeros((2048, 2048), np.uint16)
        cmb[400:412, 300:900] = 430; cmb[412:418, 300:900] = 230        # a weak edge: thr 0.4 takes it, 0.5 does not
        cmb[1600:1612, 200:700] = 480; cmb[1800:1830, 1500:1540] = 220       # 0.43: only at thr 0.4
        self.cmb = cmb
        W.G['znew']['cmb'] = {k: [{f: cmb for f in self.files}] for k in (0, 3)}

    def tearDown(self):
        W.configure(M.DEFAULT_WEIGHTS); T.Images.tearDown(self)

    def test_branch_and_rule(self):
        f = self.files[1]; W.configure([.25, .55], None, dict(thr=.4))
        a = W.branch_maps(f); b = W.branch_maps(f, 'cmb')
        self.assertTrue(np.array_equal(a['full'], b['full']) and np.array_equal(a['local'], b['local']))
        self.assertTrue(np.array_equal(b['m2f_public'], self.cmb.astype(np.float32) / np.float32(510)))
        self.assertTrue(np.array_equal(W.branch_maps(f, 'w55')['m2f_public'], M.mean_levels([self.new, self.new2])))   # the weight arms keep the chain2 maps
        masks, p, rows = W.candidate_rows(f, 'cmb')
        self.assertTrue(np.array_equal(p, b['m2f_public']))
        want = [np.asarray(m, bool) for m in iter_instances(p, SHIP_R3.with_(thr=.4))]
        self.assertEqual([W.packed(m) for m in masks], [W.packed(m) for m in want])
        self.assertNotEqual(len(want), len(list(iter_instances(p, SHIP_R3))))
        self.assertEqual(W.rule_of('cmb'), SHIP_R3.with_(thr=.4)); self.assertIs(W.rule_of('w55'), SHIP_R3); self.assertIs(W.rule_of('stored'), SHIP_R3)
        for r, g in zip(rows, MF.image_features(masks, b['m2f_public'])): self.assertEqual({n: r[n] for n in MF.FEATURES}, g)
        for r, g in zip(rows, NB.neighbour_features(masks, p)): self.assertEqual({n: r[n] for n in NB.FEATURES}, g)
        for r, m in zip(rows, masks): self.assertEqual({n: r[n] for n in C.WINDOW_FEATURES}, C.window_features(m, p))
        self.assertIn('thr=0.4', W.bank_text('cmb')); self.assertIn('combined', W.bank_text('cmb'))
        self.assertEqual(W.ship_role('cmb'), 'deployment_refit_chain2_cmb_full707_mf_mixture')

    def test_replay_arms_unchanged(self):
        f = self.files[1]
        W.configure([.25, .55]); x0 = {v: W.image_pass_variant(v, f)[1] for v in ('w25', 'w55')}; r0 = W.candidate_rows(f, 'w55')
        W.configure([.25, .55], None, dict(thr=.4)); x1 = {v: W.image_pass_variant(v, f)[1] for v in ('w25', 'w55')}; r1 = W.candidate_rows(f, 'w55')
        self.assertEqual(x0, x1); self.assertTrue(np.array_equal(r0[1], r1[1])); self.assertEqual(r0[2], r1[2])
        self.assertEqual([W.packed(m) for m in r1[0]], [W.packed(np.asarray(m, bool)) for m in iter_instances(r1[1], SHIP_R3)])
        self.assertEqual(W.bank_text('w55'), W.bank_text_chain2('w55'))


class Inventory(unittest.TestCase):
    """worker.inputs_inventory reads the combined archives (pinned, their own dataset scope) next to the chain2 map sets."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.t = Path(self.tmp.name)
        self.folds = {f'2011010{k}000000Ah': k for k in range(5)}; self.saved = (W.H, W.unique)
        W.H = self.t; W.unique = lambda name, scope: self.t / scope / name

    def tearDown(self):
        W.H, W.unique = self.saved; self.tmp.cleanup(); W.configure(M.DEFAULT_WEIGHTS)

    def write(self, scope, name, arrays):
        import hashlib
        d = self.t / scope; d.mkdir(exist_ok=True); p = d / name; np.savez(p, **arrays); b = p.read_bytes()
        return dict(file=name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), md5=hashlib.md5(b).hexdigest())

    def pins(self, cmb_dtype=np.uint16, cmb_keys=None):
        z = lambda k, dt=np.uint16: {f: np.zeros((2048, 2048), dt) for f, v in self.folds.items() if v == k}
        oof = {str(k): [dict(self.write('maps', f'oof_f{k}.npz', z(k)), run_id=f'r_f{k}', set=0)] for k in range(5)}
        c = {str(k): [self.write('cmbscope', f'cmb-oof_f{k}.npz', z(k, cmb_dtype) if cmb_keys is None or k else cmb_keys)] for k in range(5)}
        p = dict(label='x', folds=[f'r_f{k}' for k in range(5)], weights=[.25, .55], sets=1, dataset_scope='maps', oof=oof,
                 combined=dict(name='fx', thr=.4, dataset='u/cmbscope', dataset_scope='cmbscope', oof=c, test={}))
        (self.t / 'chain_inputs.json').write_text(json.dumps(p)); return p

    def test_reads_the_archives(self):
        self.pins(); out, rows = W.inputs_inventory(self.folds)
        self.assertEqual(sorted(out), ['cmb', 'new']); self.assertEqual(len(out['cmb'][0]['locals']), 1); self.assertEqual(out['cmb'][0]['denominator'], 510)
        self.assertTrue(out['cmb'][3]['locals'][0].endswith('cmbscope/cmb-oof_f3.npz'))
        self.assertEqual(rows['combined']['thr'], .4); self.assertEqual(rows['combined']['name'], 'fx'); self.assertTrue(rows['format_checked'])
        self.assertEqual(sorted(rows['combined']['archives']), [f'fold{k}' for k in range(5)])

    def test_rejects(self):
        self.pins(cmb_dtype=np.uint8)
        with self.assertRaises(AssertionError): W.inputs_inventory(self.folds)
        self.pins(cmb_keys={'other': np.zeros((2048, 2048), np.uint16)})
        with self.assertRaises(AssertionError): W.inputs_inventory(self.folds)
        p = self.pins(); p['combined']['oof']['2'][0]['sha256'] = '0' * 64; (self.t / 'chain_inputs.json').write_text(json.dumps(p))
        with self.assertRaises(AssertionError): W.inputs_inventory(self.folds)
        p = self.pins(); p['levels'] = [[.5]]; (self.t / 'chain_inputs.json').write_text(json.dumps(p))
        with self.assertRaises(AssertionError): W.inputs_inventory(self.folds)

    def test_without_option(self):
        p = self.pins(); del p['combined']; (self.t / 'chain_inputs.json').write_text(json.dumps(p))
        out, rows = W.inputs_inventory(self.folds); self.assertEqual(sorted(out), ['new']); self.assertNotIn('combined', rows)


class CombinedPlumbing(T.Plumbing):
    def fake_variant(self, v, f):
        f_, d = super().fake_variant(v, f)
        if v == 'cmb': d['map_change'] = dict(d['map_change'], mixture_mean_abs=.02)
        return f_, d

    def run_call(self, t, combined):
        orig = W.configure
        W.configure = lambda w, l=None, c=None: orig(w, l, combined)
        try: return self.run_main(t, weights=[.25, .55])
        finally: W.configure = orig; W.configure(M.DEFAULT_WEIGHTS)

    def test_main_end_to_end(self): pass                      # inherited tests run in test_worker_synthetic
    def test_variant_v1_taus_come_from_the_stored_bank_regressors(self): pass
    def test_selector_fits_exclude_query_groups(self): pass

    def test_combined_call(self):
        with tempfile.TemporaryDirectory() as t0, tempfile.TemporaryDirectory() as t1:
            a = self.run_call(Path(t0), None); b = self.run_call(Path(t1), dict(thr=.4, name='fixture'))
            self.assertEqual(b['counts.json']['arms'], ['inc', 'inc_pnat', 'inc_pcand_v1', 'pcombo', 'w25', 'w55', 'cmb'])
            rb = b['science_result.json']; self.assertTrue(rb['integrity_passed'])
            self.assertEqual(rb['combined'], dict(arm='cmb', thr=.4, name='fixture')); self.assertNotIn('combined', a['science_result.json'])
            self.assertEqual(rb['weight_arms']['cmb'], 1.0)
            for n in ('cmb_minus_pcombo', 'cmb_minus_w25'): self.assertIn(n, rb['subsets']['all707']['contrasts'])
            sel = [x for x in b['selectors.json']['models'] if x['arm'] == 'cmb']; self.assertEqual(len(sel), 100); self.assertEqual(len(sel[0]['coef'][0]), 68)
            self.assertEqual(rb['selector_fits'], 4 * 100 * 4)
            for s in a['counts.json']['per_partition_counts']:
                for arm in ('pcombo', 'w25', 'w55'): self.assertEqual(a['counts.json']['per_partition_counts'][s][arm], b['counts.json']['per_partition_counts'][s][arm])
            strip = lambda o: {f: {s: {k: v for k, v in kp.items() if k != 'cmb'} for s, kp in x.items()} for f, x in o['keeps.json']['keeps'].items()}
            self.assertEqual(strip(a), strip(b))
            self.assertEqual([x for x in a['selectors.json']['models']], [x for x in b['selectors.json']['models'] if x['arm'] != 'cmb'])
            for arm in ('pcombo', 'w25', 'w55'):
                self.assertEqual((Path(t0) / f'out/ship/{arm}_selector.json').read_bytes(), (Path(t1) / f'out/ship/{arm}_selector.json').read_bytes())
            rec = json.loads((Path(t1) / 'out/ship/cmb_selector.json').read_text()); self.assertEqual(rec['feature_order'], M.feature_names(T.FAKE39, T.CTX))
            self.assertIn('combined', rec['training']['bank'])
            ra = a['science_result.json']
            for c in ra['subsets']:
                self.assertEqual(ra['subsets'][c]['scores'], {k: v for k, v in rb['subsets'][c]['scores'].items() if k != 'cmb'})
                for n, v in ra['subsets'][c]['contrasts'].items(): self.assertEqual(v, rb['subsets'][c]['contrasts'][n], (c, n))


if __name__ == '__main__':
    unittest.main()
