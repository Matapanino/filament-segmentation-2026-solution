"""chainF (order INT1) tests, written before the code: the P-cand v1 refit of pcand (order PC1) inside the chain kernel.

Reference = pcand's own pworker.py (byte copy in case/; never run as a main here): with the flag the chain worker's arm pass
adds exactly pworker.image_pass_arm's keep patterns and 9-tau tables (the chain keys unchanged); the 100 refits, the
deployment fit and the refit pass 2 equal pworker's fit_task_arm, deployment_fit and pass2_arm('refit') on the same state.
Without the flag nothing is added (chain2m exactly). The flag refuses arms not in the call and multi-level arms (the C5 LR39
row rule is not defined for overlapping level versions; a design decision, 2026-11-14).
"""
import hashlib, json, os, sys, tempfile, types, unittest
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'tests')]
import test_pworker as TP                     # pcand's synthetic scene and fake banks (byte copy)
from test_pworker import W, PW, T, C, M, FakeRegressor, FakePool

CHAIN_KEYS = {'n', 'rle', 'hashes', 'features', 'n_gt_sets', 'raw_iou', 'map_change'}
PCAND_KEYS = ('patterns', 'pattern_of', 'scores', 'counts', 'tau_path_equal')


class Images(TP.Images):
    def tearDown(self):
        W.set_pcand([]); super().tearDown()

    def test_flag_absent_adds_nothing(self):
        W.set_pcand([]); f = self.files[1]
        for v in ('w40', 'w55'):
            _, a = W.image_pass_variant(v, f); self.assertEqual(set(a), CHAIN_KEYS, v)

    def test_flagged_arm_pass_equals_pworker(self):
        f = self.files[1]; W.set_pcand([]); _, plain = W.image_pass_variant('w55', f)
        W.set_pcand(['w55']); _, a = W.image_pass_variant('w55', f); _, ref = PW.image_pass_arm('w55', f)
        self.assertEqual(set(a), CHAIN_KEYS | set(PCAND_KEYS))
        for k in CHAIN_KEYS: self.assertEqual(a[k], plain[k], k)                     # the chain arm is unchanged
        for k in PCAND_KEYS: self.assertEqual(a[k], ref[k], k)                       # pcand's phase A exactly
        _, other = W.image_pass_variant('w40', f); self.assertEqual(set(other), CHAIN_KEYS)

    def test_refit_pass2_equals_pworker(self):
        f = self.files[1]; tau = lambda n, salt: [C.GRID[(2 * j + salt) % 9] for j in range(n)]
        V = self.pass2_state(f, tau); W.G['p2'] = ['refit']; W.G['p2_arms'] = ['w55', 'w40']
        _, ref, ref_info = PW.pass2_arm(f)
        W.set_pcand(['w55', 'w40'])
        W.G.update(keeps2={f: {7: {v: W.G['keeps_arm'][v][f][7] for v in V}}}, taus_refit={v: W.G['taus'][v]['refit'] for v in V})
        _, rows, info = W.pc_pass2_refit(f)
        self.assertEqual(sorted(rows), ['w40', 'w55'])
        for v in ('w40', 'w55'):
            self.assertEqual(rows[v][7], ref[f'{v}_refit'][7])
            self.assertEqual(info[7]['stats'][v], ref_info[7]['stats'][f'{v}_refit']); self.assertEqual(info[7]['dup'][v], ref_info[7]['dup'][f'{v}_refit'])


class Plumbing(TP.Plumbing):
    def state(self):
        AV = {v: {f: self.fake_pass_a(v, f)[1] for f in self.files} for v in self.VS}
        W.G.clear(); W.G.update(AV=AV, files=self.files, names_v1=self.names_v1, outer=self.outer, groups=self.groups, models=self.models)
        return AV

    def setUp(self):
        super().setUp(); FakeRegressor.fits = []; self.saved = (W.make_regressor, PW.make_regressor); W.make_regressor = FakeRegressor; PW.make_regressor = FakeRegressor

    def tearDown(self):
        W.make_regressor, PW.make_regressor = self.saved; W.G.clear(); W.set_pcand([]); W.configure(M.DEFAULT_WEIGHTS)

    def test_fit_task_equals_pworker(self):
        self.state()
        for v in self.VS:
            for s in self.seeds[:3]:
                for k in range(5):
                    a = W.pc_fit_task((v, s, k)); b = PW.fit_task_arm((v, s, k)); self.assertEqual(a, b)

    def test_deployment_fit_equals_pworker(self):
        AV = self.state(); cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            try:
                os.chdir(t1); a = W.pc_deployment_fit('w55', AV['w55'], self.files, self.names_v1); pa = Path('out/deploy/w55_regressor.pkl').read_bytes()
                os.chdir(t2); b = PW.deployment_fit('w55', AV['w55'], self.files, self.names_v1); pb = Path('out/deploy/w55_regressor.pkl').read_bytes()
            finally: os.chdir(cwd)
        self.assertEqual(a, b); self.assertEqual(pa, pb); self.assertEqual(a['regressor_sha256'], hashlib.sha256(pa).hexdigest())

    def test_phase_reports_in_pcand_format(self):
        """The refit phase: per-partition counts named as pcand (pcombo, <v>_raw, <v>_frozen = the chain arm, <v>_refit), the
        contrasts of pcand without the global-tau ones, and the same statistics as pt.summarize on those counts."""
        AV = self.state(); W.configure([.25, .4, .55]); W.set_pcand(['w55'])
        seeds = self.seeds; files = self.files
        per = {s: dict(pcombo=[[1.3, 2., 1., 1., 0., 0.]] * len(files), w55=[[1.2 + .01 * i, 2., 1., 1., 0., 0.] for i in range(len(files))]) for s in seeds}
        E = {f: [None, {s: dict(raw_counts=dict(w55=[1., 2., 1., 1., 0., 0.])) for s in seeds}, 0] for f in files}
        keeps2 = {f: {s: dict(w55=tuple(range(AV['w55'][f]['n']))) for s in seeds} for f in files}
        fake_mp = types.SimpleNamespace(get_context=lambda name: types.SimpleNamespace(Pool=FakePool))
        dz = dict(overlapping_prediction_pairs=0, gt_matched_by_more_than_one_prediction=0)
        fake_p2 = lambda f: (f, {'w55': {s: [1.4, 2., 1., 1., 0., 0.] for s in seeds}}, {s: dict(dup={'w55': dz}, stats={'w55': {}}) for s in seeds})
        saved = (W.mp, W.pc_pass2_refit, W.pool_map, W.progress); cwd = os.getcwd()
        cfg = dict(partition_seeds=seeds, bootstrap=dict(draws=50, seed=1))
        query = [f for f in files if self.folds[f] in (0, 1, 2)]
        with tempfile.TemporaryDirectory() as t:
            try:
                os.chdir(t); W.mp = fake_mp; W.progress = lambda **kw: None
                W.pool_map = lambda fn, items, stage, state, every=20: {f: list(fake_p2(f)[1:]) for f in items}
                out = W.pcand_refit_phase(AV, keeps2, E, per, files, self.folds, self.groups, self.outer, seeds, cfg, query, {}, self.names_v1)
                saved_files = sorted(p.name for p in Path('out').rglob('*') if p.is_file())
            finally:
                os.chdir(cwd); W.mp, W.pc_pass2_refit, W.pool_map, W.progress = saved
        self.assertEqual(out['arms'], ['pcombo', 'w55_raw', 'w55_frozen', 'w55_refit'])
        names = {n for n, _, _ in out['contrasts_spec']}
        self.assertEqual(names, {n for n, _, _ in T.contrasts_of(['w55']) if 'gtau' not in n})
        p2 = out['per_partition_counts']
        for s in seeds:
            self.assertEqual(p2[s]['w55_frozen'], per[s]['w55']); self.assertEqual(p2[s]['pcombo'], per[s]['pcombo'])
        ref = T.summarize(p2, files, files, self.groups, cfg, out['arms'], out['contrasts_spec'])
        self.assertEqual(out['subsets']['all707']['contrasts']['w55_refit_minus_w55_frozen'], ref['contrasts']['w55_refit_minus_w55_frozen'])
        self.assertIn('no_fold2', out['subsets'])
        self.assertEqual(len(FakeRegressor.fits), 100 + 2)          # 100 outer refits + the deployment fit twice
        for n in ('pcand_refit.json', 'pcand_counts.json', 'pcand_taus.json', 'pcand_fits.json', 'deploy_fit.json', 'w55_regressor.pkl'): self.assertIn(n, saved_files)


class Flag(unittest.TestCase):
    def tearDown(self): W.set_pcand([]); W.configure(M.DEFAULT_WEIGHTS)

    def test_absent_is_empty(self):
        W.configure([.25, .55]); W.set_pcand(None); self.assertEqual(W.PCAND_ARMS, [])

    def test_refuses_unknown_and_multilevel(self):
        W.configure([.25, .55]); W.set_pcand(['w55']); self.assertEqual(W.PCAND_ARMS, ['w55'])
        with self.assertRaises(AssertionError): W.set_pcand(['w70'])
        with self.assertRaises(AssertionError): W.set_pcand(['pcombo'])
        with self.assertRaises(AssertionError): W.set_pcand(['w55', 'w55'])
        W.configure([.25, .55], [[.5], [.5, .4]]); W.set_pcand(['w55'])           # [0.5] = one level: the plain arm, allowed
        with self.assertRaises(AssertionError): W.set_pcand(['w55l50l40'])


if __name__ == '__main__': unittest.main()
