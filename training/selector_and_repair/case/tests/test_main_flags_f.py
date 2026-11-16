"""chainF (order INT1): the worker's main() end to end on the synthetic harness (case/tests/test_worker_synthetic; stand-in
fits, no pools) with --no-w25 and --pcand-refit w55,w70: the chain outputs have no w25 arm, the refit phase runs after them
(100 refits per arm + the deployment fit twice), writes pcand_refit.json / deploy_fit.json / the regressor pickles, and
records no failure. Catches wiring errors in main() that the unit tests of the pieces cannot."""
import json, os, sys, tempfile, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / 'tests'))
import numpy as np
import test_worker_synthetic as T
W = T.W; M = T.M; C = T.C


class FlagsMain(T.Plumbing):
    def fake_variant(self, v, f):
        f_, d = super().fake_variant(v, f)
        if v in W.PCAND_ARMS:
            n = d['n']; rng = self.rng(f, 'pc' + v); patterns = [list(range(n)), []] if n else [[]]
            d.update(patterns=patterns, pattern_of=dict({f'{s}:{k}': 0 for s in self.seeds for k in range(5)}, deploy=0),
                     scores=[[[float(x) for x in rng.random(9)] for _ in key] for key in patterns],
                     counts=[[[1. + .01 * j, 2., 1., 1., 0., 0.] for j in range(9)] for _ in patterns], tau_path_equal=True)
        return f_, d

    def test_main_end_to_end(self):
        seeds = self.seeds
        def fake_p2(f):
            dz = dict(overlapping_prediction_pairs=0, gt_matched_by_more_than_one_prediction=0)
            return f, {v: {s: [1.3, 2., 1., 1., 0., 0.] for s in seeds} for v in W.PCAND_ARMS}, {s: dict(dup={v: dz for v in W.PCAND_ARMS}, stats={v: {} for v in W.PCAND_ARMS}) for s in seeds}
        saved = (W.main, W.deployed_selector, W.pc_pass2_refit); orig_main = W.main; T.FakeRegressor.fits = []
        def main():
            inv = W.inputs_inventory
            W.inputs_inventory = lambda fo: (lambda o: (o[0], dict(o[1], no_w25=True, pcand_refit=dict(arms=['w55', 'w70'], deployed_selector={}))))(inv(fo))
            return orig_main()
        try:
            W.main = main; W.deployed_selector = lambda pin: dict(feature_order=list(T.FAKE39), threshold=.35, scaler_mean=[0.] * 39, scaler_scale=[1.] * 39, coef=[[0.] * 39], intercept=[5.])
            W.pc_pass2_refit = fake_p2
            with tempfile.TemporaryDirectory() as t:
                o = self.run_main(Path(t), weights=[.55, .7]); out = Path(t) / 'out'
                self.assertFalse((out / 'pcand_refit_failure.json').is_file(), (out / 'pcand_refit_failure.json').read_text() if (out / 'pcand_refit_failure.json').is_file() else '')
                r = o['science_result.json']; self.assertEqual(r['weights'], [.55, .7]); self.assertNotIn('w25', r['weight_arms'])
                self.assertEqual(sorted({x['arm'] for x in o['selectors.json']['models']}), ['pcombo', 'w55', 'w70'])
                pc = json.loads((out / 'pcand_refit.json').read_text()); self.assertEqual(pc['status'], 'complete')
                self.assertEqual(pc['arms'], ['pcombo', 'w55_raw', 'w55_frozen', 'w55_refit', 'w70_raw', 'w70_frozen', 'w70_refit']); self.assertEqual(pc['v1_refits'], 200)
                self.assertIn('w55_refit_minus_w55_frozen', pc['subsets']['all707']['contrasts'])
                d = json.loads((out / 'deploy_fit.json').read_text()); self.assertEqual(sorted(d), ['w55', 'w70'])
                for v in ('w55', 'w70'): self.assertTrue((out / f'deploy/{v}_regressor.pkl').is_file())
                cnt = json.loads((out / 'pcand_counts.json').read_text())['per_partition_counts']
                for s in seeds: self.assertEqual(cnt[str(s)]['w55_frozen'], o['counts.json']['per_partition_counts'][str(s)]['w55'])
        finally:
            W.main, W.deployed_selector, W.pc_pass2_refit = saved; W.set_pcand([])

    def test_variant_v1_taus_come_from_the_stored_bank_regressors(self): pass
    def test_selector_fits_exclude_query_groups(self): pass


if __name__ == '__main__': unittest.main()
