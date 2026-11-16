"""Tests of mt.py (chain2 / order CH2; written before mt.py was changed; adapted from chain CH1): the weight-arm layout,
the fold-token parser, the multi-set MF map average, the weighted blend (w = 0.25 identical to the shipped mixture), size
bins, the 707 gate and the verdict rule."""
import sys, unittest
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H / 'support'), str(H)]
import numpy as np
import mt as M
import neigh as NB
import mf_features as MF
from contract import mix


def contrast(lower=.001, pos=18, delta=.002):
    ds = [delta] * pos + [-delta] * (20 - pos)
    return dict(delta=delta, mean_paired_partition_delta=sum(ds) / 20, positive_of20=pos, partition_deltas=ds, paired_group_CI95=[lower, lower + .004])


def result(**kw):
    c = {M.GATE_CONTRAST: contrast(**kw)}
    subsets = dict(all707=dict(images=707, contrasts=c), **{f'fold{k}': dict(images=141, contrasts={M.GATE_CONTRAST: dict(mean_paired_partition_delta=.001)}) for k in range(5)})
    return dict(replay_passed=True, integrity_passed=True, inputs_passed=True, subsets=subsets)


class Weights(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(M.parse_weights('0.25,0.40,0.55'), [.25, .40, .55])
        self.assertEqual(M.parse_weights('0.55,0.25'), [.25, .55])                      # sorted
        self.assertEqual(M.parse_weights([.4, .25]), [.25, .4])
        for bad in ('0.40,0.55', '0.25,0.25', '0.25,1.0', '0.25,0', '0.25,0.333', '0.25,x', '', '0.25,-0.2'):
            with self.assertRaises(ValueError, msg=bad): M.parse_weights(bad)

    def test_arm_names(self):
        self.assertEqual([M.arm_name(w) for w in (.25, .4, .55, .05)], ['w25', 'w40', 'w55', 'w05'])

    def test_default_layout(self):
        self.assertEqual(M.DEFAULT_WEIGHTS, (.25, .40, .55))
        L = M.layout(M.DEFAULT_WEIGHTS)
        self.assertEqual(L['ARMS'], ['inc', 'inc_pnat', 'inc_pcand_v1', 'pcombo', 'w25', 'w40', 'w55'])
        self.assertEqual(L['PASS2_ARMS'], ['inc_pcand_v1', 'pcombo', 'w25', 'w40', 'w55'])
        self.assertEqual(L['SELECTOR_ARMS'], ['pcombo', 'w25', 'w40', 'w55'])
        self.assertEqual(L['NEW_VARIANTS'], ['w25', 'w40', 'w55'])
        self.assertEqual([L['VARIANT'][a] for a in L['ARMS']], ['stored'] * 4 + ['w25', 'w40', 'w55'])
        self.assertEqual(L['WEIGHT'], dict(w25=.25, w40=.40, w55=.55))
        names = [c[0] for c in L['CONTRASTS']]
        self.assertEqual(names[:5], ['w25_minus_pcombo', 'w40_minus_pcombo', 'w55_minus_pcombo', 'w40_minus_w25', 'w55_minus_w25'])
        self.assertEqual(names[5:], M.REPLAY_CONTRASTS)
        self.assertEqual(L['ARM_CONTRASTS'], names[:5])
        for n, a, b in L['CONTRASTS']: self.assertEqual(n, f'{a}_minus_{b}'); self.assertIn(a, L['ARMS']); self.assertIn(b, L['ARMS'])
        self.assertEqual(M.GATE_CONTRAST, 'w25_minus_pcombo')

    def test_configure_sets_module_layout(self):
        try:
            M.configure([.25, .4])
            self.assertEqual(M.ARMS[-2:], ['w25', 'w40']); self.assertEqual(M.NEW_VARIANTS, ['w25', 'w40']); self.assertEqual(M.WEIGHTS, [.25, .4])
            self.assertEqual([c[0] for c in M.CONTRASTS][:3], ['w25_minus_pcombo', 'w40_minus_pcombo', 'w40_minus_w25'])
        finally:
            M.configure(M.DEFAULT_WEIGHTS)
        self.assertEqual(M.NEW_VARIANTS, ['w25', 'w40', 'w55'])

    def test_feature_names_are_pcombo_68(self):
        o39 = [f'q{i}' for i in range(39)]
        ctx = [f'context_{r}_{s}' for r in (16, 64, 128) for s in ('mean', 'std', 'dark_fraction')] + ['mf_mean', 'full_mf_abs_mean', 'local_mf_abs_mean',
               'branches_range_mean', 'branches_all_high_fraction', 'branches_one_high_fraction']
        n = M.feature_names(o39, ctx); self.assertEqual(len(n), 68); self.assertEqual(n, o39 + ctx + NB.FEATURES + MF.FEATURES)


class Tokens(unittest.TestCase):
    def test_single_and_multi(self):
        self.assertEqual(M.parse_token('c768_f0'), ['c768_f0'])
        self.assertEqual(M.parse_token('c768_f0+c1024_f0'), ['c768_f0', 'c1024_f0'])
        for bad in ('', 'a+', '+a', 'a+a', 'stored', 'a+stored', 'a b', '../x', 'a/b'):
            with self.assertRaises(ValueError, msg=bad): M.parse_token(bad)

    def test_fold_sources(self):
        s = M.fold_sources(['a0+b0', 'a1+b1', 'w1+w2', 'a3+b3', 'a4+b4'])
        self.assertEqual(s, {0: ['a0', 'b0'], 1: ['a1', 'b1'], 2: ['w1', 'w2'], 3: ['a3', 'b3'], 4: ['a4', 'b4']})
        self.assertEqual(M.n_sets(s), 2)
        self.assertEqual(M.fold_sources(['c768_f0', 'c768_f1', 'w1_crop768_4k', 'c768_f3', 'c768_f4'])[2], ['w1_crop768_4k'])
        with self.assertRaises(ValueError): M.fold_sources(['a'] * 4)                         # five folds
        with self.assertRaises(ValueError): M.fold_sources(['a0+b0', 'a1', 'a2+b2', 'a3+b3', 'a4+b4'])   # same set count per fold
        with self.assertRaises(ValueError): M.fold_sources(['stored', 'a', 'b', 'c', 'd'])   # no stored folds in chain2


class Averaging(unittest.TestCase):
    def test_one_set_is_the_single_map_rule_bit_for_bit(self):
        a = np.random.default_rng(0).integers(0, 511, size=(64, 64)).astype(np.uint16)
        m = M.mean_levels([a]); self.assertEqual(m.dtype, np.float32)
        self.assertTrue(np.array_equal(m, a.astype(np.float32) / 510))
        self.assertTrue(np.array_equal(m, a.astype(np.float32) / M.DENOMINATOR))           # chain CH1 arm new: new / 510

    def test_two_sets_level_sum_over_1020(self):
        r = np.random.default_rng(1); a = r.integers(0, 511, size=(64, 64)).astype(np.uint16); b = r.integers(0, 511, size=(64, 64)).astype(np.uint16)
        m = M.mean_levels([a, b]); self.assertEqual(m.dtype, np.float32)
        self.assertTrue(np.array_equal(m, (a.astype(np.float32) + b.astype(np.float32)) / np.float32(1020)))
        self.assertTrue(np.allclose(m, (a.astype(float) + b.astype(float)) / 1020, rtol=0, atol=1e-7))
        full = np.full((2, 2), 510, np.uint16); self.assertTrue(np.array_equal(M.mean_levels([full, full, full]), np.ones((2, 2), np.float32)))
        self.assertTrue(np.array_equal(M.mean_levels([a, a]), M.mean_levels([a])))        # equal maps average to the map itself

    def test_rejects(self):
        a = np.zeros((4, 4), np.uint16)
        with self.assertRaises(AssertionError): M.mean_levels([a.astype(np.int32)])
        with self.assertRaises(AssertionError): M.mean_levels([np.full((4, 4), 511, np.uint16)])
        with self.assertRaises(AssertionError): M.mean_levels([a, np.zeros((4, 5), np.uint16)])
        with self.assertRaises(AssertionError): M.mean_levels([])

    def test_variant_mf(self):
        a = np.array([[0, 510, 100, 7]], np.uint16); b = np.array([[10, 400, 301, 7]], np.uint16)
        for v in M.NEW_VARIANTS:
            self.assertTrue(np.array_equal(M.variant_mf(v, dict(new=[a, b])), M.mean_levels([a, b])))
        with self.assertRaises(KeyError): M.variant_mf('stored', dict(new=[a]))


class Blend(unittest.TestCase):
    def maps(self, seed=0, shape=(64, 64)):
        r = np.random.default_rng(seed)
        return dict(m2f_public=r.integers(0, 511, size=shape).astype(np.float32) / 510, full=r.integers(0, 256, size=shape).astype(np.float32) / 255,
                    local=r.integers(0, 256, size=shape).astype(np.float32) / 255)

    def test_w25_is_the_shipped_mixture_bit_for_bit(self):
        for s in range(5):
            m = self.maps(s); p = M.mix_w(m, .25)
            self.assertEqual(p.dtype, np.float32); self.assertTrue(np.array_equal(p, mix(m, 'incumbent')))
        m = self.maps(9); self.assertTrue(np.array_equal(M.mix_w(dict(local=m['local'], full=m['full'], m2f_public=m['m2f_public']), .25), mix(m, 'incumbent')))

    def test_other_weights(self):
        m = self.maps(3)
        for w in (.4, .55):
            p = M.mix_w(m, w); self.assertEqual(p.dtype, np.float32)
            self.assertTrue(np.allclose(p, w * m['m2f_public'].astype(float) + (1 - w) / 2 * (m['full'].astype(float) + m['local'].astype(float)), atol=1e-6))
            self.assertFalse(np.array_equal(p, mix(m, 'incumbent')))
            one = {k: np.ones((2, 2), np.float32) for k in m}; self.assertTrue((M.mix_w(one, w) <= 1).all())
        with self.assertRaises(ValueError): M.mix_w(m, 1.2)

    def test_mixture_dispatch(self):
        m = self.maps(4)
        self.assertTrue(np.array_equal(M.mixture(m, 'stored'), mix(m, 'incumbent')))
        for v, w in M.WEIGHT.items(): self.assertTrue(np.array_equal(M.mixture(m, v), M.mix_w(m, w)))


class Protocol(unittest.TestCase):
    def test_protocol_matches_the_worker_asserts(self):
        import json
        import presel as P
        cfg = json.loads((H / 'protocol.json').read_text())
        self.assertEqual(cfg['fixed_arms'], M.FIXED_ARMS); self.assertEqual(cfg['gate_contrast'], M.GATE_CONTRAST)
        self.assertEqual(cfg['replay_contrasts'], M.REPLAY_CONTRASTS); self.assertEqual(cfg['threshold_grid'], P.THRESHOLD_GRID)
        self.assertEqual(cfg['mf_features'], MF.FEATURES); self.assertEqual(cfg['neighbour_features'], NB.FEATURES); self.assertIs(cfg['score_started'], False)


class SizeBins(unittest.TestCase):
    def test_bins(self):
        self.assertEqual([M.size_bin(x) for x in (0, 255, 256, 1024, 1025)], ['lt256', 'lt256', '256to1024', '256to1024', 'gt1024'])

    def test_counts(self):
        q0 = np.array([[.8, 0., 0.], [.3, .2, 0.]]); q1 = np.array([[0., .6, .1]])
        out = M.size_counts([q0, q1], [[100, 2000], [500]], [300, 50, 5000])
        self.assertEqual(out['tp'], dict(lt256=1, **{'256to1024': 1}, gt1024=0))
        self.assertEqual(out['fn'], dict(lt256=0, **{'256to1024': 0}, gt1024=1))
        self.assertEqual(out['fp'], dict(lt256=1, **{'256to1024': 1}, gt1024=2))


class Gate(unittest.TestCase):
    def test_passes(self): self.assertTrue(M.gate707(result()))

    def test_ci_lower_must_be_positive(self):
        self.assertFalse(M.gate707(result(lower=0.))); self.assertFalse(M.gate707(result(lower=-.001)))

    def test_sixteen_of_twenty(self):
        self.assertTrue(M.gate707(result(pos=16))); self.assertFalse(M.gate707(result(pos=15)))

    def test_each_fold_mean_positive(self):
        r = result(); r['subsets']['fold3']['contrasts'][M.GATE_CONTRAST]['mean_paired_partition_delta'] = 0.
        self.assertFalse(M.gate707(r))

    def test_blocking_checks(self):
        for k in ('replay_passed', 'integrity_passed', 'inputs_passed'):
            r = result(); r[k] = False; self.assertFalse(M.gate707(r))

    def test_verdict(self):
        self.assertEqual(M.final_verdict(True, True), 'confirmed')
        self.assertEqual(M.final_verdict(False, True), 'Improvement could not be confirmed')
        self.assertTrue(M.final_verdict(True, False).startswith('no verdict'))


if __name__ == '__main__': unittest.main()
