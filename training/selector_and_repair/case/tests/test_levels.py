"""chain2m (order MT1): multi-threshold candidates. Pure helpers in levels.py, tested on synthetic inputs.

- level sets: '0.5;0.5,0.4' -> [[0.5], [0.5, 0.4]]; each level on the 0.01 grid in (0, 1), strictly descending inside a set
- arm tag: [0.5] -> '' (the SHIP_R3 arm itself), [0.5, 0.4] -> 'l50l40'
- overlap resolution: greedy, highest selector probability first (ties: lower candidate index first), a candidate that shares
  a pixel with an already kept candidate is dropped; applied after the cutoff it equals the greedy survivors with p >= cutoff
- nesting: parent of a candidate = the candidate of the next (lower) level that overlaps it most
"""
import sys, unittest
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / 'frozen'), str(H / 'frozen/eval'), str(H)]
import levels as L


def brute_greedy(p, pairs, tau):
    """Reference: threshold first, then greedy on the survivors of the threshold."""
    nb = {i: set() for i in range(len(p))}
    for i, j in pairs: nb[i].add(j); nb[j].add(i)
    cand = sorted([i for i in range(len(p)) if p[i] >= tau], key=lambda i: (-p[i], i)); kept = []
    for i in cand:
        if not nb[i] & set(kept): kept.append(i)
    return sorted(kept)


class LevelSets(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(L.parse_levelsets('0.5;0.5,0.4'), [[0.5], [0.5, 0.4]])
        self.assertEqual(L.parse_levelsets('0.5,0.4,0.3'), [[0.5, 0.4, 0.3]])
        self.assertEqual(L.parse_levelsets(' 0.5 , 0.40 '), [[0.5, 0.4]])

    def test_bad(self):
        for bad in ('', '0.4,0.5', '0.5,0.5', '0.5;0.5', '0.455', '1.0', '0', 'a', '0.5,', '0.5;;0.4'):
            with self.assertRaises(ValueError, msg=bad): L.parse_levelsets(bad)

    def test_tag(self):
        self.assertEqual(L.level_tag([0.5]), ''); self.assertEqual(L.level_tag([0.5, 0.4]), 'l50l40')
        self.assertEqual(L.level_tag([0.5, 0.4, 0.3]), 'l50l40l30'); self.assertEqual(L.level_tag([0.45]), 'l45')
        self.assertEqual(L.level_tag(None), '')


class Resolution(unittest.TestCase):
    def test_no_pairs_keeps_everything_over_the_cutoff(self):
        p = np.array([.9, .2, .5, .35]); self.assertTrue(L.survivors(p, []).all())
        self.assertEqual(L.resolve_keep(p, .35, []), [0, 2, 3])
        self.assertEqual(L.resolve_keep(p, .35, []), np.flatnonzero(p >= .35).tolist())

    def test_highest_first_drops_overlapping(self):
        # 0: a 0.5 piece, 1: another 0.5 piece, 2: the 0.4 parent that contains both
        pairs = [(0, 2), (1, 2)]
        self.assertEqual(L.resolve_keep(np.array([.6, .7, .8]), .35, pairs), [2])         # the parent wins
        self.assertEqual(L.resolve_keep(np.array([.6, .7, .65]), .35, pairs), [0, 1])     # the children win; parent dropped
        self.assertEqual(L.resolve_keep(np.array([.9, .1, .8]), .35, pairs), [0])         # child 1 under the cutoff
        self.assertEqual(L.resolve_keep(np.array([.3, .3, .8]), .35, pairs), [2])

    def test_dropped_candidate_does_not_block(self):
        # chain 0-1-2: 0 kept, 1 dropped (overlaps 0), 2 kept (overlaps only the dropped 1)
        self.assertEqual(L.resolve_keep(np.array([.9, .8, .7]), 0., [(0, 1), (1, 2)]), [0, 2])

    def test_ties_lower_index_first(self):
        self.assertEqual(L.resolve_keep(np.array([.5, .5]), 0., [(0, 1)]), [0])

    def test_cutoff_commutes_with_greedy(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(1, 9)); p = rng.random(n).round(2)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < .3]
            for tau in (0., .25, .5, .75):
                self.assertEqual(L.resolve_keep(p, tau, pairs), brute_greedy(p, pairs, tau))
                s = L.survivors(p, pairs); self.assertEqual(sorted(np.flatnonzero(s & (p >= tau)).tolist()), brute_greedy(p, pairs, tau))

    def test_kept_never_overlap(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            n = int(rng.integers(2, 9)); p = rng.random(n); pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < .5]
            k = set(L.resolve_keep(p, .2, pairs)); self.assertFalse(any(i in k and j in k for i, j in pairs))

    def test_resolve_probabilities_for_the_sweep(self):
        # two images in a bank: ids 0..2 (pairs (0,2)) and 3..4 (no pair); a non-survivor gets -1 (below every grid cutoff)
        records = {'a': dict(candidate_ids=[0, 1, 2]), 'b': dict(candidate_ids=[3, 4])}
        prob = np.array([.6, .4, .8, .3, .9]); out = L.resolve_probabilities(prob, records, ['a', 'b'], {'a': [(0, 2)], 'b': []})
        self.assertEqual(out.tolist(), [-1., .4, .8, .3, .9]); self.assertEqual(prob.tolist(), [.6, .4, .8, .3, .9])   # input untouched
        for t in (0., .35, .7):
            keep = out >= t
            self.assertEqual(np.flatnonzero(keep[:3]).tolist(), L.resolve_keep(prob[:3], t, [(0, 2)]))
        same = L.resolve_probabilities(prob, records, ['a', 'b'], {'a': [], 'b': []}); self.assertTrue(np.array_equal(same, prob))


def box(y0, y1, x0, x1, n=64):
    m = np.zeros((n, n), bool); m[y0:y1, x0:x1] = True; return m


class Nesting(unittest.TestCase):
    def test_pairs_and_parents(self):
        # level 0.5: two pieces; level 0.4: one component containing both, plus a separate one
        masks = [box(10, 20, 5, 15), box(10, 20, 20, 30), box(8, 22, 3, 32), box(40, 50, 40, 50)]
        lev = [.5, .5, .4, .4]
        self.assertEqual(L.overlap_pairs(masks), [(0, 2), (1, 2)])
        nest = L.nesting(masks, lev, [.5, .4])
        self.assertEqual(nest['parent'], [2, 2, -1, -1])
        self.assertEqual(nest['children'], [[], [], [0, 1], []])
        self.assertEqual(nest['identical_to_parent'], [False, False, False, False])

    def test_largest_overlap_parent_and_identity(self):
        masks = [box(10, 20, 10, 20), box(0, 15, 0, 15), box(14, 30, 14, 30), box(10, 20, 10, 20)]
        lev = [.5, .4, .4, .3]
        # candidate 0 overlaps 1 (5x5) and 2 (6x6) -> parent 2; 1 and 2 at 0.4 overlap 3 (0.3, identical to 0)
        nest = L.nesting(masks, lev, [.5, .4, .3])
        self.assertEqual(nest['parent'][0], 2); self.assertEqual(nest['parent'][3], -1)
        self.assertEqual(nest['parent'][1], 3); self.assertEqual(nest['parent'][2], 3)
        self.assertEqual(L.nesting([box(1, 5, 1, 5), box(1, 5, 1, 5)], [.5, .4], [.5, .4])['identical_to_parent'], [True, False])

    def test_one_level(self):
        masks = [box(1, 5, 1, 5), box(10, 15, 10, 15)]
        self.assertEqual(L.overlap_pairs(masks), [])
        self.assertEqual(L.nesting(masks, [.5, .5], [.5])['parent'], [-1, -1])


if __name__ == '__main__':
    unittest.main()
