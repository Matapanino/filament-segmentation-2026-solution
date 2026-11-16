"""Synthetic specification tests for mf_features.py (the 8 MF-derived candidate features), written before mf_features.py.

Each feature is checked against a hand-computed value on small synthetic maps. The no-annotation guarantee is checked
structurally: the functions accept only a candidate mask and the MF probability map (plus the components derived from that
map), and the module reads nothing else.
"""
import ast, inspect, sys, unittest
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H)]
import mf_features as MF


def blank(n=96): return np.zeros((n, n), np.float32)


class Names(unittest.TestCase):
    def test_fixed_list(self):
        self.assertEqual(MF.FEATURES, ['mfr_q10', 'mfr_q50', 'mfr_q90', 'mfr_frac_ge50', 'mfr_blob_iou', 'mfr_ring_area_ratio', 'mfr_ring_max', 'mfr_blob_count'])
        self.assertEqual((MF.THRESHOLD, MF.RING, MF.CONNECTIVITY), (.5, 16, 8))


class Values(unittest.TestCase):
    def test_percentiles_and_fraction(self):
        mf = blank(); mask = np.zeros(mf.shape, bool); mask[40:50, 40:50] = True
        vals = np.linspace(0, .99, 100, dtype=np.float32).reshape(10, 10); mf[40:50, 40:50] = vals
        r = MF.image_features([mask], mf)[0]
        v = mf[mask].astype(np.float64)
        self.assertEqual(r['mfr_q10'], float(np.percentile(v, 10))); self.assertEqual(r['mfr_q50'], float(np.percentile(v, 50)))
        self.assertEqual(r['mfr_q90'], float(np.percentile(v, 90)))
        self.assertEqual(r['mfr_frac_ge50'], float((mf[mask] >= .5).mean()))
        self.assertEqual(list(r), MF.FEATURES)

    def test_blob_iou_union_of_overlapping_components(self):
        # candidate 10x10 = 100 px; component A (8-connected) covers 50 px of it and extends 50 px outside; component B lies
        # elsewhere and does not overlap; component C overlaps 10 px and has 30 px in total.
        mf = blank(); mask = np.zeros(mf.shape, bool); mask[20:30, 20:30] = True
        mf[20:25, 20:40] = .8                      # A: 5x20 = 100 px, 50 inside the candidate
        mf[70:80, 70:80] = .9                      # B: no overlap
        mf[29, 20:30] = .6; mf[30:31, 20:30] = .6; mf[31, 20:30] = .6     # C: 3 rows x 10 = 30 px, 10 inside (row 29)
        r = MF.image_features([mask], mf)[0]
        inter = 50 + 10; union = 100 + 100 + 30 - inter
        self.assertAlmostEqual(r['mfr_blob_iou'], inter / union, places=15)
        self.assertEqual(r['mfr_blob_count'], 2.)
        self.assertEqual(r['mfr_frac_ge50'], .6)

    def test_diagonal_contact_is_one_component(self):
        mf = blank(); mask = np.zeros(mf.shape, bool); mask[10:12, 10:12] = True
        mf[10:12, 10:12] = .7; mf[12, 12] = .7; mf[13, 13] = .7          # 8-connected chain
        r = MF.image_features([mask], mf)[0]
        self.assertEqual(r['mfr_blob_count'], 1.); self.assertAlmostEqual(r['mfr_blob_iou'], 4 / 6, places=15)

    def test_no_overlap(self):
        mf = blank(); mask = np.zeros(mf.shape, bool); mask[10:20, 10:20] = True; mf[60:70, 60:70] = .95
        r = MF.image_features([mask], mf)[0]
        self.assertEqual((r['mfr_blob_iou'], r['mfr_blob_count'], r['mfr_frac_ge50']), (0., 0., 0.))

    def test_threshold_is_inclusive(self):
        mf = blank(); mask = np.zeros(mf.shape, bool); mask[10:20, 10:20] = True; mf[10:20, 10:20] = .5
        r = MF.image_features([mask], mf)[0]
        self.assertEqual((r['mfr_frac_ge50'], r['mfr_blob_count'], r['mfr_blob_iou']), (1., 1., 1.))
        mf[10:20, 10:20] = np.nextafter(np.float32(.5), np.float32(0))
        r = MF.image_features([mask], mf)[0]
        self.assertEqual((r['mfr_frac_ge50'], r['mfr_blob_count']), (0., 0.))

    def test_ring(self):
        # 16 px ring = square dilation of radius 16 minus the candidate (as context_features' ring), clipped at the border.
        mf = blank(128); mask = np.zeros(mf.shape, bool); mask[50:60, 50:60] = True          # area 100
        mf[50:60, 61:66] = .7          # 50 px inside the ring (columns 61-65, within 16 px)
        mf[50:60, 80:90] = .99         # outside the ring (column 80 is 21 px away): not counted, not the ring max
        mf[40, 40] = .3                # inside the ring (10 px diagonal), below 0.5
        r = MF.image_features([mask], mf)[0]
        self.assertAlmostEqual(r['mfr_ring_area_ratio'], 50 / 100, places=15)
        self.assertAlmostEqual(r['mfr_ring_max'], float(np.float32(.7)), places=7)
        mf[50:60, 61:66] = 0.; r = MF.image_features([mask], mf)[0]
        self.assertAlmostEqual(r['mfr_ring_max'], float(np.float32(.3)), places=7); self.assertEqual(r['mfr_ring_area_ratio'], 0.)
        # the ring corner: (50 - 16, 50 - 16) is inside the square ring, (50 - 17, 50) is not
        mf[:] = 0; mf[34, 34] = .6; r = MF.image_features([mask], mf)[0]; self.assertEqual(r['mfr_ring_area_ratio'], .01)
        mf[:] = 0; mf[33, 55] = .6; r = MF.image_features([mask], mf)[0]; self.assertEqual(r['mfr_ring_area_ratio'], 0.)

    def test_ring_excludes_candidate_and_clips_border(self):
        mf = blank(64); mask = np.zeros(mf.shape, bool); mask[0:5, 0:5] = True; mf[0:5, 0:5] = 1.
        r = MF.image_features([mask], mf)[0]
        self.assertEqual((r['mfr_ring_area_ratio'], r['mfr_ring_max']), (0., 0.))
        mf[0:21, 0:21] = .55; mf[0:5, 0:5] = 1.          # ring inside the image = 21x21 - 25 = 416 px, all >= 0.5
        r = MF.image_features([mask], mf)[0]
        self.assertAlmostEqual(r['mfr_ring_area_ratio'], 416 / 25, places=12); self.assertAlmostEqual(r['mfr_ring_max'], float(np.float32(.55)), places=7)

    def test_several_candidates_share_components(self):
        mf = blank(); a = np.zeros(mf.shape, bool); b = a.copy(); a[10:20, 10:20] = True; b[10:20, 30:40] = True
        mf[12:14, 12:38] = .8                          # one component touching both
        ra, rb = MF.image_features([a, b], mf)
        self.assertEqual((ra['mfr_blob_count'], rb['mfr_blob_count']), (1., 1.))
        self.assertAlmostEqual(ra['mfr_blob_iou'], 16 / (100 + 52 - 16), places=15)
        self.assertEqual(MF.image_features([], mf), [])

    def test_only_mf_map_matters(self):
        mf = blank(); mask = np.zeros(mf.shape, bool); mask[20:30, 20:30] = True; mf[20:30, 15:35] = .7
        r1 = MF.image_features([mask], mf)[0]; r2 = MF.image_features([mask], mf.copy())[0]
        self.assertEqual(r1, r2)
        mf2 = mf.copy(); mf2[20:30, 15:35] = .4; self.assertNotEqual(MF.image_features([mask], mf2)[0], r1)

    def test_rejects_bad_inputs(self):
        mf = blank(); mask = np.zeros(mf.shape, bool)
        with self.assertRaises(AssertionError): MF.image_features([mask], mf)              # empty candidate
        mask[5, 5] = True
        with self.assertRaises(AssertionError): MF.image_features([mask], mf.astype(np.float64) + 2.)   # not a probability
        with self.assertRaises(AssertionError): MF.image_features([mask[:50]], mf)           # shape mismatch


class NoAnnotation(unittest.TestCase):
    """No annotation input can reach any of the 8 features: the only inputs are the candidate masks and the MF map."""
    def test_signatures(self):
        self.assertEqual(list(inspect.signature(MF.image_features).parameters), ['masks', 'mf'])
        self.assertEqual(list(inspect.signature(MF.candidate_features).parameters), ['mask', 'mf', 'labels', 'areas'])
        self.assertEqual(list(inspect.signature(MF.components).parameters), ['mf'])

    def test_module_reads_nothing_else(self):
        tree = ast.parse((H / 'mf_features.py').read_text())
        imports = sorted({a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
                         | {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module})
        self.assertEqual(imports, ['cv2', 'numpy'])
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for bad in ('open', 'read', 'load', 'annotations', 'gt_sets', 'G', 'globals', 'eval', 'exec', '__import__'):
            self.assertNotIn(bad, names)
        text = (H / 'mf_features.py').read_text().lower()
        self.assertNotIn('annotation_rles', text); self.assertNotIn('pq_metric', text)


if __name__ == '__main__':
    unittest.main()
