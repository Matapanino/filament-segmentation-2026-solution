"""Run the lane's test suite (unittest discover over tests/) minus tests that belong to another lane's worker.

Excluded: test_pmerge.Classifier.test_real_classifier_recipe. It is a Kaggle-only test of pmerge's own worker.make_classifier
(the P-merge pair classifier), which this lane neither has nor uses. The copy of tests/test_pmerge.py stays byte-identical.
Every other test in tests/test_pmerge.py runs. Exit code 1 on any failure or error.
"""
import sys, unittest
from pathlib import Path

EXCLUDED = {'test_pmerge.Classifier.test_real_classifier_recipe': "tests pmerge's worker.make_classifier, not part of this lane"}


def flat(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite): yield from flat(t)
        else: yield t


def main():
    tests = list(flat(unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / 'tests'))))
    ids = {t.id() for t in tests}; assert set(EXCLUDED) <= ids, ('EXCLUDED_TEST_MISSING', set(EXCLUDED) - ids)
    kept = [t for t in tests if t.id() not in EXCLUDED]
    print(f'running {len(kept)} of {len(tests)} tests; excluded: {sorted(EXCLUDED)}', flush=True)
    r = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(kept))
    sys.exit(0 if r.wasSuccessful() else 1)


if __name__ == '__main__': main()
