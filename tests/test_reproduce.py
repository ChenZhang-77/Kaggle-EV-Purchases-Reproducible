import unittest

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from reproduce import (
    fit_nonnegative_logistic, fold_ids, fold_rank, interaction_groups,
    rank, validate_submission,
)


class ReproductionTests(unittest.TestCase):
    def test_rank_and_fold_assignment(self):
        y = np.array([0, 1] * 25)
        folds = fold_ids(y, 42)
        np.testing.assert_array_equal(folds, fold_ids(y, 42))
        self.assertEqual(set(folds), set(range(5)))
        values = np.arange(len(y), dtype=float)
        expected = np.empty(len(y))
        for fold in range(5):
            selected = folds == fold
            expected[selected] = rankdata(values[selected]) / selected.sum()
        np.testing.assert_allclose(fold_rank(values, folds), expected)
        np.testing.assert_allclose(rank([2, 1, 1]), [1, .5, .5])

    def test_interaction_groups_cover_features(self):
        groups = interaction_groups([
            "Annual_Income_USD", "Annual_Income_USD_TE_auto",
            "income100_floor", "Gender", "Gender_TE_10",
        ])
        self.assertEqual(sorted(i for group in groups for i in group), list(range(5)))
        self.assertEqual(len(groups), 2)

    def test_nonnegative_stack_and_submission_schema(self):
        X = np.array([[-1., -1.], [-.5, .5], [.5, -.5], [1., 1.]] * 5)
        y = np.array([0, 0, 1, 1] * 5)
        intercept, weights = fit_nonnegative_logistic(X, y)
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        sample = pd.DataFrame({"id": [1, 2], "Will_Buy_EV": [0.0, 0.0]})
        good = sample.assign(Will_Buy_EV=[.2, .8])
        validate_submission(good, sample)
        with self.assertRaises(ValueError):
            validate_submission(good.iloc[::-1].reset_index(drop=True), sample)


if __name__ == "__main__":
    unittest.main()
