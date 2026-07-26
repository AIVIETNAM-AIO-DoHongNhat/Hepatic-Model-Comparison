import unittest

import numpy as np

from src import stats


class StatisticalUtilitiesTests(unittest.TestCase):
    def test_identical_inputs_have_zero_difference(self):
        values = [0.4, 0.5, 0.45, 0.48, 0.43]
        t_result = stats.paired_ttest(values, values)
        w_result = stats.wilcoxon_test(values, values)
        ci = stats.ci_diff_t(values, values)
        self.assertEqual(t_result.p_value, 1.0)
        self.assertEqual(w_result.p_value, 1.0)
        self.assertEqual((ci.low, ci.high), (0.0, 0.0))

    def test_xgboost_like_vector_is_better(self):
        reference = [0.38, 0.37, 0.39, 0.38, 0.39]
        comparator = [0.48, 0.47, 0.50, 0.49, 0.51]
        result = stats.paired_ttest(reference, comparator)
        ci = stats.ci_diff_t(reference, comparator)
        self.assertLess(result.mean_difference, 0)
        self.assertLess(ci.high, 0)
        self.assertLess(result.p_value, 0.05)

    def test_difference_sign_convention(self):
        better = stats.paired_ttest([1.0, 2.1, 2.9], [2.0, 3.0, 4.1])
        worse = stats.paired_ttest([2.0, 3.0, 4.1], [1.0, 2.1, 2.9])
        self.assertLess(better.mean_difference, 0)
        self.assertGreater(worse.mean_difference, 0)

    def test_skewed_outlier_difference_is_finite(self):
        a = [0.4, 0.41, 0.39, 0.4, 2.0]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = stats.wilcoxon_test(a, b)
        self.assertTrue(np.isfinite(result.statistic))
        self.assertTrue(0 <= result.p_value <= 1)

    def test_bootstrap_is_deterministic(self):
        a = [0.4, 0.5, 0.45, 0.48, 0.43]
        b = [0.5, 0.6, 0.55, 0.58, 0.53]
        first = stats.ci_diff_bootstrap(a, b, n_boot=1000, random_state=7)
        second = stats.ci_diff_bootstrap(a, b, n_boot=1000, random_state=7)
        self.assertEqual(first, second)

    def test_bonferroni_bounds_and_family(self):
        result = stats.bonferroni_correction([0.001, 0.02, 0.2, 0.4, 1.0])
        self.assertEqual(result.family_size, 5)
        self.assertAlmostEqual(result.adjusted_alpha, 0.01)
        self.assertTrue(np.all((result.adjusted_p_values >= 0) & (result.adjusted_p_values <= 1)))
        np.testing.assert_allclose(result.adjusted_p_values, [0.005, 0.1, 1, 1, 1])

    def test_invalid_inputs_raise_clear_errors(self):
        invalid_pairs = [
            ([1, 2], [1]),
            ([1, np.nan], [1, 2]),
            ([[1, 2]], [[1, 2]]),
            ([1], [1]),
        ]
        for a, b in invalid_pairs:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                stats.paired_ttest(a, b)
        with self.assertRaises(ValueError):
            stats.bonferroni_correction([0.1, 1.2])


if __name__ == "__main__":
    unittest.main()
