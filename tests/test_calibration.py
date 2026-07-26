import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src import stats


class CalibrationUtilitiesTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 2, 0, 1, 2])
        self.perfect = np.eye(3)[self.y]

    def test_perfect_predictions(self):
        self.assertEqual(stats.brier_multiclass(self.y, self.perfect), 0.0)
        self.assertEqual(stats.top_label_ece(self.y, self.perfect), 0.0)
        self.assertEqual(stats.macro_classwise_ece(self.y, self.perfect), 0.0)

    def test_calibrated_like_beats_overconfident_wrong(self):
        calibrated = 0.7 * self.perfect + 0.3 / 3
        wrong_class = (self.y + 1) % 3
        wrong = 0.99 * np.eye(3)[wrong_class] + 0.01 / 3
        self.assertLess(stats.brier_multiclass(self.y, calibrated), stats.brier_multiclass(self.y, wrong))
        self.assertLess(stats.top_label_ece(self.y, calibrated), stats.top_label_ece(self.y, wrong))

    def test_empty_bins_are_retained(self):
        bins = stats.top_label_reliability_bins(self.y, self.perfect, n_bins=10)
        self.assertEqual(len(bins), 10)
        self.assertGreater((bins["count"] == 0).sum(), 0)
        self.assertEqual(int(bins["count"].sum()), len(self.y))

    def test_invalid_probabilities_raise(self):
        cases = [
            np.full((6, 3), 0.2),
            np.array(self.perfect, dtype=float) * 1.1,
            np.where(self.perfect == 1, np.nan, self.perfect),
            np.where(self.perfect == 1, np.inf, self.perfect),
            np.ones((6, 2)) / 2,
        ]
        for proba in cases:
            with self.subTest(shape=proba.shape), self.assertRaises(ValueError):
                stats.top_label_ece(self.y, proba)

    def test_reliability_figure_is_created(self):
        fig, axes, top_bins, class_bins = stats.plot_reliability_diagram(
            self.y, self.perfect, n_bins=5, include_classwise=True
        )
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(top_bins), 5)
        self.assertEqual(len(class_bins), 15)
        self.assertGreater(len(fig.axes), 0)
        matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    unittest.main()
