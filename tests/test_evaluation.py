import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from evaluation import (
    accuracy_metrics,
    precision_recall_at_level,
    save_figure_and_close,
)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        matplotlib.use("Agg")

    def test_precision_recall_uses_maximum_recall_for_non_monotone_precision(
        self,
    ) -> None:
        precision, recall, thresholds = precision_recall_curve(
            [0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6]
        )
        precision_level = 0.34
        old_index = int(np.searchsorted(precision, precision_level))

        recall_value, threshold = precision_recall_at_level(
            precision, recall, thresholds, precision_level
        )

        self.assertEqual(recall_value, 1.0)
        self.assertEqual(recall[old_index], 0.5)
        self.assertEqual(thresholds[old_index], 0.8)
        self.assertEqual(threshold, 0.6)

    def test_precision_sentinel_does_not_produce_a_threshold(self) -> None:
        precision = np.array([0.40, 1.00])
        recall = np.array([1.00, 0.00])
        thresholds = np.array([0.50])

        recall_value, threshold = precision_recall_at_level(
            precision, recall, thresholds, 0.95
        )

        self.assertEqual(recall_value, 0.0)
        self.assertIsNone(threshold)

    def test_precision_curve_shape_must_match_threshold_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "one more value than thresholds"):
            precision_recall_at_level(
                np.array([0.40, 1.00]),
                np.array([1.00, 0.00]),
                np.array([0.50, 0.60]),
                0.95,
            )

    def test_accuracy_metrics_reports_plain_and_balanced_accuracy(self) -> None:
        accuracy, balanced_accuracy = accuracy_metrics([0, 0, 0, 1], [0, 1, 0, 1])

        self.assertAlmostEqual(accuracy, 0.75)
        self.assertAlmostEqual(balanced_accuracy, 5 / 6)

    def test_save_figure_and_close_saves_and_releases_figure(self) -> None:
        with TemporaryDirectory() as directory:
            figure = plt.figure()
            figure_number = figure.number
            output = Path(directory) / "plot.png"

            save_figure_and_close(figure, output)

            self.assertTrue(output.is_file())
            self.assertFalse(plt.fignum_exists(figure_number))

    def test_save_figure_and_close_releases_figure_after_failure(self) -> None:
        with TemporaryDirectory() as directory:
            figure = plt.figure()
            figure_number = figure.number
            output = Path(directory) / "missing" / "plot.png"

            with self.assertRaises(FileNotFoundError):
                save_figure_and_close(figure, output)

            self.assertFalse(plt.fignum_exists(figure_number))


if __name__ == "__main__":
    unittest.main()
