from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve

from evaluation import (
    accuracy_metrics,
    precision_recall_at_level,
    save_figure_and_close,
)


@pytest.fixture(scope="module", autouse=True)
def use_noninteractive_backend() -> None:
    matplotlib.use("Agg")


def test_precision_recall_uses_maximum_recall_for_non_monotone_precision() -> None:
    precision, recall, thresholds = precision_recall_curve(
        [0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6]
    )
    precision_level = 0.34
    old_index = int(np.searchsorted(precision, precision_level))

    recall_value, threshold = precision_recall_at_level(
        precision, recall, thresholds, precision_level
    )

    assert recall_value == pytest.approx(1.00)
    assert recall[old_index] == pytest.approx(0.50)
    assert thresholds[old_index] == pytest.approx(0.8)
    assert threshold == pytest.approx(0.6)


def test_precision_sentinel_does_not_produce_a_threshold() -> None:
    precision = np.array([0.40, 1.00])
    recall = np.array([1.00, 0.00])
    thresholds = np.array([0.50])

    recall_value, threshold = precision_recall_at_level(
        precision, recall, thresholds, 0.95
    )

    assert recall_value == pytest.approx(0.00)
    assert threshold is None


def test_precision_curve_shape_must_match_threshold_contract() -> None:
    with pytest.raises(ValueError, match="one more value than thresholds"):
        precision_recall_at_level(
            np.array([0.40, 1.00]),
            np.array([1.00, 0.00]),
            np.array([0.50, 0.60]),
            0.95,
        )


def test_accuracy_metrics_reports_plain_and_balanced_accuracy() -> None:
    accuracy, balanced_accuracy = accuracy_metrics([0, 0, 0, 1], [0, 1, 0, 1])

    assert accuracy == pytest.approx(0.75)
    assert balanced_accuracy == pytest.approx(5 / 6)


def test_save_figure_and_close_saves_and_releases_figure(tmp_path: Path) -> None:
    figure = plt.figure()
    figure_number = figure.number
    output = tmp_path / "plot.png"

    save_figure_and_close(figure, output)

    assert output.is_file()
    assert not plt.fignum_exists(figure_number)


def test_save_figure_and_close_releases_figure_after_failure(tmp_path: Path) -> None:
    figure = plt.figure()
    figure_number = figure.number
    output = tmp_path / "missing" / "plot.png"

    with pytest.raises(FileNotFoundError):
        save_figure_and_close(figure, output)

    assert not plt.fignum_exists(figure_number)
