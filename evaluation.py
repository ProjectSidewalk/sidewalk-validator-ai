from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def precision_recall_at_level(
    precision: NDArray[np.float64],
    recall: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    precision_level: float,
) -> tuple[float | None, float | None]:
    if (
        precision.ndim != 1
        or recall.ndim != 1
        or thresholds.ndim != 1
        or precision.shape != recall.shape
        or thresholds.size != precision.size - 1
    ):
        raise ValueError(
            "Expected precision/recall arrays with one more value than thresholds"
        )

    eligible = precision >= precision_level
    recall_value = float(recall[eligible].max()) if eligible.any() else None

    threshold_indices = np.flatnonzero(precision[:-1] >= precision_level)
    if not threshold_indices.size:
        return recall_value, None

    best_index = int(threshold_indices[np.argmax(recall[threshold_indices])])
    return recall_value, float(thresholds[best_index])


def accuracy_metrics(
    true_labels: ArrayLike,
    predicted_labels: ArrayLike,
) -> tuple[float, float]:
    return (
        float(accuracy_score(true_labels, predicted_labels)),
        float(balanced_accuracy_score(true_labels, predicted_labels)),
    )


def save_figure_and_close(figure: Figure, path: str | Path) -> None:
    try:
        figure.savefig(path)
    finally:
        plt.close(figure)
