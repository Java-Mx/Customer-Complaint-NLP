"""Model evaluation module for complaint classification.

Computes accuracy, precision, recall, F1-scores, and generates
confusion matrix plots for classical NLP models.
"""

from __future__ import annotations

from typing import Any, Dict
import matplotlib.figure


def evaluate_classifier(
    y_true: Any,
    y_pred: Any
) -> Dict[str, float]:
    """Compute standard classification evaluation metrics.

    Metrics calculated:
    - Accuracy
    - Precision (macro and weighted)
    - Recall (macro and weighted)
    - F1-Score (macro and weighted)

    Parameters
    ----------
    y_true : Any
        Ground truth class labels.
    y_pred : Any
        Predicted class labels.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to their computed scores.
    """
    raise NotImplementedError(
        "Evaluation metrics computation will be implemented in the evaluation milestone."
    )


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: list[str] | None = None,
    save_path: str | None = None
) -> matplotlib.figure.Figure:
    """Generate and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : Any
        Ground truth class labels.
    y_pred : Any
        Predicted class labels.
    labels : list[str] | None, optional
        Display labels for categories.
    save_path : str | None, optional
        File path to save the generated plot image.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the plot.
    """
    raise NotImplementedError(
        "Confusion matrix generation will be implemented in the evaluation milestone."
    )
