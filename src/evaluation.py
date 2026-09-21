"""Model evaluation module for customer complaint classification.

Computes classical NLP evaluation metrics using Scikit-learn:
- Accuracy
- Precision, Recall, F1 (macro and weighted)
- Classification report
- Confusion matrix computation and visualization
- Per-category granular diagnostics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def _validate_inputs(y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate that y_true and y_pred are non-empty sequences of equal length.

    Parameters
    ----------
    y_true : Any
        Ground truth target labels.
    y_pred : Any
        Predicted target labels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (1D array y_true, 1D array y_pred).

    Raises
    ------
    TypeError
        If y_true or y_pred is None.
    ValueError
        If arrays are empty or lengths do not match.
    """
    if y_true is None:
        raise TypeError("y_true cannot be None.")
    if y_pred is None:
        raise TypeError("y_pred cannot be None.")

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if len(y_t) == 0:
        raise ValueError("y_true is empty. Cannot compute evaluation metrics on 0 samples.")
    if len(y_p) == 0:
        raise ValueError("y_pred is empty. Cannot compute evaluation metrics on 0 samples.")

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Dimension mismatch: y_true has {len(y_t)} samples, but y_pred has {len(y_p)} predictions."
        )

    return y_t, y_p


def calculate_accuracy(y_true: Any, y_pred: Any) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.

    Returns
    -------
    float
        Fraction of correctly classified samples in range [0.0, 1.0].
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return float(accuracy_score(y_t, y_p))


def calculate_precision(
    y_true: Any,
    y_pred: Any,
    average: str = "weighted",
    zero_division: int = 0
) -> float:
    """Compute precision score.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    average : str, default='weighted'
        Averaging strategy ('macro', 'weighted', 'micro').
    zero_division : int, default=0
        Value to return when a zero-division error occurs.

    Returns
    -------
    float
        Precision score in range [0.0, 1.0].
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return float(precision_score(y_t, y_p, average=average, zero_division=zero_division))


def calculate_recall(
    y_true: Any,
    y_pred: Any,
    average: str = "weighted",
    zero_division: int = 0
) -> float:
    """Compute recall score.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    average : str, default='weighted'
        Averaging strategy ('macro', 'weighted', 'micro').
    zero_division : int, default=0
        Value to return when a zero-division error occurs.

    Returns
    -------
    float
        Recall score in range [0.0, 1.0].
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return float(recall_score(y_t, y_p, average=average, zero_division=zero_division))


def calculate_f1(
    y_true: Any,
    y_pred: Any,
    average: str = "weighted",
    zero_division: int = 0
) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    average : str, default='weighted'
        Averaging strategy ('macro', 'weighted', 'micro').
    zero_division : int, default=0
        Value to return when a zero-division error occurs.

    Returns
    -------
    float
        F1 score in range [0.0, 1.0].
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return float(f1_score(y_t, y_p, average=average, zero_division=zero_division))


def calculate_macro_metrics(
    y_true: Any,
    y_pred: Any,
    zero_division: int = 0
) -> Dict[str, float]:
    """Calculate unweighted macro-averaged precision, recall, and F1.

    Gives equal weight to all product categories regardless of class imbalance.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    zero_division : int, default=0
        Handling of zero-division cases.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'macro_precision', 'macro_recall', and 'macro_f1'.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return {
        "macro_precision": float(precision_score(y_t, y_p, average="macro", zero_division=zero_division)),
        "macro_recall": float(recall_score(y_t, y_p, average="macro", zero_division=zero_division)),
        "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=zero_division)),
    }


def calculate_weighted_metrics(
    y_true: Any,
    y_pred: Any,
    zero_division: int = 0
) -> Dict[str, float]:
    """Calculate support-weighted precision, recall, and F1.

    Accounts for category imbalance by weighting each class by its support.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    zero_division : int, default=0
        Handling of zero-division cases.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'weighted_precision', 'weighted_recall', and 'weighted_f1'.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return {
        "weighted_precision": float(precision_score(y_t, y_p, average="weighted", zero_division=zero_division)),
        "weighted_recall": float(recall_score(y_t, y_p, average="weighted", zero_division=zero_division)),
        "weighted_f1": float(f1_score(y_t, y_p, average="weighted", zero_division=zero_division)),
    }


def generate_classification_report(
    y_true: Any,
    y_pred: Any,
    output_dict: bool = False,
    zero_division: int = 0
) -> Union[str, Dict[str, Any]]:
    """Build a comprehensive text report showing main classification metrics.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    output_dict : bool, default=False
        If True, returns output as a dictionary; else formatted string.
    zero_division : int, default=0
        Value for zero division.

    Returns
    -------
    str | Dict[str, Any]
        Text or dictionary representation of the classification report.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return classification_report(
        y_t,
        y_p,
        output_dict=output_dict,
        zero_division=zero_division
    )


def compute_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[str]] = None
) -> np.ndarray:
    """Compute confusion matrix to evaluate classification accuracy across classes.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    labels : Optional[Sequence[str]], optional
        List of labels to index the matrix.

    Returns
    -------
    np.ndarray
        Confusion matrix array of shape (n_classes, n_classes).
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return confusion_matrix(y_t, y_p, labels=labels)


def compute_per_category_metrics(
    y_true: Any,
    y_pred: Any,
    zero_division: int = 0
) -> pd.DataFrame:
    """Compute precision, recall, F1, and support for each individual category.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    zero_division : int, default=0
        Value for zero division.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by category name with columns:
        ['precision', 'recall', 'f1-score', 'support'].
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    report_dict = classification_report(
        y_t,
        y_p,
        output_dict=True,
        zero_division=zero_division
    )

    rows = []
    excluded = {"accuracy", "macro avg", "weighted avg"}
    for cat, metrics in report_dict.items():
        if cat not in excluded and isinstance(metrics, dict):
            rows.append({
                "category": cat,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1-score": float(metrics.get("f1-score", 0.0)),
                "support": int(metrics.get("support", 0)),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="support", ascending=False).reset_index(drop=True)
    return df


def evaluate_classifier(
    y_true: Any,
    y_pred: Any,
    zero_division: int = 0
) -> Dict[str, float]:
    """Compute summary classification evaluation metrics.

    Preserves scaffold function signature.

    Parameters
    ----------
    y_true : Any
        Ground truth class labels.
    y_pred : Any
        Predicted class labels.
    zero_division : int, default=0
        Value to return on zero-division.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, macro metrics, and weighted metrics.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    macro = calculate_macro_metrics(y_t, y_p, zero_division=zero_division)
    weighted = calculate_weighted_metrics(y_t, y_p, zero_division=zero_division)

    return {
        "accuracy": calculate_accuracy(y_t, y_p),
        "macro_precision": macro["macro_precision"],
        "macro_recall": macro["macro_recall"],
        "macro_f1": macro["macro_f1"],
        "weighted_precision": weighted["weighted_precision"],
        "weighted_recall": weighted["weighted_recall"],
        "weighted_f1": weighted["weighted_f1"],
    }


def evaluate_model(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[str]] = None,
    zero_division: int = 0
) -> Dict[str, Any]:
    """Generate a comprehensive evaluation bundle including metrics, report, and matrix.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    labels : Optional[Sequence[str]], optional
        Class labels order for confusion matrix.
    zero_division : int, default=0
        Value on zero division.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary metrics, per-category DataFrame,
        classification report (text + dict), and confusion matrix.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    summary = evaluate_classifier(y_t, y_p, zero_division=zero_division)
    per_cat = compute_per_category_metrics(y_t, y_p, zero_division=zero_division)
    cm = compute_confusion_matrix(y_t, y_p, labels=labels)
    rep_text = generate_classification_report(y_t, y_p, output_dict=False, zero_division=zero_division)
    rep_dict = generate_classification_report(y_t, y_p, output_dict=True, zero_division=zero_division)

    unique_classes = list(labels) if labels is not None else sorted(list(set(y_t).union(set(y_p))))

    return {
        "metrics": summary,
        "per_category": per_cat,
        "confusion_matrix": cm,
        "classification_report_text": rep_text,
        "classification_report_dict": rep_dict,
        "classes": unique_classes,
        "total_samples": len(y_t),
    }


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[list[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    normalize: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = "Complaint Categorisation Confusion Matrix"
) -> plt.Figure:
    """Generate and optionally save a confusion matrix heatmap plot.

    Parameters
    ----------
    y_true : Any
        Ground truth class labels.
    y_pred : Any
        Predicted class labels.
    labels : list[str] | None, optional
        Display labels for categories.
    save_path : str | Path | None, optional
        File path to save the generated figure.
    normalize : str | None, optional
        'true', 'pred', or 'all' for normalization.
    figsize : Tuple[int, int], default=(10, 8)
        Dimensions of the matplotlib figure.
    title : str | None, default='Complaint Categorisation Confusion Matrix'
        Title text displayed above the heatmap.

    Returns
    -------
    plt.Figure
        Matplotlib figure object containing the styled heatmap.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    class_names = list(labels) if labels is not None else sorted(list(set(y_t).union(set(y_p))))

    cm = confusion_matrix(y_t, y_p, labels=class_names, normalize=normalize)

    fig, ax = plt.subplots(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    plot_title = title if title else "Complaint Categorisation Confusion Matrix"
    ax.set_title(plot_title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted Product Category", fontsize=11)
    ax.set_ylabel("Actual Product Category", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to {out_path}")

    return fig
