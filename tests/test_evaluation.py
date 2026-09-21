"""Unit tests for classification evaluation metrics and diagnostic reports.

Tests accuracy, precision, recall, F1, macro/weighted averaging, classification
report generation, confusion matrix computation, per-category breakdowns,
plot generation, and input error handling.
"""

from pathlib import Path
import tempfile
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_accuracy,
    calculate_f1,
    calculate_macro_metrics,
    calculate_precision,
    calculate_recall,
    calculate_weighted_metrics,
    compute_confusion_matrix,
    compute_per_category_metrics,
    evaluate_classifier,
    evaluate_model,
    generate_classification_report,
    plot_confusion_matrix,
)


@pytest.fixture
def synthetic_eval_data():
    """Deterministic true and predicted multi-class labels."""
    # 10 samples, 3 classes: Credit Card (4), Mortgage (3), Debt Collection (3)
    y_true = np.array([
        "Credit Card", "Credit Card", "Credit Card", "Credit Card",
        "Mortgage", "Mortgage", "Mortgage",
        "Debt Collection", "Debt Collection", "Debt Collection"
    ])
    # 8 correct, 2 mistakes
    y_pred = np.array([
        "Credit Card", "Credit Card", "Credit Card", "Mortgage",  # 1 mistake
        "Mortgage", "Mortgage", "Debt Collection",                 # 1 mistake
        "Debt Collection", "Debt Collection", "Debt Collection"
    ])
    return y_true, y_pred


# ======================================================================
# 1. Scalar Metric Calculation Tests
# ======================================================================

class TestMetricCalculations:
    """Test suite for individual classification metrics."""

    def test_perfect_predictions_metrics_equal_one(self):
        """When y_pred == y_true, accuracy, precision, recall, and F1 must be 1.0."""
        y = ["Credit Card", "Mortgage", "Student Loan"]
        assert calculate_accuracy(y, y) == 1.0
        assert calculate_precision(y, y, average="macro") == 1.0
        assert calculate_recall(y, y, average="macro") == 1.0
        assert calculate_f1(y, y, average="macro") == 1.0

    def test_accuracy_partial_match(self, synthetic_eval_data):
        """Accuracy must equal exactly 8 / 10 = 0.80."""
        y_true, y_pred = synthetic_eval_data
        acc = calculate_accuracy(y_true, y_pred)
        assert np.isclose(acc, 0.80, atol=1e-5)

    def test_precision_recall_f1_valid_ranges(self, synthetic_eval_data):
        """All computed scores must be bounded in [0.0, 1.0]."""
        y_true, y_pred = synthetic_eval_data
        p = calculate_precision(y_true, y_pred, average="weighted")
        r = calculate_recall(y_true, y_pred, average="weighted")
        f1 = calculate_f1(y_true, y_pred, average="weighted")

        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_macro_averaging_metrics(self, synthetic_eval_data):
        """Macro metrics dictionary must contain macro_precision, macro_recall, macro_f1."""
        y_true, y_pred = synthetic_eval_data
        macro = calculate_macro_metrics(y_true, y_pred)
        assert set(macro.keys()) == {"macro_precision", "macro_recall", "macro_f1"}
        for score in macro.values():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_weighted_averaging_metrics(self, synthetic_eval_data):
        """Weighted metrics dictionary must contain weighted_precision, weighted_recall, weighted_f1."""
        y_true, y_pred = synthetic_eval_data
        weighted = calculate_weighted_metrics(y_true, y_pred)
        assert set(weighted.keys()) == {"weighted_precision", "weighted_recall", "weighted_f1"}
        for score in weighted.values():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_evaluate_classifier_wrapper(self, synthetic_eval_data):
        """evaluate_classifier returns complete dictionary of scalar metrics."""
        y_true, y_pred = synthetic_eval_data
        res = evaluate_classifier(y_true, y_pred)
        expected_keys = {
            "accuracy",
            "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1"
        }
        assert expected_keys.issubset(set(res.keys()))
        assert np.isclose(res["accuracy"], 0.80)


# ======================================================================
# 2. Diagnostic Reports and Matrices Tests
# ======================================================================

class TestReportsAndMatrices:
    """Test suite for classification report, confusion matrix, and plots."""

    def test_classification_report_text_format(self, synthetic_eval_data):
        """Text classification report must contain all unique class names."""
        y_true, y_pred = synthetic_eval_data
        report_text = generate_classification_report(y_true, y_pred, output_dict=False)
        assert isinstance(report_text, str)
        assert "Credit Card" in report_text
        assert "Mortgage" in report_text
        assert "Debt Collection" in report_text
        assert "accuracy" in report_text

    def test_classification_report_dict_format(self, synthetic_eval_data):
        """Dict classification report must parse into structured dictionary."""
        y_true, y_pred = synthetic_eval_data
        report_dict = generate_classification_report(y_true, y_pred, output_dict=True)
        assert isinstance(report_dict, dict)
        assert "accuracy" in report_dict
        assert "macro avg" in report_dict
        assert "weighted avg" in report_dict
        assert "Credit Card" in report_dict

    def test_confusion_matrix_shape_and_values(self, synthetic_eval_data):
        """Confusion matrix must have shape (3, 3) and correct diagonal elements."""
        y_true, y_pred = synthetic_eval_data
        labels = ["Credit Card", "Debt Collection", "Mortgage"]
        cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
        assert cm.shape == (3, 3)
        assert isinstance(cm, np.ndarray)
        # Sum of confusion matrix equals total samples
        assert cm.sum() == 10
        # Correct predictions on diagonal: 3 Credit Card, 3 Debt Collection, 2 Mortgage
        assert cm[0, 0] == 3  # Credit Card
        assert cm[1, 1] == 3  # Debt Collection
        assert cm[2, 2] == 2  # Mortgage

    def test_per_category_metrics_dataframe(self, synthetic_eval_data):
        """per-category metrics must return DataFrame with precision, recall, f1, support."""
        y_true, y_pred = synthetic_eval_data
        df_per_cat = compute_per_category_metrics(y_true, y_pred)
        assert isinstance(df_per_cat, pd.DataFrame)
        assert len(df_per_cat) == 3
        expected_cols = {"category", "precision", "recall", "f1-score", "support"}
        assert expected_cols.issubset(set(df_per_cat.columns))
        assert df_per_cat["support"].sum() == 10

    def test_evaluate_model_comprehensive_bundle(self, synthetic_eval_data):
        """evaluate_model produces unified evaluation dictionary."""
        y_true, y_pred = synthetic_eval_data
        bundle = evaluate_model(y_true, y_pred)
        assert "metrics" in bundle
        assert "per_category" in bundle
        assert "confusion_matrix" in bundle
        assert "classification_report_text" in bundle
        assert "classification_report_dict" in bundle
        assert bundle["total_samples"] == 10

    def test_plot_confusion_matrix_generation_and_saving(self, synthetic_eval_data):
        """plot_confusion_matrix generates Figure and saves cleanly to disk."""
        y_true, y_pred = synthetic_eval_data
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_img = Path(tmp_dir) / "cm_plot.png"
            fig = plot_confusion_matrix(y_true, y_pred, save_path=save_img)
            assert isinstance(fig, matplotlib.figure.Figure)
            assert save_img.exists()
            assert save_img.stat().st_size > 0


# ======================================================================
# 3. Input Validation Tests
# ======================================================================

class TestEvaluationInputValidation:
    """Test suite for error handling on invalid or mismatched inputs."""

    def test_none_inputs_raise_type_error(self):
        """Passing None for y_true or y_pred must raise TypeError."""
        with pytest.raises(TypeError):
            calculate_accuracy(None, ["A", "B"])
        with pytest.raises(TypeError):
            calculate_accuracy(["A", "B"], None)

    def test_empty_inputs_raise_value_error(self):
        """Passing empty arrays must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_accuracy([], [])

    def test_mismatched_lengths_raise_value_error(self):
        """Passing arrays of different lengths must raise ValueError."""
        with pytest.raises(ValueError, match="mismatch"):
            calculate_accuracy(["A", "B", "C"], ["A", "B"])

    def test_input_type_flexibility(self):
        """Functions must accept lists, numpy arrays, and pandas Series interchangeably."""
        y_true_list = ["Credit Card", "Mortgage"]
        y_pred_series = pd.Series(["Credit Card", "Credit Card"])
        y_pred_arr = np.array(["Credit Card", "Credit Card"])

        acc_series = calculate_accuracy(y_true_list, y_pred_series)
        acc_arr = calculate_accuracy(y_true_list, y_pred_arr)
        assert np.isclose(acc_series, 0.5)
        assert np.isclose(acc_arr, 0.5)
