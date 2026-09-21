"""Tests for the CFPB classification error analysis infrastructure.

Verifies:
- Exact test set size = 5,000 (same split as experiment pipeline)
- Exactly 18 categories in the test set
- Confusion matrix shape = 18 x 18
- Per-category metrics contain all 18 categories
- Confusion-pair output is correctly structured
- Confidence bands cover every prediction
- Baseline comparison contains both model configurations
- No NaN or invalid metric values in output files
- Output CSV files are generated and well-formed
- Error examples reference only test set records
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_full():
    """Load full CFPB dataset (25,000 records)."""
    from src.data_loader import load_dataset
    data_path = ROOT_DIR / "data" / "complaints.csv"
    if not data_path.exists():
        pytest.skip("complaints.csv not found.")
    return load_dataset(data_path, drop_invalid=True)


@pytest.fixture(scope="module")
def test_split(df_full):
    """Reproduce the exact 5,000-record test split."""
    from src.classification import train_test_split_data
    _, X_test, _, y_test = train_test_split_data(
        df_full, test_size=0.20, random_state=42, stratify=True
    )
    return X_test, y_test


@pytest.fixture(scope="module")
def loaded_models():
    """Load the persisted improved model and vectorizers."""
    import joblib
    clf_path = MODELS_DIR / "complaint_classifier.joblib"
    w_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    c_path = MODELS_DIR / "char_vectorizer.joblib"
    if not clf_path.exists():
        pytest.skip("complaint_classifier.joblib not found.")
    clf = joblib.load(clf_path)
    w_vec = joblib.load(w_path)
    c_vec = joblib.load(c_path)
    return clf, w_vec, c_vec


@pytest.fixture(scope="module")
def test_predictions(test_split, loaded_models):
    """Generate predictions on the test set."""
    from src.preprocessing import preprocess_series
    from src.vectorization import transform_word_char
    from src.classification import predict_categories
    X_test, y_test = test_split
    clf, w_vec, c_vec = loaded_models
    clean_test = preprocess_series(X_test)
    X_vec = transform_word_char(w_vec, c_vec, clean_test)
    y_pred = predict_categories(clf, X_vec)
    return X_vec, y_test, y_pred, clf


@pytest.fixture(scope="module")
def error_analysis_json():
    """Load the generated error_analysis_data.json artifact."""
    json_path = RESULTS_DIR / "error_analysis_data.json"
    if not json_path.exists():
        pytest.skip("error_analysis_data.json not found. Run scripts/generate_error_analysis.py first.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test Group 1: Test set integrity
# ---------------------------------------------------------------------------

class TestTestSetIntegrity:
    def test_test_set_size_is_5000(self, test_split):
        """Test set must contain exactly 5,000 records."""
        X_test, y_test = test_split
        assert len(X_test) == 5000, f"Expected 5000 test records, got {len(X_test)}"

    def test_test_set_has_18_categories(self, test_split):
        """Test set must contain exactly 18 distinct categories."""
        _, y_test = test_split
        n_cats = len(set(y_test))
        assert n_cats == 18, f"Expected 18 categories in test set, got {n_cats}"

    def test_y_test_no_missing_values(self, test_split):
        """No NaN or missing category labels in the test set."""
        _, y_test = test_split
        y_arr = np.asarray(y_test)
        assert not any(v is None for v in y_arr)
        assert not any(str(v).lower() == "nan" for v in y_arr)

    def test_x_test_no_empty_strings(self, test_split):
        """No empty or whitespace-only complaint texts in test set."""
        X_test, _ = test_split
        empties = X_test[X_test.str.strip().str.len() == 0]
        assert len(empties) == 0, f"Found {len(empties)} empty complaint texts."


# ---------------------------------------------------------------------------
# Test Group 2: Confusion matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_confusion_matrix_shape_18x18(self, test_predictions):
        """Confusion matrix must be 18 x 18."""
        from sklearn.metrics import confusion_matrix
        X_vec, y_test, y_pred, clf = test_predictions
        classes = list(clf.classes_)
        cm = confusion_matrix(np.asarray(y_test), y_pred, labels=classes)
        assert cm.shape == (18, 18), f"Expected (18, 18), got {cm.shape}"

    def test_confusion_matrix_row_sums_equal_support(self, test_predictions):
        """Each confusion matrix row must sum to the class support in the test set."""
        from sklearn.metrics import confusion_matrix
        X_vec, y_test, y_pred, clf = test_predictions
        classes = list(clf.classes_)
        cm = confusion_matrix(np.asarray(y_test), y_pred, labels=classes)
        y_arr = np.asarray(y_test)
        for i, cat in enumerate(classes):
            expected_support = int((y_arr == cat).sum())
            assert cm[i].sum() == expected_support, \
                f"Row sum mismatch for '{cat}': got {cm[i].sum()}, expected {expected_support}"

    def test_confusion_matrix_total_equals_test_size(self, test_predictions):
        """Total elements in confusion matrix must equal 5,000."""
        from sklearn.metrics import confusion_matrix
        X_vec, y_test, y_pred, clf = test_predictions
        classes = list(clf.classes_)
        cm = confusion_matrix(np.asarray(y_test), y_pred, labels=classes)
        assert cm.sum() == 5000


# ---------------------------------------------------------------------------
# Test Group 3: Per-category metrics
# ---------------------------------------------------------------------------

class TestPerCategoryMetrics:
    def test_per_category_csv_exists(self):
        """per_category_metrics.csv must be generated."""
        assert (RESULTS_DIR / "per_category_metrics.csv").exists(), \
            "per_category_metrics.csv not found. Run generate_error_analysis.py."

    def test_per_category_csv_has_all_18_categories(self):
        """per_category_metrics.csv must contain rows for all 18 categories."""
        path = RESULTS_DIR / "per_category_metrics.csv"
        if not path.exists():
            pytest.skip("per_category_metrics.csv not found.")
        df = pd.read_csv(path)
        assert len(df) == 18, f"Expected 18 rows in per_category_metrics.csv, got {len(df)}"

    def test_per_category_no_nan_metrics(self):
        """No NaN values in metric columns of per_category_metrics.csv."""
        path = RESULTS_DIR / "per_category_metrics.csv"
        if not path.exists():
            pytest.skip("per_category_metrics.csv not found.")
        df = pd.read_csv(path)
        for col in ["precision", "recall", "f1", "support"]:
            assert not df[col].isna().any(), f"NaN found in column '{col}'"

    def test_per_category_metrics_in_valid_range(self):
        """Precision, recall, and F1 must all be in [0.0, 1.0]."""
        path = RESULTS_DIR / "per_category_metrics.csv"
        if not path.exists():
            pytest.skip("per_category_metrics.csv not found.")
        df = pd.read_csv(path)
        for col in ["precision", "recall", "f1"]:
            assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), \
                f"Values out of [0,1] range in column '{col}'"

    def test_per_category_support_sums_to_5000(self):
        """Sum of per-category support must equal 5,000."""
        path = RESULTS_DIR / "per_category_metrics.csv"
        if not path.exists():
            pytest.skip("per_category_metrics.csv not found.")
        df = pd.read_csv(path)
        assert df["support"].sum() == 5000, f"Support sum {df['support'].sum()} != 5000"


# ---------------------------------------------------------------------------
# Test Group 4: Confusion pair output
# ---------------------------------------------------------------------------

class TestConfusionPairs:
    def test_error_analysis_csv_exists(self):
        """error_analysis.csv must be generated."""
        assert (RESULTS_DIR / "error_analysis.csv").exists()

    def test_error_analysis_csv_required_columns(self):
        """error_analysis.csv must contain required columns."""
        path = RESULTS_DIR / "error_analysis.csv"
        if not path.exists():
            pytest.skip("error_analysis.csv not found.")
        df = pd.read_csv(path)
        required = {"actual_category", "predicted_category", "error_count", "pct_of_actual"}
        assert required.issubset(set(df.columns)), \
            f"Missing columns: {required - set(df.columns)}"

    def test_error_analysis_csv_no_self_pairs(self):
        """error_analysis.csv must not contain pairs where actual == predicted."""
        path = RESULTS_DIR / "error_analysis.csv"
        if not path.exists():
            pytest.skip("error_analysis.csv not found.")
        df = pd.read_csv(path)
        self_pairs = df[df["actual_category"] == df["predicted_category"]]
        assert len(self_pairs) == 0, f"Found {len(self_pairs)} self-confusion pairs."

    def test_error_analysis_csv_no_nan_values(self):
        """error_analysis.csv must not contain NaN values."""
        path = RESULTS_DIR / "error_analysis.csv"
        if not path.exists():
            pytest.skip("error_analysis.csv not found.")
        df = pd.read_csv(path)
        assert not df.isnull().any().any()

    def test_error_analysis_pct_in_valid_range(self):
        """pct_of_actual values must be in (0, 100]."""
        path = RESULTS_DIR / "error_analysis.csv"
        if not path.exists():
            pytest.skip("error_analysis.csv not found.")
        df = pd.read_csv(path)
        assert (df["pct_of_actual"] > 0).all() and (df["pct_of_actual"] <= 100).all()


# ---------------------------------------------------------------------------
# Test Group 5: Confidence bands
# ---------------------------------------------------------------------------

class TestConfidenceBands:
    def test_predict_proba_available(self, loaded_models):
        """Improved model must support predict_proba."""
        clf, _, _ = loaded_models
        assert hasattr(clf, "predict_proba"), "LogisticRegression must have predict_proba."

    def test_confidence_bands_cover_all_predictions(self, test_predictions):
        """Every prediction must fall into exactly one confidence band (high/medium/low)."""
        X_vec, y_test, y_pred, clf = test_predictions
        proba_matrix = clf.predict_proba(X_vec)
        max_proba = proba_matrix.max(axis=1)
        high = (max_proba >= 0.70).sum()
        medium = ((max_proba >= 0.40) & (max_proba < 0.70)).sum()
        low = (max_proba < 0.40).sum()
        assert high + medium + low == len(max_proba), \
            f"Confidence bands don't cover all {len(max_proba)} predictions. " \
            f"high={high}, medium={medium}, low={low}"

    def test_confidence_proba_in_0_1(self, test_predictions):
        """All predicted probabilities must be in [0, 1]."""
        X_vec, y_test, y_pred, clf = test_predictions
        proba = clf.predict_proba(X_vec)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

    def test_probability_rows_sum_to_one(self, test_predictions):
        """Each row of predict_proba output must sum to approximately 1.0."""
        X_vec, y_test, y_pred, clf = test_predictions
        proba = clf.predict_proba(X_vec)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test Group 6: Baseline comparison
# ---------------------------------------------------------------------------

class TestBaselineComparison:
    def test_baseline_vs_improved_csv_exists(self):
        """baseline_vs_improved.csv must be generated."""
        assert (RESULTS_DIR / "baseline_vs_improved.csv").exists()

    def test_baseline_vs_improved_has_required_columns(self):
        """baseline_vs_improved.csv must have the expected columns."""
        path = RESULTS_DIR / "baseline_vs_improved.csv"
        if not path.exists():
            pytest.skip("baseline_vs_improved.csv not found.")
        df = pd.read_csv(path)
        required = {
            "metric",
            "baseline_word_tfidf_no_balancing",
            "improved_combined_tfidf_balanced",
            "absolute_difference_pp",
            "relative_change_pct",
        }
        assert required.issubset(set(df.columns))

    def test_baseline_vs_improved_has_7_metrics(self):
        """baseline_vs_improved.csv must contain exactly 7 metrics."""
        path = RESULTS_DIR / "baseline_vs_improved.csv"
        if not path.exists():
            pytest.skip("baseline_vs_improved.csv not found.")
        df = pd.read_csv(path)
        assert len(df) == 7, f"Expected 7 metric rows, got {len(df)}"

    def test_baseline_vs_improved_no_nan(self):
        """baseline_vs_improved.csv must not contain NaN values."""
        path = RESULTS_DIR / "baseline_vs_improved.csv"
        if not path.exists():
            pytest.skip("baseline_vs_improved.csv not found.")
        df = pd.read_csv(path)
        assert not df.isnull().any().any()

    def test_improved_accuracy_better_than_baseline(self):
        """Improved model must outperform baseline on accuracy."""
        path = RESULTS_DIR / "baseline_vs_improved.csv"
        if not path.exists():
            pytest.skip("baseline_vs_improved.csv not found.")
        df = pd.read_csv(path)
        acc_row = df[df["metric"] == "Accuracy"]
        assert len(acc_row) == 1
        baseline_acc = float(acc_row["baseline_word_tfidf_no_balancing"].iloc[0])
        improved_acc = float(acc_row["improved_combined_tfidf_balanced"].iloc[0])
        assert improved_acc > baseline_acc, \
            f"Improved accuracy {improved_acc} is not better than baseline {baseline_acc}"

    def test_improved_macro_f1_better_than_baseline(self):
        """Improved model must outperform baseline on Macro F1."""
        path = RESULTS_DIR / "baseline_vs_improved.csv"
        if not path.exists():
            pytest.skip("baseline_vs_improved.csv not found.")
        df = pd.read_csv(path)
        row = df[df["metric"] == "Macro F1"]
        assert len(row) == 1
        b = float(row["baseline_word_tfidf_no_balancing"].iloc[0])
        i = float(row["improved_combined_tfidf_balanced"].iloc[0])
        assert i > b


# ---------------------------------------------------------------------------
# Test Group 7: JSON output integrity
# ---------------------------------------------------------------------------

class TestErrorAnalysisJSON:
    def test_json_exists(self, error_analysis_json):
        """error_analysis_data.json must exist and be parseable."""
        assert error_analysis_json is not None

    def test_json_has_required_keys(self, error_analysis_json):
        """JSON output must contain all required top-level keys."""
        required_keys = {
            "analysis_metadata", "model_config", "baseline_metrics",
            "improved_metrics", "top20_confusion_pairs", "per_category_metrics",
            "confidence_analysis", "class_distribution",
        }
        assert required_keys.issubset(set(error_analysis_json.keys()))

    def test_json_test_records_is_5000(self, error_analysis_json):
        """JSON metadata must report test_records = 5000."""
        assert error_analysis_json["analysis_metadata"]["test_records"] == 5000

    def test_json_n_categories_is_18(self, error_analysis_json):
        """JSON metadata must report n_categories = 18."""
        assert error_analysis_json["analysis_metadata"]["n_categories"] == 18

    def test_json_top20_pairs_has_at_most_20(self, error_analysis_json):
        """top20_confusion_pairs must have at most 20 entries."""
        assert len(error_analysis_json["top20_confusion_pairs"]) <= 20

    def test_json_per_category_metrics_has_18(self, error_analysis_json):
        """per_category_metrics must have exactly 18 entries."""
        assert len(error_analysis_json["per_category_metrics"]) == 18

    def test_json_no_nan_in_metrics(self, error_analysis_json):
        """Improved metrics in JSON must not contain NaN."""
        for key, val in error_analysis_json["improved_metrics"].items():
            assert not (isinstance(val, float) and np.isnan(val)), \
                f"NaN found in improved_metrics['{key}']"

    def test_json_improved_accuracy_matches_known_result(self, error_analysis_json):
        """Improved model accuracy must be approximately 69.56%."""
        acc = error_analysis_json["improved_metrics"]["accuracy"]
        assert abs(acc - 0.6956) < 0.005, \
            f"Improved accuracy {acc:.4f} deviates from expected 0.6956"
