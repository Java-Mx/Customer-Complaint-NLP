"""Strict deployment artifact and readiness tests for production models.

Validates that real serialized model artifacts (complaint_classifier.joblib,
tfidf_vectorizer.joblib, and char_vectorizer.joblib) are present in the models/
directory, can be deserialized cleanly, match the expected feature dimensions
(114,493 features), and produce valid predictions on authentic complaint text.

NOTE: These tests do NOT mock model files; they test the actual production artifacts.
"""

from pathlib import Path
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse

from src.classification import (
    load_classifier,
    predict_category_proba,
    predict_complaint_category,
    validate_model_artifacts,
)
from src.preprocessing import preprocess_text
from src.vectorization import transform_word_char

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def test_production_model_artifacts_exist():
    """Verify all three production model files exist in the models/ directory."""
    clf_path = MODELS_DIR / "complaint_classifier.joblib"
    w_vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    c_vec_path = MODELS_DIR / "char_vectorizer.joblib"

    assert clf_path.exists(), f"Missing production model artifact: {clf_path}"
    assert w_vec_path.exists(), f"Missing production word vectorizer: {w_vec_path}"
    assert c_vec_path.exists(), f"Missing production char vectorizer: {c_vec_path}"

    assert clf_path.stat().st_size > 0, "complaint_classifier.joblib is empty"
    assert w_vec_path.stat().st_size > 0, "tfidf_vectorizer.joblib is empty"
    assert c_vec_path.stat().st_size > 0, "char_vectorizer.joblib is empty"


def test_production_word_vectorizer_exists():
    """Verify that models/tfidf_vectorizer.joblib exists and is non-empty."""
    w_vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    assert w_vec_path.exists()
    assert w_vec_path.is_file()
    assert w_vec_path.stat().st_size > 10_000, "tfidf_vectorizer.joblib appears truncated"


def test_production_char_vectorizer_exists():
    """Verify that models/char_vectorizer.joblib exists and is non-empty."""
    c_vec_path = MODELS_DIR / "char_vectorizer.joblib"
    assert c_vec_path.exists()
    assert c_vec_path.is_file()
    assert c_vec_path.stat().st_size > 100_000, "char_vectorizer.joblib appears truncated"


def test_production_artifacts_load():
    """Verify that all production artifacts deserialize cleanly using joblib."""
    clf = joblib.load(MODELS_DIR / "complaint_classifier.joblib")
    w_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    c_vec = joblib.load(MODELS_DIR / "char_vectorizer.joblib")

    assert clf is not None
    assert w_vec is not None
    assert c_vec is not None
    assert hasattr(clf, "predict")
    assert hasattr(w_vec, "transform")
    assert hasattr(c_vec, "transform")


def test_production_artifacts_are_compatible():
    """Verify that classifier and vectorizers correspond to the same trained model."""
    clf = joblib.load(MODELS_DIR / "complaint_classifier.joblib")
    w_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    c_vec = joblib.load(MODELS_DIR / "char_vectorizer.joblib")

    # Classifier classes
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 18

    # Vocabulary attributes
    assert hasattr(w_vec, "vocabulary_")
    assert hasattr(c_vec, "vocabulary_")

    w_dim = len(w_vec.vocabulary_)
    c_dim = len(c_vec.vocabulary_)
    combined_dim = w_dim + c_dim

    # Classifier input feature dimension
    clf_features = getattr(clf, "n_features_in_", None)
    if clf_features is None and hasattr(clf, "coef_"):
        clf_features = clf.coef_.shape[1]

    assert clf_features == combined_dim, (
        f"Classifier expected {clf_features} features, but combined vectorizer has {combined_dim}"
    )


def test_expected_feature_dimension():
    """Verify exact audited production feature dimensions (114,493 features)."""
    w_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    c_vec = joblib.load(MODELS_DIR / "char_vectorizer.joblib")
    clf = joblib.load(MODELS_DIR / "complaint_classifier.joblib")

    w_dim = len(w_vec.vocabulary_)
    c_dim = len(c_vec.vocabulary_)
    total_dim = w_dim + c_dim

    assert w_dim == 14493, f"Expected 14,493 word features, got {w_dim}"
    assert c_dim == 100000, f"Expected 100,000 char features, got {c_dim}"
    assert total_dim == 114493, f"Expected 114,493 combined features, got {total_dim}"

    clf_features = getattr(clf, "n_features_in_", None)
    if clf_features is None and hasattr(clf, "coef_"):
        clf_features = clf.coef_.shape[1]
    assert clf_features == 114493, f"Classifier n_features_in_ expected 114,493, got {clf_features}"


def test_validate_model_artifacts_function():
    """Test the reusable validate_model_artifacts function on real and invalid paths."""
    res = validate_model_artifacts(MODELS_DIR)
    assert res["is_valid"] is True
    assert len(res["errors"]) == 0
    assert res["feature_dim"] == 114493
    assert len(res["classes"]) == 18

    # Test on empty directory
    res_empty = validate_model_artifacts(ROOT_DIR / "docs")
    assert res_empty["is_valid"] is False
    assert len(res_empty["errors"]) > 0

    with pytest.raises(FileNotFoundError):
        validate_model_artifacts(ROOT_DIR / "docs", raise_on_error=True)


def test_deployment_smoke_prediction():
    """Deployment smoke test: authenticate end-to-end transformation and prediction.

    Pipeline:
    raw text -> clean text -> Word TF-IDF + Char TF-IDF -> combined sparse matrix -> classifier -> prediction
    """
    sample_complaint = (
        "I noticed multiple unauthorized transactions and recurring charges on my credit card "
        "statement from a merchant I never visited. I contacted customer service immediately to dispute "
        "the charges, but the bank refused to issue a provisional credit and continues to bill monthly finance charges."
    )

    clf = joblib.load(MODELS_DIR / "complaint_classifier.joblib")
    w_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    c_vec = joblib.load(MODELS_DIR / "char_vectorizer.joblib")

    clean_q = preprocess_text(sample_complaint)
    assert len(clean_q) > 0

    X_q = transform_word_char(w_vec, c_vec, [clean_q])
    assert issparse(X_q)
    assert X_q.shape == (1, 114493)
    assert X_q.nnz > 0

    pred_category, conf = predict_complaint_category(
        model=clf,
        vectorizer=(w_vec, c_vec),
        narrative=sample_complaint,
        preprocess=True,
    )

    assert pred_category in clf.classes_
    assert pred_category in ["Credit card", "Credit card or prepaid card"]
    assert 0.0 <= conf <= 1.0

    probas = predict_category_proba(clf, X_q)[0]
    assert len(probas) == 18
    assert np.isclose(np.sum(probas), 1.0, atol=1e-3)
