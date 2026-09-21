"""Unit tests for complaint classification and supervised training module.

Tests classifier creation, fitting on sparse feature matrices, prediction,
multiclass support, probability estimation, single-complaint pipeline,
data splitting, and persistence.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from src.classification import (
    create_classifier,
    fit_classifier,
    load_classifier,
    predict_categories,
    predict_category,
    predict_category_proba,
    predict_complaint_category,
    save_classifier,
    train_logistic_regression,
    train_test_split_data,
)
from src.vectorization import create_vectorizer, fit_transform_corpus


@pytest.fixture
def synthetic_training_data():
    """Small deterministic synthetic training set with 3 classes and sparse features."""
    # 6 samples, 5 features, 3 classes
    X = sp.csr_matrix([
        [2.0, 1.0, 0.0, 0.0, 0.0],  # Class A
        [1.8, 1.2, 0.0, 0.0, 0.0],  # Class A
        [0.0, 0.0, 2.5, 1.0, 0.0],  # Class B
        [0.0, 0.0, 1.9, 1.5, 0.0],  # Class B
        [0.0, 0.0, 0.0, 0.5, 3.0],  # Class C
        [0.0, 0.0, 0.0, 0.2, 2.8],  # Class C
    ])
    y = np.array(["Credit Card", "Credit Card", "Mortgage", "Mortgage", "Student Loan", "Student Loan"])
    return X, y


@pytest.fixture
def fitted_synthetic_model(synthetic_training_data):
    """Fitted Logistic Regression model on synthetic data."""
    X, y = synthetic_training_data
    clf = create_classifier(random_state=42)
    return fit_classifier(clf, X, y)


@pytest.fixture
def sample_text_corpus():
    """Small deterministic text corpus with category labels."""
    df = pd.DataFrame([
        {"complaint_id": "1", "text": "unauthorized charge credit card late fee dispute", "category": "Credit Card"},
        {"complaint_id": "2", "text": "fraudulent transaction on credit card statement", "category": "Credit Card"},
        {"complaint_id": "3", "text": "mortgage servicer foreclosure loan modification", "category": "Mortgage"},
        {"complaint_id": "4", "text": "mortgage lender escrow payment delay", "category": "Mortgage"},
        {"complaint_id": "5", "text": "student loan payments interest allocation problem", "category": "Student Loan"},
        {"complaint_id": "6", "text": "student loan servicer misapplied monthly amount", "category": "Student Loan"},
    ])
    return df


# ======================================================================
# 1. Classifier Creation Tests
# ======================================================================

class TestClassifierCreation:
    """Test suite for classifier instantiation and configuration."""

    def test_default_classifier_creation(self):
        """Default classifier must have C=1.0, max_iter=1000, random_state=42."""
        clf = create_classifier()
        assert isinstance(clf, LogisticRegression)
        assert clf.C == 1.0
        assert clf.max_iter == 1000
        assert clf.random_state == 42
        assert clf.solver == "lbfgs"

    def test_custom_parameters_configuration(self):
        """Custom hyperparameters must be set correctly."""
        clf = create_classifier(C=0.5, max_iter=500, random_state=123)
        assert clf.C == 0.5
        assert clf.max_iter == 500
        assert clf.random_state == 123

    def test_invalid_parameters_raise_value_error(self):
        """Invalid C <= 0 or max_iter <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            create_classifier(C=0.0)
        with pytest.raises(ValueError, match="positive"):
            create_classifier(C=-1.5)
        with pytest.raises(ValueError, match="positive integer"):
            create_classifier(max_iter=0)
        with pytest.raises(ValueError, match="positive integer"):
            create_classifier(max_iter=-100)


# ======================================================================
# 2. Classifier Fitting Tests
# ======================================================================

class TestClassifierFitting:
    """Test suite for training classifier on sparse matrices."""

    def test_fit_on_deterministic_sparse_data(self, synthetic_training_data):
        """Fitting on sparse CSR matrix must succeed and establish classes_."""
        X, y = synthetic_training_data
        clf = create_classifier(random_state=42)
        fitted = fit_classifier(clf, X, y)
        assert hasattr(fitted, "classes_")
        assert len(fitted.classes_) == 3
        np.testing.assert_array_equal(np.sort(fitted.classes_), ["Credit Card", "Mortgage", "Student Loan"])

    def test_train_logistic_regression_wrapper(self, synthetic_training_data):
        """train_logistic_regression convenience function creates and fits classifier."""
        X, y = synthetic_training_data
        fitted = train_logistic_regression(X, y, c_param=2.0, max_iter=800, random_state=42)
        assert isinstance(fitted, LogisticRegression)
        assert fitted.C == 2.0
        assert hasattr(fitted, "classes_")

    def test_dense_ndarray_compatibility(self, synthetic_training_data):
        """Classifier must fit dense numpy ndarray inputs without error."""
        X, y = synthetic_training_data
        dense_X = X.toarray()
        clf = create_classifier(random_state=42)
        fitted = fit_classifier(clf, dense_X, y)
        assert hasattr(fitted, "classes_")

    def test_empty_training_data_raises_value_error(self):
        """Empty feature matrix or labels must raise ValueError."""
        empty_sparse = sp.csr_matrix((0, 5))
        clf = create_classifier()
        with pytest.raises(ValueError, match="empty"):
            fit_classifier(clf, empty_sparse, np.array([]))

    def test_mismatched_sample_lengths_raises_value_error(self, synthetic_training_data):
        """Mismatched sample counts between X and y must raise ValueError."""
        X, y = synthetic_training_data
        truncated_y = y[:3]  # X has 6 rows
        clf = create_classifier()
        with pytest.raises(ValueError, match="mismatch"):
            fit_classifier(clf, X, truncated_y)

    def test_single_class_raises_value_error(self):
        """Training data with only 1 class must raise ValueError."""
        X = sp.csr_matrix([[1.0, 2.0], [2.0, 3.0]])
        y = np.array(["SingleClass", "SingleClass"])
        clf = create_classifier()
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            fit_classifier(clf, X, y)

    def test_none_inputs_raise_type_error(self, synthetic_training_data):
        """None inputs must raise TypeError."""
        X, y = synthetic_training_data
        clf = create_classifier()
        with pytest.raises(TypeError):
            fit_classifier(None, X, y)
        with pytest.raises(TypeError):
            fit_classifier(clf, None, y)
        with pytest.raises(TypeError):
            fit_classifier(clf, X, None)


# ======================================================================
# 3. Inference and Prediction Tests
# ======================================================================

class TestInferenceAndPrediction:
    """Test suite for predictions, shapes, labels, and probability distributions."""

    def test_prediction_output_shape_and_classes(self, fitted_synthetic_model):
        """Predictions must have length N and belong to known classes."""
        X_test = sp.csr_matrix([
            [1.5, 0.9, 0.0, 0.0, 0.0],  # Clear Class A (Credit Card)
            [0.0, 0.0, 2.0, 1.2, 0.0],  # Clear Class B (Mortgage)
            [0.0, 0.0, 0.0, 0.1, 2.5],  # Clear Class C (Student Loan)
        ])
        preds = predict_categories(fitted_synthetic_model, X_test)
        assert len(preds) == 3
        assert preds[0] == "Credit Card"
        assert preds[1] == "Mortgage"
        assert preds[2] == "Student Loan"
        for label in preds:
            assert label in fitted_synthetic_model.classes_

    def test_unfitted_classifier_raises_not_fitted_error(self):
        """Calling predict on un-fitted classifier must raise NotFittedError."""
        clf = create_classifier()
        X = sp.csr_matrix([[1.0, 0.0, 0.0]])
        with pytest.raises(NotFittedError):
            predict_categories(clf, X)

    def test_predict_category_proba_output(self, fitted_synthetic_model):
        """predict_category_proba returns (N, num_classes) probabilities summing to 1.0."""
        X_test = sp.csr_matrix([
            [1.5, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.2, 0.0],
        ])
        probas = predict_category_proba(fitted_synthetic_model, X_test)
        assert probas.shape == (2, len(fitted_synthetic_model.classes_))
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(2), atol=1e-5)

    def test_unfitted_predict_proba_raises_error(self):
        """Calling predict_category_proba on un-fitted model raises NotFittedError."""
        clf = create_classifier()
        X = sp.csr_matrix([[1.0, 0.0]])
        with pytest.raises(NotFittedError):
            predict_category_proba(clf, X)

    def test_feature_dimension_mismatch_raises_value_error(self, fitted_synthetic_model):
        """Feature dimension mismatch must raise ValueError."""
        # Model trained on 5 features; test with 3 features
        mismatched_X = sp.csr_matrix([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="dimension mismatch"):
            predict_categories(fitted_synthetic_model, mismatched_X)

    def test_empty_inference_matrix_raises_value_error(self, fitted_synthetic_model):
        """Empty feature matrix for inference must raise ValueError."""
        empty_X = sp.csr_matrix((0, 5))
        with pytest.raises(ValueError, match="empty"):
            predict_categories(fitted_synthetic_model, empty_X)

    def test_none_input_raises_type_error(self, fitted_synthetic_model):
        """None input for inference raises TypeError."""
        with pytest.raises(TypeError):
            predict_categories(fitted_synthetic_model, None)


# ======================================================================
# 4. Single Complaint Prediction Pipeline Tests
# ======================================================================

class TestSingleComplaintPipeline:
    """Test suite for end-to-end single complaint prediction helper."""

    def test_predict_complaint_category_pipeline(self, sample_text_corpus):
        """Raw text narrative is preprocessed, vectorized, and classified with confidence."""
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        fitted_vec, X_train = fit_transform_corpus(sample_text_corpus["text"], vectorizer=vec)
        clf = train_logistic_regression(X_train, sample_text_corpus["category"])

        query_text = "I have an unauthorized charge on my credit card statement"
        predicted_cat, confidence = predict_complaint_category(clf, fitted_vec, query_text)

        assert isinstance(predicted_cat, str)
        assert predicted_cat == "Credit Card"
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_category_alias(self, sample_text_corpus):
        """predict_category alias behaves identically to predict_complaint_category."""
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        fitted_vec, X_train = fit_transform_corpus(sample_text_corpus["text"], vectorizer=vec)
        clf = train_logistic_regression(X_train, sample_text_corpus["category"])

        query_text = "lender mortgage foreclosure process"
        cat, conf = predict_category(clf, fitted_vec, query_text)
        assert cat == "Mortgage"
        assert 0.0 <= conf <= 1.0

    def test_empty_complaint_text_raises_value_error(self, sample_text_corpus):
        """Empty complaint string must raise ValueError."""
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1)
        fitted_vec, X_train = fit_transform_corpus(sample_text_corpus["text"], vectorizer=vec)
        clf = train_logistic_regression(X_train, sample_text_corpus["category"])

        with pytest.raises(ValueError, match="empty"):
            predict_complaint_category(clf, fitted_vec, "   ")

    def test_none_complaint_text_raises_type_error(self, sample_text_corpus):
        """None complaint text must raise TypeError."""
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1)
        fitted_vec, X_train = fit_transform_corpus(sample_text_corpus["text"], vectorizer=vec)
        clf = train_logistic_regression(X_train, sample_text_corpus["category"])

        with pytest.raises(TypeError):
            predict_complaint_category(clf, fitted_vec, None)


# ======================================================================
# 5. Data Splitting and Model Persistence Tests
# ======================================================================

class TestDataSplittingAndPersistence:
    """Test suite for train/test partition and model serialization."""

    def test_train_test_split_shapes_and_types(self, sample_text_corpus):
        """train_test_split_data correctly partitions text and labels."""
        X_train, X_test, y_train, y_test = train_test_split_data(
            sample_text_corpus, test_size=0.33, random_state=42, stratify=True
        )
        assert len(X_train) == 4
        assert len(X_test) == 2
        assert len(y_train) == 4
        assert len(y_test) == 2
        assert isinstance(X_train, pd.Series)
        assert isinstance(y_train, pd.Series)

    def test_missing_required_columns_raises_value_error(self):
        """DataFrame missing required text or category columns raises ValueError."""
        invalid_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        with pytest.raises(ValueError, match="text column"):
            train_test_split_data(invalid_df)

    def test_model_save_and_load_roundtrip(self, fitted_synthetic_model, synthetic_training_data):
        """Saved model can be reloaded and yields identical predictions."""
        X, _ = synthetic_training_data
        original_preds = predict_categories(fitted_synthetic_model, X)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.joblib"
            saved_file = save_classifier(fitted_synthetic_model, save_path)
            assert saved_file.exists()

            loaded_model = load_classifier(save_path)
            loaded_preds = predict_categories(loaded_model, X)
            np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_nonexistent_model_raises_file_not_found(self):
        """Attempting to load a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_classifier("nonexistent_path/model.joblib")
