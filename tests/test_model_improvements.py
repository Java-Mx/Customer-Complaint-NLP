"""Unit tests for model improvements, character TF-IDF, and LinearSVC classification.

Tests character-level vectorization, sparse matrix combination, LinearSVC instantiation,
fitting, predictions, normalized confidence score extraction, and composite vectorizer inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from src.vectorization import (
    create_char_vectorizer,
    combine_sparse_matrices,
    fit_transform_word_char,
    transform_word_char,
    create_vectorizer,
)
from src.classification import (
    create_classifier,
    create_linear_svc,
    fit_classifier,
    predict_categories,
    predict_category_proba,
    predict_complaint_category,
    train_test_split_data,
)


class TestCharacterAndCombinedVectorization:
    """Tests for character-level and stacked sparse TF-IDF vectorization."""

    def test_create_char_vectorizer_defaults(self):
        """Test default parameters of character vectorizer."""
        vec = create_char_vectorizer()
        assert vec.analyzer == "char_wb"
        assert vec.ngram_range == (3, 5)
        assert vec.sublinear_tf is True
        assert vec.lowercase is False

    def test_char_vectorizer_feature_extraction(self):
        """Test character n-grams extracted within word boundaries."""
        texts = ["mortgage loan dispute", "credit card payment issue"]
        vec = create_char_vectorizer(ngram_range=(3, 4), min_df=1)
        matrix = vec.fit_transform(texts)
        assert issparse(matrix)
        assert matrix.shape[0] == 2
        features = vec.get_feature_names_out()
        # char_wb should contain character n-grams
        assert any(len(f) in (3, 4) for f in features)

    def test_combine_sparse_matrices_success(self):
        """Test horizontal stacking preserves CSR sparse format and column sum."""
        X1 = csr_matrix([[1, 2], [3, 4]])
        X2 = csr_matrix([[5, 6, 7], [8, 9, 10]])
        combined = combine_sparse_matrices(X1, X2)
        assert issparse(combined)
        assert combined.format == "csr"
        assert combined.shape == (2, 5)
        np.testing.assert_array_equal(combined.toarray(), [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]])

    def test_combine_sparse_matrices_validation(self):
        """Test that invalid types or row mismatches raise errors."""
        X1 = csr_matrix([[1, 2], [3, 4]])
        X_dense = np.array([[1, 2], [3, 4]])
        X_mismatch = csr_matrix([[1], [2], [3]])

        with pytest.raises(TypeError, match="must be scipy sparse matrices"):
            combine_sparse_matrices(X1, X_dense)

        with pytest.raises(ValueError, match="Row count mismatch"):
            combine_sparse_matrices(X1, X_mismatch)

    def test_fit_transform_and_transform_word_char(self):
        """Test end-to-end composite word + character vectorizer fitting and transformation."""
        corpus = [
            "debt collector calling daily",
            "fraudulent credit card charges",
            "mortgage servicer escrow problem",
            "student loan interest calculation"
        ]
        w_vec = create_vectorizer(ngram_range=(1, 2), min_df=1)
        c_vec = create_char_vectorizer(ngram_range=(3, 4), min_df=1)

        w_fit, c_fit, X_train = fit_transform_word_char(w_vec, c_vec, corpus)
        assert issparse(X_train)
        assert X_train.shape[0] == 4
        expected_cols = len(w_fit.vocabulary_) + len(c_fit.vocabulary_)
        assert X_train.shape[1] == expected_cols

        # Test transform on new texts
        new_texts = ["credit card dispute", "debt call"]
        X_test = transform_word_char(w_fit, c_fit, new_texts)
        assert issparse(X_test)
        assert X_test.shape == (2, expected_cols)


class TestClassifierImprovements:
    """Tests for LinearSVC, class weighting, and confidence extraction."""

    def test_create_classifier_types(self):
        """Test factory creation of both LogisticRegression and LinearSVC."""
        lr = create_classifier(classifier_type="logistic_regression", C=2.0)
        assert isinstance(lr, LogisticRegression)
        assert lr.C == 2.0

        svc = create_classifier(classifier_type="linear_svc", C=0.5, class_weight="balanced")
        assert isinstance(svc, LinearSVC)
        assert svc.C == 0.5
        assert svc.class_weight == "balanced"

    def test_create_linear_svc_helper(self):
        """Test dedicated create_linear_svc helper."""
        svc = create_linear_svc(C=1.5, class_weight="balanced", max_iter=3000)
        assert isinstance(svc, LinearSVC)
        assert svc.C == 1.5
        assert svc.class_weight == "balanced"
        assert svc.max_iter == 3000

    def test_fit_and_predict_linear_svc(self):
        """Test fitting LinearSVC on sparse matrices and predicting categories."""
        X = csr_matrix([
            [1.0, 0.0, 0.5],
            [0.8, 0.1, 0.4],
            [0.0, 1.2, 0.9],
            [0.1, 0.9, 1.1]
        ])
        y = np.array(["Credit Card", "Credit Card", "Mortgage", "Mortgage"])

        svc = create_linear_svc(C=1.0, random_state=42)
        fitted_svc = fit_classifier(svc, X, y)
        preds = predict_categories(fitted_svc, X)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 4
        assert preds[0] == "Credit Card"
        assert preds[2] == "Mortgage"

    def test_predict_category_proba_linear_svc(self):
        """Test normalized confidence scores for LinearSVC sum to 1.0."""
        X = csr_matrix([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5]
        ])
        y = np.array(["Credit Card", "Mortgage"])
        svc = create_linear_svc(C=1.0, random_state=42)
        fit_classifier(svc, X, y)

        scores = predict_category_proba(svc, X)
        assert scores.shape == (2, 2)
        np.testing.assert_allclose(np.sum(scores, axis=1), [1.0, 1.0], rtol=1e-5)
        # Score for Credit Card should be highest for row 0
        assert scores[0, 0] > scores[0, 1]
        # Score for Mortgage should be highest for row 1
        assert scores[1, 1] > scores[1, 0]

    def test_predict_complaint_category_composite_vectorizers(self):
        """Test single narrative prediction using composite (word, char) vectorizers and LinearSVC."""
        corpus = [
            "unauthorized credit card billing dispute",
            "mortgage escrow payment delay notice",
            "debt collector calling employer repeatedly"
        ]
        labels = ["Credit Card", "Mortgage", "Debt Collection"]
        w_vec = create_vectorizer(ngram_range=(1, 2), min_df=1)
        c_vec = create_char_vectorizer(ngram_range=(3, 4), min_df=1)
        w_fit, c_fit, X = fit_transform_word_char(w_vec, c_vec, corpus)

        svc = create_linear_svc(C=1.0, random_state=42)
        fit_classifier(svc, X, labels)

        # Single narrative prediction
        pred_cat, conf = predict_complaint_category(
            model=svc,
            vectorizer=(w_fit, c_fit),
            narrative="I want to dispute an unauthorized charge on my credit card statement",
            preprocess=True
        )
        assert isinstance(pred_cat, str)
        assert pred_cat == "Credit Card"
        assert 0.0 <= conf <= 1.0

    def test_train_test_split_fallback_resilience(self):
        """Test train_test_split_data handles edge case without crashing."""
        df = pd.DataFrame({
            "text": ["complaint one", "complaint two", "complaint three", "complaint four"],
            "category": ["A", "A", "B", "C"]
        })
        X_tr, X_te, y_tr, y_te = train_test_split_data(df, test_size=0.25, random_state=42, stratify=True)
        assert len(X_tr) == 3
        assert len(X_te) == 1
