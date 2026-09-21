"""Unit tests for the TF-IDF vectorization module."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

from src.vectorization import (
    create_vectorizer,
    build_tfidf_vectorizer,
    fit_tfidf,
    transform_tfidf,
    fit_transform_tfidf,
    fit_transform_corpus,
)


@pytest.fixture
def sample_corpus() -> list[str]:
    """Sample preprocessed complaints corpus with unigrams and repeated bigrams."""
    return [
        "unauthorized charge credit card account dispute fee",
        "late payment fee credit card statement dispute",
        "mortgage loan modification request denied by lender",
        "identity theft reported fraudulent loan account opened",
        "credit card payment processed late fee charged again",
    ]


class TestVectorizerCreation:
    """Tests for vectorizer instantiation and default parameters."""

    def test_default_parameters(self):
        vec = create_vectorizer()
        assert isinstance(vec, TfidfVectorizer)
        assert vec.ngram_range == (1, 2)
        assert vec.min_df == 2
        assert vec.max_df == 0.95
        assert vec.sublinear_tf is True
        assert vec.lowercase is False

    def test_alias_creation(self):
        vec = build_tfidf_vectorizer(ngram_range=(1, 1), min_df=1)
        assert vec.ngram_range == (1, 1)
        assert vec.min_df == 1


class TestFitAndTransform:
    """Tests for fitting, transforming, and vocabulary generation."""

    def test_fit_transform_on_sample(self, sample_corpus):
        vec = create_vectorizer(min_df=1)
        fitted_vec, matrix = fit_transform_tfidf(vec, sample_corpus)

        assert fitted_vec is vec
        # Output must be a scipy sparse matrix
        assert issparse(matrix)
        assert matrix.shape[0] == len(sample_corpus)
        assert matrix.shape[1] > 0

    def test_vocabulary_created(self, sample_corpus):
        vec = create_vectorizer(min_df=1)
        fitted_vec = fit_tfidf(vec, sample_corpus)

        assert hasattr(fitted_vec, "vocabulary_")
        assert len(fitted_vec.vocabulary_) > 0

    def test_unigram_and_bigram_generation(self, sample_corpus):
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1)
        fitted_vec, _ = fit_transform_tfidf(vec, sample_corpus)
        vocab = fitted_vec.vocabulary_

        # Verify both unigram and bigram features exist
        assert "credit" in vocab
        assert "card" in vocab
        assert "credit card" in vocab
        assert "late payment" in vocab
        assert "loan modification" in vocab
        assert "identity theft" in vocab

    def test_transform_with_fitted_vectorizer(self, sample_corpus):
        vec = create_vectorizer(min_df=1)
        fitted_vec = fit_tfidf(vec, sample_corpus)

        new_doc = ["credit card dispute payment"]
        transformed = transform_tfidf(fitted_vec, new_doc)

        assert issparse(transformed)
        assert transformed.shape == (1, len(fitted_vec.vocabulary_))

    def test_consistent_feature_dimensions(self, sample_corpus):
        vec = create_vectorizer(min_df=1)
        _, train_matrix = fit_transform_tfidf(vec, sample_corpus)

        new_texts = [
            "fraudulent credit card charge",
            "mortgage payment fee applied incorrectly"
        ]
        test_matrix = transform_tfidf(vec, new_texts)

        assert test_matrix.shape[0] == 2
        assert test_matrix.shape[1] == train_matrix.shape[1]

    def test_min_df_filtering(self):
        # In a 4-document corpus:
        # 'recurring_term' appears in 2 docs (50% <= 95% max_df, >= 2 min_df) -> kept
        # 'rare_term' appears in only 1 doc (< 2 min_df) -> pruned
        docs = [
            "recurring_term rare_term alpha",
            "recurring_term beta",
            "different_term gamma",
            "different_term delta",
        ]
        vec = create_vectorizer(min_df=2)
        fit_tfidf(vec, docs)
        vocab = vec.vocabulary_

        assert "rare_term" not in vocab
        assert "recurring_term" in vocab
        assert "different_term" in vocab

    def test_convenience_fit_transform_corpus(self, sample_corpus):
        vec, matrix = fit_transform_corpus(sample_corpus)
        assert issparse(matrix)
        assert matrix.shape[0] == len(sample_corpus)


class TestInputHandlingAndErrors:
    """Tests for input validation, edge cases, and error states."""

    def test_transform_before_fit_raises_error(self):
        vec = create_vectorizer()
        with pytest.raises(NotFittedError, match="not fitted"):
            transform_tfidf(vec, ["some complaint text"])

    def test_empty_document_collection_raises_value_error(self):
        vec = create_vectorizer()
        with pytest.raises(ValueError, match="Document collection is empty"):
            fit_tfidf(vec, [])

        with pytest.raises(ValueError, match="Document collection is empty"):
            fit_transform_tfidf(vec, [])

    def test_none_input_raises_type_error(self):
        vec = create_vectorizer()
        with pytest.raises(TypeError, match="cannot be None"):
            fit_tfidf(vec, None)

    def test_invalid_vectorizer_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected TfidfVectorizer"):
            fit_tfidf("not_a_vectorizer", ["some text"])

        with pytest.raises(TypeError, match="Expected TfidfVectorizer"):
            transform_tfidf("not_a_vectorizer", ["some text"])

    def test_series_and_array_inputs(self, sample_corpus):
        vec = create_vectorizer(min_df=1)
        series_input = pd.Series(sample_corpus)
        fitted_vec, matrix = fit_transform_tfidf(vec, series_input)
        assert matrix.shape[0] == len(sample_corpus)

        array_input = np.array(["credit card dispute", "loan fee"])
        transformed = transform_tfidf(fitted_vec, array_input)
        assert transformed.shape[0] == 2

    def test_single_string_convenience_transform(self, sample_corpus):
        vec = create_vectorizer(min_df=1)
        fit_tfidf(vec, sample_corpus)
        single_doc = "credit card payment"
        transformed = transform_tfidf(vec, single_doc)
        assert transformed.shape == (1, len(vec.vocabulary_))
