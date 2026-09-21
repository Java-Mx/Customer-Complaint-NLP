"""Unit tests for cosine similarity and nearest-neighbor complaint retrieval.

Tests pairwise cosine similarity calculation, query-to-corpus retrieval,
sparse matrix handling, top-k ordering, boundary conditions, and input validation.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.similarity import (
    compute_cosine_similarity,
    cosine_similarity_matrix,
    find_similar_complaints,
    get_top_k_indices,
    get_top_k_similar,
)
from src.vectorization import create_vectorizer, fit_transform_corpus


@pytest.fixture
def sample_corpus_data():
    """Deterministic sample complaint corpus for similarity testing."""
    records = [
        {"complaint_id": "C001", "Product": "Credit Card", "Consumer Complaint": "unauthorized fraudulent charge on my credit card"},
        {"complaint_id": "C002", "Product": "Credit Card", "Consumer Complaint": "late fee and billing dispute on credit card statement"},
        {"complaint_id": "C003", "Product": "Mortgage", "Consumer Complaint": "mortgage lender denied loan modification request"},
        {"complaint_id": "C004", "Product": "Debt Collection", "Consumer Complaint": "harassing calls from debt collection agency for unknown debt"},
        {"complaint_id": "C005", "Product": "Student Loan", "Consumer Complaint": "student loan servicer misapplied monthly payment"},
    ]
    df = pd.DataFrame(records)
    # Add standardized aliases
    df["text"] = df["Consumer Complaint"]
    df["category"] = df["Product"]
    return df


@pytest.fixture
def sample_vectorizer_and_matrix(sample_corpus_data):
    """Fitted vectorizer and sparse TF-IDF matrix for sample corpus."""
    vectorizer = create_vectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)
    vec, matrix = fit_transform_corpus(sample_corpus_data["text"], vectorizer=vectorizer)
    return vec, matrix


# ======================================================================
# 1. Cosine Similarity Matrix Tests
# ======================================================================

class TestCosineSimilarityMatrix:
    """Test suite for pairwise cosine similarity matrix computations."""

    def test_identical_vectors_similarity_one(self):
        """Identical vectors must have a cosine similarity of exactly 1.0."""
        vec = sp.csr_matrix([[0.5, 0.5, 0.7071]])
        sim = cosine_similarity_matrix(vec, vec)
        assert sim.shape == (1, 1)
        assert np.isclose(sim[0, 0], 1.0, atol=1e-5)

    def test_orthogonal_vectors_similarity_zero(self):
        """Orthogonal vectors with disjoint non-zero features must have similarity 0.0."""
        vec_a = sp.csr_matrix([[1.0, 0.0, 0.0]])
        vec_b = sp.csr_matrix([[0.0, 1.0, 1.0]])
        sim = cosine_similarity_matrix(vec_a, vec_b)
        assert sim.shape == (1, 1)
        assert np.isclose(sim[0, 0], 0.0, atol=1e-5)

    def test_similarity_symmetry(self):
        """Pairwise similarity between two matrices must be transpose symmetric: sim(A, B) == sim(B, A).T."""
        matrix_a = sp.csr_matrix([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
        matrix_b = sp.csr_matrix([[0.5, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
        sim_ab = cosine_similarity_matrix(matrix_a, matrix_b)
        sim_ba = cosine_similarity_matrix(matrix_b, matrix_a)
        assert np.allclose(sim_ab, sim_ba.T, atol=1e-5)

    def test_self_similarity_matrix_shape(self):
        """When matrix_b is None, shape must be (M, M) with diagonal entries equal to 1.0."""
        matrix = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        sim = cosine_similarity_matrix(matrix)
        assert sim.shape == (3, 3)
        np.testing.assert_allclose(np.diag(sim), np.ones(3), atol=1e-5)

    def test_sparse_and_dense_matrix_compatibility(self):
        """Must accept scipy CSR, CSC matrices, and dense numpy ndarrays."""
        dense_a = np.array([[1.0, 0.5], [0.2, 0.8]])
        csr_b = sp.csr_matrix([[1.0, 0.5], [0.0, 1.0]])
        csc_b = sp.csc_matrix([[1.0, 0.5], [0.0, 1.0]])

        sim_dense_csr = cosine_similarity_matrix(dense_a, csr_b)
        sim_dense_csc = cosine_similarity_matrix(dense_a, csc_b)
        assert sim_dense_csr.shape == (2, 2)
        np.testing.assert_allclose(sim_dense_csr, sim_dense_csc, atol=1e-5)

    def test_dimension_mismatch_raises_error(self):
        """Mismatched feature columns between matrix_a and matrix_b must raise ValueError."""
        matrix_a = sp.csr_matrix(np.ones((2, 5)))
        matrix_b = sp.csr_matrix(np.ones((3, 4)))
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity_matrix(matrix_a, matrix_b)

    def test_empty_matrix_raises_error(self):
        """Empty matrix input must raise ValueError."""
        empty_matrix = sp.csr_matrix((0, 5))
        with pytest.raises(ValueError, match="empty"):
            cosine_similarity_matrix(empty_matrix)

    def test_none_input_raises_type_error(self):
        """None input must raise TypeError."""
        with pytest.raises(TypeError):
            cosine_similarity_matrix(None)


# ======================================================================
# 2. Query to Corpus Cosine Similarity Tests
# ======================================================================

class TestComputeCosineSimilarity:
    """Test suite for single-query to corpus similarity matching."""

    def test_query_to_corpus_sparse_returns_1d_array(self):
        """Query against N-document corpus must return 1D array of length N."""
        query = sp.csr_matrix([[1.0, 0.0, 1.0]])
        corpus = sp.csr_matrix([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
        ])
        scores = compute_cosine_similarity(query, corpus)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
        assert len(scores) == 4
        assert np.isclose(scores[0], 1.0, atol=1e-5)
        assert np.isclose(scores[1], 0.0, atol=1e-5)
        assert np.isclose(scores[3], 1.0, atol=1e-5)

    def test_scores_within_valid_range(self):
        """All computed cosine similarities must be in range [0.0, 1.0]."""
        np.random.seed(42)
        query = sp.csr_matrix(np.random.uniform(0, 1, size=(1, 20)))
        corpus = sp.csr_matrix(np.random.uniform(0, 1, size=(50, 20)))
        scores = compute_cosine_similarity(query, corpus)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-6)

    def test_1d_query_array_handled_cleanly(self):
        """1D numpy array query of shape (D,) must be handled without error."""
        query_1d = np.array([1.0, 0.0, 1.0])
        corpus = sp.csr_matrix([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        scores = compute_cosine_similarity(query_1d, corpus)
        assert len(scores) == 2
        assert np.isclose(scores[0], 1.0, atol=1e-5)

    def test_multi_row_query_raises_value_error(self):
        """Query matrix with more than 1 row must raise ValueError."""
        query_multi = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        corpus = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="single query"):
            compute_cosine_similarity(query_multi, corpus)

    def test_empty_corpus_raises_value_error(self):
        """Empty corpus must raise ValueError."""
        query = sp.csr_matrix([[1.0, 0.0]])
        empty_corpus = sp.csr_matrix((0, 2))
        with pytest.raises(ValueError, match="empty"):
            compute_cosine_similarity(query, empty_corpus)

    def test_none_inputs_raise_type_error(self):
        """None query or corpus must raise TypeError."""
        valid_vec = sp.csr_matrix([[1.0, 0.0]])
        with pytest.raises(TypeError):
            compute_cosine_similarity(None, valid_vec)
        with pytest.raises(TypeError):
            compute_cosine_similarity(valid_vec, None)


# ======================================================================
# 3. Top-K Retrieval and Ranking Tests
# ======================================================================

class TestTopKRetrieval:
    """Test suite for top-k ranking, ordering, and boundary handling."""

    def test_descending_top_k_ordering(self):
        """Top-k indices and scores must be strictly ordered in descending order."""
        scores = np.array([0.15, 0.85, 0.42, 0.99, 0.03, 0.77])
        top_indices = get_top_k_indices(scores, top_k=4)
        top_scores = scores[top_indices]

        expected_indices = [3, 1, 5, 2]  # scores: 0.99, 0.85, 0.77, 0.42
        assert list(top_indices) == expected_indices
        assert all(top_scores[i] >= top_scores[i + 1] for i in range(len(top_scores) - 1))

    def test_k_smaller_than_corpus_size(self):
        """When k < N, exactly k elements must be returned."""
        scores = np.array([0.1, 0.4, 0.9, 0.2, 0.7])
        indices, retrieved_scores = get_top_k_similar(scores, top_k=2)
        assert len(indices) == 2
        assert len(retrieved_scores) == 2
        assert list(indices) == [2, 4]

    def test_k_equal_to_corpus_size(self):
        """When k == N, all N elements must be returned in descending order."""
        scores = np.array([0.3, 0.1, 0.9])
        indices, retrieved_scores = get_top_k_similar(scores, top_k=3)
        assert len(indices) == 3
        assert list(indices) == [2, 0, 1]

    def test_k_larger_than_corpus_size(self):
        """When k > N, safely clamps to N without IndexError or crash."""
        scores = np.array([0.2, 0.8, 0.5])
        indices, retrieved_scores = get_top_k_similar(scores, top_k=10)
        assert len(indices) == 3
        assert list(indices) == [1, 2, 0]

    def test_invalid_k_zero_or_negative_raises_value_error(self):
        """k <= 0 must raise ValueError."""
        scores = np.array([0.5, 0.6])
        with pytest.raises(ValueError, match="positive integer"):
            get_top_k_indices(scores, top_k=0)
        with pytest.raises(ValueError, match="positive integer"):
            get_top_k_indices(scores, top_k=-2)

    def test_invalid_k_type_raises_type_error(self):
        """Non-integer k must raise TypeError."""
        scores = np.array([0.5, 0.6])
        with pytest.raises(TypeError, match="integer"):
            get_top_k_indices(scores, top_k="3")  # type: ignore

    def test_empty_scores_raises_value_error(self):
        """Empty scores array must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            get_top_k_indices(np.array([]), top_k=5)

    def test_get_top_k_similar_with_vector_and_corpus(self):
        """get_top_k_similar(query_vec, corpus_mat, top_k) computes and retrieves top results."""
        query = sp.csr_matrix([[1.0, 0.0]])
        corpus = sp.csr_matrix([[0.0, 1.0], [0.9, 0.1], [0.5, 0.5]])
        indices, scores = get_top_k_similar(query, corpus, top_k=2)
        assert len(indices) == 2
        assert indices[0] == 1
        assert scores[0] >= scores[1]


# ======================================================================
# 4. End-to-End Complaint Search Tests
# ======================================================================

class TestFindSimilarComplaints:
    """Test suite for full end-to-end complaint similarity retrieval."""

    def test_end_to_end_search_with_fitted_vectorizer(self, sample_corpus_data, sample_vectorizer_and_matrix):
        """Raw string query successfully retrieves top matches with ranked DataFrame."""
        vec, matrix = sample_vectorizer_and_matrix
        query = "fraudulent unauthorized credit card charges"

        result_df = find_similar_complaints(
            query_text=query,
            vectorizer=vec,
            corpus_matrix=matrix,
            df_corpus=sample_corpus_data,
            top_k=3,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        # First match should be complaint C001 (credit card fraud)
        assert result_df.iloc[0]["complaint_id"] == "C001"
        assert result_df.iloc[0]["rank"] == 1
        assert result_df.iloc[0]["similarity_score"] > 0.3

    def test_preservation_of_complaint_ids_and_text(self, sample_corpus_data, sample_vectorizer_and_matrix):
        """Result DataFrame preserves complaint IDs and text faithfully from df_corpus."""
        vec, matrix = sample_vectorizer_and_matrix
        query = "mortgage payment loan modification"

        result_df = find_similar_complaints(
            query_text=query,
            vectorizer=vec,
            corpus_matrix=matrix,
            df_corpus=sample_corpus_data,
            top_k=2,
        )

        expected_columns = {"rank", "complaint_id", "similarity_score", "complaint_text", "text"}
        assert expected_columns.issubset(set(result_df.columns))
        # Best match should be C003 (mortgage loan modification)
        assert result_df.iloc[0]["complaint_id"] == "C003"
        assert "mortgage" in result_df.iloc[0]["complaint_text"].lower()

    def test_scores_are_sorted_descending(self, sample_corpus_data, sample_vectorizer_and_matrix):
        """Result DataFrame similarity scores must be sorted descending."""
        vec, matrix = sample_vectorizer_and_matrix
        result_df = find_similar_complaints(
            query_text="credit card dispute",
            vectorizer=vec,
            corpus_matrix=matrix,
            df_corpus=sample_corpus_data,
            top_k=5,
        )
        scores = result_df["similarity_score"].tolist()
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_preprocessed_query_retrieval(self, sample_corpus_data, sample_vectorizer_and_matrix):
        """Query with punctuation and noise is preprocessed before vectorization."""
        vec, matrix = sample_vectorizer_and_matrix
        noisy_query = "UNAUTHORIZED CHARGE!!! https://scam.org XXXX fee on my credit card!"

        result_df = find_similar_complaints(
            query_text=noisy_query,
            vectorizer=vec,
            corpus_matrix=matrix,
            df_corpus=sample_corpus_data,
            top_k=1,
            preprocess=True,
        )
        assert len(result_df) == 1
        assert result_df.iloc[0]["complaint_id"] in ["C001", "C002"]

    def test_precomputed_vector_query(self, sample_corpus_data, sample_vectorizer_and_matrix):
        """Query provided directly as a precomputed sparse vector works without vectorizer."""
        vec, matrix = sample_vectorizer_and_matrix
        query_vec = vec.transform(["mortgage lender loan"])

        result_df = find_similar_complaints(
            query_text=query_vec,
            vectorizer=None,
            corpus_matrix=matrix,
            df_corpus=sample_corpus_data,
            top_k=2,
        )
        assert len(result_df) == 2
        assert result_df.iloc[0]["complaint_id"] == "C003"

    def test_empty_query_raises_value_error(self, sample_corpus_data, sample_vectorizer_and_matrix):
        """Empty or whitespace-only query string raises ValueError."""
        vec, matrix = sample_vectorizer_and_matrix
        with pytest.raises(ValueError, match="empty"):
            find_similar_complaints(
                query_text="   ",
                vectorizer=vec,
                corpus_matrix=matrix,
                df_corpus=sample_corpus_data,
                top_k=3,
            )

    def test_corpus_size_mismatch_raises_value_error(self, sample_corpus_data, sample_vectorizer_and_matrix):
        """Mismatch between df_corpus rows and corpus_matrix rows raises ValueError."""
        vec, matrix = sample_vectorizer_and_matrix
        truncated_df = sample_corpus_data.iloc[:2]
        with pytest.raises(ValueError, match="mismatch"):
            find_similar_complaints(
                query_text="credit card",
                vectorizer=vec,
                corpus_matrix=matrix,  # has 5 rows
                df_corpus=truncated_df,  # has 2 rows
                top_k=2,
            )

    def test_empty_corpus_raises_value_error(self, sample_vectorizer_and_matrix):
        """Empty corpus DataFrame raises ValueError."""
        vec, _ = sample_vectorizer_and_matrix
        empty_df = pd.DataFrame()
        empty_matrix = sp.csr_matrix((0, 10))
        with pytest.raises(ValueError, match="empty"):
            find_similar_complaints(
                query_text="credit card",
                vectorizer=vec,
                corpus_matrix=empty_matrix,
                df_corpus=empty_df,
                top_k=2,
            )
