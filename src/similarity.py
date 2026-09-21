"""Cosine similarity module for customer complaint retrieval.

Computes pairwise cosine similarities across TF-IDF document representations
to identify historically similar customer complaints using classical Scikit-learn
sparse metrics without converting large matrices to dense representations.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import issparse, spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


def cosine_similarity_matrix(
    matrix_a: Any,
    matrix_b: Optional[Any] = None
) -> np.ndarray:
    """Compute pairwise cosine similarity between two feature matrices.

    Operates natively on sparse matrices (e.g., scipy.sparse.csr_matrix)
    to prevent dense memory allocation on large corpora.

    Parameters
    ----------
    matrix_a : spmatrix | np.ndarray
        First document-term matrix (M x D).
    matrix_b : spmatrix | np.ndarray | None, optional
        Second document-term matrix (N x D). If None, computes pairwise
        similarity of matrix_a with itself (M x M).

    Returns
    -------
    np.ndarray
        2D array of cosine similarity scores in range [0.0, 1.0].
        Shape is (M, N) or (M, M).

    Raises
    ------
    TypeError
        If matrix_a or matrix_b is None when expected, or has an invalid type.
    ValueError
        If either matrix is empty or feature dimensions (columns) do not match.
    """
    if matrix_a is None:
        raise TypeError("matrix_a cannot be None.")

    # Validate matrix_a shape and emptiness
    if not (issparse(matrix_a) or isinstance(matrix_a, (np.ndarray, list))):
        raise TypeError(f"Expected sparse matrix or ndarray for matrix_a, got {type(matrix_a).__name__}.")

    if isinstance(matrix_a, list):
        matrix_a = np.asarray(matrix_a)

    if matrix_a.shape[0] == 0 or (len(matrix_a.shape) > 1 and matrix_a.shape[1] == 0):
        raise ValueError("matrix_a is empty. Cannot compute cosine similarity.")

    if len(matrix_a.shape) == 1:
        matrix_a = matrix_a.reshape(1, -1)

    if matrix_b is not None:
        if not (issparse(matrix_b) or isinstance(matrix_b, (np.ndarray, list))):
            raise TypeError(f"Expected sparse matrix or ndarray for matrix_b, got {type(matrix_b).__name__}.")

        if isinstance(matrix_b, list):
            matrix_b = np.asarray(matrix_b)

        if matrix_b.shape[0] == 0 or (len(matrix_b.shape) > 1 and matrix_b.shape[1] == 0):
            raise ValueError("matrix_b is empty. Cannot compute cosine similarity.")

        if len(matrix_b.shape) == 1:
            matrix_b = matrix_b.reshape(1, -1)

        if matrix_a.shape[1] != matrix_b.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: matrix_a has {matrix_a.shape[1]} features, "
                f"but matrix_b has {matrix_b.shape[1]} features."
            )

    similarity = cosine_similarity(matrix_a, matrix_b, dense_output=True)
    # Clip to [0.0, 1.0] for numerical stability on non-negative TF-IDF representations
    return np.clip(similarity, 0.0, 1.0)


def compute_cosine_similarity(
    query_vector: Any,
    corpus_matrix: Any
) -> np.ndarray:
    """Compute cosine similarity between a single query TF-IDF vector and a corpus matrix.

    Scalable: Computes only 1 x N similarities against the sparse corpus matrix
    without allocating an N x N similarity matrix.

    Parameters
    ----------
    query_vector : spmatrix | np.ndarray
        Sparse or dense 1xD vector (or 1D array of length D) representing the query.
    corpus_matrix : spmatrix | np.ndarray
        Sparse NxD matrix representing the indexed complaint corpus.

    Returns
    -------
    np.ndarray
        1D array of similarity scores of length N in range [0.0, 1.0].

    Raises
    ------
    TypeError
        If query_vector or corpus_matrix is None or not a matrix-like structure.
    ValueError
        If inputs are empty, query is multi-row, or feature dimensions mismatch.
    """
    if query_vector is None:
        raise TypeError("query_vector cannot be None.")
    if corpus_matrix is None:
        raise TypeError("corpus_matrix cannot be None.")

    if not (issparse(query_vector) or isinstance(query_vector, (np.ndarray, list))):
        raise TypeError(f"Expected sparse or ndarray query_vector, got {type(query_vector).__name__}.")
    if not (issparse(corpus_matrix) or isinstance(corpus_matrix, (np.ndarray, list))):
        raise TypeError(f"Expected sparse or ndarray corpus_matrix, got {type(corpus_matrix).__name__}.")

    if isinstance(query_vector, list):
        query_vector = np.asarray(query_vector)
    if isinstance(corpus_matrix, list):
        corpus_matrix = np.asarray(corpus_matrix)

    if corpus_matrix.shape[0] == 0 or (len(corpus_matrix.shape) > 1 and corpus_matrix.shape[1] == 0):
        raise ValueError("Corpus matrix is empty. Cannot compute cosine similarity.")

    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    elif query_vector.shape[0] != 1:
        raise ValueError(f"query_vector must represent a single query (1 row), got shape {query_vector.shape}.")

    if query_vector.shape[1] != corpus_matrix.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: query has {query_vector.shape[1]} features, "
            f"but corpus has {corpus_matrix.shape[1]} features."
        )

    similarity_2d = cosine_similarity_matrix(query_vector, corpus_matrix)
    return similarity_2d.flatten()


def get_top_k_indices(
    similarity_scores: Any,
    top_k: int = 5
) -> np.ndarray:
    """Retrieve the indices of the top-k highest similarity scores in descending order.

    Safely clamps top_k to corpus size if top_k exceeds available documents.

    Parameters
    ----------
    similarity_scores : Sequence[float] | np.ndarray
        1D sequence of similarity scores.
    top_k : int, default=5
        Number of top indices to retrieve. Must be positive.

    Returns
    -------
    np.ndarray
        1D array of integer indices corresponding to highest scores descending.

    Raises
    ------
    TypeError
        If similarity_scores is None.
    ValueError
        If top_k <= 0 or similarity_scores is empty.
    """
    if similarity_scores is None:
        raise TypeError("similarity_scores cannot be None.")

    scores = np.asarray(similarity_scores).ravel()
    num_scores = len(scores)

    if not isinstance(top_k, (int, np.integer)):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")
    if num_scores == 0:
        raise ValueError("similarity_scores array is empty.")

    k_clamped = min(top_k, num_scores)
    # Sort descending
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices[:k_clamped]


def get_top_k_similar(
    query_or_scores: Any,
    corpus_matrix: Optional[Any] = None,
    top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve top-k indices and similarity scores in descending order.

    Flexible invocation:
    1. get_top_k_similar(query_vector, corpus_matrix, top_k=5)
    2. get_top_k_similar(similarity_scores, top_k=5)

    Parameters
    ----------
    query_or_scores : Any
        Query TF-IDF vector (if corpus_matrix provided) or precomputed scores array.
    corpus_matrix : Any | None, optional
        Corpus TF-IDF sparse matrix (if computing similarity), or omitted/None.
    top_k : int, default=5
        Number of top records to retrieve.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (top_indices, top_scores) in descending order of similarity.
    """
    if isinstance(corpus_matrix, (int, np.integer)):
        top_k = int(corpus_matrix)
        corpus_matrix = None

    if corpus_matrix is not None:
        scores = compute_cosine_similarity(query_or_scores, corpus_matrix)
    else:
        if query_or_scores is None:
            raise TypeError("similarity_scores cannot be None.")
        scores = np.asarray(query_or_scores).ravel()

    top_indices = get_top_k_indices(scores, top_k=top_k)
    top_scores = scores[top_indices]
    return top_indices, top_scores


def find_similar_complaints(
    query_text: Union[str, Any],
    vectorizer: Any,
    corpus_matrix: Any,
    df_corpus: pd.DataFrame,
    top_k: int = 5,
    preprocess: bool = True
) -> pd.DataFrame:
    """Find the top-k most similar complaints in the corpus given a query complaint.

    Executes the end-to-end similarity retrieval pipeline:
    Query -> [Preprocessing] -> TF-IDF Vectorization -> Cosine Similarity -> Ranked DataFrame.

    Parameters
    ----------
    query_text : str | Any
        Raw input complaint narrative, preprocessed text, or pre-vectorized query.
    vectorizer : Any
        Fitted TfidfVectorizer instance.
    corpus_matrix : Any
        Sparse TF-IDF matrix (NxD) representing indexed corpus complaints.
    df_corpus : pd.DataFrame
        DataFrame containing original complaints and metadata (N rows).
    top_k : int, default=5
        Number of top matches to retrieve.
    preprocess : bool, default=True
        Whether to apply src.preprocessing.preprocess_text to query_text if a string.

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame containing columns:
        - rank (1 to k)
        - complaint_id
        - similarity_score
        - category (if present in df_corpus)
        - complaint_text
        - text

    Raises
    ------
    TypeError
        If required arguments are None.
    ValueError
        If query is empty, corpus is empty, or dimensions mismatch.
    """
    if query_text is None:
        raise TypeError("query_text cannot be None.")
    if corpus_matrix is None:
        raise TypeError("corpus_matrix cannot be None.")
    if df_corpus is None:
        raise TypeError("df_corpus cannot be None.")
    if not isinstance(df_corpus, pd.DataFrame):
        raise TypeError(f"df_corpus must be a pandas DataFrame, got {type(df_corpus).__name__}.")

    if df_corpus.empty or corpus_matrix.shape[0] == 0:
        raise ValueError("Corpus is empty. Cannot retrieve similar complaints.")

    if len(df_corpus) != corpus_matrix.shape[0]:
        raise ValueError(
            f"Corpus size mismatch: df_corpus has {len(df_corpus)} rows, "
            f"but corpus_matrix has {corpus_matrix.shape[0]} rows."
        )

    # Transform query to TF-IDF vector
    if isinstance(query_text, str):
        if not query_text.strip():
            raise ValueError("query_text cannot be empty.")
        processed_query = preprocess_text(query_text) if preprocess else query_text
        if vectorizer is None:
            raise ValueError("vectorizer is required when query_text is a string.")
        query_vector = vectorizer.transform([processed_query])
    else:
        # Query is already a vector or matrix
        query_vector = query_text

    # Compute top-k indices and scores
    top_indices, top_scores = get_top_k_similar(query_vector, corpus_matrix, top_k=top_k)

    # Extract matched rows from df_corpus
    matched_df = df_corpus.iloc[top_indices].copy()

    # Identify complaint_id
    if "complaint_id" in matched_df.columns:
        complaint_ids = matched_df["complaint_id"].tolist()
    elif "Complaint ID" in matched_df.columns:
        complaint_ids = matched_df["Complaint ID"].tolist()
    else:
        complaint_ids = matched_df.index.tolist()

    # Identify complaint text
    if "complaint_text" in matched_df.columns:
        texts = matched_df["complaint_text"].tolist()
    elif "Consumer Complaint" in matched_df.columns:
        texts = matched_df["Consumer Complaint"].tolist()
    elif "text" in matched_df.columns:
        texts = matched_df["text"].tolist()
    else:
        texts = matched_df.iloc[:, 0].astype(str).tolist()

    # Identify category if present
    category_col = None
    if "category" in matched_df.columns:
        category_col = matched_df["category"].tolist()
    elif "Product" in matched_df.columns:
        category_col = matched_df["Product"].tolist()

    data = {
        "rank": list(range(1, len(top_indices) + 1)),
        "complaint_id": complaint_ids,
        "similarity_score": top_scores.astype(float),
    }

    if category_col is not None:
        data["category"] = category_col

    data["complaint_text"] = texts
    # Provide alias 'text' for compatibility
    data["text"] = texts

    return pd.DataFrame(data)
