"""TF-IDF vectorization module for consumer complaint narratives.

Transforms preprocessed text into term frequency-inverse document frequency
feature matrices using classical Scikit-learn sparse vector representations.

Architectural Design:
- ngram_range=(1, 2): Captures both unigrams and bigrams. Unigrams encode
  individual domain terms (e.g., 'dispute', 'mortgage', 'fraud'), while bigrams
  capture compound financial phrases (e.g., 'credit card', 'late payment',
  'loan modification', 'identity theft') that carry crucial semantic signals
  for complaint categorization and similarity matching.
- lowercase=False: Lowercase normalization is already performed upstream in
  src.preprocessing. Disabling redundant lowercasing here preserves efficiency.
- sublinear_tf=True: Replaces term frequency (TF) with 1 + log(TF), dampening the
  influence of repetitive word occurrences within lengthy complaint descriptions.
- Sparse Matrices: Feature matrices are strictly retained as scipy.sparse.csr_matrix
  structures to ensure minimal memory footprint on corpora containing 25,000+ documents.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse, spmatrix
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted


def create_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    max_df: Union[int, float] = 0.95,
    sublinear_tf: bool = True,
    lowercase: bool = False,
    max_features: Optional[int] = None,
    **kwargs: Any
) -> TfidfVectorizer:
    """Initialize and configure a classical TfidfVectorizer instance.

    Parameters
    ----------
    ngram_range : Tuple[int, int], default=(1, 2)
        Range of n-values for extracted n-grams. Defaults to unigrams + bigrams.
    min_df : int | float, default=2
        Minimum document frequency threshold. Terms appearing in fewer documents
        are pruned as corpus noise.
    max_df : int | float, default=0.95
        Maximum document frequency threshold. Terms appearing in more than 95%
        of documents are pruned as corpus-specific stop words.
    sublinear_tf : bool, default=True
        Applies sublinear scaling (1 + log(tf)) to damp the effect of repeated terms.
    lowercase : bool, default=False
        Set to False because text normalization is handled in src.preprocessing.
    max_features : int | None, optional
        Maximum vocabulary size to retain ordered by term frequency.
    **kwargs : Any
        Additional keyword arguments forwarded to Scikit-learn's TfidfVectorizer.

    Returns
    -------
    TfidfVectorizer
        Configured un-fitted vectorizer instance.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=lowercase,
        max_features=max_features,
        **kwargs
    )


# Alias preserving earlier scaffold function name
build_tfidf_vectorizer = create_vectorizer


def _validate_input_documents(texts: Any) -> Sequence[str]:
    """Validate and convert input document collections to a suitable sequence of strings.

    Parameters
    ----------
    texts : Any
        Input texts (list, tuple, pd.Series, or single string).

    Returns
    -------
    Sequence[str]
        Sequence of text strings.

    Raises
    ------
    TypeError
        If texts is None or not an iterable.
    ValueError
        If texts collection is empty.
    """
    if texts is None:
        raise TypeError("Input document collection cannot be None.")

    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (pd.Series, np.ndarray)):
        texts = texts.tolist()
    elif not isinstance(texts, Iterable):
        raise TypeError(f"Expected iterable of strings, got {type(texts).__name__}.")
    else:
        texts = list(texts)

    if len(texts) == 0:
        raise ValueError("Document collection is empty. Cannot fit or transform TF-IDF vectorizer.")

    # Ensure all elements are strings or convert
    return [str(t) if t is not None else "" for t in texts]


def fit_tfidf(
    vectorizer: TfidfVectorizer,
    texts: Iterable[str]
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a collection of preprocessed training documents.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Configured TfidfVectorizer instance.
    texts : Iterable[str]
        Collection of preprocessed complaint text strings.

    Returns
    -------
    TfidfVectorizer
        Fitted vectorizer with established vocabulary.

    Raises
    ------
    TypeError
        If vectorizer or texts are of invalid type.
    ValueError
        If texts collection is empty.
    """
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Expected TfidfVectorizer, got {type(vectorizer).__name__}.")

    validated_texts = _validate_input_documents(texts)
    vectorizer.fit(validated_texts)
    return vectorizer


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts: Iterable[str]
) -> spmatrix:
    """Transform preprocessed documents into a sparse TF-IDF feature matrix.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TfidfVectorizer instance.
    texts : Iterable[str]
        Collection of preprocessed text strings to transform.

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse feature matrix in CSR format (num_documents x vocabulary_size).

    Raises
    ------
    NotFittedError
        If the vectorizer has not been fitted prior to transform.
    TypeError
        If vectorizer or texts are of invalid type.
    ValueError
        If texts collection is empty.
    """
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Expected TfidfVectorizer, got {type(vectorizer).__name__}.")

    # Validate that vectorizer is already fitted
    try:
        check_is_fitted(vectorizer)
    except NotFittedError:
        raise NotFittedError(
            "The TfidfVectorizer is not fitted. Call fit_tfidf() or fit_transform_tfidf() before transforming."
        )

    validated_texts = _validate_input_documents(texts)
    return vectorizer.transform(validated_texts)


def fit_transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts: Iterable[str]
) -> Tuple[TfidfVectorizer, spmatrix]:
    """Fit a TF-IDF vectorizer and transform documents into a sparse matrix in a single step.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Configured TfidfVectorizer instance.
    texts : Iterable[str]
        Collection of preprocessed complaint texts.

    Returns
    -------
    Tuple[TfidfVectorizer, scipy.sparse.spmatrix]
        (fitted_vectorizer, sparse_tfidf_matrix)
    """
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Expected TfidfVectorizer, got {type(vectorizer).__name__}.")

    validated_texts = _validate_input_documents(texts)
    matrix = vectorizer.fit_transform(validated_texts)
    return vectorizer, matrix


def fit_transform_corpus(
    corpus: Iterable[str],
    vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[TfidfVectorizer, spmatrix]:
    """Convenience helper to fit and transform a corpus with a default or provided vectorizer.

    Parameters
    ----------
    corpus : Iterable[str]
        Collection of preprocessed narrative strings.
    vectorizer : Optional[TfidfVectorizer], optional
        Pre-configured vectorizer. If None, create_vectorizer() is called.

    Returns
    -------
    Tuple[TfidfVectorizer, scipy.sparse.spmatrix]
        (fitted_vectorizer, sparse_tfidf_matrix)
    """
    vec = vectorizer if vectorizer is not None else create_vectorizer()
    return fit_transform_tfidf(vec, corpus)


def create_char_vectorizer(
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: Union[int, float] = 2,
    max_df: Union[int, float] = 0.95,
    sublinear_tf: bool = True,
    lowercase: bool = False,
    max_features: Optional[int] = None,
    analyzer: str = "char_wb",
    **kwargs: Any
) -> TfidfVectorizer:
    """Initialize and configure a character-level TfidfVectorizer instance.

    Character n-grams within word boundaries ('char_wb') capture subword roots,
    prefixes, suffixes, financial acronyms, typos, and terminology variations.

    Parameters
    ----------
    ngram_range : Tuple[int, int], default=(3, 5)
        Character n-gram range (e.g. 3 to 5 characters).
    min_df : int | float, default=2
        Minimum document frequency threshold.
    max_df : int | float, default=0.95
        Maximum document frequency threshold.
    sublinear_tf : bool, default=True
        Apply sublinear term frequency scaling (1 + log(tf)).
    lowercase : bool, default=False
        Assumes text is already lowercased in preprocessing.
    max_features : int | None, optional
        Maximum features to retain.
    analyzer : str, default='char_wb'
        Feature analyzer type. 'char_wb' creates character n-grams from text
        within word boundaries.
    **kwargs : Any
        Additional keyword arguments forwarded to TfidfVectorizer.

    Returns
    -------
    TfidfVectorizer
        Configured un-fitted character vectorizer instance.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=lowercase,
        max_features=max_features,
        analyzer=analyzer,
        **kwargs
    )


def combine_sparse_matrices(
    X1: spmatrix,
    X2: spmatrix
) -> spmatrix:
    """Horizontally stack two sparse feature matrices into a single sparse CSR matrix.

    Ensures zero dense memory conversion, maintaining strict scalability.

    Parameters
    ----------
    X1 : spmatrix
        First sparse matrix of shape (N, D1).
    X2 : spmatrix
        Second sparse matrix of shape (N, D2).

    Returns
    -------
    scipy.sparse.spmatrix
        Combined sparse matrix of shape (N, D1 + D2) in CSR format.

    Raises
    ------
    TypeError
        If X1 or X2 are not sparse matrices.
    ValueError
        If row dimensions do not match.
    """
    if not issparse(X1) or not issparse(X2):
        raise TypeError(
            f"Both inputs must be scipy sparse matrices. Got {type(X1).__name__} and {type(X2).__name__}."
        )
    if X1.shape[0] != X2.shape[0]:
        raise ValueError(
            f"Row count mismatch: X1 has {X1.shape[0]} rows, X2 has {X2.shape[0]} rows."
        )

    return hstack([X1, X2], format="csr")


def fit_transform_word_char(
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    texts: Iterable[str]
) -> Tuple[TfidfVectorizer, TfidfVectorizer, spmatrix]:
    """Fit both word and character vectorizers and return combined sparse CSR matrix.

    Parameters
    ----------
    word_vec : TfidfVectorizer
        Un-fitted word-level TfidfVectorizer.
    char_vec : TfidfVectorizer
        Un-fitted character-level TfidfVectorizer.
    texts : Iterable[str]
        Collection of preprocessed text strings.

    Returns
    -------
    Tuple[TfidfVectorizer, TfidfVectorizer, scipy.sparse.spmatrix]
        (fitted_word_vec, fitted_char_vec, combined_sparse_matrix)
    """
    validated_texts = _validate_input_documents(texts)
    X_word = word_vec.fit_transform(validated_texts)
    X_char = char_vec.fit_transform(validated_texts)
    X_comb = combine_sparse_matrices(X_word, X_char)
    return word_vec, char_vec, X_comb


def transform_word_char(
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    texts: Iterable[str]
) -> spmatrix:
    """Transform documents using fitted word and character vectorizers into a combined sparse CSR matrix.

    Parameters
    ----------
    word_vec : TfidfVectorizer
        Fitted word-level TfidfVectorizer.
    char_vec : TfidfVectorizer
        Fitted character-level TfidfVectorizer.
    texts : Iterable[str]
        Collection of preprocessed text strings.

    Returns
    -------
    scipy.sparse.spmatrix
        Combined sparse CSR feature matrix of shape (N, D_word + D_char).
    """
    check_is_fitted(word_vec)
    check_is_fitted(char_vec)
    validated_texts = _validate_input_documents(texts)
    X_word = word_vec.transform(validated_texts)
    X_char = char_vec.transform(validated_texts)
    return combine_sparse_matrices(X_word, X_char)
