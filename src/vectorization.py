"""TF-IDF vectorization module for complaint narratives.

Transforms preprocessed complaint text into term frequency-inverse document
frequency feature matrices using classical Scikit-learn vectorizers.
"""

from __future__ import annotations

from typing import Any, Tuple
import pandas as pd


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.85
) -> Any:
    """Initialize and configure a TfidfVectorizer instance.

    Parameters
    ----------
    max_features : int, default=5000
        Maximum vocabulary size to retain based on term frequency across corpus.
    ngram_range : Tuple[int, int], default=(1, 2)
        Lower and upper boundary of range of n-values for different n-grams.
    min_df : int, default=2
        Minimum number of documents a word must appear in.
    max_df : float, default=0.85
        Ignore terms that appear in more than this proportion of documents.

    Returns
    -------
    TfidfVectorizer
        Configured un-fitted vectorizer instance.
    """
    raise NotImplementedError(
        "TF-IDF vectorizer configuration will be implemented in the vectorization milestone."
    )


def fit_transform_corpus(
    corpus: pd.Series | list[str],
    vectorizer: Any | None = None
) -> Tuple[Any, Any]:
    """Fit a vectorizer on the input text corpus and transform it into a TF-IDF matrix.

    Parameters
    ----------
    corpus : pd.Series | list[str]
        Iterable of preprocessed narrative strings.
    vectorizer : Any | None, optional
        Pre-configured vectorizer instance. If None, a default one will be created.

    Returns
    -------
    Tuple[Any, Any]
        (fitted_vectorizer, tfidf_matrix)
    """
    raise NotImplementedError(
        "Corpus vectorization fitting will be implemented in the vectorization milestone."
    )
