"""Text preprocessing module for consumer complaint narratives.

Provides functions for text normalization, cleaning, tokenization,
and stopword removal using classical NLP techniques.
"""

from __future__ import annotations

import re
from typing import Iterable, Set

# Standard English stop words collection (built-in baseline)
STANDARD_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}


def clean_text(text: str | None) -> str:
    """Clean raw complaint narrative text by removing URLs, special characters, and extra spaces.

    Parameters
    ----------
    text : str | None
        The input narrative text.

    Returns
    -------
    str
        Lowercased, sanitized string containing only alphanumeric tokens and single spaces.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    cleaned = text.lower()

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove email addresses
    cleaned = re.sub(r"\S+@\S+", " ", cleaned)

    # Remove CFPB redaction placeholders (e.g., 'xxxx', 'xx/xx/xxxx')
    cleaned = re.sub(r"x{2,}", " ", cleaned)

    # Remove punctuation, symbols, and non-alphanumeric characters (keep words and numbers)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)

    # Collapse repeated whitespace to a single space and strip boundaries
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def tokenize(text: str | None) -> list[str]:
    """Split cleaned text into a sequence of alphanumeric word tokens.

    Parameters
    ----------
    text : str | None
        The text string to tokenize.

    Returns
    -------
    list[str]
        List of individual word tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    cleaned = clean_text(text)
    if not cleaned:
        return []

    return cleaned.split()


def remove_stopwords(
    tokens: Iterable[str],
    custom_stopwords: Set[str] | None = None
) -> list[str]:
    """Filter out common English stop words from a token sequence.

    Parameters
    ----------
    tokens : Iterable[str]
        Input iterable of word tokens.
    custom_stopwords : Set[str] | None, optional
        Custom stop words set to override or extend the default list.

    Returns
    -------
    list[str]
        Filtered list of tokens with stop words excluded.
    """
    stopwords = custom_stopwords if custom_stopwords is not None else STANDARD_STOPWORDS
    return [token for token in tokens if token and token not in stopwords]


def preprocess_text(
    text: str | None,
    custom_stopwords: Set[str] | None = None
) -> str:
    """Execute complete end-to-end preprocessing pipeline on a single narrative string.

    Pipeline steps:
    1. Lowercase normalization & noise removal (clean_text)
    2. Tokenization (tokenize)
    3. Stopword elimination (remove_stopwords)
    4. Rejoining into a standardized space-delimited string

    Parameters
    ----------
    text : str | None
        The raw input narrative text.
    custom_stopwords : Set[str] | None, optional
        Custom set of stop words.

    Returns
    -------
    str
        Preprocessed, normalized text string ready for vectorization.
    """
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens, custom_stopwords=custom_stopwords)
    return " ".join(filtered)
