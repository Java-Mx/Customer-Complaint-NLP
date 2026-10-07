"""Text preprocessing module for consumer complaint narratives.

Provides classical NLP text normalization, cleaning, tokenization,
and stopword removal routines for financial complaint data.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set
import pandas as pd

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


def clean_text(text: str | None, mode: str = "standard") -> str:
    """Clean raw complaint narrative text using classical normalization steps.

    Design Decisions:
    - Standard Mode:
      - Lowercase normalization: Harmonizes casing variations.
      - URL & Email removal: Discards web and contact noise that does not inform
        financial categorization.
      - Punctuation & symbol stripping: Replaces non-alphanumeric symbols with spaces.
      - Numeric token retention: Numeric values (e.g., years like '2023', dollar sums
        like '50', percentages, and account digits) are intentionally preserved
        because they offer discriminative financial context.
      - CFPB Redaction removal: Strips privacy placeholders (e.g., 'xxxx', 'xx/xx/xxxx')
        using word-boundary matching (\\bx{2,}\\b).
      - Whitespace normalization: Collapses multi-spaces and strips boundaries.
    - Minimal Mode:
      - Lowercase normalization and whitespace collapsing only.
      - Retains punctuation, digits, stopwords, negation words ('not', 'never', 'n't'),
        redaction markers, and syntactic boundaries to preserve character n-gram context.

    Parameters
    ----------
    text : str | None
        Raw customer complaint narrative string.
    mode : str, default='standard'
        Cleaning mode: 'standard' or 'minimal'.

    Returns
    -------
    str
        Sanitized, lowercased string.

    Raises
    ------
    ValueError
        If mode is not 'standard' or 'minimal'.
    """
    if not isinstance(mode, str):
        raise ValueError(f"Cleaning mode must be a string, got {type(mode).__name__}.")
    mode_norm = mode.lower().strip()
    if mode_norm not in ("standard", "minimal"):
        raise ValueError(f"Unsupported cleaning mode '{mode}'. Choose 'standard' or 'minimal'.")

    if not isinstance(text, str) or not text.strip():
        return ""

    if mode_norm == "minimal":
        return re.sub(r"\s+", " ", text.lower()).strip()

    # 1. Lowercase normalization
    cleaned = text.lower()

    # 2. Remove web URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # 3. Remove email addresses
    cleaned = re.sub(r"\S+@\S+", " ", cleaned)

    # 4. Remove punctuation and non-alphanumeric symbols (preserve words and digits)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)

    # 5. Remove standalone CFPB privacy redaction masks (e.g. 'xx', 'xxxx')
    cleaned = re.sub(r"\bx{2,}\b", " ", cleaned)

    # 6. Collapse repeated whitespace and strip ends
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def tokenize(text: str | None) -> List[str]:
    """Split cleaned text into a sequence of alphanumeric word and numeric tokens.

    Parameters
    ----------
    text : str | None
        The text string to tokenize.

    Returns
    -------
    List[str]
        List of individual token strings.
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
) -> List[str]:
    """Filter out common stop words from a token sequence.

    Parameters
    ----------
    tokens : Iterable[str]
        Input iterable of word tokens.
    custom_stopwords : Set[str] | None, optional
        Custom stop words set to override or extend default stop words.

    Returns
    -------
    List[str]
        Filtered list of tokens with stop words excluded.
    """
    stopwords = custom_stopwords if custom_stopwords is not None else STANDARD_STOPWORDS
    return [token for token in tokens if token and token not in stopwords]


def preprocess_text(
    text: str | None,
    custom_stopwords: Set[str] | None = None,
    mode: str = "standard"
) -> str:
    """Execute complete end-to-end preprocessing pipeline on a single narrative string.

    Pipeline:
    - Standard: Raw complaint → clean_text → tokenize → remove_stopwords → standardized string.
    - Minimal: Raw complaint → clean_text(mode='minimal') → lowercase + whitespace-normalized string
      (preserves punctuation, digits, stopwords, negations, character boundary syntax).

    Parameters
    ----------
    text : str | None
        The raw input narrative text.
    custom_stopwords : Set[str] | None, optional
        Custom set of stop words (used in 'standard' mode).
    mode : str, default='standard'
        Preprocessing mode: 'standard' or 'minimal'.

    Returns
    -------
    str
        Preprocessed, normalized text string ready for vectorization.

    Raises
    ------
    ValueError
        If mode is not 'standard' or 'minimal'.
    """
    if isinstance(custom_stopwords, str) and mode == "standard":
        mode = custom_stopwords
        custom_stopwords = None

    if not isinstance(mode, str):
        raise ValueError(f"Preprocessing mode must be a string, got {type(mode).__name__}.")
    mode_norm = mode.lower().strip()
    if mode_norm not in ("standard", "minimal"):
        raise ValueError(f"Unsupported preprocessing mode '{mode}'. Choose 'standard' or 'minimal'.")

    if not isinstance(text, str) or not text.strip():
        return ""

    if mode_norm == "minimal":
        return clean_text(text, mode="minimal")

    tokens = tokenize(text)
    filtered = remove_stopwords(tokens, custom_stopwords=custom_stopwords)
    return " ".join(filtered)


def preprocess_series(
    series: pd.Series,
    custom_stopwords: Set[str] | None = None,
    mode: str = "standard"
) -> pd.Series:
    """Apply the text preprocessing pipeline across a pandas Series of complaint texts.

    Preserves the original Series index and does not mutate the source data.

    Parameters
    ----------
    series : pd.Series
        Pandas Series containing raw complaint text records.
    custom_stopwords : Set[str] | None, optional
        Custom stop words to filter out during preprocessing (standard mode).
    mode : str, default='standard'
        Preprocessing mode: 'standard' or 'minimal'.

    Returns
    -------
    pd.Series
        New Series containing cleaned and preprocessed narrative strings.

    Raises
    ------
    TypeError
        If input is not a pandas Series.
    ValueError
        If mode is not 'standard' or 'minimal'.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pandas Series, got {type(series).__name__}")

    if isinstance(custom_stopwords, str) and mode == "standard":
        mode = custom_stopwords
        custom_stopwords = None

    if not isinstance(mode, str):
        raise ValueError(f"Preprocessing mode must be a string, got {type(mode).__name__}.")
    mode_norm = mode.lower().strip()
    if mode_norm not in ("standard", "minimal"):
        raise ValueError(f"Unsupported preprocessing mode '{mode}'. Choose 'standard' or 'minimal'.")

    return series.fillna("").astype(str).apply(
        lambda val: preprocess_text(val, custom_stopwords=custom_stopwords, mode=mode_norm)
    )
