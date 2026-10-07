"""Unit tests for the text preprocessing module."""

import pandas as pd
import pytest
from src.preprocessing import (
    clean_text,
    tokenize,
    remove_stopwords,
    preprocess_text,
    preprocess_series,
    STANDARD_STOPWORDS,
)


class TestCleanText:
    """Tests for the clean_text normalization function."""

    def test_lowercasing(self):
        text = "UNEXPECTED Charges on Credit Card!"
        result = clean_text(text)
        assert result == "unexpected charges on credit card"

    def test_url_removal(self):
        text = "Please check https://bank.example.com/dispute or www.example.org for details"
        result = clean_text(text)
        assert "http" not in result
        assert "www" not in result
        assert "please check or for details" in result

    def test_email_removal(self):
        text = "Contact me at consumer_support@bank.com immediately."
        result = clean_text(text)
        assert "@" not in result
        assert "consumer_support" not in result
        assert result == "contact me at immediately"

    def test_cfpb_redaction_removal(self):
        text = "I called customer care on XX/XX/XXXX and spoke with agent XXXX at Exxon."
        result = clean_text(text)
        assert "xxxx" not in result
        # 'Exxon' has 'xx' inside the word and should be preserved
        assert "exxon" in result
        assert result == "i called customer care on and spoke with agent at exxon"

    def test_punctuation_and_symbols_removal(self):
        text = "Overdraft fee: $35.00!! Account #12345 -- why was this applied???"
        result = clean_text(text)
        assert "$" not in result
        assert "!" not in result
        assert "?" not in result
        assert "#" not in result
        assert "overdraft fee 35 00 account 12345 why was this applied" == result

    def test_whitespace_normalization(self):
        text = "   too    many   \n\t  spaces   in    here   "
        result = clean_text(text)
        assert result == "too many spaces in here"

    def test_empty_string(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_none_and_non_string_input(self):
        assert clean_text(None) == ""
        assert clean_text(12345) == ""
        assert clean_text(float("nan")) == ""

    def test_numeric_token_retention(self):
        text = "Charged $500 in 2022 on account 987654 at 5% interest rate."
        cleaned = clean_text(text)
        # Verify numbers are preserved as informative tokens
        assert "500" in cleaned
        assert "2022" in cleaned
        assert "987654" in cleaned
        assert "5" in cleaned


class TestTokenize:
    """Tests for the tokenize function."""

    def test_basic_tokenization(self):
        text = "Debt collection agency called repeatedly in 2023"
        tokens = tokenize(text)
        assert tokens == ["debt", "collection", "agency", "called", "repeatedly", "in", "2023"]

    def test_empty_and_none_tokenization(self):
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize(None) == []


class TestRemoveStopwords:
    """Tests for the remove_stopwords function."""

    def test_standard_stopwords_filtering(self):
        tokens = ["this", "is", "a", "fraudulent", "charge", "on", "my", "card"]
        filtered = remove_stopwords(tokens)
        assert filtered == ["fraudulent", "charge", "card"]

    def test_custom_stopwords(self):
        tokens = ["unauthorized", "fee", "charged", "bank"]
        custom_stops = {"fee", "bank"}
        filtered = remove_stopwords(tokens, custom_stopwords=custom_stops)
        assert filtered == ["unauthorized", "charged"]

    def test_empty_token_list(self):
        assert remove_stopwords([]) == []


class TestPreprocessText:
    """Tests for complete end-to-end preprocess_text pipeline."""

    def test_full_pipeline(self):
        raw_narrative = (
            "I noticed an UNKNOWN fee of $50.00 on XX/XX/2023 from https://fraud.com! "
            "Please investigate this issue."
        )
        preprocessed = preprocess_text(raw_narrative)

        # Stop words like 'i', 'an', 'of', 'on', 'from', 'this' should be removed
        # Special characters, URLs, redactions removed
        assert "unknown" in preprocessed
        assert "fee" in preprocessed
        assert "investigate" in preprocessed
        assert "issue" in preprocessed
        assert "https" not in preprocessed
        assert "$" not in preprocessed
        assert "xxxx" not in preprocessed
        # Numbers preserved
        assert "50" in preprocessed
        assert "2023" in preprocessed

    def test_empty_and_none_handling(self):
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""
        assert preprocess_text(None) == ""


class TestPreprocessSeries:
    """Tests for pandas Series preprocessing helper."""

    def test_series_preprocessing(self):
        series = pd.Series([
            "Late fee of $25 on credit card.",
            "Mortgage escrow error in 2021.",
            None,
            "   "
        ], index=["c1", "c2", "c3", "c4"])

        processed = preprocess_series(series)

        assert isinstance(processed, pd.Series)
        assert list(processed.index) == ["c1", "c2", "c3", "c4"]
        assert processed["c1"] == "late fee 25 credit card"
        assert processed["c2"] == "mortgage escrow error 2021"
        assert processed["c3"] == ""
        assert processed["c4"] == ""

    def test_series_non_mutation(self):
        original = pd.Series(["Original raw text with $100 fee."])
        copy_original = original.copy()
        _ = preprocess_series(original)
        pd.testing.assert_series_equal(original, copy_original)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected pandas Series"):
            preprocess_series(["not", "a", "series"])


class TestMinimalPreprocessing:
    """Unit tests for minimal preprocessing mode (retains syntax, punctuation, stopwords, and negation)."""

    def test_minimal_lowercasing_and_whitespace(self):
        text = "  UNEXPECTED \n Charges \t on  Account!  "
        result = clean_text(text, mode="minimal")
        assert result == "unexpected charges on account!"

    def test_minimal_retains_punctuation_and_symbols(self):
        text = "Overdraft fee: $35.00!! Account #12345 -- why was this applied???"
        result = preprocess_text(text, mode="minimal")
        assert ":" in result
        assert "$35.00!!" in result
        assert "#12345" in result
        assert "--" in result
        assert "???" in result
        assert result == "overdraft fee: $35.00!! account #12345 -- why was this applied???"

    def test_minimal_retains_digits(self):
        text = "Charged $500 in 2022 on account 987654 at 5% rate."
        result = preprocess_text(text, mode="minimal")
        assert "500" in result
        assert "2022" in result
        assert "987654" in result
        assert "5%" in result

    def test_minimal_retains_stopwords_and_negations(self):
        text = "I did not receive the statement and I wasn't notified about this fee."
        result = preprocess_text(text, mode="minimal")
        # In standard mode, 'i', 'did', 'the', 'and', 'about', 'this' are stripped.
        # In minimal mode, all words and negations are preserved.
        assert "i did not receive the statement and i wasn't notified about this fee." == result

    def test_minimal_retains_redaction_masks(self):
        text = "Contacted XXXX on XX/XX/2023 regarding account XXXX."
        result = preprocess_text(text, mode="minimal")
        assert "xxxx" in result
        assert "xx/xx/2023" in result

    def test_empty_and_none_minimal(self):
        assert preprocess_text("", mode="minimal") == ""
        assert preprocess_text("   ", mode="minimal") == ""
        assert preprocess_text(None, mode="minimal") == ""
        assert clean_text(None, mode="minimal") == ""

    def test_mode_parameter_standard_vs_minimal(self):
        sample = "I did NOT get a refund of $50 on my credit card!"
        std = preprocess_text(sample, mode="standard")
        min_p = preprocess_text(sample, mode="minimal")

        # Standard strips stopwords and punctuation
        assert "get refund 50 credit card" == std
        # Minimal preserves stopwords, punctuation, negation
        assert "i did not get a refund of $50 on my credit card!" == min_p

    def test_preprocess_series_minimal(self):
        series = pd.Series([
            "Late fee of $25 on card!",
            "Did not receive notice.",
            None,
        ], index=["c1", "c2", "c3"])

        processed = preprocess_series(series, mode="minimal")
        assert processed["c1"] == "late fee of $25 on card!"
        assert processed["c2"] == "did not receive notice."
        assert processed["c3"] == ""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported.*mode"):
            clean_text("test", mode="unsupported")
        with pytest.raises(ValueError, match="Unsupported.*mode"):
            preprocess_text("test", mode="unsupported")
        with pytest.raises(ValueError, match="Unsupported.*mode"):
            preprocess_series(pd.Series(["test"]), mode="unsupported")
