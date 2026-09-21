"""Unit tests for the text preprocessing module."""

import pytest
from src.preprocessing import (
    clean_text,
    tokenize,
    remove_stopwords,
    preprocess_text,
    STANDARD_STOPWORDS,
)


class TestCleanText:
    """Tests for the clean_text function."""

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
        text = "I called customer care on XX/XX/XXXX and spoke with agent XXXX."
        result = clean_text(text)
        assert "xxxx" not in result
        assert result == "i called customer care on and spoke with agent"

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

    def test_empty_and_none_input(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text(None) == ""


class TestTokenize:
    """Tests for the tokenize function."""

    def test_basic_tokenization(self):
        text = "Debt collection agency called repeatedly"
        tokens = tokenize(text)
        assert tokens == ["debt", "collection", "agency", "called", "repeatedly"]

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
    """Tests for end-to-end preprocess_text pipeline."""

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

    def test_empty_and_none_handling(self):
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
