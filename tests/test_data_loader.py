"""Unit tests for dataset loading, validation, and schema detection."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    load_dataset,
    resolve_columns,
    validate_columns,
    get_dataset_summary,
)


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing a mock CFPB DataFrame with valid columns and rows."""
    return pd.DataFrame({
        "Complaint ID": [101, 102, 103, 104],
        "Product": ["Credit card", "Mortgage", "Debt collection", "Student loan"],
        "Consumer Complaint": [
            "Unauthorized charge of $45 on my statement.",
            "Escrow account calculation error by loan servicer.",
            "Collector calling at workplace after verbal cease notice.",
            "Loan balance did not update after direct debit payment.",
        ],
        "Issue": ["Billing dispute", "Escrow calculation", "Harassment", "Payment processing"],
    })


@pytest.fixture
def sample_alternative_column_df() -> pd.DataFrame:
    """Fixture providing a mock DataFrame with alternative CFPB column names."""
    return pd.DataFrame({
        "complaint_id": [201, 202],
        "product": ["Bank account or service", "Consumer Loan"],
        "Consumer complaint narrative": [
            "Overdraft fee charged during deposit hold.",
            "Vehicle repossession without proper notification window.",
        ],
    })


class TestColumnResolutionAndValidation:
    """Tests for column resolution and schema validation."""

    def test_standard_column_resolution(self, sample_valid_df):
        mapping = resolve_columns(sample_valid_df)
        assert mapping["text"] == "Consumer Complaint"
        assert mapping["category"] == "Product"
        assert mapping["id"] == "Complaint ID"

    def test_alternative_column_resolution(self, sample_alternative_column_df):
        mapping = resolve_columns(sample_alternative_column_df)
        assert mapping["text"] == "Consumer complaint narrative"
        assert mapping["category"] == "product"
        assert mapping["id"] == "complaint_id"

    def test_missing_text_column_raises_error(self):
        invalid_df = pd.DataFrame({
            "Product": ["Credit card"],
            "Complaint ID": [101],
            "Issue": ["Billing dispute"]
        })
        with pytest.raises(ValueError, match="Missing required complaint text column"):
            validate_columns(invalid_df)

    def test_missing_category_column_raises_error(self):
        invalid_df = pd.DataFrame({
            "Consumer Complaint": ["Some complaint text"],
            "Complaint ID": [101]
        })
        with pytest.raises(ValueError, match="Missing required category/product column"):
            validate_columns(invalid_df)

    def test_empty_dataframe_raises_error(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_columns(empty_df)


class TestDatasetSummary:
    """Tests for dataset summary generation."""

    def test_summary_metrics(self, sample_valid_df):
        summary = get_dataset_summary(sample_valid_df)
        assert summary["num_rows"] == 4
        assert summary["num_columns"] == 4
        assert summary["text_column"] == "Consumer Complaint"
        assert summary["category_column"] == "Product"
        assert summary["id_column"] == "Complaint ID"
        assert summary["missing_text_count"] == 0
        assert summary["missing_category_count"] == 0
        assert summary["num_unique_categories"] == 4


class TestDataLoaderIntegration:
    """Integration tests for dataset loading with temporary CSV and actual dataset."""

    def test_missing_file_raises_not_found(self):
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("data/non_existent_file.csv")

    def test_load_from_temp_csv(self, tmp_path, sample_valid_df):
        temp_csv = tmp_path / "test_complaints.csv"
        sample_valid_df.to_csv(temp_csv, index=False)

        df_loaded = load_dataset(temp_csv, standardize_columns=True)
        assert not df_loaded.empty
        assert len(df_loaded) == 4
        assert "complaint_text" in df_loaded.columns
        assert "product_category" in df_loaded.columns
        assert "complaint_id" in df_loaded.columns
        assert df_loaded["complaint_text"].iloc[0] == "Unauthorized charge of $45 on my statement."

    def test_empty_narrative_filtering(self, tmp_path):
        df_with_blanks = pd.DataFrame({
            "Product": ["Mortgage", "Credit card", "Mortgage"],
            "Consumer Complaint": ["Valid complaint text", "", "   "],
            "Complaint ID": [1, 2, 3]
        })
        temp_csv = tmp_path / "blanks.csv"
        df_with_blanks.to_csv(temp_csv, index=False)

        df_loaded = load_dataset(temp_csv, drop_empty=True)
        assert len(df_loaded) == 1
        assert df_loaded["complaint_text"].iloc[0] == "Valid complaint text"

    def test_real_dataset_sample_load(self):
        """Verify loading the local CFPB dataset slice if available."""
        dataset_path = Path("data/complaints.csv")
        if dataset_path.exists():
            df = load_dataset(dataset_path, nrows=50, standardize_columns=True)
            assert not df.empty
            assert len(df) == 50
            assert "complaint_text" in df.columns
            assert "product_category" in df.columns
            assert df["complaint_text"].str.len().min() > 0
            assert df["product_category"].nunique() > 0
