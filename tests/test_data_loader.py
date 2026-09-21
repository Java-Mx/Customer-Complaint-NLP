"""Unit tests for dataset loading, schema validation, and column resolution."""

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
    """Fixture providing a mock CFPB DataFrame with standard columns."""
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
def sample_narrative_variant_df() -> pd.DataFrame:
    """Fixture providing a mock DataFrame with 'Consumer complaint narrative' naming."""
    return pd.DataFrame({
        "Complaint ID": [201, 202],
        "Product": ["Bank account or service", "Consumer Loan"],
        "Consumer complaint narrative": [
            "Overdraft fee charged during deposit hold.",
            "Vehicle repossession without proper notification window.",
        ],
    })


@pytest.fixture
def sample_api_variant_df() -> pd.DataFrame:
    """Fixture providing a mock DataFrame with API naming ('complaint_what_happened')."""
    return pd.DataFrame({
        "complaint_id": [301, 302],
        "product": ["Credit reporting", "Debt collection"],
        "complaint_what_happened": [
            "Inaccurate late payment reported on credit file.",
            "Unknown medical debt sent to collections without invoice.",
        ],
    })


class TestColumnResolution:
    """Tests for resolving column names across CFPB naming variants."""

    def test_standard_columns_resolution(self, sample_valid_df):
        mapping = resolve_columns(sample_valid_df)
        assert mapping["text"] == "Consumer Complaint"
        assert mapping["category"] == "Product"
        assert mapping["id"] == "Complaint ID"

    def test_narrative_variant_resolution(self, sample_narrative_variant_df):
        mapping = resolve_columns(sample_narrative_variant_df)
        assert mapping["text"] == "Consumer complaint narrative"
        assert mapping["category"] == "Product"
        assert mapping["id"] == "Complaint ID"

    def test_api_variant_resolution(self, sample_api_variant_df):
        mapping = resolve_columns(sample_api_variant_df)
        assert mapping["text"] == "complaint_what_happened"
        assert mapping["category"] == "product"
        assert mapping["id"] == "complaint_id"

    def test_missing_text_column_raises_error(self):
        invalid_df = pd.DataFrame({
            "Product": ["Credit card"],
            "Complaint ID": [101],
            "Issue": ["Billing dispute"]
        })
        with pytest.raises(ValueError, match="Missing required complaint text column"):
            resolve_columns(invalid_df)

    def test_missing_category_column_raises_error(self):
        invalid_df = pd.DataFrame({
            "Consumer Complaint": ["Unauthorized charge"],
            "Complaint ID": [101]
        })
        with pytest.raises(ValueError, match="Missing required category/product column"):
            resolve_columns(invalid_df)

    def test_empty_dataframe_raises_error(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_columns(empty_df)


class TestDatasetSummaryAndValidation:
    """Tests for dataset summary and missing-data auditing."""

    def test_summary_clean_dataset(self, sample_valid_df):
        summary = get_dataset_summary(sample_valid_df)
        assert summary["total_rows"] == 4
        assert summary["total_columns"] == 4
        assert summary["text_column"] == "Consumer Complaint"
        assert summary["category_column"] == "Product"
        assert summary["id_column"] == "Complaint ID"
        assert summary["missing_text_count"] == 0
        assert summary["missing_category_count"] == 0
        assert summary["usable_rows_count"] == 4
        assert summary["num_unique_categories"] == 4

    def test_summary_with_missing_and_blank_records(self):
        df_with_missing = pd.DataFrame({
            "Complaint ID": [1, 2, 3, 4],
            "Product": ["Credit card", None, "Mortgage", "Student loan"],
            "Consumer Complaint": ["Valid complaint", "Another valid", "", "   "],
        })
        summary = get_dataset_summary(df_with_missing)
        assert summary["total_rows"] == 4
        assert summary["missing_category_count"] == 1
        assert summary["missing_text_count"] == 2  # 2 empty string narratives
        assert summary["usable_rows_count"] == 1   # only record 1 is complete and non-empty


class TestDataLoaderLoading:
    """Tests for load_dataset loading, path handling, and column standardization."""

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("data/non_existent_dummy_file.csv")

    def test_standardized_fields_and_original_preservation(self, tmp_path, sample_valid_df):
        test_csv = tmp_path / "temp_complaints.csv"
        sample_valid_df.to_csv(test_csv, index=False)

        df_loaded = load_dataset(test_csv, standardize_columns=True)
        assert not df_loaded.empty
        assert len(df_loaded) == 4

        # Verify standardized canonical fields exist
        assert "text" in df_loaded.columns
        assert "category" in df_loaded.columns
        assert "complaint_id" in df_loaded.columns

        # Verify original columns are preserved
        assert "Consumer Complaint" in df_loaded.columns
        assert "Product" in df_loaded.columns
        assert "Complaint ID" in df_loaded.columns
        assert "Issue" in df_loaded.columns

        # Verify values align
        assert df_loaded["text"].iloc[0] == sample_valid_df["Consumer Complaint"].iloc[0]
        assert df_loaded["category"].iloc[0] == sample_valid_df["Product"].iloc[0]
        assert df_loaded["complaint_id"].iloc[0] == sample_valid_df["Complaint ID"].iloc[0]

    def test_drop_invalid_flag(self, tmp_path):
        df_mixed = pd.DataFrame({
            "Product": ["Mortgage", "Credit card", None],
            "Consumer Complaint": ["Valid narrative", "", "Another narrative"],
            "Complaint ID": [1, 2, 3]
        })
        test_csv = tmp_path / "mixed.csv"
        df_mixed.to_csv(test_csv, index=False)

        # Default drop_invalid=False keeps all rows
        df_retained = load_dataset(test_csv, drop_invalid=False)
        assert len(df_retained) == 3

        # drop_invalid=True filters invalid rows
        df_filtered = load_dataset(test_csv, drop_invalid=True)
        assert len(df_filtered) == 1
        assert df_filtered["text"].iloc[0] == "Valid narrative"

    def test_empty_csv_raises_value_error(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)
        with pytest.raises(ValueError):
            load_dataset(empty_csv)
