"""Unit and integration tests for post-audit classification improvements."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.preprocessing import clean_text, preprocess_text, preprocess_series

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_standard_preprocessing_backward_compatibility():
    """Verify that mode='standard' produces identical results to default calls."""
    sample = "I didn't receive my statement on XX/XX/2023 for account #12345 from http://bank.com!"
    res_default = preprocess_text(sample)
    res_explicit = preprocess_text(sample, mode="standard")
    assert res_default == res_explicit
    assert "xxxx" not in res_explicit
    assert "http" not in res_explicit
    assert "statement" in res_explicit


def test_minimal_preprocessing_preservation():
    """Verify that mode='minimal' retains punctuation, stopwords, digits, and negations."""
    sample = "I did not receive my statement on 05/12/2023 for account #12345. It wasn't fair!"
    res_min = preprocess_text(sample, mode="minimal")
    assert "did not receive" in res_min
    assert "wasn't fair!" in res_min
    assert "05/12/2023" in res_min
    assert "#12345." in res_min


def test_taxonomy_v1_conservative_mapping_completeness():
    """Verify that taxonomy_v1_conservative.json maps all 18 CFPB categories to 11 categories."""
    tax_path = ROOT_DIR / "config" / "taxonomy_v1_conservative.json"
    assert tax_path.exists()
    with open(tax_path, "r", encoding="utf-8") as f:
        tax_data = json.load(f)

    mapping = tax_data["mapping"]
    assert len(mapping) == 18
    norm_cats = {v["normalized_category"] for v in mapping.values()}
    assert len(norm_cats) == 11
    assert "Consumer Loan" in norm_cats


def test_model_improvement_comparison_csv_schema():
    """Verify results/model_improvement_comparison.csv schema if file exists."""
    csv_path = ROOT_DIR / "results" / "model_improvement_comparison.csv"
    if not csv_path.exists():
        pytest.skip("model_improvement_comparison.csv not yet generated.")

    df = pd.read_csv(csv_path)
    required_cols = [
        "experiment",
        "taxonomy",
        "preprocessing",
        "classifier",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "feature_count",
        "training_size",
        "validation_size",
        "test_size",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) >= 4


def test_taxonomy_model_comparison_csv_schema():
    """Verify results/taxonomy_model_comparison.csv schema if file exists."""
    csv_path = ROOT_DIR / "results" / "taxonomy_model_comparison.csv"
    if not csv_path.exists():
        pytest.skip("taxonomy_model_comparison.csv not yet generated.")

    df = pd.read_csv(csv_path)
    assert "model_formulation" in df.columns
    assert "val_accuracy" in df.columns
    assert "val_macro_f1" in df.columns
    assert len(df) >= 4


def test_hierarchical_comparison_csv_schema():
    """Verify results/hierarchical_comparison.csv schema if file exists."""
    csv_path = ROOT_DIR / "results" / "hierarchical_comparison.csv"
    if not csv_path.exists():
        pytest.skip("hierarchical_comparison.csv not yet generated.")

    df = pd.read_csv(csv_path)
    assert "architecture" in df.columns
    assert "val_accuracy" in df.columns
    assert "val_macro_f1" in df.columns
    assert len(df) >= 4
