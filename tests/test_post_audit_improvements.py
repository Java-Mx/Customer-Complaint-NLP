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


def test_post_audit_error_analysis_json():
    """Verify results/post_audit_error_analysis.json structure and key metrics."""
    json_path = ROOT_DIR / "results" / "post_audit_error_analysis.json"
    if not json_path.exists():
        pytest.skip("post_audit_error_analysis.json not yet generated.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["validation_size"] == 4000
    assert data["baseline_total_errors"] == 1202
    assert data["improved_total_errors"] == 1147
    assert data["net_errors_eliminated"] == 55
    assert data["baseline_sibling_errors"] == 486
    assert data["improved_sibling_errors"] == 462
    assert len(data["top_10_confusion_pairs"]) == 10
    assert "confidence_statistics" in data
    assert "why_errors_remain" in data


def test_post_audit_test_metrics_json():
    """Verify results/post_audit_test_metrics.json structure and exact metrics."""
    json_path = ROOT_DIR / "results" / "post_audit_test_metrics.json"
    if not json_path.exists():
        pytest.skip("post_audit_test_metrics.json not yet generated.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["baseline_18_class_test"]["accuracy"] == 0.6982
    assert data["baseline_18_class_test"]["macro_f1"] == 0.5088
    assert round(data["improved_18_class_test"]["accuracy"], 4) == 0.7116
    assert round(data["improved_18_class_test"]["macro_f1"], 4) == 0.5243
    assert round(data["normalized_11_class_test"]["accuracy"], 4) == 0.8190
    assert round(data["normalized_11_class_test"]["macro_f1"], 4) == 0.6329
    assert data["deltas_18_class"]["accuracy_delta_pp"] == 1.34
    assert data["deltas_18_class"]["macro_f1_delta_pp"] == 1.55


def test_post_audit_confusion_matrix_and_per_category_csv():
    """Verify post_audit_confusion_matrix.csv and post_audit_per_category.csv if generated."""
    cm_path = ROOT_DIR / "results" / "post_audit_confusion_matrix.csv"
    per_cat_path = ROOT_DIR / "results" / "post_audit_per_category.csv"
    if not cm_path.exists() or not per_cat_path.exists():
        pytest.skip("post-audit error CSVs pending generation.")

    cm_df = pd.read_csv(cm_path, index_col=0)
    assert cm_df.shape in ((17, 17), (18, 18))

    cat_df = pd.read_csv(per_cat_path)
    assert len(cat_df) in (17, 18)
    assert "improved_f1" in cat_df.columns
    assert "baseline_f1" in cat_df.columns
    assert "delta_f1" in cat_df.columns


def test_apptest_model_evaluation_post_audit_navigation():
    """Verify that Streamlit app initializes, navigates to MODEL EVALUATION, and renders tab0 cleanly."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/app.py")
    at.run(timeout=30)
    assert not at.exception

    # Find the MODEL EVALUATION navigation button
    eval_btns = [b for b in at.button if b.key and "nav_btn_model_evaluation" in b.key]
    assert len(eval_btns) == 1
    eval_btns[0].click().run(timeout=30)
    assert not at.exception
    assert at.session_state["active_module"] == "MODEL EVALUATION"
