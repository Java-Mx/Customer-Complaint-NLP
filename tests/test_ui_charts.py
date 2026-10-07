"""Tests for UI chart generation routines in app/charts.py."""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.charts import (
    create_baseline_comparison_chart,
    create_taxonomy_comparison_chart,
    create_class_distribution_chart,
    create_per_category_f1_chart,
    create_confusion_pairs_chart,
    create_confusion_matrix_heatmap,
    create_post_audit_model_comparison_chart,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_baseline_comparison_chart():
    csv_path = ROOT_DIR / "results" / "baseline_vs_improved.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_baseline_comparison_chart(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Controlled Baseline"
    assert fig.data[1].name == "Improved Model"
    assert "Controlled Baseline vs. Improved Model" in fig.layout.title.text


def test_taxonomy_comparison_chart():
    csv_path = ROOT_DIR / "results" / "taxonomy_experiment.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_taxonomy_comparison_chart(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert "Cross-Taxonomy Formulation Comparison" in fig.layout.title.text


def test_class_distribution_chart():
    csv_path = ROOT_DIR / "results" / "class_distribution.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_class_distribution_chart(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"
    assert len(fig.data[0].y) == 18
    assert "CFPB Dataset Class Distribution" in fig.layout.title.text


def test_per_category_f1_chart():
    csv_path = ROOT_DIR / "results" / "per_category_metrics.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_per_category_f1_chart(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"
    assert len(fig.data[0].y) == 18
    assert "Per-Category F1 Performance" in fig.layout.title.text


def test_confusion_pairs_chart():
    csv_path = ROOT_DIR / "results" / "error_analysis.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_confusion_pairs_chart(df, top_n=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"
    assert len(fig.data[0].y) == 10
    assert "Top 10 Classification Confusion Pairs" in fig.layout.title.text


def test_confusion_matrix_heatmap():
    err_path = ROOT_DIR / "results" / "error_analysis.csv"
    cat_path = ROOT_DIR / "results" / "per_category_metrics.csv"
    assert err_path.exists() and cat_path.exists()
    err_df = pd.read_csv(err_path)
    cat_df = pd.read_csv(cat_path)
    fig = create_confusion_matrix_heatmap(err_df, cat_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    assert fig.data[0].z.shape == (18, 18)
    assert "Multi-Class Confusion Matrix" in fig.layout.title.text


def test_post_audit_model_comparison_chart():
    df = pd.DataFrame([
        {
            "experiment": "18-class Baseline (Standard Preprocessing)",
            "taxonomy": "CFPB Original (18 classes)",
            "test_accuracy": 0.6982,
            "test_macro_f1": 0.5088,
            "test_weighted_f1": 0.6978,
            "accuracy": 0.6995,
            "macro_f1": 0.5184,
            "weighted_f1": 0.6983,
        },
        {
            "experiment": "18-class Improved Representation (Minimal Preprocessing)",
            "taxonomy": "CFPB Original (18 classes)",
            "test_accuracy": 0.7100,
            "test_macro_f1": 0.5300,
            "test_weighted_f1": 0.7100,
            "accuracy": 0.7133,
            "macro_f1": 0.5350,
            "weighted_f1": 0.7125,
        },
        {
            "experiment": "11-class Normalized Taxonomy (Minimal Preprocessing)",
            "taxonomy": "Conservative v1 (11 classes)",
            "test_accuracy": 0.8250,
            "test_macro_f1": 0.6400,
            "test_weighted_f1": 0.8250,
            "accuracy": 0.8288,
            "macro_f1": 0.6450,
            "weighted_f1": 0.8290,
        },
    ])
    fig = create_post_audit_model_comparison_chart(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert fig.data[0].name == "Original 18-Class Baseline (Standard Preprocessing)"
    assert fig.data[1].name == "Improved 18-Class (Minimal Preprocessing)"
    assert fig.data[2].name == "Normalized 11-Class (Conservative Taxonomy)"
    assert "Post-Audit Model Comparison" in fig.layout.title.text

