"""Tests for UI chart generation routines in app/charts.py."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from app.charts import (
    create_baseline_comparison_chart,
    create_taxonomy_comparison_chart,
    create_class_distribution_chart,
    create_per_category_f1_chart,
    create_confusion_pairs_chart,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_baseline_comparison_chart():
    csv_path = ROOT_DIR / "results" / "baseline_vs_improved.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_baseline_comparison_chart(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_taxonomy_comparison_chart():
    csv_path = ROOT_DIR / "results" / "taxonomy_experiment.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_taxonomy_comparison_chart(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_class_distribution_chart():
    csv_path = ROOT_DIR / "results" / "class_distribution.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_class_distribution_chart(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_per_category_f1_chart():
    csv_path = ROOT_DIR / "results" / "per_category_metrics.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_per_category_f1_chart(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confusion_pairs_chart():
    csv_path = ROOT_DIR / "results" / "error_analysis.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    fig = create_confusion_pairs_chart(df, top_n=10)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
