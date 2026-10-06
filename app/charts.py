"""Chart generation routines for the Customer Complaint NLP Streamlit application.

Renders publication-ready, interactive, dark-mode compatible academic charts using Plotly.
All charts rely strictly on authentic empirical results from the results/ directory.
Supports rich hover tooltips, zoom, pan, and legend toggling directly in Streamlit.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Academic Dark-Theme Palette matching application styling
BG_COLOR = "#0f172a"        # Dark slate background
PANEL_COLOR = "#1e293b"     # Container panel
GRID_COLOR = "rgba(255, 255, 255, 0.08)"  # Subtle grid lines
TEXT_COLOR = "#f8fafc"      # Crisp white-slate text
MUTED_TEXT = "#94a3b8"      # Slate label text
PRIMARY_BLUE = "#3b82f6"    # Standard accent blue
SECONDARY_SLATE = "#64748b" # Baseline neutral slate
ACCENT_EMERALD = "#10b981"  # Conservative taxonomy emerald
ACCENT_PURPLE = "#8b5cf6"   # Broad taxonomy purple
ACCENT_AMBER = "#f59e0b"    # Warning/Amber accent
ACCENT_CORAL = "#ef4444"    # Error/Coral accent


def _apply_dark_theme(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 420,
    margin: Optional[dict] = None,
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent academic dark-theme styling to a Plotly figure."""
    m = margin or dict(l=30, r=30, t=50, b=40)
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=TEXT_COLOR, size=13),
            x=0.01,
            y=0.97,
            xanchor="left",
            yanchor="top",
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            color=TEXT_COLOR,
            size=11,
        ),
        height=height,
        margin=m,
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor=PANEL_COLOR,
            bordercolor=PRIMARY_BLUE,
            font=dict(color="#ffffff", size=11),
        ),
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=MUTED_TEXT, size=11)) if x_title else None,
        tickfont=dict(color=MUTED_TEXT, size=10),
        gridcolor=GRID_COLOR,
        zerolinecolor="rgba(255, 255, 255, 0.15)",
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=MUTED_TEXT, size=11)) if y_title else None,
        tickfont=dict(color=MUTED_TEXT, size=10),
        gridcolor=GRID_COLOR,
        zerolinecolor="rgba(255, 255, 255, 0.15)",
    )
    return fig


def create_baseline_comparison_chart(bvsi_df: pd.DataFrame) -> go.Figure:
    """Create interactive grouped bar chart comparing Controlled Baseline vs Improved Model."""
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    baseline_vals = []
    improved_vals = []

    for m in metrics:
        row = bvsi_df[bvsi_df["metric"] == m]
        if not row.empty:
            b_val = float(row["baseline_word_tfidf_no_balancing"].values[0]) * 100
            i_val = float(row["improved_combined_tfidf_balanced"].values[0]) * 100
        else:
            b_val, i_val = 0.0, 0.0
        baseline_vals.append(b_val)
        improved_vals.append(i_val)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Controlled Baseline",
            x=metrics,
            y=baseline_vals,
            text=[f"{v:.2f}%" for v in baseline_vals],
            textposition="outside",
            textfont=dict(color=MUTED_TEXT, size=10),
            marker_color=SECONDARY_SLATE,
            hovertemplate="<b>Controlled Baseline</b><br>Metric: %{x}<br>Score: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Improved Model",
            x=metrics,
            y=improved_vals,
            text=[f"{v:.2f}%" for v in improved_vals],
            textposition="outside",
            textfont=dict(color="#93c5fd", size=10),
            marker_color=PRIMARY_BLUE,
            hovertemplate="<b>Improved Model</b><br>Metric: %{x}<br>Score: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            font=dict(color=MUTED_TEXT, size=10),
        ),
    )
    fig.update_yaxes(range=[0, 85])
    _apply_dark_theme(
        fig,
        title="Controlled Baseline vs. Improved Model (Same Holdout N=5,000 Test Set)",
        y_title="Score (%)",
        height=380,
    )
    return fig


def create_taxonomy_comparison_chart(tax_df: pd.DataFrame) -> go.Figure:
    """Create interactive grouped bar chart comparing Reference 18 vs Conservative 11 vs Broad 10 taxonomies."""
    metrics = ["Overall Accuracy", "Macro F1-Score", "Weighted F1-Score"]
    display_names = ["Accuracy", "Macro F1", "Weighted F1"]

    ref_vals = []
    v1_vals = []
    v2_vals = []

    for m in metrics:
        row = tax_df[tax_df["Metric"] == m]
        if not row.empty:
            r_str = str(row["Reference (18 Categories)"].values[0]).rstrip("%")
            v1_str = str(row["v1 Conservative (11 Categories)"].values[0]).rstrip("%")
            v2_str = str(row["v2 Broad (10 Categories)"].values[0]).rstrip("%")
            r_val = float(r_str) * 100 if float(r_str) <= 1.0 else float(r_str)
            v1_val = float(v1_str) * 100 if float(v1_str) <= 1.0 else float(v1_str)
            v2_val = float(v2_str) * 100 if float(v2_str) <= 1.0 else float(v2_str)
        else:
            r_val, v1_val, v2_val = 0.0, 0.0, 0.0
        ref_vals.append(r_val)
        v1_vals.append(v1_val)
        v2_vals.append(v2_val)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Original 18-category reference",
            x=display_names,
            y=ref_vals,
            text=[f"{v:.2f}%" for v in ref_vals],
            textposition="outside",
            textfont=dict(color="#93c5fd", size=10),
            marker_color=PRIMARY_BLUE,
            hovertemplate="<b>Original 18-Category Reference</b><br>Metric: %{x}<br>Score: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Conservative taxonomy (11 Classes)",
            x=display_names,
            y=v1_vals,
            text=[f"{v:.2f}%" for v in v1_vals],
            textposition="outside",
            textfont=dict(color="#a7f3d0", size=10),
            marker_color=ACCENT_EMERALD,
            hovertemplate="<b>Conservative Taxonomy (11 Classes)</b><br>Metric: %{x}<br>Score: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Broad taxonomy (10 Classes)",
            x=display_names,
            y=v2_vals,
            text=[f"{v:.2f}%" for v in v2_vals],
            textposition="outside",
            textfont=dict(color="#ddd6fe", size=10),
            marker_color=ACCENT_PURPLE,
            hovertemplate="<b>Broad Taxonomy (10 Classes)</b><br>Metric: %{x}<br>Score: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            font=dict(color=MUTED_TEXT, size=10),
        ),
    )
    fig.update_yaxes(range=[0, 100])
    _apply_dark_theme(
        fig,
        title="Cross-Taxonomy Formulation Comparison (Holdout N=5,000 Test Set)",
        y_title="Score (%)",
        height=380,
    )
    return fig


def create_class_distribution_chart(dist_df: pd.DataFrame) -> go.Figure:
    """Create interactive horizontal bar chart of CFPB dataset class distribution (18 categories)."""
    sorted_df = dist_df.sort_values(by="total_count", ascending=True).copy()
    categories = sorted_df["category"].tolist()
    counts = sorted_df["total_count"].tolist()
    percentages = sorted_df["total_pct"].tolist()

    n_items = len(categories)
    colors = [PRIMARY_BLUE] * n_items
    for i in range(max(0, n_items - 3), n_items):
        colors[i] = "#2563eb"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=categories,
            x=counts,
            orientation="h",
            text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts, percentages)],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            marker=dict(color=colors, line=dict(color=BG_COLOR, width=1)),
            customdata=percentages,
            hovertemplate="<b>%{y}</b><br>Complaints: %{x:,}<br>Dataset Share: %{customdata:.2f}%<extra></extra>",
        )
    )

    max_count = max(counts) if counts else 1000
    fig.update_xaxes(range=[0, max_count * 1.22])
    _apply_dark_theme(
        fig,
        title="CFPB Dataset Class Distribution (Total N=25,000 Records across 18 Categories)",
        x_title="Number of Complaint Records",
        height=540,
        margin=dict(l=260, r=40, t=50, b=40),
        show_legend=False,
    )
    fig.update_yaxes(automargin=True, tickfont=dict(color=TEXT_COLOR, size=10))
    return fig


def create_per_category_f1_chart(per_cat_df: pd.DataFrame) -> go.Figure:
    """Create interactive horizontal bar chart of per-category F1 scores on the 5,000-record test set."""
    sorted_df = per_cat_df.sort_values(by="f1", ascending=True).copy()
    categories = sorted_df["category"].tolist()
    f1_scores = sorted_df["f1"].tolist()
    supports = sorted_df["support"].tolist()
    precisions = sorted_df["precision"].tolist()
    recalls = sorted_df["recall"].tolist()

    colors = []
    for s in f1_scores:
        if s >= 0.70:
            colors.append(ACCENT_EMERALD)  # High performance
        elif s >= 0.45:
            colors.append(PRIMARY_BLUE)     # Moderate performance
        else:
            colors.append(ACCENT_AMBER)    # Minority / challenged

    f1_percentages = [s * 100 for s in f1_scores]
    custom_tuples = list(zip(supports, precisions, recalls, f1_scores))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=categories,
            x=f1_percentages,
            orientation="h",
            text=[f"{f1:.1f}% (N={supp:,})" for f1, supp in zip(f1_percentages, supports)],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            marker=dict(color=colors, line=dict(color=BG_COLOR, width=1)),
            customdata=custom_tuples,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "F1-Score: %{customdata[3]:.4f} (%{x:.2f}%)<br>"
                "Precision: %{customdata[1]:.2%}<br>"
                "Recall: %{customdata[2]:.2%}<br>"
                "Test Support: N=%{customdata[0]:,}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(range=[0, 115])
    _apply_dark_theme(
        fig,
        title="Per-Category F1 Performance (Holdout N=5,000 Test Set)",
        x_title="F1-Score (%)",
        height=540,
        margin=dict(l=260, r=40, t=50, b=40),
        show_legend=False,
    )
    fig.update_yaxes(automargin=True, tickfont=dict(color=TEXT_COLOR, size=10))
    return fig


def create_confusion_pairs_chart(err_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Create interactive horizontal bar chart of top confusion pairs from error analysis."""
    top_pairs = err_df.head(top_n).iloc[::-1].copy()
    actual_cats = top_pairs["actual_category"].tolist()
    pred_cats = top_pairs["predicted_category"].tolist()
    counts = top_pairs["error_count"].tolist()
    pcts = top_pairs["pct_of_actual"].tolist()

    labels = []
    for actual, pred in zip(actual_cats, pred_cats):
        act_short = f"{actual[:22]}..." if len(actual) > 24 else actual
        pred_short = f"{pred[:22]}..." if len(pred) > 24 else pred
        labels.append(f"{act_short} → {pred_short}")

    custom_tuples = list(zip(actual_cats, pred_cats, pcts))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            text=[f"{cnt:,} errors ({pct:.1f}% of class)" for cnt, pct in zip(counts, pcts)],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            marker=dict(color=ACCENT_CORAL, line=dict(color=BG_COLOR, width=1)),
            customdata=custom_tuples,
            hovertemplate=(
                "<b>Actual:</b> %{customdata[0]}<br>"
                "<b>Predicted:</b> %{customdata[1]}<br>"
                "<b>Misclassifications:</b> %{x:,} errors<br>"
                "<b>Impact:</b> %{customdata[2]:.2f}% of actual category<extra></extra>"
            ),
        )
    )

    max_count = max(counts) if counts else 100
    fig.update_xaxes(range=[0, max_count * 1.35])
    _apply_dark_theme(
        fig,
        title=f"Top {top_n} Classification Confusion Pairs (Actual → Predicted)",
        x_title="Recorded Misclassifications (Test Set N=5,000)",
        height=420,
        margin=dict(l=300, r=40, t=50, b=40),
        show_legend=False,
    )
    fig.update_yaxes(automargin=True, tickfont=dict(color=TEXT_COLOR, size=10))
    return fig


def create_confusion_matrix_heatmap(err_df: pd.DataFrame, per_cat_df: pd.DataFrame) -> go.Figure:
    """Create interactive 18x18 confusion matrix heatmap using Plotly."""
    categories = per_cat_df["category"].tolist()
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    n = len(categories)

    matrix = np.zeros((n, n), dtype=int)

    # Diagonal: correct predictions
    for _, row in per_cat_df.iterrows():
        cat = row["category"]
        if cat in cat_to_idx:
            idx = cat_to_idx[cat]
            matrix[idx, idx] = int(row["correct"])

    # Off-diagonal: misclassifications
    for _, row in err_df.iterrows():
        act = row["actual_category"]
        pred = row["predicted_category"]
        if act in cat_to_idx and pred in cat_to_idx:
            matrix[cat_to_idx[act], cat_to_idx[pred]] = int(row["error_count"])

    short_labels = [f"{c[:18]}.." if len(c) > 20 else c for c in categories]

    row_sums = matrix.sum(axis=1)
    customdata = np.empty((n, n, 3), dtype=object)
    for r in range(n):
        for c in range(n):
            cnt = matrix[r, c]
            pct = (cnt / row_sums[r] * 100.0) if row_sums[r] > 0 else 0.0
            customdata[r, c] = [categories[r], categories[c], pct]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=short_labels,
            y=short_labels,
            customdata=customdata,
            colorscale=[
                [0.0, "#0f172a"],
                [0.05, "#1e3a8a"],
                [0.2, "#2563eb"],
                [0.6, "#38bdf8"],
                [1.0, "#f8fafc"],
            ],
            colorbar=dict(
                title=dict(text="Records", font=dict(color=TEXT_COLOR, size=11)),
                tickfont=dict(color=MUTED_TEXT, size=10),
            ),
            hovertemplate=(
                "<b>Actual:</b> %{customdata[0]}<br>"
                "<b>Predicted:</b> %{customdata[1]}<br>"
                "<b>Count:</b> %{z:,} complaints (%{customdata[2]:.1f}% of class)<extra></extra>"
            ),
        )
    )

    _apply_dark_theme(
        fig,
        title="Multi-Class Confusion Matrix Heatmap (N=5,000 Test Set)",
        x_title="Predicted Product Category",
        y_title="Actual Ground Truth Category",
        height=620,
        margin=dict(l=140, r=40, t=50, b=90),
        show_legend=False,
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(color=TEXT_COLOR, size=9))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_COLOR, size=9))
    return fig
