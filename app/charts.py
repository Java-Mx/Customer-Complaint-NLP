"""Chart generation routines for the Customer Complaint NLP Streamlit application.

Renders publication-ready, dark-mode compatible academic charts using Matplotlib.
All charts rely strictly on authentic empirical results from the results/ directory.
"""

from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Academic Dark-Theme Palette
BG_COLOR = "#0f172a"        # Dark slate background
PANEL_COLOR = "#1e293b"     # Slightly lighter container panel
GRID_COLOR = "#334155"      # Subtle gridlines
TEXT_COLOR = "#f8fafc"      # Crisp white-slate text
MUTED_TEXT = "#94a3b8"      # Slate label text
PRIMARY_BLUE = "#3b82f6"    # Standard accent blue
SECONDARY_SLATE = "#64748b" # Baseline neutral slate
ACCENT_EMERALD = "#10b981"  # Conservative taxonomy emerald
ACCENT_PURPLE = "#8b5cf6"   # Broad taxonomy purple
ACCENT_AMBER = "#f59e0b"    # Warning/Amber accent
ACCENT_CORAL = "#ef4444"    # Error/Coral accent


def _apply_dark_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply consistent academic dark-theme styling to a Matplotlib axis and figure."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)


def create_baseline_comparison_chart(bvsi_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart comparing Controlled Baseline vs Improved Model on the 5,000-test set."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=100)
    _apply_dark_style(ax, fig)

    # Extract metrics for same-split comparison
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

    x = np.arange(len(metrics))
    width = 0.32

    rects1 = ax.bar(x - width / 2, baseline_vals, width, label="Controlled Baseline (Word TF-IDF, No Balancing)",
                    color=SECONDARY_SLATE, edgecolor=BG_COLOR, linewidth=1.2, zorder=3)
    rects2 = ax.bar(x + width / 2, improved_vals, width, label="Improved Model (Word+Char TF-IDF, Balanced)",
                    color=PRIMARY_BLUE, edgecolor=BG_COLOR, linewidth=1.2, zorder=3)

    ax.set_ylabel("Metric Score (%)", fontsize=10, fontweight="medium")
    ax.set_title("Controlled Baseline vs. Improved Model (Same Holdout N=5,000 Test Set)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 85)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper left")

    # Add direct value labels above bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=MUTED_TEXT, fontweight="semibold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color="#93c5fd", fontweight="bold")

    fig.tight_layout()
    return fig


def create_taxonomy_comparison_chart(tax_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart comparing Reference 18 vs Conservative 11 vs Broad 10 taxonomies."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=100)
    _apply_dark_style(ax, fig)

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
            # If values were decimals (like 0.5056), scale to percentage
            r_val = float(r_str) * 100 if float(r_str) <= 1.0 else float(r_str)
            v1_val = float(v1_str) * 100 if float(v1_str) <= 1.0 else float(v1_str)
            v2_val = float(v2_str) * 100 if float(v2_str) <= 1.0 else float(v2_str)
        else:
            r_val, v1_val, v2_val = 0.0, 0.0, 0.0
        ref_vals.append(r_val)
        v1_vals.append(v1_val)
        v2_vals.append(v2_val)

    x = np.arange(len(metrics))
    width = 0.25

    r1 = ax.bar(x - width, ref_vals, width, label="Original Reference (18 Classes)", color=PRIMARY_BLUE, zorder=3)
    r2 = ax.bar(x, v1_vals, width, label="v1 Conservative (11 Classes)", color=ACCENT_EMERALD, zorder=3)
    r3 = ax.bar(x + width, v2_vals, width, label="v2 Broad (10 Classes)", color=ACCENT_PURPLE, zorder=3)

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="medium")
    ax.set_title("Cross-Taxonomy Formulation Comparison (Holdout N=5,000 Test Set)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 100)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper left")

    for rects, color in [(r1, "#93c5fd"), (r2, "#a7f3d0"), (r3, "#ddd6fe")]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=color, fontweight="semibold")

    fig.tight_layout()
    return fig


def create_class_distribution_chart(dist_df: pd.DataFrame) -> plt.Figure:
    """Create horizontal bar chart of CFPB dataset class distribution (18 categories)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=100)
    _apply_dark_style(ax, fig)

    sorted_df = dist_df.sort_values(by="total_count", ascending=True).copy()
    categories = sorted_df["category"].tolist()
    counts = sorted_df["total_count"].tolist()
    percentages = sorted_df["total_pct"].tolist()

    y = np.arange(len(categories))
    bars = ax.barh(y, counts, color=PRIMARY_BLUE, alpha=0.85, edgecolor=BG_COLOR, height=0.65, zorder=3)

    # Highlight top 3 high-volume categories with a distinctive shade
    for i in range(len(bars) - 3, len(bars)):
        bars[i].set_color("#2563eb")

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8.5)
    ax.set_xlabel("Number of Complaint Records", fontsize=10, fontweight="medium")
    ax.set_title("CFPB Dataset Class Distribution (Total N=25,000 Records across 18 Categories)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, max(counts) * 1.18)

    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax.annotate(f"{count:,} ({pct:.1f}%)", xy=(count, y[i]),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=TEXT_COLOR, fontweight="medium")

    fig.tight_layout()
    return fig


def create_per_category_f1_chart(per_cat_df: pd.DataFrame) -> plt.Figure:
    """Create horizontal bar chart of per-category F1 scores on the 5,000-record test set."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=100)
    _apply_dark_style(ax, fig)

    sorted_df = per_cat_df.sort_values(by="f1", ascending=True).copy()
    categories = sorted_df["category"].tolist()
    f1_scores = sorted_df["f1"].tolist()
    supports = sorted_df["support"].tolist()

    y = np.arange(len(categories))

    # Color code by performance tier
    colors = []
    for s in f1_scores:
        if s >= 0.70:
            colors.append(ACCENT_EMERALD)  # High performance
        elif s >= 0.45:
            colors.append(PRIMARY_BLUE)     # Moderate performance
        else:
            colors.append(ACCENT_AMBER)    # Challenged / low support

    bars = ax.barh(y, [s * 100 for s in f1_scores], color=colors, alpha=0.88, edgecolor=BG_COLOR, height=0.65, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8.5)
    ax.set_xlabel("F1-Score (%)", fontsize=10, fontweight="medium")
    ax.set_title("Per-Category F1 Performance (Holdout N=5,000 Test Set)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)

    for i, (f1_val, supp) in enumerate(zip(f1_scores, supports)):
        score_pct = f1_val * 100
        ax.annotate(f"{score_pct:.1f}% (N={supp:,})", xy=(score_pct, y[i]),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=TEXT_COLOR, fontweight="medium")

    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_EMERALD, label="F1 >= 70%"),
        Patch(facecolor=PRIMARY_BLUE, label="45% <= F1 < 70%"),
        Patch(facecolor=ACCENT_AMBER, label="F1 < 45% (Minority / Disputed)"),
    ]
    ax.legend(handles=legend_elements, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8, loc="lower right")

    fig.tight_layout()
    return fig


def create_confusion_pairs_chart(err_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Create horizontal bar chart of top confusion pairs from error analysis."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=100)
    _apply_dark_style(ax, fig)

    top_pairs = err_df.head(top_n).iloc[::-1].copy()
    pair_labels = [
        f"{str(row['actual_category'])[:22]}... -> {str(row['predicted_category'])[:22]}..."
        if len(str(row['actual_category'])) > 22 or len(str(row['predicted_category'])) > 22
        else f"{row['actual_category']} -> {row['predicted_category']}"
        for _, row in top_pairs.iterrows()
    ]
    counts = top_pairs["error_count"].tolist()
    pcts = top_pairs["pct_of_actual"].tolist()

    y = np.arange(len(pair_labels))
    ax.barh(y, counts, color=ACCENT_CORAL, alpha=0.82, edgecolor=BG_COLOR, height=0.65, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(pair_labels, fontsize=8.2)
    ax.set_xlabel("Recorded Misclassifications (Test Set N=5,000)", fontsize=10, fontweight="medium")
    ax.set_title(f"Top {top_n} Classification Confusion Pairs (Actual -> Predicted)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, max(counts) * 1.25)

    for i, (cnt, pct) in enumerate(zip(counts, pcts)):
        ax.annotate(f"{cnt:,} errors ({pct:.1f}% of class)", xy=(cnt, y[i]),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=TEXT_COLOR, fontweight="medium")

    fig.tight_layout()
    return fig
