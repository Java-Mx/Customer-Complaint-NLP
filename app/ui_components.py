"""UI components and presentation helpers for the Customer Complaint NLP Streamlit application.

Provides a unified academic design system, centralized design tokens,
standardized metric cards, compact responsive layouts, and model insight
visualizations without altering underlying NLP methodologies.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure project root is on sys.path before importing from src or app
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import preprocess_text
from src.vectorization import transform_word_char, get_top_active_features
from src.similarity import find_similar_complaints
from src.classification import predict_complaint_category, predict_category_proba
try:
    from app.charts import (
        create_baseline_comparison_chart,
        create_taxonomy_comparison_chart,
        create_class_distribution_chart,
        create_per_category_f1_chart,
        create_confusion_pairs_chart,
        create_confusion_matrix_heatmap,
    )
except (ImportError, ModuleNotFoundError):
    from charts import (
        create_baseline_comparison_chart,
        create_taxonomy_comparison_chart,
        create_class_distribution_chart,
        create_per_category_f1_chart,
        create_confusion_pairs_chart,
        create_confusion_matrix_heatmap,
    )

# ==============================================================================
# CENTRALIZED DESIGN TOKENS
# ==============================================================================
DESIGN_TOKENS: Dict[str, Any] = {
    "colors": {
        "bg_main": "#0a0e17",
        "bg_surface": "#101726",
        "bg_surface_elevated": "#162032",
        "bg_surface_hover": "#1e2c44",
        "border_subtle": "rgba(255, 255, 255, 0.08)",
        "border_medium": "rgba(255, 255, 255, 0.14)",
        "border_accent": "rgba(59, 130, 246, 0.35)",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#94a3b8",
        "accent_primary": "#2563eb",
        "accent_hover": "#1d4ed8",
        "accent_border": "#3b82f6",
        "status_success": "#10b981",
        "status_warning": "#f59e0b",
        "status_error": "#ef4444",
        "status_info": "#38bdf8",
    },
    "spacing": {
        "xs": "0.25rem",
        "sm": "0.5rem",
        "md": "0.75rem",
        "lg": "1.25rem",
        "xl": "1.75rem",
    },
    "radius": {
        "sm": "4px",
        "md": "6px",
        "lg": "8px",
    },
}

# Realistic demonstration complaint examples (6 domain-standard + 3 intentionally ambiguous for viva defense)
DEMO_COMPLAINT_EXAMPLES: Dict[str, Dict[str, str]] = {
    "Unauthorized Credit Card Payment": {
        "label": "Unauthorized Credit Card Payment",
        "category_hint": "Credit card",
        "is_ambiguous": False,
        "text": (
            "I noticed multiple unauthorized transactions and recurring charges on my credit card "
            "statement from a merchant I never visited. I contacted customer service immediately to dispute "
            "the charges, but the bank refused to issue a provisional credit and continues to bill monthly finance charges."
        ),
    },
    "Debt Collection Complaint": {
        "label": "Debt Collection Complaint",
        "category_hint": "Debt collection",
        "is_ambiguous": False,
        "text": (
            "A debt collection agency has been repeatedly calling my personal cell phone and workplace "
            "attempting to collect on a medical debt that was already settled in full in 2022. They refuse to "
            "provide written debt verification and threatened legal action in direct violation of the FDCPA."
        ),
    },
    "Credit Report Error": {
        "label": "Credit Report Error",
        "category_hint": "Credit reporting",
        "is_ambiguous": False,
        "text": (
            "My credit report contains an inaccurate 90-day delinquent account and an erroneous public record "
            "that belongs to another individual with a similar name. I submitted formal written dispute requests "
            "with Experian, Equifax, and TransUnion along with supporting identity documents, but they failed to investigate."
        ),
    },
    "Mortgage Problem": {
        "label": "Mortgage Problem",
        "category_hint": "Mortgage",
        "is_ambiguous": False,
        "text": (
            "Our mortgage loan was recently transferred to a new loan servicer who failed to properly credit "
            "our monthly escrow and principal payments. They misapplied escrow disbursements, improperly claimed the "
            "account was in default, and assessed unwarranted late fees despite timely automated bank transfers."
        ),
    },
    "Student Loan Problem": {
        "label": "Student Loan Problem",
        "category_hint": "Student loan",
        "is_ambiguous": False,
        "text": (
            "My federal student loan servicer failed to process my Income-Driven Repayment (IDR) recertification "
            "paperwork on time. Because of their administrative delay, my required monthly payment increased tenfold "
            "and my account was improperly reported as delinquent to consumer credit bureaus."
        ),
    },
    "Bank Account Problem": {
        "label": "Bank Account Problem",
        "category_hint": "Bank account or service",
        "is_ambiguous": False,
        "text": (
            "The bank charged multiple consecutive overdraft and NSF fees on my checking account in a single "
            "afternoon by reordering debit transactions from highest to lowest amount. Additionally, they froze "
            "access to my direct-deposited payroll funds without prior warning or justification."
        ),
    },
    "Ambiguous: Card Dispute vs Credit Bureau Reporting": {
        "label": "Ambiguous: Card Dispute vs Credit Bureau Reporting",
        "category_hint": "Cross-boundary (Credit Card x Credit Reporting)",
        "is_ambiguous": True,
        "text": (
            "I opened a formal billing dispute with my credit card company regarding an unauthorized merchant charge. "
            "While the dispute was still under active investigation, the bank reported the contested balance as a 90-day "
            "delinquent derogatory account to all three major credit bureaus, severely damaging my credit score."
        ),
    },
    "Ambiguous: Debt Collection vs Identity Theft Tradeline": {
        "label": "Ambiguous: Debt Collection vs Identity Theft Tradeline",
        "category_hint": "Cross-boundary (Debt Collection x Credit Reporting)",
        "is_ambiguous": True,
        "text": (
            "A third-party collection agency placed an unverified collection account on my credit report for an "
            "identity theft loan that I never opened. When I sent them an official FTC identity theft report, the collection "
            "agency refused to cease collections and refused to delete the fraudulent tradeline from my consumer reports."
        ),
    },
    "Ambiguous: Checking Overdraft vs Payday Loan ACH": {
        "label": "Ambiguous: Checking Overdraft vs Payday Loan ACH",
        "category_hint": "Cross-boundary (Checking Account x Payday Loan)",
        "is_ambiguous": True,
        "text": (
            "An online payday lender executed repeated unauthorized electronic ACH debits against my checking account, "
            "causing my bank account balance to become negative. The bank then assessed consecutive overdraft penalty "
            "fees while the high-interest loan company continued daily withdrawal attempts."
        ),
    },
}


# ==============================================================================
# STATUS ICON AND TICK HELPERS (PURE SVG, NO UNICODE EMOJIS/SYMBOLS)
# ==============================================================================

def render_status_tick(status: str = "success") -> str:
    """Return inline SVG check icon or non-success state for status indicators.

    Parameters
    ----------
    status : str
        'success' (green checkmark), 'warning' (amber indicator),
        or 'error' / 'offline' (coral indicator).

    Returns
    -------
    str
        Accessible inline SVG markup with exact color tokens.
    """
    if status == "success":
        return (
            '<svg class="status-svg" viewBox="0 0 16 16" width="14" height="14" fill="none" '
            'stroke="#22c55e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
            'aria-label="Success" role="img">'
            '<polyline points="3 8.5 6.5 12 13 4.5"/>'
            '</svg>'
        )
    elif status == "warning":
        return (
            '<svg class="status-svg" viewBox="0 0 16 16" width="14" height="14" fill="none" '
            'stroke="#f59e0b" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round" '
            'aria-label="Warning" role="img">'
            '<circle cx="8" cy="8" r="6.5"/>'
            '<line x1="8" y1="5" x2="8" y2="8.5"/>'
            '<line x1="8" y1="11" x2="8.01" y2="11"/>'
            '</svg>'
        )
    elif status == "info":
        return (
            '<svg class="status-svg" viewBox="0 0 16 16" width="14" height="14" fill="none" '
            'stroke="#38bdf8" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round" '
            'aria-label="Info" role="img">'
            '<circle cx="8" cy="8" r="6.5"/>'
            '<line x1="8" y1="8" x2="8" y2="11.5"/>'
            '<line x1="8" y1="5" x2="8.01" y2="5"/>'
            '</svg>'
        )
    else:
        return (
            '<svg class="status-svg" viewBox="0 0 16 16" width="14" height="14" fill="none" '
            'stroke="#ef4444" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round" '
            'aria-label="Error or Offline" role="img">'
            '<circle cx="8" cy="8" r="6.5"/>'
            '<line x1="5.5" y1="5.5" x2="10.5" y2="10.5"/>'
            '<line x1="10.5" y1="5.5" x2="5.5" y2="10.5"/>'
            '</svg>'
        )


def render_status_row(label: str, subtext: str, status: str = "success") -> str:
    """Return HTML snippet for a single status item within the unified status card."""
    icon_svg = render_status_tick(status)
    return f"""
    <div class="status-item">
        <div class="status-icon-wrap">{icon_svg}</div>
        <div class="status-info">
            <div class="status-label">{label}</div>
            <div class="status-sub">{subtext}</div>
        </div>
    </div>
    """


# ==============================================================================
# CSS STYLESHEET INJECTION (UNIFIED DESIGN SYSTEM)
# ==============================================================================

def apply_custom_styles() -> None:
    """Inject polished, presentation-ready CSS with centralized design tokens."""
    css = """
    <style>
    /* -------------------------------------------------------------------------
       DESIGN SYSTEM TOKENS & CSS VARIABLES
       ------------------------------------------------------------------------- */
    :root {
        --bg-main: #0a0e17;
        --bg-surface: #101726;
        --bg-surface-elevated: #162032;
        --bg-surface-hover: #1e2c44;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-medium: rgba(255, 255, 255, 0.14);
        --border-accent: rgba(59, 130, 246, 0.35);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --accent-primary: #2563eb;
        --accent-hover: #1d4ed8;
        --accent-border: #3b82f6;
        --status-success: #10b981;
        --status-warning: #f59e0b;
        --status-error: #ef4444;
        --status-info: #38bdf8;
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 0.75rem;
        --space-lg: 1.25rem;
        --space-xl: 1.75rem;
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
        --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --font-mono: "JetBrains Mono", "SF Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    }

    /* Global application canvas */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
    }

    /* Compact layout: remove excessive top whitespace */
    .block-container, [data-testid="stMainBlockContainer"], .stMainBlockContainer {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Subtle divider */
    hr {
        margin: 1.15rem 0 !important;
        border: 0 !important;
        border-top: 1px solid var(--border-subtle) !important;
    }

    /* -------------------------------------------------------------------------
       SIDEBAR & NAVIGATION
       ------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebarContent"] {
        padding: 1rem 0.85rem !important;
    }

    .sidebar-brand {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        padding: 0.35rem 0.4rem 0.65rem 0.4rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 0.75rem;
    }

    /* Sidebar rectangular button navigation */
    [data-testid="stSidebar"] div.stButton > button {
        border-radius: var(--radius-md) !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.03em !important;
        padding: 0.38rem 0.65rem !important;
        min-height: 2.15rem !important;
        height: 2.15rem !important;
        margin-bottom: 0.2rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
        text-transform: uppercase !important;
        transition: all 0.15s ease-in-out !important;
        width: 100% !important;
    }

    /* Inactive sidebar buttons */
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"],
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
        background-color: var(--bg-surface-elevated) !important;
        border-color: var(--border-medium) !important;
        color: var(--text-primary) !important;
    }

    /* Active sidebar button - Restrained Technical Highlight */
    [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
        background-color: rgba(37, 99, 235, 0.18) !important;
        border: 1px solid var(--accent-border) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* -------------------------------------------------------------------------
       BUTTONS ACROSS APPLICATION
       ------------------------------------------------------------------------- */
    div.stButton > button {
        border-radius: var(--radius-md) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: var(--accent-primary) !important;
        border: 1px solid var(--accent-border) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
        border-color: #60a5fa !important;
    }
    div.stButton > button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: var(--bg-surface-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-secondary) !important;
        padding: 0.45rem 1rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: var(--bg-surface-hover) !important;
        border-color: var(--border-medium) !important;
        color: var(--text-primary) !important;
    }

    /* Demonstration example buttons */
    .example-btn-area div.stButton > button {
        font-size: 0.76rem !important;
        padding: 0.32rem 0.55rem !important;
        border-radius: var(--radius-sm) !important;
        background-color: var(--bg-surface-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-secondary) !important;
        text-transform: none !important;
    }
    .example-btn-area div.stButton > button:hover {
        border-color: var(--accent-border) !important;
        color: var(--text-primary) !important;
        background-color: var(--bg-surface-hover) !important;
    }

    /* -------------------------------------------------------------------------
       STANDARDIZED PAGE & SECTION HEADERS
       ------------------------------------------------------------------------- */
    .app-page-header {
        margin-bottom: 1rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .page-title-row {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .page-title {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        padding: 0 !important;
        text-transform: uppercase !important;
    }
    .page-subtitle {
        font-size: 0.84rem !important;
        color: var(--text-muted) !important;
        margin-top: 0.25rem !important;
        line-height: 1.45 !important;
    }

    .app-section-header {
        margin-top: 0.85rem;
        margin-bottom: 0.55rem;
    }
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .section-desc {
        font-size: 0.80rem;
        color: var(--text-muted);
        margin-top: 0.12rem;
        line-height: 1.4;
    }

    /* -------------------------------------------------------------------------
       STANDARDIZED METRIC CARDS
       ------------------------------------------------------------------------- */
    [data-testid="stMetric"] {
        background-color: var(--bg-surface-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.65rem 0.85rem !important;
        min-height: 82px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        transition: border-color 0.15s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-medium) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.70rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--text-muted) !important;
        margin-bottom: 0.15rem !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.30rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        line-height: 1.2 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        margin-top: 0.2rem !important;
    }

    /* -------------------------------------------------------------------------
       CONTEXTUAL INFO & STATUS CARDS
       ------------------------------------------------------------------------- */
    .app-info-card {
        background-color: rgba(30, 41, 59, 0.45);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent-border);
        border-radius: var(--radius-sm);
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.85rem;
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }
    .info-card-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
        font-size: 0.82rem;
    }

    /* Unified Tabs */
    [data-testid="stTabs"] {
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        gap: 0.25rem !important;
        padding-bottom: 0 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        padding: 0.45rem 0.80rem !important;
        transition: all 0.15s ease !important;
        text-transform: uppercase !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        border-bottom-color: var(--border-medium) !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--accent-border) !important;
        background-color: transparent !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--accent-border) !important;
    }

    /* System Status Card in Sidebar */
    .status-card {
        background-color: var(--bg-surface-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 0.75rem 0.85rem;
        font-size: 0.78rem;
        line-height: 1.4;
        color: var(--text-secondary);
        margin-top: 0.85rem;
        box-sizing: border-box;
        width: 100%;
    }
    .status-header {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        padding-bottom: 0.35rem;
        margin-bottom: 0.55rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .status-list {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }
    .status-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }
    .status-icon-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        padding-top: 2px;
        flex-shrink: 0;
    }
    .status-svg {
        display: block;
    }
    .status-info {
        display: flex;
        flex-direction: column;
        gap: 0.05rem;
        min-width: 0;
    }
    .status-label {
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.25;
    }
    .status-sub {
        font-size: 0.70rem;
        color: var(--text-muted);
        line-height: 1.3;
        word-break: normal;
    }

    /* Feature Representation Card */
    .feature-rep-card {
        background-color: var(--bg-surface-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 0.65rem 0.85rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .feature-rep-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 0.2rem;
    }
    .feature-rep-val {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.35;
    }

    /* Prediction Result Card */
    .prediction-card {
        background-color: var(--bg-surface-elevated);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius-md);
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .card-label {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.2rem;
    }
    .card-value-pred {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }
    .card-value-conf {
        font-size: 1.35rem;
        font-weight: 700;
        color: #60a5fa;
        letter-spacing: -0.01em;
    }

    /* Unified Tables */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        background-color: var(--bg-surface) !important;
    }
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 0.80rem !important;
        color: var(--text-secondary) !important;
    }
    table th {
        background-color: var(--bg-surface-elevated) !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 0.5rem 0.7rem !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        text-align: left !important;
    }
    table td {
        padding: 0.45rem 0.7rem !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        line-height: 1.4 !important;
    }
    table tr:hover td {
        background-color: var(--bg-surface-elevated) !important;
    }

    /* Unified Inputs and Text Areas */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-border) !important;
        box-shadow: 0 0 0 1px var(--accent-border) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: var(--bg-surface-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        margin-bottom: 0.55rem !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        padding: 0.45rem 0.75rem !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--text-primary) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==============================================================================
# REUSABLE RENDERING HELPERS
# ==============================================================================

def render_page_header(
    title: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """Render standardized top page header across all application modules."""
    badge_html = f'<span class="page-badge">{badge}</span>' if badge else ""
    sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div class="app-page-header">
        <div class="page-title-row">
            <h1 class="page-title">{title}</h1>
            {badge_html}
        </div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, description: Optional[str] = None) -> None:
    """Render standardized section header across all application modules."""
    desc_html = f'<div class="section-desc">{description}</div>' if description else ""
    html = f"""
    <div class="app-section-header">
        <div class="section-title">{title}</div>
        {desc_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    help: Optional[str] = None,
    delta_color: str = "normal",
) -> None:
    """Render standardized metric card using the unified design system.

    Parameters
    ----------
    label : str
        Descriptive label (rendered in uppercase muted text).
    value : Any
        Primary metric value (formatted percentage, integer, or string).
    delta : Optional[str], optional
        Optional contextual change or benchmark delta (e.g. '+7.99 pp').
    help : Optional[str], optional
        Optional tooltip string explaining the metric.
    delta_color : str, optional
        Delta display color mode: 'normal', 'inverse', or 'off'.
    """
    st.metric(
        label=label,
        value=str(value),
        delta=delta,
        help=help,
        delta_color=delta_color,
    )


def render_metric_grid(metrics: List[Dict[str, Any]], cols: int = 4) -> None:
    """Render a responsive grid of standardized metric cards.

    Parameters
    ----------
    metrics : list of dict
        Each dict contains 'label', 'value', and optionally 'delta' and 'help'.
    cols : int
        Number of columns per row.
    """
    for i in range(0, len(metrics), cols):
        chunk = metrics[i : i + cols]
        columns = st.columns(cols)
        for col, m in zip(columns, chunk):
            with col:
                render_metric_card(
                    label=m["label"],
                    value=m["value"],
                    delta=m.get("delta"),
                    help=m.get("help"),
                    delta_color=m.get("delta_color", "normal"),
                )


def render_info_card(text: str, title: Optional[str] = None) -> None:
    """Render a subtle contextual callout card."""
    title_html = f'<div class="info-card-title">{title}</div>' if title else ""
    html = f"""
    <div class="app-info-card">
        {title_html}
        <div class="info-card-body">{text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_chart_card(
    fig: Any,
    title: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[dict] = None,
) -> None:
    """Render an interactive Plotly chart within a standardized card container."""
    cfg = config or {"displayModeBar": True, "displaylogo": False}
    if title:
        render_section_header(title, description)
    st.plotly_chart(fig, width="stretch", config=cfg)


def render_status_card(
    api_status: str,
    api_str: str,
    data_status: str,
    data_str: str,
    model_status: str,
    model_str: str,
    rep_status: str,
    rep_str: str,
) -> None:
    """Render unified system status card for sidebar."""
    card_html = f"""
    <div class="status-card">
        <div class="status-header">SYSTEM STATUS</div>
        <div class="status-list">
            {render_status_row("CFPB Search API", api_str, api_status)}
            {render_status_row("Local Dataset", data_str, data_status)}
            {render_status_row("Classifier", model_str, model_status)}
            {render_status_row("Representation", rep_str, rep_status)}
        </div>
    </div>
    """
    st.sidebar.markdown(card_html, unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ==============================================================================

def render_sidebar_navigation(
    api_online: bool,
    data_csv_exists: bool,
    clf_loaded: Any,
    vec_loaded: Any,
    dataset_records: Optional[int] = None,
) -> str:
    """Render rounded rectangular sidebar navigation and compact system status.

    Parameters
    ----------
    api_online : bool
        Whether CFPB API health check succeeded.
    data_csv_exists : bool
        Whether local complaints CSV exists on disk.
    clf_loaded : Any
        Trained classifier instance if loaded.
    vec_loaded : Any
        Vectorizer or tuple of vectorizers if loaded.
    dataset_records : Optional[int], optional
        Dynamically retrieved total record count for the dataset.

    Returns
    -------
    str
        Selected navigation module key.
    """
    st.sidebar.markdown('<div class="sidebar-brand">CUSTOMER COMPLAINT NLP</div>', unsafe_allow_html=True)

    modules = [
        ("LIVE DEMO", "live_demo"),
        ("DATA EXPLORER", "data_explorer"),
        ("MODEL EVALUATION", "model_evaluation"),
        ("ERROR ANALYSIS", "error_analysis"),
        ("TAXONOMY ANALYSIS", "taxonomy_analysis"),
        ("SIMILARITY RETRIEVAL", "similarity_retrieval"),
        ("CLASSIFICATION", "classification"),
        ("PREPROCESSING", "preprocessing"),
        ("SYSTEM ARCHITECTURE", "system_architecture"),
    ]

    if "active_module" not in st.session_state:
        st.session_state["active_module"] = "LIVE DEMO"

    active_module = st.session_state["active_module"]

    def _select_nav_module(label: str) -> None:
        st.session_state["active_module"] = label

    for label, mod_key in modules:
        is_active = (active_module == label)
        btn_type = "primary" if is_active else "secondary"
        st.sidebar.button(
            label,
            key=f"nav_btn_{mod_key}",
            width="stretch",
            type=btn_type,
            on_click=_select_nav_module,
            args=(label,),
        )

    # Compute status rows
    api_str = "Online (HTTP 200)" if api_online else "Offline / Rate Limited"
    api_status = "success" if api_online else "error"

    if data_csv_exists and dataset_records:
        data_str = f"complaints.csv ({dataset_records:,} records)"
        data_status = "success"
    elif data_csv_exists:
        data_str = "complaints.csv (Available)"
        data_status = "success"
    else:
        data_str = "complaints.csv Not Found"
        data_status = "error"

    if clf_loaded is not None:
        raw_name = type(clf_loaded).__name__
        model_str = "Logistic Regression" if raw_name == "LogisticRegression" else raw_name
        model_status = "success"
    else:
        model_str = "Not Loaded"
        model_status = "warning"

    if vec_loaded is not None:
        feat_str = "Combined Word + Char TF-IDF" if isinstance(vec_loaded, tuple) else "Word TF-IDF"
        rep_status = "success"
    else:
        feat_str = "Not Loaded"
        rep_status = "warning"

    render_status_card(
        api_status=api_status,
        api_str=api_str,
        data_status=data_status,
        data_str=data_str,
        model_status=model_status,
        model_str=model_str,
        rep_status=rep_status,
        rep_str=feat_str,
    )

    return st.session_state["active_module"]


# ==============================================================================
# LIVE DEMO COMPONENTS
# ==============================================================================

def render_live_demo_header() -> None:
    """Render standardized header for LIVE complaint analysis module."""
    render_page_header(
        title="Live Complaint Analysis",
        subtitle="Paste a consumer complaint to evaluate supervised classification and sparse cosine similarity retrieval in real time.",
    )


def render_example_complaint_buttons() -> None:
    """Render clickable buttons to populate the complaint input with authentic examples."""
    render_section_header(
        "Demonstration Examples",
        "Click any example to populate the narrative input without triggering immediate execution.",
    )

    def _populate_example_complaint(text: str) -> None:
        st.session_state["text_area_live_complaint"] = text
        st.session_state["live_complaint_text"] = text
        st.session_state["live_analyzed_data"] = None

    # Standard Domain Examples
    st.caption("Standard CFPB Product Grievances:")
    st.markdown('<div class="example-btn-area">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    standard_keys = [
        "Unauthorized Credit Card Payment",
        "Debt Collection Complaint",
        "Credit Report Error",
        "Mortgage Problem",
        "Student Loan Problem",
        "Bank Account Problem",
    ]

    for idx, key in enumerate(standard_keys):
        col = [c1, c2, c3][idx % 3]
        item = DEMO_COMPLAINT_EXAMPLES[key]
        with col:
            st.button(
                f"[ {item['label']} ]",
                key=f"ex_btn_{idx}",
                width="stretch",
                on_click=_populate_example_complaint,
                args=(item["text"],),
            )

    # Intentionally Ambiguous Examples for Viva / Boundary Discussion
    st.caption("Intentionally Ambiguous Grievances (Decision Boundary & Viva Discussion):")
    a1, a2, a3 = st.columns(3)
    ambig_keys = [
        "Ambiguous: Card Dispute vs Credit Bureau Reporting",
        "Ambiguous: Debt Collection vs Identity Theft Tradeline",
        "Ambiguous: Checking Overdraft vs Payday Loan ACH",
    ]

    for idx, key in enumerate(ambig_keys):
        col = [a1, a2, a3][idx % 3]
        item = DEMO_COMPLAINT_EXAMPLES[key]
        with col:
            st.button(
                f"[ {item['label']} ]",
                key=f"ex_ambig_{idx}",
                width="stretch",
                on_click=_populate_example_complaint,
                args=(item["text"],),
            )

    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_result(analyzed_data: Dict[str, Any]) -> None:
    """Render prominently formatted prediction result and top 5 categories distribution."""
    render_section_header("Prediction Result", "Supervised classification output and normalized model confidence score.")

    pcol1, pcol2 = st.columns([2, 1])
    with pcol1:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="card-label">PREDICTED CATEGORY</div>
                <div class="card-value-pred">{analyzed_data['pred_category']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with pcol2:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="card-label">MODEL CONFIDENCE SCORE</div>
                <div class="card-value-conf">{analyzed_data['confidence']:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "The confidence score reflects the Logistic Regression class probability distribution and is "
        "not a calibrated probability or individual prediction accuracy."
    )

    # Top 5 distribution
    st.markdown("##### TOP 5 PRODUCT CATEGORIES BY MODEL CONFIDENCE")
    probs = analyzed_data["probabilities"]
    classes = analyzed_data["classes"]
    top5_indices = np.argsort(probs)[::-1][:5]

    for idx in top5_indices:
        cat_label = classes[idx]
        cat_prob = float(probs[idx])
        col_lbl, col_val = st.columns([4, 1])
        with col_lbl:
            st.write(f"**{cat_label}**")
        with col_val:
            st.write(f"`{cat_prob:.2%}`")
        st.progress(min(max(cat_prob, 0.0), 1.0))


def render_pipeline_trace(analyzed_data: Dict[str, Any]) -> None:
    """Render pipeline trace and representation summary."""
    render_section_header("Pipeline Transformation Trace", "Step-by-step intermediate representations from raw complaint to prediction.")
    st.markdown(
        """
        ```text
        Raw Complaint
              ↓
        Text Preprocessing (Lowercasing, Redaction Cleaning, Tokenization, Stopwords)
              ↓
        Tokenisation / Cleaned Text
              ↓
        Combined Word + Character TF-IDF (Sparse CSR: 114,493 dimensions)
              ↓
        Logistic Regression (class_weight='balanced')
              ↓
        Predicted Category
        ```
        """
    )

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("**Raw Consumer Complaint:**")
        st.text_area("Original Raw Narrative", value=analyzed_data["raw_complaint"], height=130, disabled=True, key="disp_raw")
    with tcol2:
        st.markdown("**Tokenised & Preprocessed Text (`preprocess_text`):**")
        st.text_area("Preprocessed Narrative", value=analyzed_data["clean_text"], height=130, disabled=True, key="disp_clean")

    render_section_header("TF-IDF Feature Representation Summary", "Dimensionality and sparsity of the extracted feature vector.")
    rep_type = "Combined Word + Character TF-IDF" if analyzed_data.get("is_composite", True) else "Word TF-IDF"
    feature_dim = analyzed_data.get("feature_dim", 114493)
    active_nnz = analyzed_data.get("active_nnz", 0)

    # ROW 1: Compact metric cards
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        render_metric_card("Total Feature Dimension", f"{feature_dim:,}")
    with r1_col2:
        render_metric_card("Active Non-Zero Features", f"{active_nnz:,}")

    # ROW 2: Wide horizontal information card for Feature Representation
    st.markdown(
        f"""
        <div class="feature-rep-card">
            <div class="feature-rep-label">FEATURE REPRESENTATION</div>
            <div class="feature-rep-val">{rep_type}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Representation: Sparse CSR (`scipy.sparse.csr_matrix`). Preserves memory efficiency without dense allocation.")


def render_highest_weighted_features(w_vec: Any, c_vec: Any, clean_text: str) -> None:
    """Render table of top active n-gram features and their TF-IDF weights."""
    df_active = get_top_active_features(w_vec, c_vec, clean_text, top_n=10)
    if not df_active.empty:
        render_section_header("Highest-Weighted Active Features", "Active vocabulary n-grams extracted from this complaint narrative with their learned TF-IDF weights.")
        disp_active = df_active.rename(columns={
            "feature": "Active Feature (N-gram)",
            "type": "N-gram Type",
            "weight": "TF-IDF Weight"
        })
        disp_active["TF-IDF Weight"] = disp_active["TF-IDF Weight"].apply(lambda v: f"{v:.4f}")
        st.table(disp_active)


def render_similarity_results(
    complaint_input: str,
    clean_q: str,
    df_corpus: pd.DataFrame,
    fitted_sim_vec: Any,
    corpus_matrix: Any,
) -> None:
    """Render top similar historical CFPB complaints via sparse cosine similarity retrieval."""
    render_section_header("Top Similar Historical Complaints", "Vector-space nearest neighbor ranking over sparse historical complaint representations.")
    st.markdown(
        """
        ```text
        Complaint Word TF-IDF Vector
                  ↓
        Cosine Similarity Retrieval (against indexed historical CFPB corpus)
                  ↓
        Top Similar Historical Complaints
        ```
        """
    )
    st.caption(
        "Classification uses the combined Word + Character TF-IDF representation, while lexical similarity "
        "retrieval uses the pre-indexed Word TF-IDF historical complaint representation for sparse cosine retrieval."
    )

    if df_corpus.empty or fitted_sim_vec is None or corpus_matrix is None:
        st.info("Historical CFPB complaint corpus is not available for similarity retrieval.")
        return

    with st.spinner("Computing sparse cosine similarities against indexed CFPB complaints..."):
        sim_results = find_similar_complaints(
            query_text=complaint_input,
            vectorizer=fitted_sim_vec,
            corpus_matrix=corpus_matrix,
            df_corpus=df_corpus,
            top_k=7,
            preprocess=True
        )

    # Exclude exact self-matches
    filtered_matches = []
    for _, row in sim_results.iterrows():
        row_text = str(row.get("complaint_text", ""))
        row_score = float(row.get("similarity_score", 0.0))
        if row_score > 0.9999 and (row_text.strip() == complaint_input.strip() or preprocess_text(row_text) == clean_q):
            continue
        filtered_matches.append(row)
        if len(filtered_matches) == 5:
            break

    if not filtered_matches:
        st.info("No distinct historical complaints retrieved above similarity threshold.")
    else:
        st.markdown("#### Retrieved Historical Complaints (Ranked by Cosine Similarity)")
        for rank, match in enumerate(filtered_matches, 1):
            c_id = match.get("complaint_id", "N/A")
            c_cat = match.get("category", "General")
            c_score = float(match.get("similarity_score", 0.0))
            c_text = str(match.get("complaint_text", ""))

            with st.container():
                st.markdown(
                    f"**{rank}. Category:** `{c_cat}` | **Cosine Similarity:** `{c_score:.4f}` | **Complaint ID:** `{c_id}`"
                )
                snippet = c_text[:220].replace("\n", " ") + ("..." if len(c_text) > 220 else "")
                st.write(f"> {snippet}")
                with st.expander(f"View Full Historical Narrative (ID: {c_id})"):
                    st.write(c_text)
                st.divider()


def render_what_this_demonstrates() -> None:
    """Render concise explanation of the classical NLP pipeline demonstration."""
    render_section_header("What This Demonstrates", "Core classical NLP principles exemplified by this interactive demonstration.")
    st.markdown(
        """
        1. **Text Preprocessing:** Case normalization, regex noise filtering, CFPB redaction removal, tokenization, and stopword filtering.
        2. **TF-IDF Vector Representation:** Unigram and bigram word features combined with character subword n-grams (`char_wb`, 3–5).
        3. **Supervised Logistic Regression Classification:** Multinomial classification with class-frequency balanced loss weighting.
        4. **Cosine Similarity Retrieval:** Vector-space nearest neighbor ranking over sparse document representations.
        5. **Sparse Vector Computation:** SciPy CSR matrices to preserve memory and enable fast dot product calculations.
        6. **Classical Statistical NLP:** Interpretable feature representations and linear decision boundaries.

        **Academic Methodology Notice:**
        - **No LLMs, transformers, pretrained embeddings, or external generative APIs are used.**
        - The confidence score reflects the Logistic Regression normalized softmax distribution over 18 product classes and is not a calibrated probability or individual prediction accuracy.
        """
    )


def render_model_insights_section(
    bvsi_df: Optional[pd.DataFrame],
    tax_df: Optional[pd.DataFrame],
    dist_df: Optional[pd.DataFrame],
    per_cat_df: Optional[pd.DataFrame],
    err_df: Optional[pd.DataFrame],
) -> None:
    """Render supporting model and dataset insight graphs below the live demo."""
    st.markdown("---")
    render_section_header("Model & Dataset Insights", "Empirical benchmarks and diagnostics loaded directly from validated experimental results in results/.")

    tab_base, tab_tax, tab_dist, tab_f1, tab_pairs = st.tabs([
        "A. Controlled Baseline vs Improved",
        "B. Taxonomy Comparison",
        "C. Dataset Class Distribution",
        "D. Per-Category F1",
        "E. Top Confusion Pairs",
    ])

    with tab_base:
        st.markdown("#### Controlled Same-Split Comparison: Baseline vs. Improved Model")
        st.markdown(
            "Both models evaluated on the **identical 5,000-record holdout test set**. "
            "Controlled baseline uses Word TF-IDF without class balancing. Improved model incorporates "
            "subword character n-grams and class weight balancing."
        )
        if bvsi_df is not None:
            fig_base = create_baseline_comparison_chart(bvsi_df)
            st.plotly_chart(fig_base, width="stretch", config={"displayModeBar": True, "displaylogo": False})
            st.caption(
                "Key Finding: Class balancing and subword character n-grams elevated Macro F1 from 34.15% to 50.88% (+16.73 pp, a 49% relative gain) "
                "while elevating overall accuracy to 69.82% on the controlled holdout test set."
            )
        else:
            st.info("baseline_vs_improved.csv not found.")

    with tab_tax:
        st.markdown("#### Cross-Taxonomy Formulation Comparison")
        st.markdown(
            "Investigating whether classification difficulty stems from NLP feature representations or from "
            "CFPB administrative form revisions (2017 & 2019) that left historical label variants in the database."
        )
        if tax_df is not None:
            fig_tax = create_taxonomy_comparison_chart(tax_df)
            st.plotly_chart(fig_tax, width="stretch", config={"displayModeBar": True, "displaylogo": False})
            st.caption(
                "Key Finding: Normalizing historical synonymous categories (v1 Conservative: 11 classes; v2 Broad: 10 classes) "
                "elevates Accuracy to 81.5% - 82.3% and Macro F1 to 63.4% - 66.6%. Over 39.9% of baseline errors are "
                "intra-variant misclassifications between administratively split categories."
            )
        else:
            st.info("taxonomy_experiment.csv not found.")

    with tab_dist:
        st.markdown("#### CFPB Dataset Class Distribution (Total N = 25,000 Records)")
        st.markdown("Severe class imbalance across the 18 CFPB product verticals motivates class-frequency balancing.")
        if dist_df is not None:
            fig_dist = create_class_distribution_chart(dist_df)
            st.plotly_chart(fig_dist, width="stretch", config={"displayModeBar": True, "displaylogo": False})
            st.caption(
                "The top 3 categories (Debt collection, Credit reporting, Mortgage) comprise over 56% of all complaints, "
                "while 7 minority categories each account for less than 1% of the corpus."
            )
        else:
            st.info("class_distribution.csv not found.")

    with tab_f1:
        st.markdown("#### Per-Category F1-Score (Holdout N = 5,000 Test Records)")
        st.markdown("18-category test performance under balanced Logistic Regression and Combined Word+Char TF-IDF.")
        if per_cat_df is not None:
            fig_f1 = create_per_category_f1_chart(per_cat_df)
            st.plotly_chart(fig_f1, width="stretch", config={"displayModeBar": True, "displaylogo": False})
            st.caption(
                "Mortgage (92.2%), Student loan (85.0%), and Debt collection (83.6%) achieve the highest F1 scores, "
                "benefiting from distinct vocabulary and substantial support."
            )
        else:
            st.info("per_category_metrics.csv not found.")

    with tab_pairs:
        st.markdown("#### Top Misclassification Confusion Pairs (Actual -> Predicted)")
        st.markdown("Most frequent confusion pairs on the 5,000-record holdout test set (1,522 total errors).")
        if err_df is not None:
            fig_pairs = create_confusion_pairs_chart(err_df, top_n=10)
            st.plotly_chart(fig_pairs, width="stretch", config={"displayModeBar": True, "displaylogo": False})
            st.caption(
                "Notice that the top 2 confusion pairs (314 combined errors) occur between Credit reporting variants, "
                "and pairs 3-4 (139 combined errors) occur between Credit card variants."
            )
        else:
            st.info("error_analysis.csv not found.")
