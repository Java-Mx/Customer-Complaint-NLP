"""UI components and presentation helpers for the Customer Complaint NLP Streamlit application.

Provides clean academic styling, rectangular button navigation, live demo workflows,
and model insight visualizations without altering underlying NLP methodologies.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is on sys.path before importing from src or app
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
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
    )
except (ImportError, ModuleNotFoundError):
    from charts import (
        create_baseline_comparison_chart,
        create_taxonomy_comparison_chart,
        create_class_distribution_chart,
        create_per_category_f1_chart,
        create_confusion_pairs_chart,
    )

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


def apply_custom_styles() -> None:
    """Inject polished, presentation-ready CSS for academic NLP demonstration."""
    css = """
    <style>
    /* Clean layout and typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Sidebar title */
    .sidebar-brand {
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        padding: 0.25rem 0.25rem 0.6rem 0.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 0.75rem;
    }

    /* Sidebar navigation buttons (rectangular, rounded corners, clean padding) */
    [data-testid="stSidebar"] div.stButton > button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.80rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.48rem 0.75rem !important;
        margin-bottom: 0.25rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: all 0.15s ease-in-out !important;
        text-transform: uppercase !important;
    }

    /* Inactive sidebar buttons */
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"],
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #273549 !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
    }

    /* Active sidebar button */
    [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
        background-color: #1d4ed8 !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.3) !important;
    }

    /* Main content buttons */
    div.stButton > button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Primary analyze button */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #3b82f6 !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
    }

    /* Example complaint buttons */
    .example-btn-area div.stButton > button {
        font-size: 0.78rem !important;
        padding: 0.35rem 0.6rem !important;
        border-radius: 5px !important;
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
        text-transform: none !important;
    }
    .example-btn-area div.stButton > button:hover {
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        background-color: #26354a !important;
    }

    /* System Status Box */
    .status-container {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 0.65rem 0.8rem;
        font-size: 0.78rem;
        line-height: 1.5;
        color: #cbd5e1;
        margin-top: 0.6rem;
    }
    .status-heading {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        margin-bottom: 0.45rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.25rem;
    }
    .status-row {
        margin-bottom: 0.3rem;
        display: flex;
        align-items: baseline;
    }
    .status-row:last-child {
        margin-bottom: 0;
    }

    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .card-value-pred {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .card-value-conf {
        font-size: 1.45rem;
        font-weight: 700;
        color: #60a5fa;
        letter-spacing: -0.01em;
    }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 0.6rem 0.85rem;
    }

    /* Section banner */
    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #60a5fa;
        margin-bottom: 0.4rem;
    }
    .hero-lead {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.45;
        margin-bottom: 1.1rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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
            use_container_width=True,
            type=btn_type,
            on_click=_select_nav_module,
            args=(label,),
        )

    # Compact System Status
    st.sidebar.markdown(
        """
        <div class="status-container">
            <div class="status-heading">SYSTEM STATUS</div>
        """,
        unsafe_allow_html=True,
    )

    api_str = "Online (HTTP 200)" if api_online else "Offline / Rate Limited"
    api_mark = "✓" if api_online else "!"
    api_color = "#4ade80" if api_online else "#f87171"

    if data_csv_exists and dataset_records:
        data_str = f"complaints.csv ({dataset_records:,} records)"
    elif data_csv_exists:
        data_str = "complaints.csv (Available)"
    else:
        data_str = "complaints.csv Not Found"
    data_mark = "✓" if data_csv_exists else "✗"
    data_color = "#4ade80" if data_csv_exists else "#f87171"

    if clf_loaded is not None:
        raw_name = type(clf_loaded).__name__
        model_str = "Logistic Regression" if raw_name == "LogisticRegression" else raw_name
        model_mark = "✓"
        model_color = "#4ade80"
    else:
        model_str = "Not Loaded"
        model_mark = "!"
        model_color = "#fbbf24"

    if vec_loaded is not None:
        feat_str = "Combined Word + Character TF-IDF" if isinstance(vec_loaded, tuple) else "Word TF-IDF"
        feat_mark = "✓"
        feat_color = "#4ade80"
    else:
        feat_str = "Not Loaded"
        feat_mark = "!"
        feat_color = "#fbbf24"

    status_html = f"""
        <div class="status-row"><span style="color: {api_color}; font-weight: bold; margin-right: 6px;">{api_mark}</span><span><strong>CFPB Search API:</strong> {api_str}</span></div>
        <div class="status-row"><span style="color: {data_color}; font-weight: bold; margin-right: 6px;">{data_mark}</span><span><strong>Local Dataset:</strong> {data_str}</span></div>
        <div class="status-row"><span style="color: {model_color}; font-weight: bold; margin-right: 6px;">{model_mark}</span><span><strong>Model:</strong> {model_str}</span></div>
        <div class="status-row"><span style="color: {feat_color}; font-weight: bold; margin-right: 6px;">{feat_mark}</span><span><strong>Representation:</strong> {feat_str}</span></div>
    </div>
    """
    st.sidebar.markdown(status_html, unsafe_allow_html=True)

    return st.session_state["active_module"]


def render_live_demo_header() -> None:
    """Render top hero header and explanation for LIVE complaint analysis."""
    st.markdown(
        """
        <div class="hero-subtitle">CUSTOMER COMPLAINT SIMILARITY & CATEGORISATION</div>
        <div class="hero-title">LIVE COMPLAINT ANALYSIS</div>
        <div class="hero-lead">
            Paste a consumer complaint and see how the classical NLP pipeline categorises it and retrieves
            similar historical CFPB complaints.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_complaint_buttons() -> None:
    """Render clickable buttons to populate the complaint input with authentic examples."""
    st.markdown("##### TRY AN EXAMPLE")
    st.caption(
        "Demonstration Examples: Click any button below to populate the input box. "
        "It will not run automatically — click **Analyze Complaint** above to inspect the pipeline."
    )

    def _populate_example_complaint(text: str) -> None:
        st.session_state["text_area_live_complaint"] = text
        st.session_state["live_complaint_text"] = text
        st.session_state["live_analyzed_data"] = None

    # Standard Domain Examples
    st.markdown("**Standard CFPB Product Grievances:**")
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
                use_container_width=True,
                on_click=_populate_example_complaint,
                args=(item["text"],),
            )

    # Intentionally Ambiguous Examples for Viva / Boundary Discussion
    st.markdown("**Intentionally Ambiguous Examples (Model Boundaries & Viva Defense):**")
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
                use_container_width=True,
                on_click=_populate_example_complaint,
                args=(item["text"],),
            )

    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_result(analyzed_data: Dict[str, Any]) -> None:
    """Render prominently formatted prediction result and top 5 categories distribution."""
    st.markdown("### PREDICTION RESULT")

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
    st.markdown("### HOW THE COMPLAINT WAS PROCESSED (PIPELINE TRACE)")
    st.markdown(
        """
        ```text
        Raw Complaint
              ↓
        Text Preprocessing (Lowercasing, Redaction Cleaning, Tokenization, Stopwords)
              ↓
        Tokenisation / Cleaned Text
              ↓
        Combined Word + Character TF-IDF (Sparse CSR: 237,148 dimensions)
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

    st.markdown("#### TF-IDF Feature Representation Summary")
    rep_type = "Combined Word + Character TF-IDF" if analyzed_data["is_composite"] else "Word TF-IDF"
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.metric("Feature Representation", rep_type)
    with fcol2:
        st.metric("Total Feature Dimension", f"{analyzed_data['feature_dim']:,}")
    with fcol3:
        st.metric("Active Non-Zero Features", f"{analyzed_data['active_nnz']:,}")

    st.caption("Representation: Sparse CSR (`scipy.sparse.csr_matrix`). Preserves memory efficiency without dense allocation.")


def render_highest_weighted_features(w_vec: Any, c_vec: Any, clean_text: str) -> None:
    """Render table of top active n-gram features and their TF-IDF weights."""
    df_active = get_top_active_features(w_vec, c_vec, clean_text, top_n=10)
    if not df_active.empty:
        st.markdown("### HIGHEST-WEIGHTED ACTIVE FEATURES")
        st.caption("Active vocabulary n-grams extracted from this complaint narrative with their learned TF-IDF weights.")
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
    st.markdown("### TOP SIMILAR HISTORICAL CFPB COMPLAINTS")
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
    st.markdown("### WHAT THIS DEMONSTRATES")
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
    st.markdown("### MODEL & DATASET INSIGHTS")
    st.caption("Empirical benchmarks and diagnostics loaded directly from validated experimental results in results/.")

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
            st.pyplot(fig_base, clear_figure=True)
            plt.close(fig_base)
            st.caption(
                "Key Finding: Class balancing elevated Macro F1 from 34.15% to 50.56% (+16.41 pp, a 48% relative gain) "
                "while maintaining overall accuracy at 69.56% on the controlled holdout test set."
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
            st.pyplot(fig_tax, clear_figure=True)
            plt.close(fig_tax)
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
            st.pyplot(fig_dist, clear_figure=True)
            plt.close(fig_dist)
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
            st.pyplot(fig_f1, clear_figure=True)
            plt.close(fig_f1)
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
            st.pyplot(fig_pairs, clear_figure=True)
            plt.close(fig_pairs)
            st.caption(
                "Notice that the top 2 confusion pairs (314 combined errors) occur between Credit reporting variants, "
                "and pairs 3-4 (139 combined errors) occur between Credit card variants."
            )
        else:
            st.info("error_analysis.csv not found.")
