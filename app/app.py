"""Streamlit Web Application for Customer Complaint Similarity & Categorisation.

Provides an academic project interface to explore real consumer complaints from the
official CFPB Consumer Complaint Database and API, demonstrate text preprocessing,
TF-IDF vectorisation, cosine similarity retrieval, supervised classification with
improved classical models (Combined Word + Character TF-IDF & Class Weight Balancing),
and comprehensive statistical model evaluation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path before importing from src
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_dataset, get_complaints_data, get_dataset_summary
from src.preprocessing import clean_text, tokenize, remove_stopwords, preprocess_text, preprocess_series
from src.vectorization import (
    create_vectorizer,
    fit_transform_corpus,
    fit_transform_tfidf,
    transform_word_char,
    get_top_active_features,
)
from src.similarity import find_similar_complaints, compute_cosine_similarity
from src.classification import (
    load_classifier,
    predict_complaint_category,
    predict_category_proba,
    train_test_split_data,
    create_classifier,
    fit_classifier,
    save_classifier,
)
from src.evaluation import (
    evaluate_classifier,
    evaluate_model,
    compute_per_category_metrics,
    generate_classification_report,
    plot_confusion_matrix,
)
from src.cfpb_api import fetch_cfpb_data, test_api_connection
from app.ui_components import (
    apply_custom_styles,
    render_sidebar_navigation,
    render_live_demo_header,
    render_example_complaint_buttons,
    render_prediction_result,
    render_pipeline_trace,
    render_highest_weighted_features,
    render_similarity_results,
    render_what_this_demonstrates,
    render_model_insights_section,
)


def get_status_icon_svg(status: str) -> str:
    """Return inline SVG for clean visual status indication without emojis."""
    if status == "check":
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
            'fill="none" stroke="#2e7d32" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;"><polyline points="20 6 9 17 4 12"/></svg>'
        )
    elif status == "cross":
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
            'fill="none" stroke="#c62828" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;"><line x1="18" y1="6" x2="6" y2="18"/>'
            '<line x1="6" y1="6" x2="18" y2="18"/></svg>'
        )
    elif status == "warning":
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
            'fill="none" stroke="#f57c00" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
            '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        )
    elif status == "info":
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
            'fill="none" stroke="#0288d1" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;"><circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
        )
    return ""


# Page Configuration
st.set_page_config(
    page_title="Customer Complaint Similarity & Categorisation",
    page_icon=None,
    layout="wide"
)

# Inject custom CSS styles
apply_custom_styles()


# ----------------------------------------------------------------------
# Cached Resource Loaders
# ----------------------------------------------------------------------

@st.cache_data
def load_local_complaints(nrows: int = 1000) -> pd.DataFrame:
    """Load real CFPB complaints from the local dataset."""
    data_path = ROOT_DIR / "data" / "complaints.csv"
    if data_path.exists():
        df = load_dataset(data_path, nrows=nrows, drop_invalid=True)
        return df
    return pd.DataFrame(columns=["complaint_id", "category", "text"])


@st.cache_data
def build_indexed_corpus(nrows: int = 300):
    """Index a real complaint corpus for live cosine similarity search."""
    df = load_local_complaints(nrows=nrows)
    if df.empty:
        return df, None, None

    clean_narratives = preprocess_series(df["text"])
    df["clean_text"] = clean_narratives
    vec = create_vectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False)
    fitted_vec, matrix = fit_transform_corpus(df["clean_text"], vectorizer=vec)
    return df, fitted_vec, matrix


@st.cache_resource
def load_trained_model():
    """Load the trained classification model and vectorizer(s)."""
    model_path = ROOT_DIR / "models" / "complaint_classifier.joblib"
    w_vec_path = ROOT_DIR / "models" / "tfidf_vectorizer.joblib"
    c_vec_path = ROOT_DIR / "models" / "char_vectorizer.joblib"

    if model_path.exists() and w_vec_path.exists():
        try:
            clf = load_classifier(model_path)
            w_vec = joblib.load(w_vec_path)
            c_vec = joblib.load(c_vec_path) if c_vec_path.exists() else None
            vec = (w_vec, c_vec) if c_vec is not None else w_vec
            return clf, vec
        except Exception as e:
            logger.warning("Failed to deserialize model artifacts: %s", e)
    else:
        logger.info("Model artifacts not present at %s or %s", model_path, w_vec_path)

    return None, None


@st.cache_data
def load_evaluation_summary():
    """Load cached final evaluation metrics and experiment comparison."""
    metrics_path = ROOT_DIR / "results" / "final_evaluation_metrics.json"
    comp_path = ROOT_DIR / "results" / "model_comparison.csv"

    metrics_data = None
    comparison_df = None

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    if comp_path.exists():
        comparison_df = pd.read_csv(comp_path)

    return metrics_data, comparison_df


@st.cache_data
def load_error_analysis_data():
    """Load precomputed error analysis artifacts from results/."""
    json_path = ROOT_DIR / "results" / "error_analysis_data.json"
    bvsi_path = ROOT_DIR / "results" / "baseline_vs_improved.csv"
    per_cat_path = ROOT_DIR / "results" / "per_category_metrics.csv"
    err_pairs_path = ROOT_DIR / "results" / "error_analysis.csv"

    ea_json = None
    bvsi_df = None
    per_cat_df = None
    err_pairs_df = None

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            ea_json = json.load(f)
    if bvsi_path.exists():
        bvsi_df = pd.read_csv(bvsi_path)
    if per_cat_path.exists():
        per_cat_df = pd.read_csv(per_cat_path)
    if err_pairs_path.exists():
        err_pairs_df = pd.read_csv(err_pairs_path)

    return ea_json, bvsi_df, per_cat_df, err_pairs_df


@st.cache_data
def load_taxonomy_analysis_data():
    """Load precomputed taxonomy analysis artifacts from results/ and config/."""
    json_path = ROOT_DIR / "results" / "taxonomy_experiment_data.json"
    comp_csv_path = ROOT_DIR / "results" / "taxonomy_experiment.csv"
    audit_json_path = ROOT_DIR / "results" / "taxonomy_audit.json"
    audit_csv_path = ROOT_DIR / "results" / "taxonomy_audit.csv"
    cfg_v1_path = ROOT_DIR / "config" / "taxonomy_v1_conservative.json"
    cfg_v2_path = ROOT_DIR / "config" / "taxonomy_v2_broad.json"

    tax_json = None
    comp_df = None
    audit_json = None
    audit_df = None
    cfg_v1 = None
    cfg_v2 = None

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            tax_json = json.load(f)
    if comp_csv_path.exists():
        comp_df = pd.read_csv(comp_csv_path)
    if audit_json_path.exists():
        with open(audit_json_path, "r", encoding="utf-8") as f:
            audit_json = json.load(f)
    if audit_csv_path.exists():
        audit_df = pd.read_csv(audit_csv_path)
    if cfg_v1_path.exists():
        with open(cfg_v1_path, "r", encoding="utf-8") as f:
            cfg_v1 = json.load(f)
    if cfg_v2_path.exists():
        with open(cfg_v2_path, "r", encoding="utf-8") as f:
            cfg_v2 = json.load(f)

    return tax_json, comp_df, audit_json, audit_df, cfg_v1, cfg_v2


@st.cache_data
def load_class_distribution_data() -> Optional[pd.DataFrame]:
    """Load precomputed class distribution artifact from results/."""
    path = ROOT_DIR / "results" / "class_distribution.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ----------------------------------------------------------------------
# Sidebar Navigation & System Status
# ----------------------------------------------------------------------

@st.cache_data(ttl=60)
def check_live_api_status() -> bool:
    """Check CFPB API connectivity with caching to prevent latency on reruns."""
    try:
        return test_api_connection(timeout=3.0)
    except Exception as exc:
        logger.warning("CFPB API connectivity check failed: %s", exc)
        return False


@st.cache_data
def get_dataset_record_count() -> Optional[int]:
    """Retrieve CFPB local dataset record count dynamically from real project artifacts."""
    dist = load_class_distribution_data()
    if dist is not None and "total_count" in dist.columns:
        return int(dist["total_count"].sum())
    ea_json, _, _, _ = load_error_analysis_data()
    if ea_json and "analysis_metadata" in ea_json:
        return int(ea_json["analysis_metadata"].get("total_records", 25000))
    csv_path = ROOT_DIR / "data" / "complaints.csv"
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    return None


api_online = check_live_api_status()
data_csv = ROOT_DIR / "data" / "complaints.csv"
clf_loaded, vec_loaded = load_trained_model()
dataset_records = get_dataset_record_count()

selected_section = render_sidebar_navigation(
    api_online=api_online,
    data_csv_exists=data_csv.exists(),
    clf_loaded=clf_loaded,
    vec_loaded=vec_loaded,
    dataset_records=dataset_records,
)


# ----------------------------------------------------------------------
# MODULE 1: LIVE COMPLAINT ANALYSIS (HERO MODULE - DEFAULT)
# ----------------------------------------------------------------------

if selected_section == "LIVE DEMO":
    render_live_demo_header()

    if "text_area_live_complaint" not in st.session_state:
        st.session_state["text_area_live_complaint"] = ""
    if "live_complaint_text" not in st.session_state:
        st.session_state["live_complaint_text"] = ""

    def _clear_live_complaint() -> None:
        st.session_state["text_area_live_complaint"] = ""
        st.session_state["live_complaint_text"] = ""
        st.session_state["live_analyzed_data"] = None

    st.markdown("##### ENTER A CUSTOMER COMPLAINT")
    complaint_input = st.text_area(
        "Consumer Complaint Narrative",
        key="text_area_live_complaint",
        height=160,
        help="Paste a consumer complaint narrative to test supervised classification and cosine retrieval.",
        placeholder="Type or paste customer complaint text here...",
        label_visibility="collapsed",
    )
    st.session_state["live_complaint_text"] = complaint_input

    # Demonstration Examples
    render_example_complaint_buttons()

    # Primary and secondary action buttons
    col_btn1, col_btn2, _ = st.columns([2, 1, 4])
    with col_btn1:
        analyze_clicked = st.button("ANALYZE COMPLAINT", type="primary", use_container_width=True)
    with col_btn2:
        st.button(
            "CLEAR",
            type="secondary",
            use_container_width=True,
            on_click=_clear_live_complaint,
        )

    if analyze_clicked:
        if not complaint_input or not complaint_input.strip():
            st.warning("Please enter a customer complaint narrative before analyzing.")
            st.session_state["live_analyzed_data"] = None
        else:
            clf_model, vec_model = load_trained_model()
            if clf_model is None or vec_model is None:
                st.warning(
                    "The trained classification model is not available. Please ensure "
                    "serialized model artifacts exist in the models/ directory."
                )
                st.session_state["live_analyzed_data"] = None
            else:
                clean_q = preprocess_text(complaint_input)
                if not clean_q.strip():
                    st.warning("The complaint contains only punctuation or stopwords and produced an empty preprocessed text.")
                    st.session_state["live_analyzed_data"] = None
                else:
                    if isinstance(vec_model, tuple):
                        w_vec, c_vec = vec_model[0], vec_model[1]
                        X_q = transform_word_char(w_vec, c_vec, [clean_q])
                    else:
                        w_vec, c_vec = vec_model, None
                        X_q = w_vec.transform([clean_q])

                    pred_cat, conf = predict_complaint_category(
                        model=clf_model,
                        vectorizer=vec_model,
                        narrative=complaint_input,
                        preprocess=True
                    )
                    probabilities = predict_category_proba(clf_model, X_q)[0]

                    st.session_state["live_analyzed_data"] = {
                        "raw_complaint": complaint_input,
                        "clean_text": clean_q,
                        "pred_category": pred_cat,
                        "confidence": conf,
                        "probabilities": probabilities,
                        "classes": clf_model.classes_,
                        "feature_dim": X_q.shape[1],
                        "active_nnz": X_q.nnz,
                        "is_composite": c_vec is not None,
                    }

    # Render prediction results if available
    analyzed_data = st.session_state.get("live_analyzed_data")
    if analyzed_data is not None and analyzed_data.get("raw_complaint") == complaint_input:
        clf_model, vec_model = load_trained_model()
        w_vec = vec_model[0] if isinstance(vec_model, tuple) else vec_model
        c_vec = vec_model[1] if isinstance(vec_model, tuple) else None

        st.markdown("---")
        render_prediction_result(analyzed_data)

        st.markdown("---")
        render_pipeline_trace(analyzed_data)

        st.markdown("---")
        render_highest_weighted_features(w_vec, c_vec, analyzed_data["clean_text"])

        st.markdown("---")
        df_corpus, fitted_sim_vec, corpus_matrix = build_indexed_corpus(nrows=500)
        render_similarity_results(
            complaint_input=analyzed_data["raw_complaint"],
            clean_q=analyzed_data["clean_text"],
            df_corpus=df_corpus,
            fitted_sim_vec=fitted_sim_vec,
            corpus_matrix=corpus_matrix,
        )

        st.markdown("---")
        render_what_this_demonstrates()

    # Empirical Model & Dataset Insights Section (Below Live Demo)
    _, bvsi_df, per_cat_df, err_pairs_df = load_error_analysis_data()
    _, tax_comp_df, _, _, _, _ = load_taxonomy_analysis_data()
    dist_df = load_class_distribution_data()

    render_model_insights_section(
        bvsi_df=bvsi_df,
        tax_df=tax_comp_df,
        dist_df=dist_df,
        per_cat_df=per_cat_df,
        err_df=err_pairs_df,
    )


# ----------------------------------------------------------------------
# MODULE 2: CFPB LIVE API & DATA EXPLORER
# ----------------------------------------------------------------------

elif selected_section == "DATA EXPLORER":
    st.subheader("Official CFPB API Integration & Data Explorer")
    st.markdown(
        """
        Query real consumer financial complaints in real time directly from the official 
        [CFPB Consumer Complaint Database API v1](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/).
        """
    )

    data_source_mode = st.radio(
        "Select Data Explorer Mode:",
        ["Query Live CFPB Search API", "Explore Local CFPB Dataset (CSV)"],
        horizontal=True
    )

    if data_source_mode == "Query Live CFPB Search API":
        st.markdown("#### Live CFPB Search API Query")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            search_query = st.text_input("Search Keyword / Term:", value="mortgage", placeholder="e.g. overdraft, credit card, foreclosure")
        with col2:
            product_filter = st.selectbox(
                "Filter by Product (Optional):",
                [
                    "All Products",
                    "Mortgage",
                    "Credit card",
                    "Credit reporting, credit repair services, or other personal consumer reports",
                    "Debt collection",
                    "Checking or savings account",
                    "Student loan",
                    "Vehicle loan or lease",
                    "Payday loan, title loan, or personal loan"
                ]
            )
        with col3:
            max_rec = st.slider("Records to Fetch:", min_value=5, max_value=30, value=10, step=5)

        prod_arg = "" if product_filter == "All Products" else product_filter

        if st.button("Fetch Live Complaints from CFPB API", type="primary"):
            with st.spinner("Connecting to official CFPB API endpoint..."):
                try:
                    df_live = fetch_cfpb_data(
                        search_term=search_query,
                        product=prod_arg,
                        max_records=max_rec,
                        timeout=12
                    )
                    st.session_state["live_cfpb_df"] = df_live
                except Exception as e:
                    st.error(f"Error communicating with CFPB API: {e}")

        if "live_cfpb_df" in st.session_state:
            df_live = st.session_state["live_cfpb_df"]
            if df_live.empty:
                st.markdown(f"{get_status_icon_svg('warning')} **No records returned matching the query criteria.**", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"{get_status_icon_svg('check')} **Successfully retrieved {len(df_live)} live complaint records from CFPB Search API.**",
                    unsafe_allow_html=True,
                )

                display_cols = [c for c in ["complaint_id", "category", "company", "date_received", "state", "issue"] if c in df_live.columns]
                st.dataframe(df_live[display_cols], use_container_width=True)

                st.markdown("#### Inspect Live Record Details")
                comp_ids = df_live["complaint_id"].tolist()
                if comp_ids:
                    selected_id = st.selectbox("Select Complaint ID to View:", comp_ids)
                    matched_rows = df_live[df_live["complaint_id"] == selected_id]
                    if not matched_rows.empty:
                        sel_row = matched_rows.iloc[0]

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Product Category", sel_row.get("category", "N/A"))
                        with c2:
                            st.metric("Company", sel_row.get("company", "N/A"))
                        with c3:
                            st.metric("Date Received", str(sel_row.get("date_received", "N/A"))[:10])

                        narrative_text = str(sel_row.get("text", "")).strip()
                        if narrative_text and narrative_text != "nan":
                            st.markdown("**Consumer Complaint Narrative:**")
                            st.text_area("", value=narrative_text, height=140, disabled=True)
                        else:
                            st.info(
                                "Note: Consumer narrative is not public for this recent record. Under CFPB publication policies, "
                                "newer complaint narratives undergo FOIA redaction review before public release. "
                                "Categorical metadata (Product, Company, Date, State, Issue) is fully available."
                            )

    else:
        st.markdown("#### Local CFPB Dataset Overview")
        sample_df = load_local_complaints(nrows=500)
        if sample_df.empty:
            st.error("No local dataset found at data/complaints.csv.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Sampled Local Records", len(sample_df))
            with c2:
                st.metric("Unique Product Categories", sample_df["category"].nunique())
            with c3:
                st.metric("Missing Complaint Texts", sample_df["text"].isna().sum())

            st.dataframe(
                sample_df[["complaint_id", "category", "text"]].head(50),
                use_container_width=True
            )


# ----------------------------------------------------------------------
# MODULE 3: MODEL EVALUATION & DIAGNOSTICS
# ----------------------------------------------------------------------

elif selected_section == "MODEL EVALUATION":
    st.subheader("Model Evaluation & Systematic Improvements")
    st.markdown(
        """
        Rigorous comparative evaluation between the **Initial Baseline Model** ($N=600$ test split) and the 
        **Improved Final Model** ($N=5,000$ test split, Combined Word+Character TF-IDF, Class Weight Balancing).
        - **Data Leakage Safeguard**: 80/20 train/test split executed strictly prior to vectorization.
        - **Model Selection Standard**: Selected using Validation Macro F1 across 30+ experimental configurations.
        """
    )

    metrics_data, comparison_df = load_evaluation_summary()

    if metrics_data is None:
        st.warning("Evaluation artifacts not found. Please run scripts/run_experiments.py to generate benchmarks.")
    else:
        base = metrics_data["baseline"]
        improved = metrics_data["final_model"]

        # Comparison KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_acc = improved["accuracy"] - base["accuracy"]
            st.metric("Overall Accuracy", f"{improved['accuracy']:.2%}", delta=f"{delta_acc:+.2%}")
        with col2:
            delta_mf1 = improved["macro_f1"] - base["macro_f1"]
            st.metric("Macro F1-Score", f"{improved['macro_f1']:.4f}", delta=f"{delta_mf1:+.4f}")
        with col3:
            delta_wf1 = improved["weighted_f1"] - base["weighted_f1"]
            st.metric("Weighted F1-Score", f"{improved['weighted_f1']:.4f}", delta=f"{delta_wf1:+.4f}")
        with col4:
            st.metric("Test Partition Size", f"{improved['test_samples']:,} samples", delta=f"+{improved['test_samples'] - base['samples']:,}")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Macro Precision", f"{improved['macro_precision']:.4f}", delta=f"{improved['macro_precision'] - base['macro_precision']:+.4f}")
        with col6:
            st.metric("Macro Recall", f"{improved['macro_recall']:.4f}", delta=f"{improved['macro_recall'] - base['macro_recall']:+.4f}")
        with col7:
            st.metric("Weighted Precision", f"{improved['weighted_precision']:.4f}", delta=f"{improved['weighted_precision'] - base['weighted_precision']:+.4f}")
        with col8:
            st.metric("Total Vocabulary Features", f"{improved['total_features']:,}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Baseline vs. Improved Comparison", "Confusion Matrix Heatmap", "Experiment Grid (30 Runs)"])

        with tab1:
            st.markdown("#### Baseline vs. Improved Model Benchmark Comparison")
            comp_table = [
                {"Metric": "Overall Accuracy", "Initial Baseline": f"{base['accuracy']:.2%}", "Improved Final Model": f"{improved['accuracy']:.2%}", "Absolute Improvement": f"{delta_acc:+.2%}"},
                {"Metric": "Macro F1-Score", "Initial Baseline": f"{base['macro_f1']:.4f}", "Improved Final Model": f"{improved['macro_f1']:.4f}", "Absolute Improvement": f"{delta_mf1:+.4f} (More than doubled)"},
                {"Metric": "Weighted F1-Score", "Initial Baseline": f"{base['weighted_f1']:.4f}", "Improved Final Model": f"{improved['weighted_f1']:.4f}", "Absolute Improvement": f"{delta_wf1:+.4f}"},
                {"Metric": "Macro Precision", "Initial Baseline": f"{base['macro_precision']:.4f}", "Improved Final Model": f"{improved['macro_precision']:.4f}", "Absolute Improvement": f"{improved['macro_precision'] - base['macro_precision']:+.4f}"},
                {"Metric": "Macro Recall", "Initial Baseline": f"{base['macro_recall']:.4f}", "Improved Final Model": f"{improved['macro_recall']:.4f}", "Absolute Improvement": f"{improved['macro_recall'] - base['macro_recall']:+.4f}"},
                {"Metric": "Weighted Precision", "Initial Baseline": f"{base['weighted_precision']:.4f}", "Improved Final Model": f"{improved['weighted_precision']:.4f}", "Absolute Improvement": f"{improved['weighted_precision'] - base['weighted_precision']:+.4f}"},
                {"Metric": "Test Samples", "Initial Baseline": f"{base['samples']:,}", "Improved Final Model": f"{improved['test_samples']:,}", "Absolute Improvement": f"+{improved['test_samples'] - base['samples']:,} samples"},
                {"Metric": "Feature Representation", "Initial Baseline": "Word TF-IDF (1,2)", "Improved Final Model": "Combined Word(1,2) + Char(3,5)", "Absolute Improvement": "Subword granularity"},
                {"Metric": "Class Imbalance Strategy", "Initial Baseline": "None (Standard)", "Improved Final Model": "class_weight='balanced'", "Absolute Improvement": "Minority classes boosted"},
            ]
            st.table(pd.DataFrame(comp_table))

        with tab2:
            st.markdown("#### Multi-Class Confusion Matrix (N = 5,000 Test Records)")
            cm_img_path = ROOT_DIR / "results" / "confusion_matrix.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), caption="Holdout Test Set Confusion Matrix (N=5,000 Unseen Complaints)", use_container_width=True)
            else:
                st.info("Confusion matrix plot file not found.")

        with tab3:
            st.markdown("#### Full Systematic Experiment Comparison Table")
            st.markdown("Results logged across 30+ validation configurations using identical training/validation splits:")
            if comparison_df is not None:
                st.dataframe(
                    comparison_df.style.format({
                        "Val Accuracy": "{:.4f}",
                        "Val Macro F1": "{:.4f}",
                        "Val Weighted F1": "{:.4f}",
                        "Val Macro Prec": "{:.4f}",
                        "Val Macro Rec": "{:.4f}",
                        "Fit Time (s)": "{:.2f}"
                    }),
                    use_container_width=True
                )
            else:
                st.info("results/model_comparison.csv not found.")


# ----------------------------------------------------------------------
# MODULE 4: ERROR ANALYSIS
# ----------------------------------------------------------------------

elif selected_section == "ERROR ANALYSIS":
    st.subheader("Classification Error Analysis")
    st.markdown(
        """
        Comprehensive diagnostic analysis of the **final trained model** evaluated on the untouched
        **5,000-record holdout test set**. All artifacts are precomputed — the model is **not** modified.

        > **Note:** This analysis is diagnostic only. The classifier, vectorizers, and train/test splits
        > are unchanged. Results are loaded from pre-generated CSV and JSON files.
        """
    )

    ea_json, bvsi_df, per_cat_df, err_pairs_df = load_error_analysis_data()

    if ea_json is None:
        st.warning(
            "Error analysis artifacts not found. "
            "Run `python scripts/generate_error_analysis.py` to generate them."
        )
    else:
        tab_overview, tab_baseline, tab_pairs, tab_percat, tab_confidence, tab_examples = st.tabs([
            "Overall Performance",
            "Controlled Baseline vs. Improved",
            "Top Confusion Pairs",
            "Per-Category Metrics",
            "Confidence Analysis",
            "Error Examples",
        ])

        with tab_overview:
            st.markdown("#### Final Model — Holdout Test Set Performance (N = 5,000)")
            m = ea_json["improved_metrics"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Accuracy", f"{m['accuracy']:.2%}")
                st.metric("Macro Precision", f"{m['macro_precision']:.4f}")
            with col2:
                st.metric("Macro F1-Score", f"{m['macro_f1']:.4f}")
                st.metric("Macro Recall", f"{m['macro_recall']:.4f}")
            with col3:
                st.metric("Weighted F1-Score", f"{m['weighted_f1']:.4f}")
                st.metric("Weighted Precision", f"{m['weighted_precision']:.4f}")

            meta = ea_json["analysis_metadata"]
            st.markdown("---")
            st.markdown("#### Dataset & Split Configuration")
            split_info = {
                "Partition": ["Full Dataset", "Training Pool", "Train Subset", "Validation Subset", "Holdout Test Set"],
                "Records": [
                    f"{meta['total_records']:,}",
                    f"{meta['training_pool']:,}",
                    f"{meta['train_subset']:,}",
                    f"{meta['val_subset']:,}",
                    f"{meta['test_records']:,}",
                ],
                "Share": ["100%", "80%", "64%", "16%", "20%"],
            }
            st.table(pd.DataFrame(split_info))

            cfg = ea_json["model_config"]
            st.markdown("#### Final Model Configuration")
            cfg_rows = {
                "Parameter": [
                    "Classifier", "Solver", "C (regularization)", "class_weight",
                    "max_iter", "Word n-gram range", "Char n-gram range",
                    "Char analyzer", "Total feature dimensions",
                ],
                "Value": [
                    cfg["type"], cfg["solver"], cfg["C"], cfg["class_weight"],
                    cfg["max_iter"], str(cfg["word_ngram_range"]),
                    str(cfg["char_ngram_range"]), cfg["char_analyzer"],
                    f"{cfg['total_features']:,}",
                ],
            }
            st.table(pd.DataFrame(cfg_rows))

        with tab_baseline:
            st.markdown("#### Controlled Same-Split Comparison: Baseline vs. Improved Model")
            st.markdown(
                """
                Both models are evaluated on the **identical 5,000-record holdout test set**.
                The baseline uses the original Word TF-IDF + no class weighting configuration.
                Only this comparison is a methodologically valid apples-to-apples contrast.
                """
            )
            if bvsi_df is not None:
                styled_bvsi = bvsi_df.copy()
                styled_bvsi.columns = [
                    "Metric",
                    "Baseline (Word TF-IDF, no balancing)",
                    "Improved (Combined TF-IDF, balanced)",
                    "Absolute Diff (pp)",
                    "Relative Change (%)",
                ]
                st.dataframe(
                    styled_bvsi.style.format({
                        "Baseline (Word TF-IDF, no balancing)": "{:.4f}",
                        "Improved (Combined TF-IDF, balanced)": "{:.4f}",
                        "Absolute Diff (pp)": "{:+.4f}",
                        "Relative Change (%)": "{:+.2f}%",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("baseline_vs_improved.csv not found.")

        with tab_pairs:
            st.markdown("#### Top Confusion Pairs (Actual → Predicted)")
            top_pairs = pd.DataFrame(ea_json["top20_confusion_pairs"])
            if not top_pairs.empty:
                top_pairs.columns = [
                    "Actual Category", "Predicted Category", "Error Count", "% of Actual Class"
                ]
                st.dataframe(
                    top_pairs.style.format({
                        "Error Count": "{:,}",
                        "% of Actual Class": "{:.2f}%",
                    }).background_gradient(subset=["Error Count"], cmap="Reds"),
                    use_container_width=True,
                )

            st.markdown("---")
            st.markdown("#### 18 × 18 Confusion Matrix Heatmap")
            cm_img = ROOT_DIR / "results" / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix — Final Model on N=5,000 Test Set",
                         use_container_width=True)
            else:
                st.info("Confusion matrix image not found at results/confusion_matrix.png.")

        with tab_percat:
            st.markdown("#### Per-Category Performance on the 5,000-Record Test Set")
            if per_cat_df is not None:
                display_df = per_cat_df[[
                    "category", "support", "correct", "incorrect",
                    "precision", "recall", "f1",
                    "primary_confusion_category", "primary_confusion_count",
                ]].copy()
                display_df.columns = [
                    "Category", "Support", "Correct", "Incorrect",
                    "Precision", "Recall", "F1",
                    "Primary Confusion Target", "Confusion Count",
                ]
                st.dataframe(
                    display_df.style.format({
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1": "{:.4f}",
                    }).background_gradient(subset=["F1"], cmap="RdYlGn"),
                    use_container_width=True,
                )

                st.markdown("---")
                st.markdown("#### Class Distribution Across Splits vs. Test Recall & F1")
                dist_data = pd.DataFrame(ea_json["class_distribution"]["by_category"])
                corr_r = ea_json["class_distribution"]["corr_test_support_vs_recall"]
                corr_f = ea_json["class_distribution"]["corr_test_support_vs_f1"]
                st.info(
                    f"Pearson correlation — test support vs recall: **{corr_r:.3f}** | "
                    f"test support vs F1: **{corr_f:.3f}**"
                )
                st.dataframe(dist_data, use_container_width=True)
            else:
                st.info("per_category_metrics.csv not found.")

        with tab_confidence:
            st.markdown("#### Prediction Probability Analysis (LogisticRegression predict_proba)")
            st.markdown(
                """
                The table below summarises the **predicted class probability** distribution
                for correctly classified and incorrectly classified predictions separately.
                """
            )
            conf = ea_json.get("confidence_analysis", {})
            if conf:
                correct_pred = conf.get("correct", {}).get("predicted_class_prob", {})
                incorrect_pred = conf.get("incorrect", {}).get("predicted_class_prob", {})

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High-Confidence Errors (≥ 0.70)", str(conf.get("high_confidence_errors", "N/A")))
                with col2:
                    st.metric("Low-Confidence Errors (< 0.40)", str(conf.get("low_confidence_errors", "N/A")))
                with col3:
                    total_incorrect = incorrect_pred.get("count", 0)
                    st.metric("Total Incorrect Predictions", f"{total_incorrect:,}")

                if correct_pred and incorrect_pred:
                    stats_table = {
                        "Statistic": ["Count", "Mean Probability", "Median", "Q1 (25th pct)",
                                      "Q3 (75th pct)", "Min", "Max",
                                      "% High (≥ 0.70)", "% Medium (0.40–0.69)", "% Low (< 0.40)"],
                        "Correct Predictions": [
                            f"{correct_pred['count']:,}",
                            f"{correct_pred['mean']:.4f}",
                            f"{correct_pred['median']:.4f}",
                            f"{correct_pred['q1']:.4f}",
                            f"{correct_pred['q3']:.4f}",
                            f"{correct_pred['min']:.4f}",
                            f"{correct_pred['max']:.4f}",
                            f"{correct_pred['pct_high']:.1f}%",
                            f"{correct_pred['pct_medium']:.1f}%",
                            f"{correct_pred['pct_low']:.1f}%",
                        ],
                        "Incorrect Predictions": [
                            f"{incorrect_pred['count']:,}",
                            f"{incorrect_pred['mean']:.4f}",
                            f"{incorrect_pred['median']:.4f}",
                            f"{incorrect_pred['q1']:.4f}",
                            f"{incorrect_pred['q3']:.4f}",
                            f"{incorrect_pred['min']:.4f}",
                            f"{incorrect_pred['max']:.4f}",
                            f"{incorrect_pred['pct_high']:.1f}%",
                            f"{incorrect_pred['pct_medium']:.1f}%",
                            f"{incorrect_pred['pct_low']:.1f}%",
                        ],
                    }
                    st.table(pd.DataFrame(stats_table))
            else:
                st.info("Confidence analysis data not found in error_analysis_data.json.")

        with tab_examples:
            st.markdown("#### Representative Error Cases (Top Confusion Pairs)")
            st.markdown("Up to 5 actual test-set complaints per confusion pair. Complaint texts are truncated for readability.")
            examples = ea_json.get("error_examples", {})
            if not examples:
                st.info("Error examples not available.")
            else:
                pair_options = list(examples.keys())
                if not pair_options:
                    st.info("No confusion pair options available.")
                else:
                    selected_pair = st.selectbox("Select confusion pair:", pair_options)
                    if selected_pair and selected_pair in examples:
                        pair_examples = examples[selected_pair]
                        st.markdown(f"**{len(pair_examples)} example(s) for: `{selected_pair}`**")
                        for i, ex in enumerate(pair_examples, 1):
                            with st.expander(
                                f"Example {i} — Test Index {ex['complaint_index']}: "
                                f"Actual `{ex['actual_category']}` → Predicted `{ex['predicted_category']}`"
                            ):
                                st.markdown(f"- **Actual:** `{ex['actual_category']}`")
                                st.markdown(f"- **Predicted:** `{ex['predicted_category']}`")
                                st.markdown("**Complaint Text (truncated):**")
                                st.text_area(
                                    "", value=ex["complaint_text_truncated"],
                                    height=150, disabled=True,
                                    key=f"ex_{selected_pair}_{i}",
                                )


# ----------------------------------------------------------------------
# MODULE 5: TAXONOMY ANALYSIS
# ----------------------------------------------------------------------

elif selected_section == "TAXONOMY ANALYSIS":
    st.subheader("Taxonomy-Aware Complaint Classification Analysis")
    st.markdown(
        """
        Investigating whether CFPB complaint classification difficulty is driven by linguistic ambiguity or by 
        **documented administrative taxonomy revisions** (April 2017 and 2019) that left historical label variants in the database.

        > **Academic Objective:** This milestone tests task formulation rather than tuning for maximum F1.
        > The original 18-category model remains the primary reference. Two deterministic taxonomies 
        > were pre-defined before evaluation based on official CFPB sources:
        > - **v1 Conservative (11 Categories):** Consolidates only CFPB-documented renames/mergers; keeps `Consumer Loan` separate.
        > - **v2 Broad (10 Categories):** Additionally merges `Consumer Loan` into the consumer/small-dollar loan group.
        """
    )

    tax_json, comp_df, audit_json, audit_df, cfg_v1, cfg_v2 = load_taxonomy_analysis_data()

    if tax_json is None or comp_df is None:
        st.warning("Taxonomy experiment artifacts not found. Please run `python scripts/run_taxonomy_experiment.py`.")
    else:
        ref_m = tax_json["models"]["reference_18"]["metrics"]
        v1_m = tax_json["models"]["v1_conservative"]["metrics"]
        v2_m = tax_json["models"]["v2_broad"]["metrics"]
        audit_counts = tax_json.get("audit_asserted_errors", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Ref. (18 Cats) Accuracy", f"{ref_m['accuracy']:.2%}")
            st.caption("Macro F1: " + f"{ref_m['macro_f1']:.4f}")
        with c2:
            st.metric("v1 (11 Cats) Accuracy", f"{v1_m['accuracy']:.2%}", delta=f"{v1_m['accuracy'] - ref_m['accuracy']:+.2%}")
            st.caption(f"Macro F1: {v1_m['macro_f1']:.4f} ({v1_m['macro_f1'] - ref_m['macro_f1']:+.4f})")
        with c3:
            st.metric("v2 (10 Cats) Accuracy", f"{v2_m['accuracy']:.2%}", delta=f"{v2_m['accuracy'] - ref_m['accuracy']:+.2%}")
            st.caption(f"Macro F1: {v2_m['macro_f1']:.4f} ({v2_m['macro_f1'] - ref_m['macro_f1']:+.4f})")
        with c4:
            st.metric("Intra-Group Errors", f"{audit_counts.get('conservative_intra_errors', 608)} / {audit_counts.get('total_test_errors', 1522)}")
            st.caption("39.9% of all test errors are intra-variant")

        st.markdown("---")

        tab_comp, tab_mapping, tab_loss, tab_cross, tab_percat = st.tabs([
            "Cross-Taxonomy Benchmarks",
            "Taxonomy Mappings & CFPB Evidence",
            "Information Loss Accounting",
            "Error Elimination Dissection",
            "Per-Category Normalized Performance"
        ])

        with tab_comp:
            st.markdown("#### Comparative Experimental Results Across Formulations")
            st.markdown(
                "All models trained using the identical 20,000-sample pool (16k train / 4k val) and evaluated on the "
                "identical 5,000-sample untouched holdout test set using Word+Char TF-IDF (237,148 features) and Logistic Regression."
            )
            st.dataframe(comp_df, use_container_width=True)
            st.info(
                "**Key Observation:** Both normalized formulations produce higher Accuracy (~81-82%) and Macro F1 (~68-73%). "
                "Crucially, this is **not** evidence that the normalized models are inherently superior classifiers; rather, "
                "it demonstrates that ~40% of baseline errors stemmed from requiring the model to separate historically synonymous labels."
            )

        with tab_mapping:
            st.markdown("#### Documented Taxonomy Formulations & CFPB Rationale")
            tax_choice = st.radio("Select Taxonomy Variant:", ["v1 Conservative (11 Categories)", "v2 Broad (10 Categories)"], horizontal=True)
            chosen_cfg = cfg_v1 if "v1" in tax_choice else cfg_v2

            if chosen_cfg and "mapping" in chosen_cfg:
                st.markdown(f"**Description:** {chosen_cfg.get('description', '')}")
                mapping_rows = []
                for orig, entry in chosen_cfg["mapping"].items():
                    mapping_rows.append({
                        "Original CFPB Category": orig,
                        "Normalized Target Category": entry["normalized_category"],
                        "Evidence Type": entry["evidence_type"],
                        "CFPB Documentation Source": entry["cfpb_source"],
                        "Historical Context / Rationale": entry["evidence_description"]
                    })
                st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True)

        with tab_loss:
            st.markdown("#### Information Loss Analysis")
            loss_choice = st.radio("Select Variant for Information Loss:", ["v1 Conservative (11 Categories)", "v2 Broad (10 Categories)"], horizontal=True, key="loss_radio")
            active_loss = tax_json["models"]["v1_conservative"]["information_loss"] if "v1" in loss_choice else tax_json["models"]["v2_broad"]["information_loss"]

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Records Affected", f"{active_loss['total_records_affected']:,}")
            with col_b:
                st.metric("Dataset Share Affected", f"{active_loss['percentage_dataset_affected']}%")
            with col_c:
                st.metric("Original Categories Merged", f"{active_loss['num_labels_merged']} -> {len(active_loss['merged_groups'])}")

            loss_rows = []
            cfg_active = cfg_v1 if "v1" in loss_choice else cfg_v2
            info_dict = cfg_active.get("information_loss", {}) if cfg_active else {}
            for grp, details in active_loss["merged_groups"].items():
                loss_rows.append({
                    "Normalized Target": grp,
                    "Merged Original Labels": ", ".join(details["original_categories"]),
                    "Records Remapped": f"{details['records_affected']:,} ({details['percentage_of_dataset']}%)",
                    "Specific Information Lost": info_dict.get(grp, "N/A")
                })
            st.dataframe(pd.DataFrame(loss_rows), use_container_width=True)

        with tab_cross:
            st.markdown("#### Dissecting Error Elimination: Task Collapse vs. Classifier Generalization")
            dissect_v1 = tax_json["models"]["v1_conservative"]["cross_task_error_dissection"]
            dissect_v2 = tax_json["models"]["v2_broad"]["cross_task_error_dissection"]

            dissect_df = pd.DataFrame([
                {
                    "Metric": "Total Errors on Original 18-Category Task",
                    "v1 Conservative (11 Cats)": f"{dissect_v1['original_errors']:,}",
                    "v2 Broad (10 Cats)": f"{dissect_v2['original_errors']:,}"
                },
                {
                    "Metric": "Errors Eliminated Mechanically by Label Collapse",
                    "v1 Conservative (11 Cats)": f"{dissect_v1['errors_eliminated_purely_by_collapse']:,} ({dissect_v1['errors_eliminated_purely_by_collapse']/1522*100:.1f}%)",
                    "v2 Broad (10 Cats)": f"{dissect_v2['errors_eliminated_purely_by_collapse']:,} ({dissect_v2['errors_eliminated_purely_by_collapse']/1522*100:.1f}%)"
                },
                {
                    "Metric": "Errors Remaining if Predictions Merely Post-Hoc Remapped",
                    "v1 Conservative (11 Cats)": f"{dissect_v1['post_hoc_collapsed_errors']:,}",
                    "v2 Broad (10 Cats)": f"{dissect_v2['post_hoc_collapsed_errors']:,}"
                },
                {
                    "Metric": "Actual Errors of Model Retrained on Normalized Labels",
                    "v1 Conservative (11 Cats)": f"{dissect_v1['retrained_normalized_errors']:,}",
                    "v2 Broad (10 Cats)": f"{dissect_v2['retrained_normalized_errors']:,}"
                },
                {
                    "Metric": "Net Error Reduction from Dedicated Retraining",
                    "v1 Conservative (11 Cats)": f"{dissect_v1['retraining_net_error_reduction']:+d} errors",
                    "v2 Broad (10 Cats)": f"{dissect_v2['retraining_net_error_reduction']:+d} errors"
                }
            ])
            st.table(dissect_df)

        with tab_percat:
            st.markdown("#### Per-Category Performance on Normalized Holdout Test Set (N=5,000)")
            variant_sel = st.selectbox("Select Model to Inspect:", ["v1 Conservative (11 Categories)", "v2 Broad (10 Categories)"])
            m_key = "v1_conservative" if "v1" in variant_sel else "v2_broad"
            per_cat_data = pd.DataFrame(tax_json["models"][m_key]["per_category"])
            st.dataframe(
                per_cat_data.style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1": "{:.4f}",
                    "support": "{:,}"
                }).background_gradient(subset=["f1"], cmap="RdYlGn"),
                use_container_width=True
            )


# ----------------------------------------------------------------------
# MODULE 6: COSINE SIMILARITY RETRIEVAL
# ----------------------------------------------------------------------

elif selected_section == "SIMILARITY RETRIEVAL":
    st.subheader("Cosine Similarity Nearest-Neighbor Complaint Retrieval")
    st.markdown(
        """
        Find historically similar customer complaints by projecting a grievance into the learned 
        **TF-IDF vector space** and computing **pairwise cosine similarities** against the indexed corpus.
        
        $$\\text{Cosine Similarity}(\\mathbf{q}, \\mathbf{d}) = \\frac{\\mathbf{q} \\cdot \\mathbf{d}}{\\|\\mathbf{q}\\|_2 \\|\\mathbf{d}\\|_2}$$
        """
    )

    df_corpus, fitted_vec, corpus_matrix = build_indexed_corpus(nrows=500)

    if df_corpus.empty or fitted_vec is None:
        st.error("No complaint corpus available for similarity search.")
    else:
        st.sidebar.markdown(f"**Indexed Corpus:** {len(df_corpus)} complaints")
        st.sidebar.markdown(f"**Vocabulary Features:** {corpus_matrix.shape[1]:,}")

        input_choice = st.radio(
            "Select Query Source:",
            ["Choose Real Complaint from Database", "Enter Custom Grievance Text"],
            horizontal=True
        )

        if input_choice == "Choose Real Complaint from Database":
            complaint_opts = {
                f"ID {row['complaint_id']} [{row['category']}]: {str(row['text'])[:75]}...": row["text"]
                for _, row in df_corpus.head(25).iterrows()
            }
            if not complaint_opts:
                st.warning("No complaints available in indexed corpus. Please enter custom grievance text below.")
                query_narrative = ""
            else:
                selected_label = st.selectbox("Select Real Complaint from Loaded Corpus:", list(complaint_opts.keys()))
                query_narrative = complaint_opts.get(selected_label, "")
                st.text_area("Selected Query Text:", value=query_narrative, height=120, disabled=True)
        else:
            query_narrative = st.text_area(
                "Enter Customer Grievance Narrative:",
                value="I noticed multiple unauthorized transactions and hidden fees charged to my credit card statement without my permission.",
                height=120
            )

        top_k = st.slider("Number of similar complaints to retrieve (k):", min_value=1, max_value=10, value=5)

        if st.button("Search Similar Complaints", type="primary"):
            if not query_narrative.strip():
                st.warning("Please provide a non-empty complaint narrative.")
            else:
                with st.spinner("Computing sparse cosine similarities across indexed corpus..."):
                    results = find_similar_complaints(
                        query_text=query_narrative,
                        vectorizer=fitted_vec,
                        corpus_matrix=corpus_matrix,
                        df_corpus=df_corpus,
                        top_k=top_k,
                        preprocess=True
                    )

                top_score = float(results.iloc[0]["similarity_score"]) if len(results) > 0 else 0.0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieved Matches", len(results))
                with col2:
                    st.metric("Corpus Size", len(df_corpus))
                with col3:
                    st.metric("Top Similarity Score", f"{top_score:.4f}")

                st.markdown("### Top Ranked Similar Complaints")
                for _, row in results.iterrows():
                    rank = int(row["rank"])
                    comp_id = row["complaint_id"]
                    score = float(row["similarity_score"])
                    cat = row.get("category", "General")
                    text = row["complaint_text"]

                    with st.container():
                        st.markdown(
                            f"**Rank {rank}** | **Complaint ID:** `{comp_id}` | "
                            f"**Similarity Score:** `{score:.4f}` | **Category:** `{cat}`"
                        )
                        snippet = text[:220].replace("\n", " ") + ("..." if len(text) > 220 else "")
                        st.write(f"> {snippet}")
                        with st.expander(f"View Full Complaint Narrative (ID: {comp_id})"):
                            st.write(text)
                        st.divider()


# ----------------------------------------------------------------------
# MODULE 7: COMPLAINT CATEGORISATION
# ----------------------------------------------------------------------

elif selected_section == "CLASSIFICATION":
    st.subheader("Supervised Complaint Categorisation")
    st.markdown(
        """
        Classify customer complaint narratives into CFPB financial product categories 
        using the improved **Class-Balanced Model** operating directly on sparse TF-IDF representations.
        """
    )

    clf_model, vec_model = load_trained_model()

    if clf_model is None or vec_model is None:
        st.error("Classifier model or vectorizer could not be loaded.")
    else:
        st.sidebar.markdown(f"**Model:** `{type(clf_model).__name__}`")
        st.sidebar.markdown(f"**Trained Classes:** {len(clf_model.classes_)}")
        dim_str = f"{vec_model[0].vocabulary_.__len__() + vec_model[1].vocabulary_.__len__():,}" if isinstance(vec_model, tuple) else f"{len(vec_model.vocabulary_):,}"
        st.sidebar.markdown(f"**Vocabulary Features:** {dim_str}")

        input_mode = st.radio(
            "Select Narrative Source:",
            ["Choose Real Complaint from Database", "Enter Custom Complaint Narrative"],
            horizontal=True
        )

        actual_cat = None
        if input_mode == "Choose Real Complaint from Database":
            df_sample = load_local_complaints(nrows=200)
            complaint_map = {
                f"ID {r['complaint_id']} [Actual: {r['category']}]: {str(r['text'])[:75]}...": (r["text"], r["category"])
                for _, r in df_sample.head(20).iterrows()
            }
            if not complaint_map:
                st.warning("Local dataset is not available at data/complaints.csv. Please switch to custom complaint narrative.")
                input_narrative = ""
                actual_cat = None
            else:
                chosen_key = st.selectbox("Select Real Complaint to Test:", list(complaint_map.keys()))
                if chosen_key and chosen_key in complaint_map:
                    input_narrative, actual_cat = complaint_map[chosen_key]
                    st.text_area("Selected Complaint Narrative:", value=input_narrative, height=120, disabled=True)
                    st.info(f"Actual Ground Truth Category: {actual_cat}")
                else:
                    input_narrative = ""
                    actual_cat = None
        else:
            input_narrative = st.text_area(
                "Enter Customer Complaint Narrative:",
                value="I noticed multiple fraudulent charges on my credit card statement and contacted customer service to dispute the unauthorized billing.",
                height=130,
                placeholder="Type or paste customer complaint text here..."
            )

        if st.button("Categorise Complaint", type="primary"):
            if not input_narrative.strip():
                st.warning("Please provide a complaint narrative before proceeding.")
            else:
                with st.spinner("Processing narrative and performing inference..."):
                    pred_category, confidence = predict_complaint_category(
                        model=clf_model,
                        vectorizer=vec_model,
                        narrative=input_narrative,
                        preprocess=True
                    )

                    clean_q = preprocess_text(input_narrative)
                    if isinstance(vec_model, tuple):
                        X_q = transform_word_char(vec_model[0], vec_model[1], [clean_q])
                    else:
                        X_q = vec_model.transform([clean_q])

                    probabilities = predict_category_proba(clf_model, X_q)[0]

                is_prob = hasattr(clf_model, "predict_proba")
                conf_label = "Prediction Confidence (Probability)" if is_prob else "Normalized confidence score"

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.success(f"### Predicted Product Category:\n**{pred_category}**")
                    if actual_cat:
                        if pred_category.strip().lower() == actual_cat.strip().lower():
                            st.markdown(
                                f"{get_status_icon_svg('check')} **Prediction matches ground truth category.**",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"{get_status_icon_svg('cross')} **Mismatch:** Model predicted `{pred_category}`, actual ground truth was `{actual_cat}`.",
                                unsafe_allow_html=True,
                            )
                with col2:
                    st.metric(conf_label, f"{confidence:.2%}")

                top5_idx = np.argsort(probabilities)[::-1][:5]
                st.markdown(f"#### Top 5 Product Categories by {conf_label}")
                for idx in top5_idx:
                    p_cat = clf_model.classes_[idx]
                    p_val = float(probabilities[idx])
                    st.write(f"- **{p_cat}**: `{p_val:.2%}`")
                    st.progress(min(max(p_val, 0.0), 1.0))


# ----------------------------------------------------------------------
# MODULE 8: TEXT PREPROCESSING & TF-IDF INSPECTOR
# ----------------------------------------------------------------------

elif selected_section == "PREPROCESSING":
    st.subheader("Text Preprocessing & TF-IDF Feature Inspector")
    st.markdown(
        """
        Inspect each stage of the classical text preprocessing pipeline and examine 
        the extracted TF-IDF feature weights for an authentic complaint narrative.
        """
    )

    sample_text = (
        "On XX/XX/2023, I was charged an UNKNOWN late fee of $45.00 on my credit card statement! "
        "Called representative XXXX regarding account # 987654. Visited https://bank-dispute.com "
        "but the dispute was rejected without explanation."
    )
    user_text = st.text_area("Input Complaint Narrative to Inspect:", value=sample_text, height=110)

    if st.button("Inspect Preprocessing & Features", type="primary"):
        if not user_text.strip():
            st.warning("Please provide text to inspect.")
        else:
            cleaned = clean_text(user_text)
            tokens = tokenize(cleaned)
            filtered_tokens = remove_stopwords(tokens)
            final_text = preprocess_text(user_text)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw Words", len(user_text.split()))
            with col2:
                st.metric("Tokens Extracted", len(tokens))
            with col3:
                st.metric("Filtered Tokens", len(filtered_tokens))

            st.markdown("#### Preprocessing Stages")
            st.markdown("**1. Cleaned Text (Noise & Redaction Removal):**")
            st.code(cleaned, language="text")

            st.markdown("**2. Tokens after Tokenization:**")
            st.write(tokens)

            st.markdown("**3. Tokens after Stopword Removal:**")
            st.write(filtered_tokens)

            st.markdown("**4. Final Reconstructed Preprocessed String:**")
            st.success(final_text)

            _, vec = load_trained_model()
            if vec is not None:
                inspect_vec = vec[0] if isinstance(vec, tuple) else vec
                vec_features = inspect_vec.get_feature_names_out()
                tfidf_vec = inspect_vec.transform([final_text])
                nonzero_indices = tfidf_vec.nonzero()[1]

                if len(nonzero_indices) > 0:
                    feature_weights = [
                        (vec_features[idx], float(tfidf_vec[0, idx]))
                        for idx in nonzero_indices
                    ]
                    feature_weights.sort(key=lambda x: x[1], reverse=True)

                    st.markdown("#### Matched Word TF-IDF Features in Vocabulary")
                    df_feats = pd.DataFrame(feature_weights[:15], columns=["Feature (N-Gram)", "TF-IDF Weight"])
                    st.dataframe(df_feats.style.format({"TF-IDF Weight": "{:.4f}"}), use_container_width=True)
                else:
                    st.info("No matching vocabulary n-grams found in the trained vectorizer.")


# ----------------------------------------------------------------------
# MODULE 9: SYSTEM ARCHITECTURE
# ----------------------------------------------------------------------

elif selected_section == "SYSTEM ARCHITECTURE":
    st.subheader("System Architecture & Pipeline Design")
    st.markdown(
        """
        ```text
                            CFPB Consumer Complaints
                      (Local CSV or Live CFPB Search API)
                                      ↓
                                Data Preprocessing
                 (Lowercasing, Redaction Cleaning, Tokenization,
                       Number Normalization, Stopwords)
                                      ↓
                     Feature Extraction: TF-IDF Engine
          ┌──────────────────────────────────────────────────────────┐
          │  Word TF-IDF: Unigrams + Bigrams (ngram_range=(1,2))    │
          │  Char TF-IDF: Subwords within boundaries (char_wb, 3-5)  │
          │  Combined: scipy.sparse.hstack (237,148 sparse features) │
          └──────────────────────────────────────────────────────────┘
                                      ↓
                    ┌───────────────────────────────────┐
                    │                                   │
                    ▼                                   ▼
          Cosine Similarity Retrieval         Supervised Classification
        (Pairwise dot product on CSR)     (Class-Balanced Logistic Regression / LinearSVC)
                    │                                   │
                    ▼                                   ▼
          Top-K Similar Grievances             Predicted Product Category
                                                        │
                                                        ▼
                                                 Model Evaluation
                                          (Accuracy: 69.56%, Macro F1: 50.56%,
                                            Weighted F1: 69.55% on N=5,000 Test)
        ```
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Real-World CFPB Data")
        st.write(
            "Trained and evaluated on the authentic CFPB Consumer Complaint Database (25,000 complaints). "
            "Partitioned with zero data leakage: 20,000 training pool and 5,000 untouched test records."
        )
    with col2:
        st.markdown("### Word + Character Subwords")
        st.write(
            "Fuses word n-grams with character n-grams (`char_wb`, 3–5) to robustly capture compound terms, "
            "prefixes, suffixes, financial acronyms, and terminology variations in sparse CSR format."
        )
    with col3:
        st.markdown("### Measured Performance Gain")
        st.write(
            "Class-balanced optimization elevated **Macro F1 from 34.15% to 50.56%** (+16.41 pp gain) "
            "and **Accuracy to 69.56%** on the controlled 5,000-record holdout test set."
        )

    st.markdown("---")
    st.markdown("### Systematic Engineering & Research Milestones")
    milestones = [
        ("1. Repository Foundation & Scaffolding", "Complete", "6b161cf"),
        ("2. CFPB Dataset Loading & Validation", "Complete", "24752a0"),
        ("3. Classical Text Preprocessing Pipeline", "Complete", "e964ec0"),
        ("4. TF-IDF Vectorisation (Unigrams + Bigrams)", "Complete", "364ebc4"),
        ("5. Cosine Similarity Complaint Search", "Complete", "99bd215"),
        ("6. Supervised Classification Baseline", "Complete", "164f722"),
        ("7. Model Evaluation & Live CFPB API Integration", "Complete", "6f6bfc4"),
        ("8. Systematic Classical Model Improvement", "Complete", "4ef78fe"),
        ("9. Classification Error Analysis", "Complete", "4cf8afa"),
        ("10. Taxonomy-Aware Classification Analysis", "Complete", "0d49099"),
        ("11. Polished Academic Live Demo & Visual Analytics", "Complete", "Current"),
    ]
    st.table(pd.DataFrame(milestones, columns=["Milestone", "Status", "Git Commit"]))
