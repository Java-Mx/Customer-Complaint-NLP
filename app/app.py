"""Streamlit Web Application for Customer Complaint Similarity & Categorisation.

Provides an interactive user interface to explore complaints, demonstrate preprocessing,
TF-IDF vectorisation, cosine-similarity-based retrieval, and supervised complaint classification.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_dataset
from src.preprocessing import clean_text, preprocess_text, preprocess_series
from src.vectorization import create_vectorizer, fit_transform_corpus, fit_transform_tfidf
from src.similarity import find_similar_complaints, compute_cosine_similarity
from src.classification import (
    create_classifier,
    fit_classifier,
    load_classifier,
    predict_categories,
    predict_category_proba,
    predict_complaint_category,
    save_classifier,
    train_test_split_data,
)

st.set_page_config(
    page_title="Customer Complaint Similarity & Categorisation",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Customer Complaint Similarity & Categorisation")
st.markdown(
    """
    **Academic NLP Mini-Project** analyzing consumer financial complaints from the 
    [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
    
    *Classical NLP Pipeline: Preprocessing → TF-IDF → Cosine Similarity & Supervised Classification.*
    """
)

@st.cache_data
def load_cached_corpus():
    """Load and index a manageable sample of complaints for real-time similarity search."""
    data_path = Path("data/complaints.csv")
    if data_path.exists():
        df = load_dataset(data_path, nrows=300, drop_invalid=True)
        clean_narratives = preprocess_series(df["text"])
        df["clean_text"] = clean_narratives
        vec = create_vectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False)
        fitted_vec, matrix = fit_transform_corpus(df["clean_text"], vectorizer=vec)
        return df, fitted_vec, matrix
    else:
        # Fallback sample when CSV is not present locally
        records = [
            {"complaint_id": "FB-001", "category": "Credit Card", "text": "Unauthorized transaction and fraudulent charge on my credit card that was never resolved."},
            {"complaint_id": "FB-002", "category": "Credit Card", "text": "Excessive late fees charged to credit card statement despite payment submitted before due date."},
            {"complaint_id": "FB-003", "category": "Mortgage", "text": "Mortgage loan servicer lost loan modification paperwork and threatened foreclosure."},
            {"complaint_id": "FB-004", "category": "Debt Collection", "text": "Debt collector continuously calling my workplace regarding an unknown medical debt."},
            {"complaint_id": "FB-005", "category": "Student Loan", "text": "Student loan payments were misapplied to interest rather than principal balance."},
            {"complaint_id": "FB-006", "category": "Credit reporting", "text": "Inaccurate inquiries and incorrect delinquent status appearing on my Experian report."},
        ]
        df = pd.DataFrame(records)
        clean_narratives = preprocess_series(df["text"])
        df["clean_text"] = clean_narratives
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=False)
        fitted_vec, matrix = fit_transform_corpus(df["clean_text"], vectorizer=vec)
        return df, fitted_vec, matrix


@st.cache_resource
def load_cached_classifier():
    """Load or train a cached Logistic Regression model and vectorizer for live inference."""
    model_path = Path("models/complaint_classifier.joblib")
    vec_path = Path("models/tfidf_vectorizer.joblib")

    if model_path.exists() and vec_path.exists():
        try:
            clf = load_classifier(model_path)
            vec = joblib.load(vec_path)
            return clf, vec
        except Exception:
            pass

    # Fallback to local training on complaints.csv sample if serialized files unavailable
    data_path = Path("data/complaints.csv")
    if data_path.exists():
        df = load_dataset(data_path, nrows=2000, drop_invalid=True)
        X_train, _, y_train, _ = train_test_split_data(df, test_size=0.20, random_state=42, stratify=True)
        clean_train = preprocess_series(X_train)
        vec = create_vectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False)
        vec, X_train_tfidf = fit_transform_corpus(clean_train, vectorizer=vec)
        clf = create_classifier(C=1.0, max_iter=1000, random_state=42)
        clf = fit_classifier(clf, X_train_tfidf, y_train)

        # Save for future use
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_classifier(clf, model_path)
        joblib.dump(vec, vec_path)
        return clf, vec
    else:
        # Curated fallback sample
        records = [
            {"text": "fraudulent charge on credit card statement", "category": "Credit Card"},
            {"text": "credit card annual fee billing dispute", "category": "Credit Card"},
            {"text": "mortgage servicer foreclosure loan modification", "category": "Mortgage"},
            {"text": "lender escrow monthly mortgage payment increase", "category": "Mortgage"},
            {"text": "harassing phone calls from debt collection agency", "category": "Debt Collection"},
            {"text": "debt collector demanding payment for invalid debt", "category": "Debt Collection"},
            {"text": "student loan repayment plan interest capitalization", "category": "Student Loan"},
            {"text": "student loan servicer misapplied payment", "category": "Student Loan"},
        ]
        df = pd.DataFrame(records)
        clean_train = preprocess_series(df["text"])
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=False)
        vec, X_train_tfidf = fit_transform_corpus(clean_train, vectorizer=vec)
        clf = create_classifier(C=1.0, max_iter=1000, random_state=42)
        clf = fit_classifier(clf, X_train_tfidf, df["category"])
        return clf, vec


st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select Mode",
    [
        "Overview & Pipeline",
        "Preprocessing Demo",
        "TF-IDF Demo",
        "Cosine Similarity Demo",
        "Complaint Categorisation Demo"
    ]
)

if section == "Overview & Pipeline":
    st.subheader("System Architecture")
    st.markdown(
        """
        ```
        Customer Complaint Narrative
                    ↓
            Text Preprocessing
         (Lowercasing, Cleaning, Tokenization, Stopword Filtering)
                    ↓
            TF-IDF Vectorisation
         (Unigrams + Bigrams, Sublinear Scaling, Sparse CSR Matrix)
                    ↓
          ┌─────────────────────────────────┐
          │                                 │
          ▼                                 ▼
        Cosine Similarity             Classification
          │                        (Logistic Regression)
          ▼                                 ▼
        Top-K Similar Complaints       Predicted Category
                                            ↓
                                      Model Evaluation
                            (Accuracy, Precision, Recall, F1)
        ```
        """
    )
    st.info(
        "Current Milestone: Text preprocessing, TF-IDF vectorisation, Cosine Similarity retrieval, "
        "and Multinomial Logistic Regression complaint classification are complete. "
        "Formal model evaluation and confusion matrix diagnostics are scheduled for the next milestone."
    )

elif section == "Preprocessing Demo":
    st.subheader("Classical Text Preprocessing Pipeline Demo")
    st.markdown(
        "Demonstrates the preprocessing steps on sample complaint text: "
        "lowercasing, URL/email removal, punctuation stripping, CFPB redaction removal (e.g. `XXXX`), "
        "number preservation, and stopword filtering."
    )
    default_text = (
        "I noticed an UNKNOWN fee of $50.00 on XX/XX/2023 from https://fraud.com! "
        "Called customer care agent XXXX regarding account # 12345 -- why was this fee applied?"
    )
    input_text = st.text_area("Input Complaint Narrative:", value=default_text, height=120)
    if st.button("Run Preprocessing"):
        if not input_text.strip():
            st.warning("Please enter text to preprocess.")
        else:
            cleaned = clean_text(input_text)
            final_text = preprocess_text(input_text)
            raw_words = len(input_text.split())
            clean_words = len(final_text.split())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Raw Word Count", raw_words)
            with col2:
                st.metric("Cleaned Word Count", clean_words)

            st.markdown("**Cleaned & Normalized Text (after noise & redaction removal):**")
            st.code(cleaned, language="text")

            st.markdown("**Final Preprocessed Text (ready for TF-IDF):**")
            st.success(final_text)

elif section == "TF-IDF Demo":
    st.subheader("TF-IDF Vectorisation Demo")
    st.markdown(
        """
        Transforms preprocessed text into term frequency-inverse document frequency feature vectors.
        - **Vocabulary**: Unigrams + Bigrams (`ngram_range=(1, 2)`)
        - **Scaling**: Sublinear Term Frequency ($1 + \\log(\\text{tf})$)
        - **Representation**: Scipy sparse CSR matrix
        """
    )

    sample_complaints = [
        "unauthorized charge credit card account dispute late fee",
        "late payment fee credit card statement billing dispute",
        "mortgage loan modification request denied lender servicer",
        "identity theft reported fraudulent loan account opened",
        "credit card payment processed late fee charged again",
    ]

    st.markdown("**Sample Corpus (5 Preprocessed Complaint Documents):**")
    for idx, text in enumerate(sample_complaints, start=1):
        st.markdown(f"- **Doc {idx}:** `{text}`")

    if st.button("Generate TF-IDF Features"):
        vec = create_vectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=False)
        fitted_vec, matrix = fit_transform_tfidf(vec, sample_complaints)
        feature_names = fitted_vec.get_feature_names_out()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Documents", matrix.shape[0])
        with col2:
            st.metric("Total Extracted Features", matrix.shape[1])
        with col3:
            st.metric("Matrix Storage Format", f"Sparse ({type(matrix).__name__})")

        unigrams = [f for f in feature_names if " " not in f]
        bigrams = [f for f in feature_names if " " in f]

        st.markdown(f"**Unigrams ({len(unigrams)}):** `{', '.join(unigrams[:10])}...`")
        st.markdown(f"**Bigrams ({len(bigrams)}):** `{', '.join(bigrams[:10])}...`")

elif section == "Cosine Similarity Demo":
    st.subheader("🔍 Cosine Similarity Nearest-Neighbor Complaint Retrieval")
    st.markdown(
        """
        Find historically similar customer complaints by projecting a grievance into the learned 
        **TF-IDF vector space** and computing **pairwise cosine similarities** against the indexed corpus.
        
        $$\\text{Cosine Similarity}(\\mathbf{q}, \\mathbf{d}) = \\frac{\\mathbf{q} \\cdot \\mathbf{d}}{\\|\\mathbf{q}\\|_2 \\|\\mathbf{d}\\|_2}$$
        """
    )

    df_corpus, fitted_vec, corpus_matrix = load_cached_corpus()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Corpus Size:** {len(df_corpus)} complaints")
    st.sidebar.markdown(f"**Vocabulary Size:** {corpus_matrix.shape[1]} features")

    query_input_mode = st.radio(
        "Choose Query Source:",
        ["Select Sample Complaint", "Enter Custom Narrative"],
        horizontal=True
    )

    preset_options = {
        "Credit Card Dispute": "I called customer care multiple times regarding an unauthorized charge of $150 on my credit card statement that the bank refused to credit back.",
        "Mortgage Loan Modification": "My mortgage servicer has delayed my loan modification application for months and initiated foreclosure proceedings despite receiving all required paperwork.",
        "Debt Collection Harassment": "A debt collection agency has been calling my cellular phone and employer continuously demanding payment for a medical debt that is not mine.",
        "Credit Report Inaccuracy": "There are several inaccurate inquiries and delinquent marks on my credit report that resulted from fraudulent accounts opened without my permission."
    }

    if query_input_mode == "Select Sample Complaint":
        selected_preset = st.selectbox("Preset Grievance Scenarios:", list(preset_options.keys()))
        query_text = preset_options[selected_preset]
        st.text_area("Selected Query Narrative:", value=query_text, height=100, disabled=True)
    else:
        query_text = st.text_area(
            "Enter customer complaint text:",
            value="I noticed multiple unauthorized transactions on my credit card statement and called to dispute the fraudulent charges.",
            height=120
        )

    top_k = st.slider("Number of similar complaints to retrieve (k):", min_value=1, max_value=10, value=5)

    if st.button("Search Similar Complaints", type="primary"):
        if not query_text.strip():
            st.warning("Please provide a non-empty complaint narrative.")
        else:
            with st.spinner("Computing cosine similarities across indexed corpus..."):
                results = find_similar_complaints(
                    query_text=query_text,
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
                st.metric("Indexed Corpus Size", len(df_corpus))
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
                    with st.expander(f"View Full Complaint Text (ID: {comp_id})"):
                        st.write(text)
                    st.divider()

elif section == "Complaint Categorisation Demo":
    st.subheader("🎯 Supervised Complaint Categorisation Demo")
    st.markdown(
        """
        Classify customer complaint narratives into CFPB financial product categories 
        using a trained **Multinomial Logistic Regression** model operating on unigram + bigram TF-IDF representations.
        """
    )

    clf_model, vec_model = load_cached_classifier()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model Type:** `{type(clf_model).__name__}`")
    st.sidebar.markdown(f"**Recognized Classes:** {len(clf_model.classes_)}")
    st.sidebar.markdown(f"**Vocabulary Features:** {len(vec_model.vocabulary_):,}")

    classify_mode = st.radio(
        "Select Narrative Input Mode:",
        ["Choose Benchmark Example", "Enter Custom Complaint Narrative"],
        horizontal=True
    )

    benchmark_narratives = {
        "Credit Card Billing Issue": "I was charged an unexpected $35 late fee on my credit card statement even though my online payment was scheduled and submitted two business days before the due date.",
        "Mortgage Servicing Problem": "My mortgage loan servicer failed to pay the county property taxes from my escrow account, resulting in tax penalties and a notice of tax lien on my home.",
        "Debt Collection Contact": "A collection agency named Allied Recovery has been calling my mobile phone five times a day regarding an alleged medical debt from four years ago that I do not owe.",
        "Student Loan Payment Allocation": "I submitted an extra principal-only payment of $500 to my federal student loan servicer, but they incorrectly allocated the entire sum to future interest.",
        "Credit Reporting Dispute": "Equifax is reporting an open delinquent account with a balance of $1,200 that belongs to someone else with a similar name. I filed a dispute with documentation but it was rejected."
    }

    if classify_mode == "Choose Benchmark Example":
        chosen_key = st.selectbox("Benchmark Grievance Scenarios:", list(benchmark_narratives.keys()))
        complaint_input = benchmark_narratives[chosen_key]
        st.text_area("Selected Complaint Narrative:", value=complaint_input, height=110, disabled=True)
    else:
        complaint_input = st.text_area(
            "Enter customer complaint narrative:",
            value="I noticed an unauthorized cash withdrawal on my checking account and contacted the bank to dispute the fraudulent transaction.",
            height=130,
            placeholder="Type or paste customer complaint text here..."
        )

    if st.button("Categorise Complaint", type="primary"):
        if not complaint_input.strip():
            st.warning("Please provide a complaint narrative before proceeding.")
        else:
            with st.spinner("Processing text narrative and performing classification..."):
                pred_category, confidence = predict_complaint_category(
                    model=clf_model,
                    vectorizer=vec_model,
                    narrative=complaint_input,
                    preprocess=True
                )

                # Compute full class probability distribution
                clean_q = preprocess_text(complaint_input)
                X_q = vec_model.transform([clean_q])
                probabilities = predict_category_proba(clf_model, X_q)[0]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"### Predicted Product Category:\n**{pred_category}**")
            with col2:
                st.metric("Prediction Confidence", f"{confidence:.2%}")

            # Top 3 predicted class probabilities
            top3_idx = np.argsort(probabilities)[::-1][:3]
            df_top3 = pd.DataFrame({
                "Product Category": [clf_model.classes_[idx] for idx in top3_idx],
                "Probability": [float(probabilities[idx]) for idx in top3_idx]
            })

            st.markdown("#### Top Class Probabilities")
            for _, r in df_top3.iterrows():
                p_cat = r["Product Category"]
                p_val = r["Probability"]
                st.write(f"- **{p_cat}**: `{p_val:.2%}`")
                st.progress(min(max(p_val, 0.0), 1.0))

            st.caption(
                "Note: Baseline Multinomial Logistic Regression model trained on CFPB consumer complaints. "
                "Formal model evaluation metrics (Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix) "
                "will be generated in the upcoming evaluation milestone."
            )
