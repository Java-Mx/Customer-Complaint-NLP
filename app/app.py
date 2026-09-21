"""Streamlit Web Application for Customer Complaint Similarity & Categorisation.

Provides an interactive user interface to explore complaints, demonstrate preprocessing,
TF-IDF vectorisation, and cosine-similarity-based nearest-neighbor complaint retrieval.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_loader import load_dataset
from src.preprocessing import clean_text, preprocess_text, preprocess_series
from src.vectorization import create_vectorizer, fit_transform_corpus, fit_transform_tfidf
from src.similarity import find_similar_complaints, compute_cosine_similarity

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

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select Mode",
    [
        "Overview & Pipeline",
        "Preprocessing Demo",
        "TF-IDF Demo",
        "Cosine Similarity Demo",
        "Classify Complaint"
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
          │                        (Supervised Classifier)
          ▼                                 ▼
        Top-K Similar Complaints       Predicted Category
                                            ↓
                                      Model Evaluation
                            (Accuracy, Precision, Recall, F1)
        ```
        """
    )
    st.info(
        "Current Milestone: Text preprocessing, TF-IDF vectorisation, and Cosine Similarity retrieval are complete. "
        "Complaint classification and model evaluation will be developed in the upcoming milestone."
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

elif section == "Classify Complaint":
    st.subheader("Predict Complaint Product Category")
    st.markdown(
        """
        Supervised classification predicting complaint product categories (e.g., *Credit Card*, 
        *Mortgage*, *Debt Collection*) will be implemented in the next development milestone.
        """
    )
    user_complaint = st.text_area(
        "Enter customer complaint narrative:",
        height=150,
        placeholder="Type or paste a complaint narrative here..."
    )
    if st.button("Categorise Complaint"):
        if not user_complaint.strip():
            st.warning("Please provide a complaint narrative before proceeding.")
        else:
            st.info(
                "Classification model training is scheduled in the upcoming development milestone. "
                "The trained classifier will categorize complaints here."
            )
