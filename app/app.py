"""Streamlit Web Application for Customer Complaint Similarity & Categorisation.

Provides an interactive user interface to explore complaints, demonstrate preprocessing
and TF-IDF vectorisation, predict product categories, and retrieve similar grievances.
"""

import pandas as pd
import streamlit as st

from src.preprocessing import clean_text, preprocess_text, preprocess_series
from src.vectorization import create_vectorizer, fit_transform_tfidf

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
    
    *Classical NLP Pipeline: Preprocessing → TF-IDF → Cosine Similarity & Logistic Regression.*
    """
)

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select Mode",
    ["Overview & Pipeline", "Preprocessing Demo", "TF-IDF Demo", "Classify Complaint", "Find Similar Complaints"]
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
        "Current Milestone: Text preprocessing and TF-IDF vectorisation complete. "
        "Cosine similarity search and classification modelling are scheduled in upcoming milestones."
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

elif section == "Classify Complaint":
    st.subheader("Predict Complaint Product Category")
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
                "Classification model training is scheduled in an upcoming development milestone. "
                "The trained Logistic Regression classifier will categorize complaints here."
            )

elif section == "Find Similar Complaints":
    st.subheader("Find Historically Similar Complaints")
    query_text = st.text_area(
        "Enter reference complaint to search similar records:",
        height=150,
        placeholder="Enter complaint narrative to match against corpus..."
    )
    top_k = st.slider("Number of similar complaints to retrieve", min_value=1, max_value=10, value=5)
    if st.button("Search Similar Complaints"):
        if not query_text.strip():
            st.warning("Please enter a query complaint narrative.")
        else:
            st.info(
                "Similarity indexing with TF-IDF and Cosine Similarity will be enabled "
                "in the upcoming similarity milestone."
            )
