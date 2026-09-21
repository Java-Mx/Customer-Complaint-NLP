"""Streamlit Web Application for Customer Complaint Similarity & Categorisation.

Provides an interactive user interface to explore complaints, predict product categories,
and retrieve historically similar grievances using classical NLP and Machine Learning.
"""

import streamlit as st

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
    ["Overview & Pipeline", "Classify Complaint", "Find Similar Complaints"]
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
         (Term Frequency - Inverse Document Frequency)
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
        "Project initialization stage complete. Model training and evaluation pipelines "
        "will be incrementally enabled in upcoming milestones."
    )

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
                "Classification model training is scheduled in the upcoming development milestone. "
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
