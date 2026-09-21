# Customer Complaint Similarity & Categorisation

An academic Natural Language Processing (NLP) system designed to analyze consumer complaints, compute pairwise semantic similarity using classical vector-space models, and categorize complaints into financial product categories using supervised machine learning.

---

## Overview

Consumer financial institutions receive thousands of customer complaints daily across numerous product verticals (e.g., credit reporting, mortgages, debt collection, credit cards). Manually triaging and finding recurring complaint themes is labor-intensive and prone to inconsistencies.

This project implements an end-to-end, interpretable classical NLP pipeline that:
1. Cleans and standardizes raw unstructured consumer complaint text.
2. Represents text documents using Term Frequency-Inverse Document Frequency (TF-IDF).
3. Computes document-to-document Cosine Similarity to identify complaints with similar textual context.
4. Classifies complaints into designated categories using supervised machine learning.
5. Evaluates model performance using rigorous statistical metrics.
6. Presents an interactive Streamlit web dashboard for exploration and inference.

---

## Problem Statement

Financial complaints submitted to regulatory bodies contain unstructured, noisy natural language narratives. Organizations need to:
- Identify recurring issues and retrieve historically similar complaints quickly.
- Automatically route new customer grievances to the appropriate department without relying on black-box, cost-prohibitive proprietary APIs or deep neural networks.

---

## Objective

- Implement a modular, transparent text preprocessing workflow (tokenization, lowercase normalization, noise reduction, and stopword removal).
- Build a vectorization engine using Scikit-learn's TF-IDF vectorizer to extract meaningful n-gram feature representations.
- Implement an efficient Cosine Similarity search mechanism to rank complaints by similarity.
- Train and validate a supervised classical classifier (Logistic Regression) to accurately predict complaint categories.
- Provide comprehensive evaluation reporting (precision, recall, F1-score, confusion matrix) and an interactive web interface.

---

## Dataset

This project utilizes the official **Consumer Complaint Database** maintained by the **Consumer Financial Protection Bureau (CFPB)**.
- **Provider**: Consumer Financial Protection Bureau (U.S. Federal Government)
- **Primary Text Field**: `Consumer complaint narrative` (unstructured consumer feedback).
- **Target Category Field**: `Product` (financial product/service classification).
- **Data Policy**: Large raw dataset files (`*.csv`) are excluded from Git version control via `.gitignore`. See [data/README.md](data/README.md) for data schema details and download instructions.

---

## Methodology

The architecture follows a strict classical NLP paradigm:

```text
Customer Complaint Narrative
            ↓
    Text Preprocessing
 (Lowercasing, Cleaning, Tokenization, Stopword Filtering)
            ↓
    TF-IDF Vectorisation
 (Sublinear Term Frequency, Document Frequency Thresholding)
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
                    (Precision, Recall, F1, Confusion Matrix)
```

---

## Technologies Used

- **Language**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Machine Learning & NLP**: Scikit-learn, NLTK
- **Visualization**: Matplotlib, Seaborn
- **Web Application**: Streamlit
- **Development & Testing**: Pytest, Jupyter Notebook

*Note: In accordance with project constraints, this system relies exclusively on classical statistical and machine learning methods. Pretrained language models (e.g., BERT, Transformers) and external LLM APIs are deliberately excluded.*

---

## Project Structure

```text
Customer-Complaint-NLP/
│
├── data/
│   ├── README.md              # Dataset download and schema documentation
│   └── .gitkeep
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb # End-to-end exploratory analysis and pipeline demo
│
├── src/
│   ├── __init__.py
│   ├── cfpb_api.py            # Official CFPB API client & data fetcher
│   ├── data_loader.py         # Unified dataset loader (local CSV & live API)
│   ├── preprocessing.py       # Text cleaning, normalization, and tokenization
│   ├── vectorization.py       # TF-IDF feature extraction (sparse CSR matrices)
│   ├── similarity.py          # Cosine similarity calculations & Top-K retrieval
│   ├── classification.py      # Classifier training & inference (Logistic Regression)
│   └── evaluation.py          # Metrics, classification reports, confusion matrices
│
├── tests/
│   ├── __init__.py
│   ├── test_cfpb_api.py       # Unit tests for CFPB API integration
│   ├── test_classification.py # Unit tests for classifier training & inference
│   ├── test_data_loader.py    # Unit tests for dataset loading and validation
│   ├── test_evaluation.py     # Unit tests for evaluation metrics & reports
│   ├── test_preprocessing.py  # Unit tests for preprocessing routines
│   ├── test_similarity.py     # Unit tests for cosine similarity search
│   └── test_vectorization.py  # Unit tests for TF-IDF vectorization
│
├── app/
│   └── app.py                 # Streamlit web application interface
│
├── models/
│   └── .gitkeep               # Serialized models and vectorizers (gitignored)
│
├── results/
│   ├── confusion_matrix.png   # Multi-class confusion matrix on holdout test set
│   └── .gitkeep
│
├── README.md                  # Project overview and documentation
├── LICENSE                    # MIT License for source code
├── CODE_OF_CONDUCT.md         # Contributor Covenant Code of Conduct
├── CONTRIBUTING.md            # Contribution guidelines
├── .gitignore                 # Files and directories excluded from git
├── requirements.txt           # Python package dependencies
└── pyproject.toml             # Project metadata and build configuration
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git
   cd Customer-Complaint-NLP
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running Tests
Execute the full test suite (130 unit tests) via pytest:
```bash
python -m pytest -v
```

### Running the Web Application
Launch the Streamlit web dashboard:
```bash
streamlit run app/app.py
```

---

## Model

The supervised classification component employs **Multinomial Logistic Regression** (with L2 regularization) trained on unigram and bigram TF-IDF representations.

### Architectural Rationale:
1. **Convex Optimization**: Logistic Regression provides stable, globally optimal convergence on sparse, high-dimensional TF-IDF feature representations ($26,900+$ features).
2. **Probability Calibration**: Outputs calibrated class probability distributions, enabling confidence-based complaint routing and triage.
3. **Interpretability**: Linear weights allow straightforward inspection of top predictive n-grams per financial product category.
4. **Computational Efficiency**: Extremely fast inference ($< 1$ ms) operating directly on SciPy sparse CSR matrices without memory-intensive dense conversions.
5. **Leakage Prevention**: Stratified 80/20 train/test split executed strictly before vocabulary learning and TF-IDF transformation.

---

## Evaluation & Official Benchmark Results

Model performance was evaluated on an unseen holdout test partition ($N = 600$ complaints, 20% stratified split) drawn from 3,000 verified CFPB complaint narratives.

| Evaluation Metric | Measured Score | Description |
|---|---|---|
| **Overall Accuracy** | **61.83%** (0.6183) | Percentage of test complaints correctly classified across all categories |
| **Weighted Precision** | **53.79%** (0.5379) | Support-weighted precision accounting for class frequency |
| **Weighted Recall** | **61.83%** (0.6183) | Support-weighted recall across all categories |
| **Weighted F1-Score** | **54.96%** (0.5496) | Harmonic mean of weighted precision and recall |
| **Macro Precision** | **26.33%** (0.2633) | Unweighted mean precision across all 17 classes |
| **Macro Recall** | **25.97%** (0.2597) | Unweighted mean recall across all 17 classes |
| **Macro F1-Score** | **24.78%** (0.2478) | Unweighted mean F1-score across all 17 classes |
| **Test Partition Size** | **600 samples** | 20% stratified holdout split |
| **Vocabulary Features** | **26,922 features** | Unigrams + bigrams ($1 \le n \le 2$, $\text{min\_df}=2$) |

### Confusion Matrix Diagnostics
The multi-class confusion matrix plot is generated and saved at [`results/confusion_matrix.png`](results/confusion_matrix.png).
- Dominant categories (*Debt collection*, *Credit reporting*, *Mortgage*) exhibit strong true-positive concentrations along the main diagonal.
- Primary confusions occur between semantically adjacent credit products (e.g. *Credit card* vs. *Credit reporting*).

---

## Live CFPB API Integration

The application integrates with the official [CFPB Consumer Complaint Database API v1](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/):
- **Endpoint**: `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`
- **Client**: `src.cfpb_api.CFPBClient` and convenience function `fetch_cfpb_data()`
- **Schema Normalization**: Maps raw API Elasticsearch hits into canonical fields (`complaint_id`, `category`, `text`, `company`, `date_received`, `state`, `issue`).
- **Unified Dispatcher**: `src.data_loader.get_complaints_data(source="csv" | "api")` allows switching seamlessly between offline CSV analysis and live CFPB querying.

---

## Limitations

- **Syntactic Context**: Classical bag-of-words and TF-IDF representations do not capture complex long-range syntactic nuances or word re-ordering beyond the defined n-gram window.
- **Out-of-Vocabulary Terms**: Words not present in the training vocabulary are ignored during inference.
- **CFPB Narrative Publication Policy**: Under the CFPB's public disclosure policy, newer complaint records undergo redaction review before consumer narratives become publicly accessible. The system gracefully handles metadata-only records.

---

## Future Improvements

- Hyperparameter tuning via grid search across TF-IDF max features and regularization parameters ($C$).
- Incorporating domain-specific financial stopword lists and entity masking.
- Support for hierarchical classification (predicting both `Product` and `Sub-product`).
- Exporting automated summary evaluation reports in PDF/Markdown.

---

## Dataset & API Sources

- **CFPB Consumer Complaint Database**:
  [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **CFPB Complaint Search API v1**:
  [https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/)

