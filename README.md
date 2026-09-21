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
Execute the full test suite (141 unit tests) via pytest:
```bash
python -m pytest -v
```

### Running the Web Application
Launch the Streamlit web dashboard:
```bash
streamlit run app/app.py
```

### Running Systematic Model Experiments
Run the non-destructive model evaluation and selection pipeline across all 30+ configurations:
```bash
python scripts/run_experiments.py
```

---

## Model & Systematic Architecture Improvements

The supervised classification engine operates on a classical machine learning pipeline enhanced with subword granularity and class-imbalance mitigation:

### Architectural Innovations:
1. **Word + Character Subword Fusion**:
   - Fuses word n-grams (`ngram_range=(1, 2)`) with character n-grams within word boundaries (`analyzer="char_wb"`, `ngram_range=(3, 5)`).
   - Subword character n-grams capture morphology, financial roots, prefixes/suffixes (e.g. *foreclos-*, *delinqu-*, *overcharg-*), acronyms (e.g. *APR*, *FCRA*, *CFPB*), and spelling variations.
   - Combined representation stacked into a single sparse matrix via `scipy.sparse.hstack(..., format="csr")` spanning **237,148 sparse features** with zero dense memory allocation.
2. **Class Imbalance Mitigation**:
   - Severe category imbalance (ranging from 5,830 *Debt collection* complaints to rare minority classes) was resolved using `class_weight="balanced"`.
   - Adjusts loss penalization inversely proportional to class frequencies, directly resolving minority-class neglect.
3. **Model Family Exploration**:
   - Evaluated **LinearSVC** (with hinge loss) and **Multinomial Logistic Regression** (with cross-entropy loss) across regularization parameters ($C \in [0.25, 0.5, 1.0, 2.0]$).
   - Logistic Regression with balanced weighting and combined word+char features emerged as the optimal configuration on the internal validation subset.
4. **Strict Leakage Prevention**:
   - Full 25,000 dataset partitioned into 20,000 Training Pool and 5,000 Untouched Test Set.
   - Training pool internally split into 16,000 train subset and 4,000 validation subset for model selection.
   - Untouched test set evaluated strictly once after final retraining on the 20,000-sample pool.

---

## Evaluation & Official Benchmark Results

Model performance was evaluated on the unseen holdout test partition ($N = 5,000$ complaints, 20% stratified split) drawn from the 25,000 verified CFPB dataset.

### Baseline vs. Improved Final Model Comparison

| Evaluation Metric | Initial Baseline ($N=600$) | Improved Final Model ($N=5,000$) | Measured Improvement |
|---|---|---|---|
| **Overall Accuracy** | **61.83%** (0.6183) | **69.56%** (0.6956) | **+7.73% absolute gain** |
| **Macro F1-Score** | **24.78%** (0.2478) | **50.56%** (0.5056) | **+25.78% (More than doubled!)** |
| **Weighted F1-Score** | **54.96%** (0.5496) | **69.55%** (0.6955) | **+14.59% absolute gain** |
| **Macro Precision** | **26.33%** (0.2633) | **49.67%** (0.4967) | **+23.34% absolute gain** |
| **Macro Recall** | **25.97%** (0.2597) | **51.97%** (0.5197) | **+26.00% absolute gain** |
| **Weighted Precision** | **53.79%** (0.5379) | **70.03%** (0.7003) | **+16.24% absolute gain** |
| **Weighted Recall** | **61.83%** (0.6183) | **69.56%** (0.6956) | **+7.73% absolute gain** |
| **Test Partition Size** | 600 samples | **5,000 samples** | Real-world statistical power |
| **Vocabulary Features** | 26,922 features | **237,148 features** | Full word + character subword coverage |

### Validation Experiment Highlights (from `results/model_comparison.csv`)

| Model | Feature Representation | Class Weight | $C$ | Val Accuracy | Val Macro F1 | Val Weighted F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression (Selected)** | **Combined Word(1,2) + Char(3,5)** | **balanced** | **1.0** | **69.20%** | **51.27%** | **69.13%** |
| Logistic Regression | Char-only (3,5) | balanced | 1.0 | 67.10% | 50.26% | 67.48% |
| Logistic Regression | Word (1,2) min_df=3 | balanced | 1.0 | 68.57% | 49.79% | 68.35% |
| LinearSVC | Combined Word(1,2) + Char(3,5) | balanced | 0.5 | 70.77% | 49.15% | 69.99% |
| LinearSVC | Word (1,3) min_df=2 | balanced | 1.0 | 71.25% | 47.94% | 70.01% |

*The complete 30-experiment validation log is tracked at [`results/model_comparison.csv`](results/model_comparison.csv).*

### Confusion Matrix Diagnostics
The updated multi-class confusion matrix on the 5,000 unseen complaints is saved at [`results/confusion_matrix.png`](results/confusion_matrix.png).
- Dominant categories (*Debt collection*, *Credit reporting*, *Mortgage*) achieve strong true-positive diagonal clustering.
- Subword character n-grams dramatically reduced false negatives in low-frequency minority categories.

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

