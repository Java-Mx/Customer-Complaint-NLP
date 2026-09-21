# Customer Complaint Similarity & Categorisation

An academic Natural Language Processing (NLP) system designed to analyze consumer complaints, compute pairwise semantic similarity using classical vector-space models, categorize complaints into financial product categories using supervised machine learning, and investigate target taxonomy formulations using regulatory data.

---

## Overview

Consumer financial institutions receive thousands of customer complaints daily across numerous product verticals (e.g., credit reporting, mortgages, debt collection, credit cards). Manually triaging and finding recurring complaint themes is labor-intensive and prone to inconsistencies.

This project implements an end-to-end, interpretable classical NLP pipeline that:
1. Cleans and standardizes raw unstructured consumer complaint text.
2. Represents text documents using Term Frequency-Inverse Document Frequency (TF-IDF) with combined word and subword character n-grams.
3. Computes document-to-document Cosine Similarity to identify complaints with similar textual context.
4. Classifies complaints into designated categories using supervised machine learning.
5. Diagnoses failure modes through rigorous confusion matrix, confidence, and error analysis.
6. Investigates how administrative product taxonomy revisions affect classification task difficulty.
7. Presents an interactive Streamlit web dashboard for live exploration and inference.

---

## Problem Statement

Financial complaints submitted to regulatory bodies contain unstructured, noisy natural language narratives. Organizations need to:
- Identify recurring issues and retrieve historically similar complaints quickly.
- Automatically route new customer grievances to the appropriate department without relying on black-box, cost-prohibitive proprietary APIs or deep neural networks.
- Formulate target classification taxonomies that reflect genuine product boundaries rather than historical administrative artifacts.

---

## Objective

- Implement a modular, transparent text preprocessing workflow (tokenization, lowercase normalization, noise reduction, and stopword removal).
- Build a vectorization engine using Scikit-learn's TF-IDF vectorizer to extract meaningful word and character n-gram feature representations.
- Implement an efficient Cosine Similarity search mechanism to rank complaints by similarity.
- Train and validate supervised classical classifiers (Logistic Regression, LinearSVC) to accurately predict complaint categories.
- Provide comprehensive evaluation reporting (precision, recall, F1-score, confusion matrix) and an interactive web interface.
- Audit the CFPB product taxonomy to disentangle linguistic classification difficulty from administrative label synonymy.

---

## Dataset

This project utilizes the official **Consumer Complaint Database** maintained by the **Consumer Financial Protection Bureau (CFPB)**.
- **Provider**: Consumer Financial Protection Bureau (U.S. Federal Government)
- **Dataset Size**: **25,000 complaints** across **18 original Product categories**.
- **Data Partitions**: 20,000 training pool (internally partitioned into 16,000 train and 4,000 validation subsets) and an untouched **5,000-record final test set** created via a stratified split with `random_state=42`.
- **Primary Text Field**: `Consumer complaint narrative` (unstructured consumer feedback).
- **Target Category Field**: `Product` (financial product/service classification).
- **Data Policy**: Raw dataset files (`*.csv`) are excluded from Git version control via `.gitignore`. See [data/README.md](data/README.md) for data schema details and download instructions.

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
 (Combined Word N-grams (1,2) + Character Subword N-grams (3,5))
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
                                    ↓
                        Taxonomy Formulation Audit
                 (Reference 18 vs. Conservative 11 vs. Broad 10)
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
├── config/
│   ├── taxonomy_v1_conservative.json # 11-category taxonomy configuration & CFPB citations
│   └── taxonomy_v2_broad.json        # 10-category taxonomy configuration & CFPB citations
│
├── data/
│   ├── README.md                     # Dataset download and schema documentation
│   └── .gitkeep
│
├── docs/
│   ├── error-analysis.md             # Diagnostic error analysis on 5,000-record test set
│   ├── project-updates.md            # Chronological engineering & milestone log
│   ├── taxonomy-analysis.md          # Complete taxonomy formulation & error report
│   └── viva-preparation.md           # Academic viva defense & technical interview Q&A
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb  # Exploratory analysis and pipeline demonstrations
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── cfpb_api.py                   # Official CFPB API client & data fetcher
│   ├── classification.py             # Classifier training, inference & thresholding
│   ├── data_loader.py                # Unified dataset loader (local CSV & live API)
│   ├── evaluation.py                 # Statistical metrics, classification reports & plots
│   ├── preprocessing.py              # Text cleaning, normalization, and tokenization
│   ├── similarity.py                 # Cosine similarity calculations & Top-K retrieval
│   └── vectorization.py              # TF-IDF feature extraction (Word + Char subword fusion)
│
├── tests/
│   ├── __init__.py
│   ├── test_cfpb_api.py              # Tests for CFPB API integration (13 tests)
│   ├── test_classification.py        # Tests for classifier training & inference (25 tests)
│   ├── test_data_loader.py           # Tests for dataset loading & validation (12 tests)
│   ├── test_error_analysis.py        # Tests for error analysis infrastructure (35 tests)
│   ├── test_evaluation.py            # Tests for evaluation metrics & validation (16 tests)
│   ├── test_model_improvements.py    # Tests for subword vectorizer & LinearSVC (11 tests)
│   ├── test_preprocessing.py         # Tests for preprocessing routines (19 tests)
│   ├── test_similarity.py            # Tests for cosine similarity search (30 tests)
│   ├── test_taxonomy.py              # Tests for taxonomy mapping & assertions (14 tests)
│   └── test_vectorization.py         # Tests for TF-IDF feature extraction (15 tests)
│
├── scripts/
│   ├── analyze_taxonomy.py           # Audits category support, baseline metrics & candidate groups
│   ├── generate_error_analysis.py    # Generates diagnostic error analysis & confusion pairs
│   ├── run_experiments.py            # Grid search across 30+ validation configurations
│   └── run_taxonomy_experiment.py    # Evaluates reference, conservative, and broad taxonomies
│
├── app/
│   └── app.py                        # Streamlit web application interface
│
├── models/
│   └── .gitkeep                      # Serialized models and vectorizers (gitignored)
│
├── results/
│   ├── confusion_matrix.png          # Multi-class confusion matrix plot
│   └── .gitkeep
│
├── README.md                         # Project overview and documentation
├── LICENSE                           # MIT License for source code
├── CODE_OF_CONDUCT.md                # Contributor Covenant Code of Conduct
├── CONTRIBUTING.md                   # Contribution guidelines & doc sync workflow
├── .gitignore                        # Files and directories excluded from git
├── requirements.txt                  # Python package dependencies
└── pyproject.toml                    # Project metadata and build configuration
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
Execute the full test suite (**190 tests passed**) via pytest:
```bash
python -m pytest -v
```

### Running the Web Application
Launch the Streamlit web dashboard:
```bash
streamlit run app/app.py
```

### Running Systematic Model Experiments
Run the model evaluation and selection pipeline across all 30+ validation configurations:
```bash
python scripts/run_experiments.py
```

### Running Error Analysis
Generate the comprehensive confusion matrix and probability error analysis on the 5,000-record test set:
```bash
python scripts/generate_error_analysis.py
```

### Running Taxonomy-Aware Experiments
Execute the taxonomy audit and benchmark the normalized taxonomy formulations:
```bash
python scripts/analyze_taxonomy.py
python scripts/run_taxonomy_experiment.py
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

## Evaluation & Benchmark Results

### Clarification on Experimental Comparisons

To maintain scientific integrity, this project distinguishes between two different comparisons:

1. **Historical Initial Baseline ($N=600$ test split, 17 categories)**:
   - Early exploratory prototype: Accuracy **61.83%**, Macro F1 **24.78%**, Weighted F1 **54.96%**.
   - *Note*: This historical baseline was evaluated on an earlier 3,000-record dataset with a 600-sample test set. It is documented for historical provenance, but is **not** an apples-to-apples controlled scientific comparison.
2. **Authoritative Controlled Model Comparison (Exact Same 5,000-Record Test Set, 18 Categories)**:
   - Evaluated on the exact same 20,000-record training pool and 5,000-record held-out test set:

| Evaluation Metric | Controlled Baseline (Word TF-IDF, No Balancing) | Improved Final Model (Combined TF-IDF, Balanced) | Absolute Difference |
|---|---:|---:|---:|
| **Overall Accuracy** | 69.14% | **69.56%** | +0.42 pp |
| **Macro F1-Score** | 34.15% | **50.56%** | **+16.41 pp (+48.1%)** |
| **Weighted F1-Score** | 65.73% | **69.55%** | +3.82 pp |
| **Macro Precision** | 41.37% | **49.67%** | +8.30 pp |
| **Macro Recall** | 34.08% | **51.97%** | +17.89 pp |
| **Weighted Precision** | 66.53% | **70.03%** | +3.50 pp |
| **Weighted Recall** | 69.14% | **69.56%** | +0.42 pp |
| **Test Partition Size** | 5,000 samples | 5,000 samples | Identical holdout |
| **Vocabulary Features** | 199,630 features | 237,148 features | Subword character fusion |

*Key finding*: The primary controlled improvement from subword fusion and class-weight balancing is on **Macro F1** (+16.41 pp gain), resolving minority-class neglect without sacrificing majority-class precision.

---

## Taxonomy-Aware Classification

Error analysis on the held-out 5,000-record test set revealed that **608 out of 1,522 errors (39.95%)** occurred between pairs of categories that represent the exact same financial products under differing names due to historical CFPB administrative revisions (April 2017 and 2019). The database preserves submission-time labels without retroactively relabeling records.

To investigate whether classification difficulty stemmed from NLP representation limits or from historical label synonymy, the project evaluates three task formulations:

| Task Formulation | Categories | Accuracy | Macro F1 | Weighted F1 | Test Errors |
|---|---:|---:|---:|---:|---:|
| **Original Reference** | 18 | 69.56% | 50.56% | 69.55% | 1,522 |
| **v1 Conservative** | 11 | 81.50% | 63.43% | 81.61% | 925 |
| **v2 Broad** | 10 | 82.32% | 66.57% | 82.42% | 884 |

### Methodological Context on Performance Changes

> **Important**: The normalized tasks change the target taxonomy. Part of the apparent performance increase therefore comes from redefining which distinctions count as separate classification errors, rather than an increase in model capability alone.

### Retraining Error Decomposition

To scientifically separate mechanical task collapse from classifier retraining effects:

$$\text{Total Error Reduction} = \Delta_{\text{mechanical collapse}} + \Delta_{\text{retraining effect}}$$

- **v1 Conservative (11 Categories)**:
  - Original 18-category errors: **1,522**
  - Mechanical collapse elimination: **608 errors (39.95%)**
  - Post-hoc collapsed errors (predictions remapped without retraining): **914**
  - Actual retrained v1 model errors: **925**
  - Retraining effect: **+11 errors** relative to post-hoc collapsed reference
- **v2 Broad (10 Categories)**:
  - Original 18-category errors: **1,522**
  - Mechanical collapse elimination: **635 errors (41.72%)**
  - Post-hoc collapsed errors (predictions remapped without retraining): **887**
  - Actual retrained v2 model errors: **884**
  - Retraining effect: **-3 errors** relative to post-hoc collapsed reference

This confirms that the jump from 69.56% to ~82% accuracy is essentially attributable to resolving label synonymy in the task definition rather than superior classifier generalization.

### Information-Loss Trade-Off

The normalized tasks trade label granularity for consistency. As documented in [docs/taxonomy-analysis.md](docs/taxonomy-analysis.md):
- **Credit Reporting**: Collapses pre-2019 narrow credit bureau disputes with broader post-2019 credit repair and tenant screening scope.
- **Card Products**: Collapses revolving credit cards (TILA-governed) with stored-value prepaid cards (EFTA-governed).
- **Banking Accounts**: Collapses legacy ancillary banking services into retail checking/savings accounts.
- **Money Movement**: Collapses international remittances with virtual currency exchanges and non-bank payment apps.
- **Consumer Loans (Broad v2 only)**: Collapses multi-year installment loans ($5,000–$25,000) with two-week payday advances ($300–$500).

No taxonomy is declared universally superior; deployment choice depends on whether the downstream objective requires high-level operational triage (favoring 11 categories) or exact historical regulatory compliance (favoring 18 categories).

---

## Documentation

Comprehensive project documentation is maintained in the repository:

- [docs/project-updates.md](docs/project-updates.md) — Complete chronological engineering log of all 11 milestones.
- [docs/taxonomy-analysis.md](docs/taxonomy-analysis.md) — Comprehensive 15-section report on taxonomy audits, formulations, and information loss.
- [docs/error-analysis.md](docs/error-analysis.md) — Statistical error analysis on the 5,000-record test set.
- [docs/viva-preparation.md](docs/viva-preparation.md) — Academic viva defense preparation and theoretical examination Q&A.
- [notebooks/01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb) — Interactive dataset exploration and pipeline verification notebook.
- [data/README.md](data/README.md) — Dataset download instructions, schema definitions, and policies.

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
- **Temporal Confounding**: Without complaint filing dates as explicit input features, a text-only classifier operating on historical data will face Bayes error induced by administrative label synonymy.

---

## Dataset & API Sources

- **CFPB Consumer Complaint Database**:
  [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **CFPB Complaint Search API v1**:
  [https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/)
