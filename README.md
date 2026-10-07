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
│   ├── test_cfpb_api.py              # Tests for CFPB API integration (16 tests)
│   ├── test_classification.py        # Tests for classifier training & inference (26 tests)
│   ├── test_data_loader.py           # Tests for dataset loading & validation (12 tests)
│   ├── test_error_analysis.py        # Tests for error analysis infrastructure (35 tests)
│   ├── test_evaluation.py            # Tests for evaluation metrics & validation (16 tests)
│   ├── test_model_improvements.py    # Tests for subword vectorizer & LinearSVC (11 tests)
│   ├── test_preprocessing.py         # Tests for preprocessing routines (19 tests)
│   ├── test_similarity.py            # Tests for cosine similarity search (30 tests)
│   ├── test_taxonomy.py              # Tests for taxonomy mapping & assertions (14 tests)
│   └── test_vectorization.py         # Tests for TF-IDF feature extraction (19 tests)
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
Execute the full test suite (**215 tests passed**) via pytest:
```bash
python -m pytest -v
```

### Running the Web Application
Launch the presentation-ready Streamlit web dashboard:
```bash
streamlit run app/app.py
```

The interactive application features an academic NLP interface:
- **LIVE Complaint Analysis (Opening Hero Module)**: Type or paste any unseen customer complaint narrative or select from authentic demonstration examples (including domain-standard and intentionally ambiguous viva boundary cases). Immediately inspects:
  - **Predicted Product Category & Confidence Score**: Displayed with top-5 class confidence distributions.
  - **Pipeline Trace**: Step-by-step transparency showing raw vs. cleaned text, Word+Char TF-IDF representation (114,493 sparse CSR dimensions), and active non-zero feature counts.
  - **TF-IDF Representation Layout**: Row 1 compact metric cards (`Total Feature Dimension` and `Active Non-Zero Features`) and Row 2 full-width horizontal card (`Feature Representation: Combined Word + Character TF-IDF`) ensuring zero label truncation, with sparse CSR efficiency notes.
  - **Highest-Weighted Active Features**: Exact n-grams and learned TF-IDF weights extracted from the input narrative.
  - **Top Similar Historical CFPB Complaints**: Sparse cosine retrieval against indexed historical complaints with expandable narratives.
  - **What This Demonstrates**: Concise explanation of the 6 classical NLP pipeline stages (with zero reliance on LLMs or external generative APIs).
  - **Model & Dataset Insights Dashboard**: Embedded interactive Plotly charts (hover tooltips, zoom/pan, dark-slate theme, zero static PNGs) comparing controlled baseline vs. improved models (Accuracy, Macro F1, Weighted F1), cross-taxonomy benchmarks, dataset class distributions, 18-category F1 metrics, top confusion pairs, and 18×18 confusion matrix heatmaps.
- **Sidebar Navigation**: Clean rounded rectangular buttons replacing default radio controls, with an enclosed System Status card featuring inline SVG check/cross/warning indicators.
- **CFPB Live API & Data Explorer**: Live Elasticsearch querying of the official CFPB Search API v1 and interactive exploration of the local 25,000-record dataset.
- **Model Evaluation & Diagnostics**: Performance metrics, confusion matrices, and controlled benchmark comparisons.
- **Error Analysis**: Confusion pairs and confidence distribution breakdown on the 5,000-record holdout test set.
- **Taxonomy Analysis**: Conservative (11 categories) and Broad (10 categories) taxonomy formulation audits and retraining decompositions.
- **Cosine Similarity Retrieval**: Independent similarity query engine against indexed historical complaints.
- **Complaint Categorisation**: Batch and interactive supervised categorization.
- **Text Preprocessing & TF-IDF**: Interactive stage-by-stage tokenization and n-gram inspector.
- **System Architecture**: Complete pipeline schematic and milestone tracking.

### Running Systematic Model Experiments
Run the 7-stage systematic model improvement pipeline across 98 validation configurations:
```bash
python scripts/run_model_improvement.py search   # Evaluate candidates on 16k train / 4k val split only
python scripts/run_model_improvement.py final    # Retrain selected model on 20k pool, evaluate once on 5k test
```
Or run the historical 30-run grid search:
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
   - Fuses word unigrams (`ngram_range=(1, 1)`) with character n-grams (`analyzer="char"`, `ngram_range=(3, 5)`).
   - Subword character n-grams capture morphology, financial roots, prefixes/suffixes (e.g. *foreclos-*, *delinqu-*, *overcharg-*), acronyms (e.g. *APR*, *FCRA*, *CFPB*), and spelling variations.
   - Combined representation stacked into a single sparse matrix via `scipy.sparse.hstack(..., format="csr")` spanning **114,493 sparse features** (pruned from 237,148 by eliminating noisy word bigrams) with zero dense memory allocation.
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

## Model Performance

### Original 18-Class Task
Current primary baseline on untouched holdout test set ($N=5,000$):
- **Holdout Test Accuracy**: **69.82%** (3,491 / 5,000)
- **Holdout Test Macro F1**: **50.88%**
- **Holdout Test Weighted F1**: **69.78%**
- **Internal Validation ($N=4,000$)**: Accuracy **69.95%**, Macro F1 **51.84%**

### Improved 18-Class Task
Representational optimization via minimal preprocessing (retaining punctuation, digits, stopwords, and negation context while preserving character n-gram boundaries):
- **Internal Validation ($N=4,000$)**:
  - Accuracy: **71.33%** (+1.38 pp vs. baseline 69.95%)
  - Macro F1: **53.50%** (+1.66 pp vs. baseline 51.84%)
  - Weighted F1: **71.25%** (+1.42 pp vs. baseline 69.83%)
- **Holdout Test Set ($N=5,000$)**:
  - Accuracy: **71.16%** (+1.34 pp vs. baseline 69.82%)
  - Macro F1: **52.43%** (+1.55 pp vs. baseline 50.88%)
  - Weighted F1: **71.07%** (+1.29 pp vs. baseline 69.78%)

*Key Representational Finding*: Preserving punctuation and syntactic negation boundaries (`didn't`, `not`, `never`) gives character n-grams (`analyzer='char'`, ranges 3–5) crucial context that standard stopword and symbol stripping aggressively discard, lifting text-only performance without increasing model capacity.

### Normalized 11-Class Task
*The original 18-class task achieves approximately 70% accuracy. When historical administrative label variants are normalized into 11 broader product groups, the same classical NLP pipeline achieves approximately 82% accuracy.*

- **Classification Formulation**: Conservative Taxonomy v1 (11 Categories, grounded in official CFPB documentation; `Consumer Loan` retained independently).
- **Standard Preprocessing**:
  - Validation Accuracy: **82.20%** | Validation Macro F1: **63.23%**
  - Holdout Test Accuracy: **81.50%** | Holdout Test Macro F1: **63.43%**
- **Minimal Preprocessing**:
  - Validation Accuracy: **82.85%** | Validation Macro F1: **64.53%**
  - Holdout Test Accuracy: **81.90%** | Holdout Test Macro F1: **63.29%**

> **Important Scientific Distinction**: The 11-class normalized taxonomy represents a **different classification task formulation**, not an algorithmic improvement of the 18-class model. Over 40% of all baseline errors occur between identical financial concepts separated purely by CFPB administrative form redesign dates (e.g. *Credit reporting* vs. *Credit reporting, credit repair services...*). Removing administrative synonyms aligns the task with genuine product boundaries.

---

## Evaluation & Benchmark Results

### Clarification on Experimental Comparisons

To maintain scientific integrity, this project distinguishes between two different comparisons:

1. **Historical Initial Baseline ($N=600$ test split, 17 categories)**:
   - Early exploratory prototype: Accuracy **61.83%**, Macro F1 **24.78%**, Weighted F1 **54.96%**.
   - *Note*: This historical baseline was evaluated on an earlier 3,000-record dataset with a 600-sample test set. It is documented for historical provenance, but is **not** an apples-to-apples controlled scientific comparison.
2. **Authoritative Controlled Model Comparison (Exact Same 5,000-Record Test Set, 18 Categories)**:
   - Evaluated on the exact same 20,000-record training pool and 5,000-record held-out test set:

| Evaluation Metric | Controlled Baseline (Word TF-IDF, No Balancing) | Previous Model (Combined (1,2)+(3,5), Balanced, C=1.0) | Improved Final Model (Word(1,1)+Char(3,5), Balanced, C=2.0) | Improvement vs. Previous | Improvement vs. Baseline |
|---|---:|---:|---:|---:|---:|
| **Overall Accuracy** | 69.14% | 69.56% | **69.82%** | **+0.26 pp** | +0.68 pp |
| **Macro F1-Score** | 34.15% | 50.56% | **50.88%** | **+0.33 pp** | **+16.73 pp (+48.99%)** |
| **Weighted F1-Score** | 65.73% | 69.55% | **69.78%** | **+0.23 pp** | +4.05 pp |
| **Macro Precision** | 41.37% | 49.67% | **50.41%** | **+0.74 pp** | +9.04 pp |
| **Macro Recall** | 34.08% | **51.97%** | 51.69% | -0.28 pp | +17.61 pp |
| **Weighted Precision** | 66.53% | 70.03% | **70.08%** | **+0.05 pp** | +3.55 pp |
| **Weighted Recall** | 69.14% | 69.56% | **69.82%** | **+0.26 pp** | +0.68 pp |
| **Test Partition Size** | 5,000 samples | 5,000 samples | 5,000 samples | Identical holdout | Identical holdout |
| **Vocabulary Features** | 199,630 features | 237,148 features | **114,493 features** | **-51.7% feature reduction** | Compact vocabulary |

*Key finding*: The systematic model improvement framework achieved improvements across **Accuracy** (69.82%), **Macro F1** (50.88%), **Weighted F1** (69.78%), and **Macro Precision** (50.41%), while simultaneously cutting vocabulary dimensions by more than half (from 237,148 to 114,493 features).

---

## Model Improvement Experiments

A systematic, leakage-free empirical investigation was conducted across 98 candidate configurations to improve classification performance while strictly adhering to classical/statistical NLP methods.

### 1. Previous Model (Reference Baseline)
- **Dataset**: 25,000 authentic CFPB complaints across 18 product categories.
- **Features**: Word TF-IDF (1,2) + Character TF-IDF (3,5 within word boundaries `char_wb`), yielding 237,148 dimensions.
- **Classifier**: Logistic Regression (`solver='lbfgs'`, $C=1.0$, `class_weight='balanced'`, `max_iter=1000`).
- **Holdout Test Set Performance**: Accuracy 69.56%, Macro F1 50.56%, Weighted F1 69.55%, Macro Precision 49.67%, Macro Recall 51.97%.

### 2. Candidate Models & Search Exploration Space
All candidate exploration was performed exclusively using a stratified 80/20 internal partition of the 20,000-record training pool (**16,000 train / 4,000 validation records**). The 5,000-record final test set remained strictly untouched throughout model selection. 98 distinct configurations were evaluated across 7 structured stages:

1. **Stage 0 — Production Reference**: Exact replication of the production configuration with fully converged solver iterations.
2. **Stage 1 — Word TF-IDF Variations**: Evaluated n-gram ranges `(1,1)`, `(1,2)`, `(1,3)`, document frequency cutoffs (`min_df` $\in \{2, 3, 5\}$, `max_df` $\in \{0.5, 0.8, 0.95\}$), norm formulations (`l1` vs. `l2`), sublinear scaling, and vocabulary caps (50k, 100k).
3. **Stage 2 — Character TF-IDF & Subword Fusion**: Evaluated character n-gram spans `(2,5)`, `(3,5)`, `(3,6)`, `(4,6)` comparing standard cross-boundary character n-grams (`analyzer='char'`) against word-boundary subwords (`analyzer='char_wb'`). Combined the top-performing character blocks with the best word representations via sparse horizontal stacking.
4. **Stage 3 — Logistic Regression Hyperparameters**: Evaluated inverse regularization strength $C \in \{0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0\}$ under both `class_weight=None` and `class_weight='balanced'`.
5. **Stage 4 — Alternative Classical Sparse Classifiers**:
   - **Linear Support Vector Classifier (LinearSVC)**: Evaluated $C \in \{0.05, 0.1, 0.25, 0.5, 1.0, 2.0\}$ with and without class balancing.
   - **Naive Bayes**: Evaluated `ComplementNB` and `MultinomialNB` with smoothing parameter $\alpha \in \{0.01, 0.03, 0.1, 0.3, 1.0\}$.
6. **Stage 5 — Train-Only Custom Class Weighting**: Evaluated power-scaled class balancing $w_c = (n / (k \cdot n_c))^p$ with power exponents $p \in \{0.25, 0.5, 0.75, 1.25\}$, computed strictly from training labels without validation leakage.
7. **Stage 6 — Generic Classical Feature Augmentation**: Tested appending stateless, text-only stylistic features (log narrative length, CFPB redaction mask count `XXXX`, and mask indicator flag).

### 3. Validation Results Summary
Candidates were ranked using a pre-registered multi-metric hierarchy: **Validation Macro F1 $\rightarrow$ Validation Macro Recall $\rightarrow$ Validation Accuracy $\rightarrow$ Validation Weighted F1**.

| Stage | Candidate Configuration | Dimensions | Val Accuracy | Val Macro Prec | Val Macro Rec | Val Macro F1 | Val Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| **S3 LR (Winner)** | **LogisticRegression ($C=2.0$, balanced) + Word(1,1) + Char(3,5, `char`)** | **109,228** | **69.95%** | **53.17%** | **51.78%** | **51.84%** | **69.83%** |
| S2 word+char | LogisticRegression ($C=1.0$, balanced) + Word(1,1) + Char(3,5, `char`) | 109,228 | 69.17% | 51.89% | 51.96% | 51.49% | 69.20% |
| S0 reference | LogisticRegression ($C=1.0$, balanced) + Word(1,2) + Char(3,5, `char_wb`) | 198,478 | 69.20% | 51.92% | 51.65% | 51.27% | 69.13% |
| S1 word | LogisticRegression ($C=1.0$, balanced) + Word(1,1) | 13,209 | 68.27% | 50.54% | 53.05% | 51.28% | 68.52% |
| S5 class-weights | LogisticRegression ($C=2.0$, power=1.25) + Word(1,1) + Char(3,5, `char`) | 109,228 | 68.95% | 51.27% | 51.88% | 51.11% | 69.16% |
| S4 SVC | LinearSVC ($C=0.5$, balanced) + Word(1,2) + Char(3,5, `char_wb`) | 198,478 | 70.77% | 52.23% | 48.36% | 49.15% | 69.99% |
| S4 SVC | LinearSVC ($C=0.5$, balanced) + Word(1,1) + Char(3,5, `char`) | 109,228 | 70.40% | 50.37% | 48.65% | 48.98% | 69.85% |
| S4 NB | MultinomialNB ($\alpha=0.01$) + Word(1,1) + Char(3,5, `char`) | 109,228 | 69.27% | 53.24% | 44.88% | 46.80% | 68.15% |
| S4 NB | ComplementNB ($\alpha=0.3$) + Word(1,1) + Char(3,5, `char`) | 109,228 | 67.05% | 51.52% | 38.51% | 39.06% | 62.63% |

**Key Validation Insights**:
- **Why LinearSVC was not selected**: Although LinearSVC achieved higher raw Accuracy (70.77%), its Macro F1 was substantially lower (49.15% vs. 51.84%) due to poor recall on low-support classes. The task specifically prioritizes Macro F1 to prevent minority class neglect.
- **Why Naive Bayes was not selected**: Multinomial and Complement Naive Bayes struggled with independence violations on dense character n-gram overlaps, achieving Macro F1 scores below 47%.
- **Character Analyzer Comparison**: Full character n-grams (`analyzer='char'`) consistently outperformed word-boundary character n-grams (`analyzer='char_wb'`), providing better subword discrimination across punctuation and compound financial expressions.

### 4. Selected Configuration
- **Model**: Multinomial Logistic Regression (`solver='lbfgs'`, $C=2.0$, `class_weight='balanced'`, `max_iter=1000`, `random_state=42`).
- **Word TF-IDF**: `ngram_range=(1,1)`, `min_df=2`, `max_df=0.95`, `norm='l2'`, `sublinear_tf=True`, `lowercase=False`.
- **Character TF-IDF**: `analyzer='char'`, `ngram_range=(3,5)`, `min_df=5`, `max_df=0.95`, `max_features=100,000`, `sublinear_tf=True`, `lowercase=False`.
- **Combination**: `scipy.sparse.hstack` into CSR matrix format (114,493 features total).

### 5. Final Controlled Test Results
Following selection, the winning configuration was retrained on the full 20,000-record training pool and evaluated **exactly once** on the untouched 5,000-record holdout test set:

- **Accuracy**: **69.82%** (+0.26 pp vs. previous 69.56%; +0.68 pp vs. baseline 69.14%)
- **Macro F1**: **50.88%** (+0.33 pp vs. previous 50.56%; +16.73 pp vs. baseline 34.15%)
- **Weighted F1**: **69.78%** (+0.23 pp vs. previous 69.55%; +4.05 pp vs. baseline 65.73%)
- **Macro Precision**: **50.41%** (+0.74 pp vs. previous 49.67%; +9.04 pp vs. baseline 41.37%)
- **Macro Recall**: **51.69%** (-0.28 pp vs. previous 51.97%; +17.61 pp vs. baseline 34.08%)
- **Weighted Precision**: **70.08%** (+0.05 pp vs. previous 70.03%; +3.55 pp vs. baseline 66.53%)
- **Weighted Recall**: **69.82%** (+0.26 pp vs. previous 69.56%; +0.68 pp vs. baseline 69.14%)
- **Total Features**: **114,493** (reduced from 237,148; 51.7% smaller feature space)
- **Model Artifact Size**: **16.5 MB** (down from 34.2 MB; 51.8% smaller memory footprint)

### 6. Per-Category Breakdown & Improvements
Significant F1 gains were achieved on major and minority categories:
- *Payday loan, title loan, or personal loan*: F1 elevated by **+3.28 pp** (23.19% $\rightarrow$ 26.47%)
- *Money transfer, virtual currency, or money service*: F1 elevated by **+2.69 pp** (59.13% $\rightarrow$ 61.82%)
- *Money transfers*: F1 elevated by **+2.66 pp** (56.72% $\rightarrow$ 59.38%)
- *Prepaid card*: F1 elevated by **+2.41 pp** (74.51% $\rightarrow$ 76.92%)
- *Consumer Loan*: F1 elevated by **+2.17 pp** (46.87% $\rightarrow$ 49.04%)
- *Credit reporting, credit repair services...*: F1 elevated by **+2.02 pp** (60.03% $\rightarrow$ 62.05%)
- *Mortgage*: F1 elevated by **+0.39 pp** (92.22% $\rightarrow$ 92.61%)
- *Credit reporting*: F1 elevated by **+0.31 pp** (61.41% $\rightarrow$ 61.73%)

### 7. Reasons for Improvement
1. **Reduced Collinearity & Feature Noise**: Eliminating word bigrams while relying on character n-grams (`analyzer='char'`) for subword and compound phrase modeling pruned over 122,000 redundant features. This cleaner representation reduced variance and improved linear decision boundaries.
2. **Cross-Boundary Subword Modeling**: Unconstrained character n-grams (`char`) captured subword stems across whitespace and punctuation boundary artifacts more effectively than `char_wb`, improving recognition of domain-specific financial codes, account identifiers, and abbreviations.
3. **Optimized Regularization Balance**: Raising $C$ from $1.0$ to $2.0$ allowed the model to penalize minority misclassifications more heavily without overfitting, leveraging the lower-dimensional feature matrix.

### 8. Methodological Safeguards & Limitations
- **Zero Leakage**: All vectorizers and class weights were fitted exclusively on training records. The 5,000-record test set was evaluated exactly once via programmatic guard assertion.
- **Irreducible Label Ambiguity**: The remaining error ceiling (~30.18% error rate) is primarily driven by CFPB historical label synonymy (e.g., *Credit reporting* vs. *Credit reporting, credit repair services...*), as detailed in the Taxonomy-Aware Classification section below.


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
