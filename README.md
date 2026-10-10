# Customer Complaint Similarity & Categorisation

An academic Natural Language Processing (NLP) system designed to analyze consumer financial complaints, compute pairwise semantic similarity using classical vector-space models, categorize complaints into financial product categories using supervised machine learning, and investigate target taxonomy formulations using regulatory data.

---

## Overview

Consumer financial institutions receive thousands of customer complaints daily across numerous product verticals, including credit reporting, debt collection, mortgages, credit cards, student loans, and bank accounts. Manually triaging incoming complaints and identifying recurring grievances is labor-intensive, slow, and prone to subjective inconsistencies.

This project implements an end-to-end, interpretable classical NLP pipeline that:
1. Cleans and standardizes raw unstructured consumer complaint narratives.
2. Represents text documents using Term Frequency-Inverse Document Frequency (TF-IDF) combining word unigrams and subword character n-grams into a compact 114,493-dimensional sparse matrix.
3. Computes document-to-document Cosine Similarity to retrieve historically similar complaints.
4. Classifies complaints into designated product categories using class-balanced Multinomial Logistic Regression.
5. Diagnoses error patterns and confidence distributions on a 5,000-sample holdout test partition.
6. Investigates how administrative product taxonomy revisions affect classification task difficulty (Reference 18-class vs. Conservative 11-class vs. Broad 10-class formulations).
7. Presents an interactive Streamlit web dashboard for live complaint exploration, similarity search, and model inference.
8. Integrates with the official Consumer Financial Protection Bureau (CFPB) API for real-time live queries without requiring an API key.

In accordance with strict classical machine learning constraints, this system relies exclusively on statistical and classical machine learning methods. Pretrained language models (such as BERT or Transformers) and external generative LLM APIs are deliberately excluded.

---

## Features

- **Supervised Multi-Class Categorisation**: Multinomial Logistic Regression trained with class-frequency balancing to handle extreme class imbalance across financial product categories.
- **Subword-Aware TF-IDF Representation**: Feature fusion combining word unigrams with cross-boundary character 3-5 n-grams, capturing domain abbreviations, financial roots, and spelling variations in 114,493 sparse CSR features.
- **Pairwise Semantic Similarity Retrieval**: High-efficiency sparse cosine similarity search over indexed historical complaints with adjustable top-K retrieval and similarity score thresholding.
- **Interactive Streamlit Web Dashboard**: Full-featured web interface featuring live complaint analysis, step-by-step pipeline tracing, interactive Plotly charts, model diagnostics, and dataset exploration.
- **Official CFPB API Integration**: Native HTTP client querying the official public CFPB Consumer Complaint Database API v1 with pagination, narrative filtering, and schema normalization.
- **Diagnostic Error Analysis**: Exhaustive evaluation on an untouched 5,000-record holdout test set, identifying top confusion pairs, probability calibration, and administrative label synonymy.
- **Taxonomy Formulation Auditing**: Empirical comparison between the official 18-class reference taxonomy, an 11-class conservative taxonomy, and a 10-class broad taxonomy with error reduction decomposition.
- **Zero Data Leakage**: Strict split discipline with a 20,000-record training pool (16,000 train / 4,000 validation) and an untouched 5,000-record holdout test partition evaluated strictly once.
- **Comprehensive Automated Test Suite**: Full unit, integration, UI regression, and deployment artifact verification tests.
- **Cross-Platform Compatibility**: Validated workflows and setup instructions for Windows, Linux, and macOS.

---

## Quick Start

For experienced developers seeking immediate setup:

```bash
# Clone repository
git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git
cd Customer-Complaint-NLP

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run test suite
python -m pytest

# Launch Streamlit dashboard
streamlit run app/app.py
```

The Streamlit dashboard will be accessible in your web browser at `http://localhost:8501`.

---

## NLP Pipeline

The pipeline follows a classical, interpretable NLP architecture:

```text
Raw Consumer Complaint Narrative
              |
              v
      Text Preprocessing
 (Lowercasing, Cleaning, Tokenization, Stopword Filtering)
              |
              v
     TF-IDF Vectorisation
 + Word Unigrams: (1, 1), min_df=2, max_df=0.95
 + Character N-grams: (3, 5), min_df=5, max_df=0.95, analyzer='char'
              |
              v
     Sparse CSR Feature Matrix
        (114,493 Dimensions)
              |
      +-------+-------+
      |               |
      v               v
Cosine Similarity    Supervised Classification
  (Retrieval)        (Logistic Regression, C=2.0, balanced)
      |               |
      v               v
Top-K Similar        Predicted Category &
 Historical Records   Confidence Distribution
                      |
                      v
             Evaluation & Diagnostics
          (Precision, Recall, F1, Confusion Matrix)
```

1. **Text Preprocessing**: Normalizes text by lowercasing, stripping punctuation artifacts and special symbols, tokenizing into words, and filtering common English stopwords.
2. **TF-IDF Feature Extraction**: Computes TF-IDF representations separately for word unigrams and character 3-5 n-grams, then stacks them horizontally into a compressed sparse row (CSR) matrix.
3. **Cosine Similarity Engine**: Measures angular distance between query vectors and pre-vectorized historical complaints to retrieve the most semantically relevant historical complaints.
4. **Supervised Classification Engine**: Applies class-balanced Multinomial Logistic Regression to compute class posterior probabilities and assign the highest-scoring product label.
5. **Evaluation and Diagnostics**: Generates classification reports, confusion matrices, and error breakdowns on held-out test data.

---

## Algorithms Used

The project strictly separates feature representation, similarity retrieval, and classification algorithms:

1. **Text Preprocessing (Normalization & Tokenization)**:
   Regex-based character cleaning, lowercasing, whitespace normalization, and NLTK-based English stopword filtering. An optional minimal preprocessing mode is also supported for character n-gram boundary preservation.

2. **Term Frequency-Inverse Document Frequency (TF-IDF)**:
   *Role*: Feature representation method (NOT a classification algorithm).
   Computes statistical weights for terms based on within-document frequency and inverse collection frequency:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left(\log \frac{1 + |D|}{1 + \text{DF}(t, D)} + 1\right)$$
   Features are L2-normalized across documents.

3. **Word N-Grams**:
   *Role*: Semantic lexical feature extraction.
   Configured with unigram span `(1, 1)` to capture individual financial keywords (such as *debt*, *credit*, *escrow*, *overdraft*) while eliminating redundant, noisy bigram features.

4. **Character Subword N-Grams**:
   *Role*: Subword morphological feature extraction.
   Configured with span `(3, 5)` using `analyzer='char'` across word boundaries. Captures grammatical affixes, financial stems (*foreclos-*, *delinqu-*), abbreviations (*FCRA*, *CFPB*, *APR*), account masking patterns (*XXXX*), and typographic misspellings.

5. **Sparse Matrix Representation (Scipy CSR)**:
   *Role*: Memory-efficient sparse feature layout.
   Uses `scipy.sparse.hstack(..., format="csr")` to join 14,493 word features and 100,000 character features into 114,493 sparse dimensions without allocating dense arrays.

6. **Cosine Similarity**:
   *Role*: Similarity and document retrieval method (NOT a classifier).
   Calculates the normalized dot product between L2-normalized sparse document vectors $u$ and $v$:
   $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
   Because TF-IDF vectors are L2-normalized during extraction, cosine similarity equals the inner product $u \cdot v^T$, computed efficiently via sparse matrix multiplication.

7. **Multinomial Logistic Regression**:
   *Role*: Supervised multi-class classification algorithm.
   Models class probabilities using a softmax linear function:
   $$P(Y = k \mid X) = \frac{\exp(w_k^T X + b_k)}{\sum_{j=1}^K \exp(w_j^T X + b_j)}$$
   Optimized with the `lbfgs` solver, inverse regularization parameter $C=2.0$, and `class_weight='balanced'` to scale loss penalties inversely proportional to class frequencies.

8. **Statistical Evaluation Metrics**:
   *Role*: Quantitative assessment of model discrimination.
   Evaluates multi-class performance using Macro Precision, Macro Recall, Macro F1, Weighted F1, Overall Accuracy, and an 18-class Confusion Matrix.

---

## Project Architecture

```text
Customer-Complaint-NLP/
|
|-- config/                          # Taxonomy mapping configurations
|   |-- taxonomy_v1_conservative.json
|   `-- taxonomy_v2_broad.json
|
|-- data/                            # Dataset directory (raw CSV excluded from Git)
|   |-- README.md                    # Data catalog documentation and download guide
|   `-- .gitkeep
|
|-- docs/                            # In-depth technical documentation
|   |-- error-analysis.md            # Test set error analysis and confusion analysis
|   |-- project-updates.md           # Engineering log of completed milestones
|   |-- taxonomy-analysis.md         # Regulatory taxonomy audit and error decomposition
|   `-- viva-preparation.md          # Theoretical examination preparation guide
|
|-- models/                          # Version-controlled production model artifacts
|   |-- complaint_classifier.joblib  # Trained Multinomial Logistic Regression model
|   |-- tfidf_vectorizer.joblib      # Fitted Word TF-IDF vectorizer (14,493 features)
|   |-- char_vectorizer.joblib       # Fitted Character TF-IDF vectorizer (100,000 features)
|   `-- .gitkeep
|
|-- notebooks/                       # Exploratory Jupyter notebooks
|   |-- 01_dataset_exploration.ipynb
|   `-- .gitkeep
|
|-- results/                         # Generated evaluation tables and figures
|   |-- baseline_vs_improved.csv
|   |-- error_analysis.csv
|   |-- per_category_metrics.csv
|   |-- confusion_matrix.png
|   `-- .gitkeep
|
|-- scripts/                         # Standalone execution and evaluation scripts
|   |-- analyze_taxonomy.py          # Taxonomy group auditing script
|   |-- diagnostic_audit.py          # System diagnostic inspection
|   |-- generate_error_analysis.py   # Test set error pair generation
|   |-- run_experiments.py           # Historical 30-configuration grid search
|   |-- run_model_improvement.py     # 98-configuration staged model optimization
|   |-- run_post_audit_experiments.py# Preprocessing and boundary experiments
|   `-- run_taxonomy_experiment.py   # Normalized taxonomy benchmark experiments
|
|-- src/                             # Core Python package modules
|   |-- __init__.py
|   |-- cfpb_api.py                  # Official CFPB API HTTP client
|   |-- classification.py            # Model training, inference, and validation
|   |-- data_loader.py               # Dataset loading and column resolution
|   |-- evaluation.py                # Metric calculations and reporting
|   |-- model_selection.py           # Staged model selection utilities
|   |-- preprocessing.py             # Text cleaning, normalization, and tokenization
|   |-- similarity.py                # Cosine similarity search engine
|   `-- vectorization.py             # Word and character TF-IDF feature extraction
|
|-- tests/                           # Pytest automated test suite
|   |-- __init__.py
|   |-- test_cfpb_api.py             # CFPB API client unit tests
|   |-- test_classification.py       # Classifier unit tests
|   |-- test_data_loader.py          # Dataset loader schema and error handling tests
|   |-- test_deployment_artifacts.py # Production artifact integrity and dimension tests
|   |-- test_error_analysis.py       # Error analysis helper tests
|   |-- test_evaluation.py           # Metric calculation tests
|   |-- test_model_improvements.py   # Subword vectorizer and classifier tests
|   |-- test_model_selection.py      # Staged search protocol tests
|   |-- test_post_audit_improvements.py # Preprocessing variant tests
|   |-- test_preprocessing.py        # Tokenization and cleaning tests
|   |-- test_rendered_ui_verification.py # UI component rendering tests
|   |-- test_run_search.py           # Improvement pipeline execution tests
|   |-- test_similarity.py           # Cosine similarity ranking tests
|   |-- test_streamlit_app.py        # Streamlit entry point and helper tests
|   |-- test_taxonomy.py             # Taxonomy mapping consistency tests
|   |-- test_ui_charts.py            # Chart generation tests
|   |-- test_ui_components.py        # Frontend card and badge tests
|   |-- test_ui_regression_and_html.py # UI layout regression tests
|   `-- test_vectorization.py        # TF-IDF feature extraction tests
|
|-- app/                             # Streamlit web application
|   |-- __init__.py
|   |-- app.py                       # Main Streamlit dashboard application
|   |-- charts.py                    # Plotly chart generation module
|   `-- ui_components.py             # UI cards, navigation, and styled widgets
|
|-- .gitignore                       # Git exclusion rules
|-- CODE_OF_CONDUCT.md               # Contributor Covenant Code of Conduct
|-- CONTRIBUTING.md                  # Contribution guidelines and workflow
|-- LICENSE                          # MIT License
|-- pyproject.toml                   # Project metadata and pytest configuration
|-- README.md                        # Project documentation
`-- requirements.txt                 # Pinned project dependencies
```

---

## Dataset

This project utilizes the official **Consumer Complaint Database** maintained by the **Consumer Financial Protection Bureau (CFPB)**, an agency of the United States federal government.

- **Official Source**: [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Data Catalog Entry**: [https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)
- **Direct Database Archive**: [https://files.consumerfinance.gov/ccdb/complaints.csv.zip](https://files.consumerfinance.gov/ccdb/complaints.csv.zip)
- **Dataset Size**: **25,000 complaints** across **18 original Product categories**.
- **Data Partitions**:
  - **Training Pool**: 20,000 records (internally split into 16,000 training records and 4,000 validation records for model selection).
  - **Holdout Test Set**: 5,000 records partitioned via stratified sampling with `random_state=42`, kept untouched throughout model selection and evaluated strictly once.
- **Primary Text Field**: `Consumer complaint narrative` (CFPB portal export format) or `Consumer Complaint` (narrative archive format) or `complaint_what_happened` (API format).
- **Target Category Field**: `Product` (financial product/service classification).
- **Identifier Field**: `Complaint ID` (numeric complaint tracking identifier).
- **Data Policy**: Raw dataset files (`*.csv`) are excluded from Git tracking via `.gitignore` to maintain repository size standards. See [data/README.md](data/README.md) for full schema documentation and column details.

---

## Repository Structure

The repository organizes code cleanly by responsibility:
- `src/`: Modular, reusable Python package containing data loading, preprocessing, vectorization, classification, similarity retrieval, evaluation, and API client logic.
- `app/`: Streamlit dashboard code, custom visual components, and interactive Plotly chart generators.
- `models/`: Production model artifacts version-controlled in Git for instant local deployment.
- `data/`: Dataset storage location and documentation (raw CSV files excluded by `.gitignore`).
- `scripts/`: Reproducible standalone scripts for model training, staged hyperparameter search, error analysis, and taxonomy evaluation.
- `tests/`: Comprehensive Pytest automated test suite covering all modules, artifacts, and UI components.
- `docs/`: In-depth reports on error analysis, taxonomy formulation, engineering updates, and viva examination preparation.

---

## Prerequisites

Before installing and running the project, verify that the following prerequisites are met:

- **Python Version**: Python 3.9 or higher (Python 3.9+) is required as specified in `pyproject.toml` (`requires-python = ">=3.9"`). The project does not enforce a single exact Python patch version.
- **Git**: Git version control is recommended for cloning and version management. Verify with:
  ```bash
  git --version
  ```
- **Operating System**: Supported on Windows 10/11, Linux (Debian, Ubuntu, Fedora, CentOS, Arch), and macOS (11+).
- **Hardware Resources**:
  - Memory: 4 GB RAM minimum (8 GB recommended for running systematic model training scripts).
  - Storage: Approximately 500 MB for repository files and Python dependencies; approximately 1.5 GB if storing the full uncompressed CFPB CSV locally.

---

## Downloading the Project

Choose one of three practical methods to download the project:

### Method 1 — Git Clone

This is the recommended method for developers and contributors.

1. Verify that Git is installed on your system:
   ```bash
   git --version
   ```
2. Clone the repository using Git:
   ```bash
   git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git
   ```
3. Navigate into the project directory:
   ```bash
   cd Customer-Complaint-NLP
   ```

*Benefits*: Retains complete Git history, enables branch switching, and allows pulling updates using `git pull`.

### Method 2 — Download ZIP

For users who do not have Git installed or prefer a direct archive download:

1. Open a web browser and navigate to the repository:
   `https://github.com/Java-Mx/Customer-Complaint-NLP`
2. Click the green **Code** button located near the top right of the file listing.
3. Select **Download ZIP** from the dropdown menu.
4. Extract the downloaded ZIP file:
   - **Windows**: Right-click `Customer-Complaint-NLP-main.zip`, select **Extract All...**, choose your target directory, and click **Extract**.
   - **Linux**: Open a terminal and run:
     ```bash
     unzip Customer-Complaint-NLP-main.zip -d Customer-Complaint-NLP
     ```
   - **macOS**: Double-click the downloaded `.zip` file in Finder, or run in Terminal:
     ```bash
     unzip Customer-Complaint-NLP-main.zip
     ```
5. Open your terminal or command prompt and change into the extracted folder:
   ```bash
   cd Customer-Complaint-NLP-main
   ```

*Note*: Downloading a ZIP archive does not include Git version tracking or the `.git` directory. You will not be able to use Git commands (`git pull`, `git status`) unless you initialize Git manually.

### Method 3 — GitHub Desktop

For users who prefer a graphical Git workflow:

1. Open GitHub Desktop.
2. Select **File > Clone Repository...** (or click **Clone a repository from the Internet...**).
3. In the repository URL or repository identifier field, enter:
   `Java-Mx/Customer-Complaint-NLP`
   or the complete clone URL:
   `https://github.com/Java-Mx/Customer-Complaint-NLP.git`
4. Choose a local destination directory on your computer.
5. Click **Clone**.
6. After cloning completes, open the project in your terminal or preferred code editor (such as VS Code or PyCharm).

---

## Installation

Follow the platform-specific instructions below to set up the project on your operating system.

### Windows

1. Open **PowerShell** or **Command Prompt**.
2. Navigate into the cloned or extracted project folder:
   ```powershell
   cd path\to\Customer-Complaint-NLP
   ```
3. Verify your Python installation:
   ```powershell
   python --version
   ```
4. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
5. Activate the virtual environment (see the [Virtual Environment](#virtual-environment) section for execution policy details):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
6. Upgrade pip and install all project dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### Linux

1. Open a terminal.
2. Navigate into the project folder:
   ```bash
   cd Customer-Complaint-NLP
   ```
3. Verify your Python 3 installation:
   ```bash
   python3 --version
   ```
4. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
   *Note*: On Debian or Ubuntu distributions, if virtual environment creation fails with a message indicating that `ensurepip` is missing, install the `python3-venv` package:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip
   ```
5. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
6. Upgrade pip and install all dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### macOS

1. Open **Terminal**.
2. Navigate into the project folder:
   ```bash
   cd Customer-Complaint-NLP
   ```
3. Verify your Python 3 installation:
   ```bash
   python3 --version
   ```
   *Note*: If command-line developer tools are prompted, install them with:
   ```bash
   xcode-select --install
   ```
4. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
5. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
6. Upgrade pip and install all dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

## Virtual Environment

Isolating dependencies inside a Python virtual environment prevents package conflicts with other projects or system tools.

### Windows PowerShell

To create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**PowerShell Execution Policy**:
If PowerShell returns an execution policy restriction error (`PSSecurityException` or `running scripts is disabled on this system`), allow script execution for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This bypasses script restrictions only for your current PowerShell window without altering system-wide administrative settings.

To deactivate the virtual environment when finished:
```powershell
deactivate
```

### Windows Command Prompt

If you prefer using standard Windows Command Prompt (`cmd.exe`):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

To deactivate:
```cmd
deactivate
```

### Linux

Using `bash` or `zsh` on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate:
```bash
deactivate
```

### macOS

Using `zsh` or `bash` on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate:
```bash
deactivate
```

---

## Install Dependencies

Once your virtual environment is active, install the project dependencies specified in `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Key Dependencies

The project relies on a verified, classical machine learning stack:

| Package | Minimum Version | Role in Project |
|---|---|---|
| `pandas` | `>=2.0.0` | Dataset manipulation, column mapping, and CSV ingestion |
| `numpy` | `>=1.24.0` | Vector operations and numerical computations |
| `scipy` | `>=1.10.0` | Sparse matrix storage (`scipy.sparse.csr_matrix`, `hstack`) |
| `joblib` | `>=1.3.0` | Serialization and loading of model and vectorizer artifacts |
| `scikit-learn` | `>=1.3.0` | TF-IDF vectorization, Logistic Regression, LinearSVC, metrics |
| `nltk` | `>=3.8.0` | English stopword lists and word tokenization |
| `matplotlib` | `>=3.7.0` | Static visualization and confusion matrix plotting |
| `seaborn` | `>=0.12.0` | Statistical heatmaps |
| `streamlit` | `>=1.30.0` | Interactive web dashboard framework |
| `plotly` | `>=5.18.0` | Interactive dashboard charts (heatmaps, comparisons, distributions) |
| `pytest` | `>=7.4.0` | Automated unit, integration, and UI testing |
| `requests` | `>=2.28.0` | HTTP client for official CFPB Search API v1 |
| `jupyter` | `>=1.0.0` | Interactive notebook exploration (`notebooks/`) |

---

## Dataset Setup

The project supports two operational modes: **offline CSV analysis** and **live CFPB API querying**.

### Mode A — Offline CSV Mode (Local Dataset)

To run offline analysis, model training scripts, or full dataset exploration:

1. Download the CFPB dataset from the official portal:
   - Official portal: [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
   - Direct download archive: [https://files.consumerfinance.gov/ccdb/complaints.csv.zip](https://files.consumerfinance.gov/ccdb/complaints.csv.zip)
2. Extract the archive and place the CSV file at:
   ```text
   data/complaints.csv
   ```
3. Alternatively, place a smaller sampled file at `data/complaints_sample.csv` for lightweight development.

**Automatic Column Resolution**:
The project loader (`src/data_loader.py`) automatically maps raw column names to standardized fields:
- **Complaint Text**: Resolves `Consumer Complaint`, `Consumer complaint narrative`, `complaint_what_happened`, or `text` into canonical `text`.
- **Product Category**: Resolves `Product` or `category` into canonical `category`.
- **Complaint ID**: Resolves `Complaint ID` or `complaint_id` into canonical `complaint_id`.

**Git Exclusion**:
Raw dataset files (`data/*.csv`) are strictly excluded from Git version control by `.gitignore`. Do not commit raw CSV data files to Git.

### Mode B — Live CFPB API Mode (Zero Local CSV Required)

If `data/complaints.csv` is not present on your system, the project continues to function:
- The Streamlit web dashboard (`app/app.py`) provides preloaded authentic demonstration complaints for instant live inference, classification, and similarity search.
- The built-in **CFPB Live API & Data Explorer** queries live complaint records directly over the network via `src.cfpb_api.CFPBClient`.
- Programmatic scripts can fetch real-time complaints via:
  ```python
  from src.data_loader import get_complaints_data
  df = get_complaints_data(source="api", max_records=50)
  ```

---

## Model Artifact Setup

The project includes verified, serialized production model artifacts stored in the `models/` directory:

```text
models/
|-- complaint_classifier.joblib   # Trained class-balanced Multinomial Logistic Regression (~15.7 MB)
|-- tfidf_vectorizer.joblib       # Fitted Word TF-IDF vectorizer (14,493 features, ~0.3 MB)
`-- char_vectorizer.joblib        # Fitted Character TF-IDF vectorizer (100,000 features, ~3.3 MB)
```

### Production Artifact Details

- **`complaint_classifier.joblib`**: A Multinomial Logistic Regression model trained on 20,000 authentic complaints across 18 product categories with `C=2.0` and `class_weight='balanced'`.
- **`tfidf_vectorizer.joblib`**: Word unigram vectorizer with `ngram_range=(1, 1)`, `min_df=2`, `max_df=0.95`, and `sublinear_tf=True`, capturing 14,493 vocabulary terms.
- **`char_vectorizer.joblib`**: Cross-boundary character subword vectorizer with `analyzer='char'`, `ngram_range=(3, 5)`, `min_df=5`, `max_df=0.95`, and `max_features=100,000`, capturing 100,000 character n-grams.
- **Combined Dimension**: Exactly **114,493 sparse features**.

### Version Control & Acquisition

- **Included with the Repository**: Unlike raw CSV files, the three production model artifacts are version-controlled and tracked directly in Git (whitelisted in `.gitignore` via `!models/complaint_classifier.joblib`, `!models/tfidf_vectorizer.joblib`, and `!models/char_vectorizer.joblib`).
- **No Retraining Required**: When you clone or download the repository, the production models are already included and immediately ready for use by Streamlit and tests.
- **Integrity Validation**: When the Streamlit application starts, `src.classification.validate_model_artifacts()` automatically verifies that all three files exist, can be deserialized cleanly, and match the expected 114,493 feature dimensions.
- **Optional Local Retraining**: If you modify the training pipeline or wish to regenerate artifacts from scratch, run:
  ```bash
  python scripts/run_model_improvement.py final
  ```

---

## Run Tests

The repository contains an automated test suite executed with Pytest.

### Running the Full Test Suite

Activate your virtual environment and run:

```bash
python -m pytest
```

To run quietly with summary-only output:

```bash
python -m pytest -q
```

To run with detailed per-test reporting:

```bash
python -m pytest -v
```

### Test Suite Coverage

The test suite thoroughly validates every layer of the repository across test modules:
- `tests/test_cfpb_api.py`: Official CFPB API client initialization, query building, pagination, error handling, and schema normalization.
- `tests/test_classification.py`: Classifier instantiation, training, prediction, posterior probabilities, and decision thresholding.
- `tests/test_data_loader.py`: CSV dataset loading, column mapping adaptation, schema validation, and missing-value handling.
- `tests/test_deployment_artifacts.py`: Production artifact existence, deserialization, feature compatibility (114,493 dimensions), and end-to-end smoke predictions.
- `tests/test_error_analysis.py`: Error analysis utilities, confusion pair extraction, and probability calibration.
- `tests/test_evaluation.py`: Metric calculation functions, classification report structures, and confusion matrix builders.
- `tests/test_model_improvements.py` & `tests/test_model_selection.py`: Subword vectorizer fusion, staged search protocols, and ranking hierarchies.
- `tests/test_post_audit_improvements.py`: Minimal preprocessing rules and punctuation boundary retention.
- `tests/test_preprocessing.py`: Text cleaning, normalization, tokenization, and stopword removal.
- `tests/test_similarity.py`: Cosine similarity computation, Top-K ranking, and edge case handling (empty strings, identical inputs).
- `tests/test_taxonomy.py`: Conservative (11-class) and broad (10-class) taxonomy mappings and error-collapse accounting.
- `tests/test_vectorization.py`: Word and character TF-IDF vectorizer configuration, fitting, and sparse horizontal stacking.
- `tests/test_streamlit_app.py`, `tests/test_ui_charts.py`, `tests/test_ui_components.py`, `tests/test_ui_regression_and_html.py`, `tests/test_rendered_ui_verification.py`: Streamlit entry points, Plotly figure builders, layout structure, and UI regression guards.

---

## Run the Streamlit Application

The Streamlit web application provides an academic, interactive dashboard for testing the NLP pipeline.

### Launching the Application

Make sure your virtual environment is active, then run:

```bash
streamlit run app/app.py
```

Or invoke Streamlit via Python explicitly:

- **Windows**:
  ```powershell
  python -m streamlit run app/app.py
  ```
- **Linux / macOS**:
  ```bash
  python3 -m streamlit run app/app.py
  ```

### Accessing the Web Interface

Streamlit will launch a local web server and display the application address in your terminal:

```text
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

Open `http://localhost:8501` in your web browser.

### Running on a Custom Port

If port 8501 is already in use by another service:

```bash
streamlit run app/app.py --server.port 8502
```

### Stopping the Application

To stop the Streamlit server, switch to your terminal window and press:

```text
Ctrl + C
```

### Application Features

The interactive application includes:
- **LIVE Complaint Analysis (Opening Module)**: Type or paste any unseen customer complaint narrative, or click one of the authentic demonstration examples. Immediately inspects:
  - Predicted product category and confidence score with top-5 class distribution bars.
  - Step-by-step pipeline trace (raw text, cleaned text, 114,493-dimensional sparse CSR representation, active feature counts).
  - Highest-weighted active word and character n-grams extracted from the input narrative.
  - Semantically similar historical complaints retrieved via sparse cosine similarity.
- **Model & Dataset Insights Dashboard**: Interactive Plotly visualizations comparing baseline vs. improved models, dataset distributions, per-category F1 metrics, top confusion pairs, and 18-class confusion matrix heatmaps.
- **CFPB Live API & Data Explorer**: Live Elasticsearch querying of the official CFPB Search API v1 and interactive exploration of the local dataset.
- **Model Evaluation & Diagnostics**: Performance metrics, confusion matrices, and controlled benchmark comparisons.
- **Error Analysis**: Confusion pairs and confidence distribution breakdown on the 5,000-record holdout test set.
- **Taxonomy Analysis**: Conservative (11 categories) and Broad (10 categories) taxonomy formulation audits and retraining decompositions.
- **Cosine Similarity Retrieval**: Independent similarity query engine against indexed historical complaints.
- **Text Preprocessing & TF-IDF Inspector**: Stage-by-stage tokenization, n-gram extraction, and vocabulary weights inspection.

---

## Run the Project from the Command Line

In addition to the Streamlit web interface, the project includes standalone command-line scripts in `scripts/`:

### 1. Systematic Model Improvement Search

To run the staged model improvement exploration across validation configurations (training pool internal 16k train / 4k validation split; does not touch the test set):

```bash
python scripts/run_model_improvement.py search
```

### 2. Retrain and Evaluate Winning Model

To retrain the selected winning configuration on the full 20,000-record training pool and evaluate once on the 5,000-record holdout test set:

```bash
python scripts/run_model_improvement.py final
```

### 3. Generate Diagnostic Error Analysis

To generate diagnostic error analysis metrics, confusion pairs, and calibration data on the 5,000-record test set:

```bash
python scripts/generate_error_analysis.py
```

### 4. Execute Taxonomy Audit and Experiments

To inspect category distributions, merge candidates, and evaluate normalized 11-category and 10-category formulations:

```bash
python scripts/analyze_taxonomy.py
python scripts/run_taxonomy_experiment.py
```

### 5. Historical 30-Run Grid Search

To run the historical validation grid search across baseline configurations:

```bash
python scripts/run_experiments.py
```

---

## CFPB API

The application provides integration with the official **Consumer Financial Protection Bureau (CFPB) Complaint Search API v1**.

- **Official Base Endpoint**:
  `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`
- **Authentication**: **No API key is required**. The CFPB search API is a publicly accessible federal service.
- **Client Implementation**: `src.cfpb_api.CFPBClient` and helper function `fetch_cfpb_data()`.
- **Fields Retrieved**: Complaint narrative (`complaint_what_happened`), financial product (`product`), sub-product, issue, company, state, date received, and unique complaint identifier (`complaint_id`).
- **Normalized Schema**: The client normalizes raw API Elasticsearch responses into standard fields (`text`, `category`, `complaint_id`).
- **Unified Dispatcher**: `src.data_loader.get_complaints_data(source="api", max_records=50)` allows fetching live complaints without requiring a local CSV file.
- **Optional vs. Required**: API access is completely optional. When working offline with local files (`data/complaints.csv` and `models/`), network connectivity is not required. When local CSV data is absent, the application uses live API access as an optional data source.

---

## Example Usage

### 1. Preprocessing and TF-IDF Representation

```python
from src.preprocessing import preprocess_text
from src.vectorization import transform_word_char, get_top_active_features
from src.classification import validate_model_artifacts

# Load verified production artifacts
artifacts = validate_model_artifacts()
w_vec = artifacts["word_vectorizer"]
c_vec = artifacts["char_vectorizer"]

# Clean and vectorize input narrative
narrative = "The bank charged an unexpected overdraft fee on my checking account without notice."
cleaned_text = preprocess_text(narrative)
X_sparse = transform_word_char(w_vec, c_vec, [cleaned_text])

print(f"Sparse matrix shape: {X_sparse.shape}")  # (1, 114493)
print(f"Non-zero features: {X_sparse.nnz}")

# Inspect highest-weighted active features
top_features = get_top_active_features(w_vec, c_vec, X_sparse, top_n=5)
for feat, weight in top_features:
    print(f"  {feat}: {weight:.4f}")
```

### 2. Complaint Classification & Probability Estimation

```python
from src.classification import predict_complaint_category, predict_category_proba

clf = artifacts["classifier"]

# Predict category and confidence
pred_cat, confidence = predict_complaint_category(
    model=clf,
    vectorizer=(w_vec, c_vec),
    narrative=narrative,
    preprocess=True,
)
print(f"Predicted Category: {pred_cat}")
print(f"Confidence: {confidence:.2%}")

# Compute probability distribution across all 18 classes
probabilities = predict_category_proba(clf, X_sparse)[0]
top_indices = probabilities.argsort()[-3:][::-1]
for idx in top_indices:
    print(f"  {clf.classes_[idx]}: {probabilities[idx]:.2%}")
```

### 3. Pairwise Cosine Similarity Search

```python
import pandas as pd
from src.similarity import find_similar_complaints

# Reference dataset
corpus_df = pd.DataFrame({
    "complaint_id": ["C1", "C2", "C3"],
    "text": [
        "Unauthorized transactions appeared on my credit card statement.",
        "Overdraft fee charged to checking account without notification.",
        "Mortgage servicer did not apply monthly escrow payment.",
    ],
    "category": [
        "Credit card",
        "Checking or savings account",
        "Mortgage",
    ]
})

similar_results = find_similar_complaints(
    query_text="Bank charged me an unfair overdraft penalty on my checking account.",
    complaints_df=corpus_df,
    vectorizer=(w_vec, c_vec),
    top_k=2,
)

print(similar_results[["complaint_id", "category", "similarity_score"]])
```

### 4. Querying the Official CFPB Live API

```python
from src.cfpb_api import CFPBClient

client = CFPBClient()
df_live = client.fetch_complaints(size=10, search_term="escrow")
print(f"Retrieved {len(df_live)} live complaints from CFPB API")
print(df_live[["complaint_id", "category", "text"]].head())
```

---

## Cross-Platform Command Table

The following comparison table shows equivalent commands across Windows PowerShell, Windows Command Prompt, Linux, and macOS:

| Task | Windows PowerShell | Windows Command Prompt | Linux (bash/zsh) | macOS (zsh/bash) |
|---|---|---|---|---|
| **Check Python** | `python --version` | `python --version` | `python3 --version` | `python3 --version` |
| **Check Git** | `git --version` | `git --version` | `git --version` | `git --version` |
| **Clone Repo** | `git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git` | `git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git` | `git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git` | `git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git` |
| **Navigate** | `cd Customer-Complaint-NLP` | `cd Customer-Complaint-NLP` | `cd Customer-Complaint-NLP` | `cd Customer-Complaint-NLP` |
| **Create venv** | `python -m venv .venv` | `python -m venv .venv` | `python3 -m venv .venv` | `python3 -m venv .venv` |
| **Activate venv** | `.\.venv\Scripts\Activate.ps1` | `.venv\Scripts\activate.bat` | `source .venv/bin/activate` | `source .venv/bin/activate` |
| **Deactivate venv** | `deactivate` | `deactivate` | `deactivate` | `deactivate` |
| **Upgrade pip** | `python -m pip install --upgrade pip` | `python -m pip install --upgrade pip` | `python -m pip install --upgrade pip` | `python -m pip install --upgrade pip` |
| **Install reqs** | `python -m pip install -r requirements.txt` | `python -m pip install -r requirements.txt` | `python -m pip install -r requirements.txt` | `python -m pip install -r requirements.txt` |
| **Run tests** | `python -m pytest` | `python -m pytest` | `python -m pytest` | `python -m pytest` |
| **Run Streamlit** | `streamlit run app/app.py` | `streamlit run app/app.py` | `streamlit run app/app.py` | `streamlit run app/app.py` |
| **Alt Streamlit** | `python -m streamlit run app/app.py` | `python -m streamlit run app/app.py` | `python3 -m streamlit run app/app.py` | `python3 -m streamlit run app/app.py` |

---

## Troubleshooting

The table below lists common setup and runtime issues along with verified solutions:

| Problem | Possible Cause | Solution |
|---|---|---|
| `'python'` or `'python3'` is not recognized as an internal or external command | Python is not installed or not added to your system's `PATH` environment variable | Install Python 3.9+ from [python.org](https://www.python.org/) or your distribution package manager. On Windows, ensure **"Add Python to PATH"** is checked during installation. |
| PowerShell returns `cannot be loaded because running scripts is disabled on this system` | Windows PowerShell execution policy restricts running unsigned activation scripts | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell window, then retry `.\.venv\Scripts\Activate.ps1`. Alternatively, use Command Prompt with `.venv\Scripts\activate.bat`. |
| Linux: `The virtual environment was not created successfully because ensurepip is not available` | Debian/Ubuntu base installations separate the virtual environment package | Install the required venv package: `sudo apt update && sudo apt install -y python3-venv python3-pip`. Then rerun `python3 -m venv .venv`. |
| `pip install -r requirements.txt` fails with compiler or permission errors | Pip is outdated or global system environment is write-protected | Ensure the virtual environment is active (terminal prompt should show `(.venv)`). Upgrade pip first: `python -m pip install --upgrade pip`, then rerun the installation command. |
| `streamlit: command not found` | Streamlit was installed inside a virtual environment that is not currently active, or PATH does not include script executables | Activate your virtual environment first, or run Streamlit via Python module syntax: `python -m streamlit run app/app.py` (Windows) or `python3 -m streamlit run app/app.py` (Linux/macOS). |
| `Port 8501 is already in use` | Another instance of Streamlit or a local application is occupying port 8501 | Launch Streamlit on an alternate port: `streamlit run app/app.py --server.port 8502`. |
| `Missing production model artifact: models/complaint_classifier.joblib` | Model files were accidentally deleted or omitted from an incomplete clone | Verify that `models/complaint_classifier.joblib`, `models/tfidf_vectorizer.joblib`, and `models/char_vectorizer.joblib` are present. If missing, pull them from Git (`git checkout models/`) or retrain with `python scripts/run_model_improvement.py final`. |
| `Dataset file not found at: data/complaints.csv` | Offline CSV file is missing when invoking CSV-dependent routines | Download the CFPB complaint dataset and place it at `data/complaints.csv` (see [Dataset Setup](#dataset-setup)). To run without the local CSV, use the live API mode or the interactive demo narratives in Streamlit. |
| `ModuleNotFoundError: No module named 'src'` | Running scripts or tests from an inner subdirectory instead of the project root | Ensure your current working directory is the repository root (`Customer-Complaint-NLP`). Run commands as `python -m pytest` or `streamlit run app/app.py` from the root directory. |
| Pytest tests fail due to environment or dependency version mismatch | Packages installed outside virtual environment or outdated dependencies | Create a fresh virtual environment: delete `.venv`, run `python -m venv .venv`, activate it, and reinstall dependencies via `python -m pip install -r requirements.txt`. |
| Network timeout when fetching live complaints from CFPB API | CFPB federal gateway experiencing temporary latency or local firewall restriction | Increase the network timeout: `CFPBClient(timeout=45.0)`. Live API access is optional; all core NLP classification and similarity functions operate fully offline using local artifacts. |

---

## Reproducibility

To ensure strict scientific reproducibility, the experimental configuration is fully specified:

### Dataset & Partitioning Protocol
- **Source**: Authentic CFPB Consumer Complaint Database (25,000 complaints, 18 classes).
- **Split Protocol**: Stratified sampling by product category with random seed `random_state=42`.
  - Full dataset ($N=25,000$) divided into:
    - **Training Pool**: 20,000 records (80%).
    - **Holdout Test Set**: 5,000 records (20%), kept untouched throughout model selection.
  - The training pool is internally split into:
    - **Train Subset**: 16,000 records (80% of pool).
    - **Validation Subset**: 4,000 records (20% of pool) for hyperparameter and feature selection.
- **Leakage Prevention**: Vectorizer vocabularies and class weighting parameters were fitted exclusively on training data. The 5,000-record holdout test set was evaluated strictly once using automated execution guards.

### TF-IDF Feature Representation
- **Word TF-IDF**:
  - N-gram range: `(1, 1)` (unigrams only)
  - Document frequency cutoffs: `min_df=2`, `max_df=0.95`
  - Sublinear term frequency scaling: `sublinear_tf=True`
  - Normalization: `norm='l2'`
  - Lowercasing: `lowercase=False` (preprocessed before vectorization)
  - Extracted vocabulary: **14,493 features**
- **Character TF-IDF**:
  - Analyzer: `analyzer='char'` (cross-boundary character n-grams)
  - N-gram range: `(3, 5)`
  - Document frequency cutoffs: `min_df=5`, `max_df=0.95`
  - Vocabulary capacity: `max_features=100,000`
  - Sublinear term frequency scaling: `sublinear_tf=True`
  - Normalization: `norm='l2'`
  - Lowercasing: `lowercase=False`
  - Extracted vocabulary: **100,000 features**
- **Combined Representation**:
  - Stacking method: `scipy.sparse.hstack(..., format="csr")`
  - Total combined dimension: **114,493 sparse features**

### Classifier Hyperparameters
- **Algorithm**: Multinomial Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Solver**: `lbfgs`
- **Regularization Parameter**: $C = 2.0$ (L2 regularization)
- **Class Weighting**: `class_weight='balanced'`
- **Maximum Iterations**: `max_iter=1000`
- **Random State**: `random_state=42`

---

## Technical Details

### Controlled Model Comparison

The table below presents the apples-to-apples controlled evaluation on the exact same 20,000-record training pool and untouched 5,000-record holdout test set across 18 product categories:

| Evaluation Metric | Controlled Baseline (Word TF-IDF, No Balancing) | Previous Model (Word(1,2)+Char(3,5), Balanced, C=1.0) | Improved Final Model (Word(1,1)+Char(3,5), Balanced, C=2.0) | Improvement vs. Previous | Improvement vs. Baseline |
|---|---:|---:|---:|---:|---:|
| **Overall Accuracy** | 69.14% | 69.56% | **69.82%** | **+0.26 pp** | +0.68 pp |
| **Macro F1-Score** | 34.15% | 50.56% | **50.88%** | **+0.33 pp** | **+16.73 pp (+48.99%)** |
| **Weighted F1-Score** | 65.73% | 69.55% | **69.78%** | **+0.23 pp** | +4.05 pp |
| **Macro Precision** | 41.37% | 49.67% | **50.41%** | **+0.74 pp** | +9.04 pp |
| **Macro Recall** | 34.08% | **51.97%** | 51.69% | -0.28 pp | +17.61 pp |
| **Weighted Precision** | 66.53% | 70.03% | **70.08%** | **+0.05 pp** | +3.55 pp |
| **Weighted Recall** | 69.14% | 69.56% | **69.82%** | **+0.26 pp** | +0.68 pp |
| **Test Partition Size** | 5,000 samples | 5,000 samples | 5,000 samples | Identical holdout | Identical holdout |
| **Vocabulary Features** | 199,630 features | 237,148 features | **114,493 features** | **-51.7% reduction** | Compact vocabulary |
| **Disk Size** | ~28 MB | 34.2 MB | **16.5 MB** | **-51.8% footprint** | Lightweight asset |

### Candidate Model Search Summary (Validation Set, N=4,000)

Candidate models were ranked during the 98-configuration search using the priority hierarchy: **Val Macro F1 -> Val Macro Recall -> Val Accuracy -> Val Weighted F1**:

| Stage | Candidate Configuration | Dimensions | Val Macro F1 | Val Macro Rec | Val Accuracy | Val Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| **S3 LR (Selected)** | **LogisticRegression ($C=2.0$, balanced) + Word(1,1) + Char(3,5, `char`)** | **109,228** | **51.84%** | **51.78%** | **69.95%** | **69.83%** |
| S2 word+char | LogisticRegression ($C=1.0$, balanced) + Word(1,1) + Char(3,5, `char`) | 109,228 | 51.49% | 51.96% | 69.17% | 69.20% |
| S0 reference | LogisticRegression ($C=1.0$, balanced) + Word(1,2) + Char(3,5, `char_wb`) | 198,478 | 51.27% | 51.65% | 69.20% | 69.13% |
| S1 word | LogisticRegression ($C=1.0$, balanced) + Word(1,1) | 13,209 | 51.28% | 53.05% | 68.27% | 68.52% |
| S4 SVC | LinearSVC ($C=0.5$, balanced) + Word(1,2) + Char(3,5, `char_wb`) | 198,478 | 49.15% | 48.36% | 70.77% | 69.99% |
| S4 SVC | LinearSVC ($C=0.5$, balanced) + Word(1,1) + Char(3,5, `char`) | 109,228 | 48.98% | 48.65% | 70.40% | 69.85% |
| S4 NB | MultinomialNB ($\alpha=0.01$) + Word(1,1) + Char(3,5, `char`) | 109,228 | 46.80% | 44.88% | 69.27% | 68.15% |
| S4 NB | ComplementNB ($\alpha=0.3$) + Word(1,1) + Char(3,5, `char`) | 109,228 | 39.06% | 38.51% | 67.05% | 62.63% |

### Taxonomy-Aware Classification & Error Decomposition

Error analysis on the held-out 5,000-record test set revealed that **608 out of 1,522 errors (39.95%)** occurred between category pairs representing identical financial products separated purely by historical administrative form revisions (April 2017 and 2019):

| Task Formulation | Categories | Accuracy | Macro F1 | Weighted F1 | Test Errors |
|---|---:|---:|---:|---:|---:|
| **Original Reference** | 18 | 69.56% | 50.56% | 69.55% | 1,522 |
| **v1 Conservative** | 11 | 81.50% | 63.43% | 81.61% | 925 |
| **v2 Broad** | 10 | 82.32% | 66.57% | 82.42% | 884 |

**Retraining Error Decomposition**:
To isolate mechanical task collapse from classifier retraining effects:
$$\text{Total Error Reduction} = \Delta_{\text{mechanical collapse}} + \Delta_{\text{retraining effect}}$$

- **v1 Conservative (11 Categories)**:
  - Original 18-category errors: **1,522**
  - Errors eliminated by resolving administrative synonyms: **608 errors (39.95%)**
  - Post-hoc collapsed errors (remapped without retraining): **914**
  - Retrained v1 model errors: **925**
  - Retraining effect: **+11 errors** relative to post-hoc mapping
- **v2 Broad (10 Categories)**:
  - Original 18-category errors: **1,522**
  - Errors eliminated by resolving administrative synonyms: **635 errors (41.72%)**
  - Post-hoc collapsed errors (remapped without retraining): **887**
  - Retrained v2 model errors: **884**
  - Retraining effect: **-3 errors** relative to post-hoc mapping

This confirms that the increase in accuracy from ~70% to ~82% is primarily driven by resolving administrative label synonymy in the task definition rather than superior classifier generalization.

10 categories — Broad v2
The Broad v2 taxonomy merges Consumer Loan into Consumer & Small Dollar Loans. The other categories remain the same:
1. Credit Reporting & Repair
2. Credit Card & Prepaid
3. Banking Accounts
4. Money Transfer & Services
5. Consumer & Small Dollar Loans
6. Vehicle Finance
7. Mortgage
8. Student Loan
9. Debt Collection
10. Other Financial Service
    
---

## Limitations

1. **Bag-of-Words & N-Gram Contextual Bounds**: Classical TF-IDF feature representations do not capture long-range syntactic dependencies or semantic word order changes beyond the defined n-gram window.
2. **Out-of-Vocabulary Terms**: Words completely absent from the training vocabulary receive zero weight during inference, though this is mitigated by character 3-5 n-gram subword modeling.
3. **CFPB Disclosure Delays**: Under CFPB narrative disclosure rules, consumer narratives undergo an administrative redaction and review period before publication. The system handles metadata-only records gracefully.
4. **Administrative Label Synonymy**: Without filing timestamps as input features, a text-only classifier operating on historical CFPB data faces an irreducible error ceiling (~30% error rate on 18 classes) caused by historical administrative label redesigns.

---

## Future Scope

1. **Sublinear Sparse Indexing**: Integrating approximate nearest neighbor algorithms (such as HNSW or Annoy) for sub-millisecond cosine similarity retrieval across millions of complaints.
2. **Hierarchical Multi-Level Classification**: Predicting primary product families first, followed by specialized sub-product classifications to mirror financial intake routing workflows.
3. **Automated Concept Drift Monitoring**: Tracking lexical and statistical distribution shifts across complaint intake quarters to identify emerging consumer finance risks.
4. **Streaming Data Ingestion**: Building background ingestion pipelines from the official CFPB API into local storage (SQLite/PostgreSQL) with automated deduplication.
5. **Multilingual Preprocessing Support**: Extending preprocessing routines to support Spanish-language complaint narratives filed with the CFPB.

---

## Key Takeaways

One of the main things we learned from this project is that traditional NLP and machine-learning techniques can be effective for analyzing customer complaints when combined with suitable text representation, class-imbalance handling, and proper evaluation.

Using both word and character TF-IDF helped us represent the complaint text in more detail. We also looked at class imbalance and used error analysis to understand where and why the model was making mistakes.

An important finding was that the model's performance depends not only on the features and machine-learning model, but also on how the complaint categories are defined. Some categories were very similar or had changed over time, which affected the results. This made taxonomy analysis an important part of the project rather than simply treating the categories as fixed.

The final system is designed to assist with complaint analysis and categorisation as a decision-support tool. It is not intended to replace human review.

## Contributors

- **Author & Maintainer**: Ashwin Chhawaniya ([Java-Mx](https://github.com/Java-Mx))
- **Repository**: [https://github.com/Java-Mx/Customer-Complaint-NLP](https://github.com/Java-Mx/Customer-Complaint-NLP)
- **License**: Released under the [MIT License](LICENSE).
