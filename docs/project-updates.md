# Project Updates & Milestone Engineering Log

**Project:** Customer Complaint NLP — CFPB Complaint Similarity & Categorisation  
**Repository:** [https://github.com/Java-Mx/Customer-Complaint-NLP](https://github.com/Java-Mx/Customer-Complaint-NLP)  

This document serves as the permanent chronological engineering, modeling, and research log of the project. Every milestone entry records the commit, technical scope, NLP/machine learning concepts introduced, empirical results, and verification status.

---

## Milestone 1: Repository Foundation & Project Scaffolding
- **Commit:** `6b161cf` (2026-09-21)
- **Commit Message:** `chore: initialize NLP project structure`
- **Scope & Changes:** Initialized repository directory structure (`src/`, `tests/`, `data/`, `notebooks/`, `models/`, `results/`, `app/`), dependency declarations (`requirements.txt`, `pyproject.toml`), MIT License, Contributor Covenant Code of Conduct, and initial placeholder modules.
- **NLP / ML Concepts Introduced:** Classical NLP project architecture, modular separation of data loading, preprocessing, feature extraction, retrieval, classification, and evaluation.
- **Implementation Details:** Configured `.gitignore` to prevent committing raw datasets (`data/*.csv`), serialized models (`models/*.joblib`), and temporary files. Established Streamlit interface scaffold in `app/app.py`.
- **Tests & Artifacts:** Initial test skeleton (`tests/test_preprocessing.py`).

---

## Milestone 2: CFPB Dataset Loading & Validation Pipeline
- **Commit:** `24752a0` (2026-09-21)
- **Commit Message:** `feat: add CFPB dataset loading and validation`
- **Scope & Changes:** Implemented robust dataset ingestion in `src/data_loader.py` with flexible column resolution, missing value handling, and data summary diagnostics.
- **NLP / ML Concepts Introduced:** Data sanitization, categorical distribution analysis, consumer narrative validation, missing data imputation strategies.
- **Implementation Details:** Handled arbitrary column namings (`Consumer complaint narrative`, `Product`, `Complaint ID`, etc.) via dynamic column resolution. Added `drop_invalid=True` filtering to purge records lacking narrative text or category labels.
- **Tests & Verification:** 12 unit tests in `tests/test_data_loader.py`.

---

## Milestone 3: Text Preprocessing Pipeline
- **Commit:** `e964ec0` (2026-09-21)
- **Commit Message:** `feat: implement text preprocessing pipeline`
- **Scope & Changes:** Built end-to-end cleaning, normalization, and tokenization pipeline in `src/preprocessing.py`.
- **NLP / ML Concepts Introduced:** Case normalization, regex-based noise filtering, CFPB redaction removal (`XXXX` masking), URL/email stripping, punctuation removal, whitespace normalization, tokenization, stopword filtering.
- **Implementation Details:** Developed `clean_text()`, `tokenize()`, `remove_stopwords()`, `preprocess_text()`, and vectorized `preprocess_series()` operating on Pandas Series without mutating original data.
- **Tests & Verification:** 19 unit tests in `tests/test_preprocessing.py`.

---

## Milestone 4: TF-IDF Feature Extraction Engine
- **Commit:** `364ebc4` (2026-09-21)
- **Commit Message:** `feat: add tfidf vectorization`
- **Scope & Changes:** Implemented Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction in `src/vectorization.py`.
- **NLP / ML Concepts Introduced:** Vector Space Model, sublinear term frequency scaling ($1 + \log(\text{tf})$), inverse document frequency thresholding (`min_df`, `max_df`), n-gram extraction (unigrams + bigrams), sparse CSR matrix representation.
- **Implementation Details:** Encapsulated Scikit-learn `TfidfVectorizer` with serialization wrappers (`save_vectorizer`, `load_vectorizer`) and safe transformation routines.
- **Tests & Verification:** 15 unit tests in `tests/test_vectorization.py`.

---

## Milestone 5: Pairwise Cosine Similarity & Complaint Retrieval
- **Commit:** `99bd215` (2026-09-21)
- **Commit Message:** `feat: add cosine similarity`
- **Scope & Changes:** Implemented sparse vector-space cosine similarity search and Top-$K$ complaint retrieval in `src/similarity.py`.
- **NLP / ML Concepts Introduced:** Dot product similarity normalized by $L_2$ norms:
  $$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
- **Implementation Details:** Optimized for sparse CSR matrices using Scikit-learn's optimized linear algebra routines. Handled edge cases (zero vectors, orthogonal vectors, empty corpus) and preserved metadata during Top-$K$ ranking.
- **Tests & Verification:** 30 unit tests in `tests/test_similarity.py`. Total test suite reached 88 tests.

---

## Milestone 6: Supervised Complaint Classification Baseline
- **Commit:** `164f722` (2026-09-21)
- **Commit Message:** `feat: add complaint classification`
- **Scope & Changes:** Implemented supervised multi-class classification pipeline in `src/classification.py` using Multinomial Logistic Regression.
- **NLP / ML Concepts Introduced:** Supervised text classification, cross-entropy minimization, L-BFGS optimization, train-test splitting with stratification, inference probability calibration.
- **Implementation Details:** Added `train_classifier()`, `predict_categories()`, `predict_category_proba()`, `predict_complaint_category()`, and `train_test_split_data()`. Integrated live categorisation interface into Streamlit.
- **Tests & Verification:** 25 unit tests in `tests/test_classification.py`. Total test suite reached 113 tests.

---

## Milestone 7: Streamlit Path Resolution Fix
- **Commit:** `8dbc36f` (2026-09-21)
- **Commit Message:** `fix: resolve streamlit src import path`
- **Scope & Changes:** Resolved `ModuleNotFoundError: No module named 'src'` when launching Streamlit from arbitrary working directories.
- **Implementation Details:** Added dynamic `pathlib`-based project root detection to `app/app.py` prior to importing from `src`.

---

## Milestone 8: Model Evaluation & Live CFPB API Integration
- **Commit:** `6f6bfc4` (2026-09-21)
- **Commit Message:** `feat: add model evaluation and CFPB API integration`
- **Scope & Changes:** Added comprehensive statistical evaluation suite in `src/evaluation.py` and official CFPB API v1 client in `src/cfpb_api.py`.
- **NLP / ML Concepts Introduced:** Multi-class evaluation metrics (Precision, Recall, Macro F1, Weighted F1), confusion matrix heatmap generation, Elasticsearch-backed REST API querying, schema mapping.
- **Implementation Details:** 
  - `compute_classification_metrics()`, `compute_per_category_metrics()`, `plot_confusion_matrix()`.
  - `CFPBClient` fetching live complaints from `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`.
- **Empirical Results (Initial Baseline, $N=600$ test split):** Accuracy: 61.83%, Macro F1: 24.78%, Weighted F1: 54.96%.
- **Tests & Verification:** 13 API tests + 16 evaluation tests. Total test suite reached 130 tests.

---

## Milestone 9: Systematic Classical Model Improvement
- **Commit:** `4ef78fe` (2026-09-21)
- **Commit Message:** `feat: improve complaint classification performance`
- **Scope & Changes:** Overhauled feature representation and imbalance handling without deep neural networks or external APIs.
- **NLP / ML Concepts Introduced:** Subword character n-grams (`analyzer="char_wb"`, n-grams 3–5), sparse matrix fusion via `scipy.sparse.hstack`, class-frequency loss weighting (`class_weight="balanced"`), LinearSVC vs. Logistic Regression grid search.
- **Implementation Details:** Built `scripts/run_experiments.py` evaluating 30+ configurations on an internal 16,000-train / 4,000-val split. Scaled vocabulary to 237,148 features.
- **Empirical Results (Held-Out $N=5,000$ Test Set, 18 Categories):**
  - Accuracy: **69.56%** (+7.73 pp over initial baseline)
  - Macro F1: **50.56%** (+25.78 pp; more than doubled)
  - Weighted F1: **69.55%** (+14.59 pp)
- **Tests & Verification:** 11 new tests in `tests/test_model_improvements.py`. Total test suite reached 141 tests.

---

## Milestone 10: Diagnostic Error Analysis
- **Commit:** `4cf8afa` (2026-09-21)
- **Commit Message:** `docs: add classification error analysis`
- **Scope & Changes:** Executed thorough, non-destructive diagnostic error analysis on the untouched 5,000-record test set via `scripts/generate_error_analysis.py`.
- **NLP / ML Concepts Introduced:** Multi-class confusion matrix diagnosis, prediction probability distribution analysis (`predict_proba()`), class support vs. recall correlation, lexical n-gram overlap between confused classes.
- **Implementation Details:** 
  - Controlled same-split baseline trained on the exact same 20k pool: Accuracy 69.14%, Macro F1 34.15%, Weighted F1 65.73%.
  - Documented 154 distinct confusion pairs among the 1,522 test errors.
  - Identified that 3 dominant clusters accounted for hundreds of errors: *Credit reporting* variants (314 errors), *Credit card* variants (148 errors), *Banking account* variants (121 errors).
- **Artifacts & Tests:** Created `docs/error-analysis.md`, `results/error_analysis.csv`, `results/baseline_vs_improved.csv`, `results/per_category_metrics.csv`. Added 35 tests in `tests/test_error_analysis.py`. Total test suite reached 176 tests.

---

## Milestone 11: Taxonomy-Aware Complaint Classification Analysis
- **Commit:** `0d49099` (2026-09-21)
- **Commit Message:** `feat: add taxonomy-aware complaint classification analysis`
- **Scope & Changes:** Investigated whether classification difficulty stemmed from NLP representation limits or from historical CFPB administrative form revisions.
- **NLP / ML Concepts Introduced:** Task formulation analysis, ground-truth label noise audit, information loss accounting, mechanical collapse vs. classifier retraining decomposition.
- **Regulatory Findings:** Researched official CFPB documentation (*Summary of product and sub-product changes*, April 24, 2017; ~2019 notices). Confirmed that the CFPB preserves original submission-time labels without backfilling, leaving pre-2017 and post-2017 label variants simultaneously in multi-year datasets.
- **Taxonomy Formulations Pre-Defined:**
  - **v1 Conservative (11 categories):** Consolidates only CFPB-documented renames/mergers; keeps `Consumer Loan` separate.
  - **v2 Broad (10 categories):** Merges `Consumer Loan` into `Consumer & Small Dollar Loans`.
- **Empirical Results (Identical 5,000-Record Holdout Test Set):**
  - Reference 18-Category Model: Accuracy **69.56%**, Macro F1 **50.56%**, Weighted F1 **69.55%**, 1,522 test errors.
  - v1 Conservative (11 Categories): Accuracy **81.50%**, Macro F1 **63.43%**, Weighted F1 **81.61%**, 925 test errors.
  - v2 Broad (10 Categories): Accuracy **82.32%**, Macro F1 **66.57%**, Weighted F1 **82.42%**, 884 test errors.
- **Error Dissection:**
  - Conservative mechanically eliminates **608 errors (39.95%)** through label collapse. Retraining effect: +11 errors.
  - Broad mechanically eliminates **635 errors (41.72%)** through label collapse. Retraining effect: -3 errors.
- **Methodological Lesson:**  
  *Taxonomy normalization changes the classification task; normalized metrics must therefore not be interpreted as a pure improvement in classifier capability. Rather, ~40% of apparent baseline errors were administrative artifacts of historical form revisions.*
- **Artifacts & Tests:** 
  - `config/taxonomy_v1_conservative.json`, `config/taxonomy_v2_broad.json`
  - `scripts/analyze_taxonomy.py`, `scripts/run_taxonomy_experiment.py`
  - `docs/taxonomy-analysis.md`, `docs/viva-preparation.md`
  - `tests/test_taxonomy.py` (14 new tests). Total test suite reached **190 tests** (all passing).
  - Streamlit "Taxonomy Analysis" module added (`HTTP 200` verified).
