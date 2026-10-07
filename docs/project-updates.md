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

---

## Milestone 12: Presentation-Ready Academic Live Demo & Visual Analytics
- **Commit:** `be9887c` (2026-09-21)
- **Commit Message:** `feat: improve live demo interface and visual analytics`
- **Scope & Changes:** Refactored the Streamlit web application (`app/app.py`, `app/ui_components.py`, `app/charts.py`) into a polished, presentation-ready academic NLP demo designed for university vivas and project defenses.
- **NLP / ML User Experience Concepts:**
  - **Live Demo as Opening Hero:** Established LIVE Complaint Analysis as the primary opening module, communicating the classical NLP pipeline within 5 seconds.
  - **Rectangular Button Navigation:** Replaced default radio buttons with rounded rectangular buttons and compact real-time System Status (CFPB API health, dynamic dataset records, model, and sparse representation).
  - **Interactive Demonstration Examples:** Provided 6 domain-standard CFPB product grievances alongside 3 intentionally ambiguous test cases illustrating linguistic overlap across product boundaries (Credit Card vs. Credit Reporting, Debt Collection vs. Identity Theft, Checking Overdraft vs. Payday Loan). Connected using Streamlit `on_click` callbacks to eliminate state collisions.
  - **End-to-End Pipeline Trace & Active Features:** Displayed original vs. preprocessed text, 237,148-dimensional Word+Char TF-IDF representation, active non-zero feature counts, and top active n-gram weights without dense allocation.
  - **Sparse Cosine Similarity Retrieval:** Retrieved top-5 historically similar CFPB complaints with cosine similarity scores, complaint IDs, and expandable narratives using the indexed Word TF-IDF corpus.
  - **Integrated Empirical Model Insights:** Embedded publication-ready dark-theme charts for controlled baseline vs. improved models (Accuracy, Macro F1, Weighted F1), cross-taxonomy benchmarks, dataset class distributions across 18 product categories, per-category F1 scores, and top confusion pairs.
- **Tests & Verification:** Added 5 chart tests (`tests/test_ui_charts.py`), 3 UI component and end-to-end inference tests (`tests/test_ui_components.py`), and 3 Streamlit AppTest and server integration tests (`tests/test_streamlit_app.py`) verifying headless startup, navigation, example button population, live inference, and clear actions. Total test suite reached **209 unit and integration tests** (all passing).

---

## Milestone 13: Interactive Plotly Visualizations, Non-Truncated Metric Cards & Inline SVG Status System
- **Commit:** Current
- **Commit Message:** `feat: interactive plotly visual analytics, card layout, and inline svg status card`
- **Scope & Changes:** Enhanced Streamlit UI layout, visual hierarchy, and dashboard analytics:
  - **TF-IDF Metric Card Hierarchy:** Re-architected representation cards into two distinct rows:
    - Row 1: Compact metric cards for `Total Feature Dimension` (237,148) and dynamic `Active Non-Zero Features`.
    - Row 2: Full-width horizontal info card for `FEATURE REPRESENTATION: Combined Word + Character TF-IDF`, completely eliminating text truncation on all screen resolutions, followed by a sparse CSR explanation caption.
  - **Interactive Plotly Visualizations (Zero Static PNGs):** Eliminated all static `st.image` PNG renders across the entire dashboard. Replaced with fully interactive, dark-slate themed Plotly figures (`plotly.graph_objects`):
    - Controlled Baseline vs. Improved Model (grouped bar chart with hover tooltips and exact score percentages).
    - Cross-Taxonomy Formulation Comparison (Reference 18 vs. Conservative 11 vs. Broad 10).
    - CFPB Dataset Class Distribution (horizontal bar chart across 18 product categories, $N=25,000$).
    - Per-Category F1 Scores on 5,000-record holdout test set with support and precision/recall tooltips.
    - Top Misclassification Confusion Pairs (horizontal bar chart showing error counts and percentage of actual class).
    - Interactive 18×18 Multi-Class Confusion Matrix Heatmap (in both Model Evaluation and Error Analysis modules).
  - **Consolidated System Status Card:** Enclosed the complete System Status section in a single rounded card container (`#0f172a`, border `#1e293b`) that expands dynamically with padding, preventing content clipping or boundary overflow.
  - **Inline SVG Status Indicators:** Replaced Unicode checkmarks (`✓`, `✔`, `☑`) with accessible, reusable inline SVG check, warning, and error icons (`render_status_tick()`).
  - **Runtime & Type Hardening:** Resolved multi-variable unpacking mismatch in Model Evaluation and variable scoping in Error Analysis; normalized mixed-type dictionary values to strings to prevent PyArrow serialization warnings.
- **Tests & Verification:** Full test suite expanded to **215 tests** (`python -m pytest -v`), with 100% pass rate. Verified clean compilation via `compileall` and headless server startup with HTTP 200 responses.

---

## Milestone 14: Systematic Classical Model Improvement & Final Controlled Evaluation
- **Commit:** Current
- **Commit Message:** `feat: improve classical complaint classification`
- **Scope & Changes:** Implemented a reproducible, leakage-free empirical framework for classical NLP model optimization across 98 candidate configurations, followed by a one-shot evaluation on the held-out 5,000-record test set:
  - **Reproducible Experimentation Framework (`scripts/run_model_improvement.py`, `src/model_selection.py`):**
    - Established strict anti-leakage protocol: Stratified 80/20 partition into 20,000 training pool and 5,000 held-out test set; 20k pool partitioned into 16,000 train and 4,000 validation records. All candidate selection performed exclusively on the 4,000 validation subset.
    - Evaluated 98 candidate configurations across 7 structured stages: production reference (S0), word TF-IDF variations (S1), character n-gram spans and analyzers (S2), Logistic Regression hyperparameter grids (S3), alternative sparse classifiers including LinearSVC and Naive Bayes (S4), train-only power class weighting (S5), and classical stylistic text statistics (S6).
    - Ranked candidates hierarchically by Validation Macro F1 $\rightarrow$ Validation Macro Recall $\rightarrow$ Validation Accuracy $\rightarrow$ Validation Weighted F1.
  - **Winning Model Configuration:**
    - Features: Word TF-IDF (1,1, `min_df=2`, `max_df=0.95`, sublinear TF) + Character TF-IDF (3,5, `analyzer='char'`, `min_df=5`, `max_df=0.95`, `max_features=100,000`, sublinear TF).
    - Classifier: Multinomial Logistic Regression (`solver='lbfgs'`, $C=2.0$, `class_weight='balanced'`, `max_iter=1000`).
    - Feature space: 114,493 dimensions (pruned 122,655 redundant features, achieving a 51.7% vocabulary reduction).
  - **Final Controlled Test Evaluation (N = 5,000 Untouched Records):**
    - Retrained winning configuration on the full 20,000-record training pool and evaluated once on the 5,000-record test set:
    - **Accuracy:** **69.82%** (+0.26 pp vs. previous 69.56%; +0.68 pp vs. baseline 69.14%).
    - **Macro F1:** **50.88%** (+0.33 pp vs. previous 50.56%; +16.73 pp vs. baseline 34.15%).
    - **Weighted F1:** **69.78%** (+0.23 pp vs. previous 69.55%; +4.05 pp vs. baseline 65.73%).
    - **Macro Precision:** **50.41%** (+0.74 pp vs. previous 49.67%).
    - **Model Artifact Size:** Serialized classifier reduced from 34.2 MB to 16.5 MB (-51.8%).
  - **Streamlit & Artifact Integration:**
    - Serialized and deployed improved classifier and vectorizers to `models/`.
    - Updated evaluation benchmarks in `results/final_evaluation_metrics.json`, `results/baseline_vs_improved.csv`, `results/model_improvement_per_category.csv`, and `results/model_improvement_confusion_matrix.csv`.
    - Preserved seamless single-narrative classification, Top-K cosine similarity retrieval, live CFPB API integration, error analysis, and cross-taxonomy diagnostics.
- **Tests & Verification:**
  - Added unit tests for leakage-safe model selection in `tests/test_model_selection.py` (9 tests) and pipeline verification in `tests/test_run_search.py` (2 tests). Total test suite expanded to **226 tests** (all passing).
  - Verified clean syntax across all modules with `compileall`.

---

### Milestone 15: Post-Audit Controlled Model Improvement & Representation Optimization

- **Objective:** Implement the highest-value classical improvements established by the comprehensive model plateau audit while rigorously preserving the original 18-category task as the primary baseline.
- **Architectural Constraints Adhered To:**
  - Strictly classical NLP: TF-IDF, unigrams, character n-grams, Logistic Regression.
  - Zero BERT, Transformers, LLMs, or external APIs.
  - Untouched 5,000 holdout test set (stratified, random_state=42) evaluated exactly once at completion.
- **Key Deliverables & Experiments:**
  1. **Reusable Preprocessing Engine (`src/preprocessing.py`):**
     - Implemented `mode="standard"` (legacy regex cleaning, URL/email/stopword/punctuation removal) and `mode="minimal"` (lowercase and whitespace collapsing only; preserving punctuation, digits, stopwords, and negation context).
     - Full backward compatibility across `clean_text()`, `tokenize()`, `preprocess_text()`, and vectorized `preprocess_series()`.
  2. **Controlled 18-Class Improvement:**
     - Evaluated minimal preprocessing against standard baseline on identical 16k train / 4k val split using production TF-IDF and Logistic Regression ($C=2.0$, balanced).
     - Minimal preprocessing elevated validation accuracy from **69.95% to 71.33% (+1.38 pp)** and Macro F1 from **51.84% to 53.50% (+1.66 pp)** by preserving syntactic boundaries for character n-grams.
  3. **Formal 11-Class Taxonomy Normalization:**
     - Benchmarked conservative taxonomy v1 (`config/taxonomy_v1_conservative.json`), collapsing CFPB-documented administrative renames.
     - Normalized 11-class task achieves **82.10% validation accuracy** and **63.33% Macro F1** under standard preprocessing, and **82.88% accuracy / 64.50% Macro F1** under minimal preprocessing.
  4. **Hierarchical Classical Classifier Investigation:**
     - Evaluated text-only 2-stage hierarchy (Stage 1 predicts 11 product groups; Stage 2 trains specialist text classifiers for multi-label variant groups).
     - Confirmed that text-only local classifiers face the identical temporal barrier: intra-group historical variants share indistinguishable vocabularies, confirming that metadata (filing date) rather than architectural complexity is needed for administrative separation.
  5. **Controlled Metrics & Artifact Generation:**
     - Generated `results/model_improvement_comparison.csv`, `results/taxonomy_model_comparison.csv`, and `results/hierarchical_comparison.csv`.
     - Produced granular error decomposition in `results/post_audit_error_analysis.json` and one-shot test metrics in `results/post_audit_test_metrics.json`.
  6. **Interactive Streamlit Web Dashboard:**
     - Added dedicated Post-Audit Model Comparison view with interactive Plotly visualization, executive KPI cards, and formal benchmark tables.
- **Testing & Verification:**
  - Added unit tests in `tests/test_preprocessing.py`, `tests/test_ui_charts.py`, and `tests/test_post_audit_improvements.py`.
  - All test suites passing; verified with `compileall` and Streamlit HTTP 200 health check.

---

### Milestone 16: Unified Streamlit Design System & API Migration

- **Objective:** Modernize and unify the Streamlit user interface across all 9 application modules with a coherent design language, eliminate deprecated Streamlit parameter warnings, and standardize metric formatting without altering any ML results or evaluation data.
- **Architectural & Compatibility Changes:**
  1. **Unified Design System & Centralized Design Tokens (`app/ui_components.py`):**
     - Established design tokens (`DESIGN_TOKENS`) for background, surfaces, borders, text hierarchy, semantic statuses, typography, and corner radius.
     - Centralized custom CSS with CSS variables (`--bg-main`, `--bg-surface`, `--border-subtle`, `--accent-primary`, etc.).
     - Built reusable UI rendering primitives: `render_page_header()`, `render_section_header()`, `render_metric_card()`, `render_metric_grid()`, `render_chart_card()`, `render_info_card()`, and `render_status_card()`.
     - Eliminated excessive vertical spacing through compact container padding (`padding-top: 1.5rem !important;`).
  2. **Standardized Navigation & Sidebar:**
     - Replaced oversized sidebar buttons with compact, rectangular items featuring restrained active selection highlights (translucent blue border rather than saturated solid fills).
     - Unified System Status indicator using accessible inline SVG icons and consistent card geometry, eliminating unicode symbols (`✓`, `✔`) and emojis.
  3. **Standardized Metric Cards & KPI Grid:**
     - Enforced consistent metric formatting: accuracy, precision, recall, and F1 values are rendered as percentages (e.g. `50.88%` instead of `0.5088`).
     - Fixed "Test Partition Size" text truncation by removing CSS ellipsis truncation and formatting cleanly as `5,000 complaints`.
     - Grouped metrics into a clean 4-column responsive grid (Row 1: Accuracy, Macro F1, Weighted F1, Test Partition; Row 2: Macro Precision, Macro Recall, Weighted Precision, Total Features).
     - Standardized delta formatting to explicit percentage-point indicators (`+7.99 pp`, `+26.10 pp`) to prevent confusion between raw decimal differences and relative percentages.
  4. **Streamlit Width API Migration (`width="stretch"`):**
     - Replaced all 29 deprecated occurrences of `use_container_width=True` across `app/app.py` and `app/ui_components.py` with `width="stretch"`.
     - Eliminated all Streamlit deprecation warnings across buttons, dataframes, and Plotly charts.
     - Added explicit labels with `label_visibility="collapsed"` for text areas to resolve accessibility warnings.
  5. **Strict Preservation of NLP & Model Behavior:**
     - No modifications to preprocessing algorithms, TF-IDF configurations, model serialization, classification logic, similarity metrics, or evaluation figures.
### Milestone 17: HTML Rendering Regression Repair, Unified Header Hierarchy & Strict UI Regression Suite

- **Objective:** Resolve broken HTML rendering and vertical title wrapping introduced during UI refactoring, enforce a single robust page-header architecture across all 9 modules, eliminate split container markdown calls, and implement a strict automated UI regression test suite.
- **Root Cause & Architectural Repairs:**
  1. **Indented Code-Block Regression Fixed:** In CommonMark/markdown-it, multiline f-strings with 4-space indentation were erroneously parsed as indented `<pre><code>` code blocks, terminating HTML containers early and exposing raw literal text such as `</div>` and `<div class="page-subtitle">...`. Resolved by formatting all custom HTML components as self-contained, unindented, single HTML fragments with zero internal blank lines.
  2. **Page Title Vertical Wrapping Fixed:** Unclosed flex containers (`.page-title-row`) caused flex items to collapse to minimum content width. Replaced `<h1>` with `<div class="page-title">`, added `writing-mode: horizontal-tb !important;`, `word-break: normal !important;`, `overflow-wrap: normal !important;`, and `flex-wrap: wrap !important;` to ensure titles remain strictly horizontal across desktop and responsive viewports.
  3. **Split Markdown Containers Eliminated:** Removed pattern where `<div class="example-btn-area">` was opened in one `st.markdown()` call and closed in another.
  4. **Standardized Header & Metric Presentation:** All 9 modules now share identical uppercase header hierarchy. MODEL EVALUATION features a balanced 3×3 metric grid displaying all 9 key metrics with explicit `pp` percentage-point deltas and unclipped values (`5,000 complaints`, `114,493 features`). Native `st.success` / `st.error` / `st.warning` replace custom HTML where Streamlit native components are optimal.
  5. **Strict UI Regression Suite Added (`tests/test_ui_regression_and_html.py`):**
     - Automated `html.parser.HTMLParser` validation verifying balanced tags and zero orphan closures.
     - Source-level regression tests asserting zero `use_container_width` occurrences and zero split containers.
     - CommonMark rendered-text regression tests verifying zero escaped raw HTML tags in visible output.
     - CSS regression tests guarding against vertical text and character-level breaking.
     - End-to-end `streamlit.testing.v1.AppTest` smoke test verifying seamless navigation across all 9 modules without exceptions.
     - Headless HTTP 200 server startup and zero deprecation warning validation.
  6. **Zero NLP / Model Modification:** Preprocessing, TF-IDF vectorization, models, confusion matrices, and benchmark numbers remain untouched.
- **Testing & Verification:**
  - Full test suite passing: 266 passed tests (`pytest -q`).
  - Python compilation validated: `python -m compileall src app scripts tests` passed cleanly.
  - Headless Streamlit server verified: HTTP 200 on health check and main application with zero warnings.

---

### Milestone 18: Final Streamlit UI Architecture Repair, Clearance Fix & Enhanced UI Regression Suite

- **Objective:** Fix top title clipping caused by Streamlit toolbar overlap, increase input box font size, scope all custom CSS selectors, eliminate potential layout regressions, and expand the UI regression test suite with the 12 required architectural assertions.
- **Root Causes Identified & Repaired:**
  1. **Page Title Top Clipping & Toolbar Overlap:** In Streamlit, `header[data-testid="stHeader"]` occupies a fixed height of `3.75rem` (60px) at the top of the browser viewport. Previous compacting CSS set `.block-container` `padding-top: 1.5rem !important;` (24px). Because 24px < 60px, the top 36px of `.block-container`—where the page header resides—was shifted underneath the fixed Streamlit header toolbar, causing the top half of the page title to be clipped and partially hidden behind the header region. Corrected by setting `.block-container` `padding-top: 5rem !important;`, providing 1.25rem of natural breathing clearance below Streamlit's chrome and ensuring the title is 100% visible in natural document flow without clipping or vertical offset hacks.
  2. **Page Header Architectural Safety:** Configured `.app-page-header`, `.page-title-row`, `.page-title`, and `.page-subtitle` with `height: auto !important;`, `overflow: visible !important;`, `position: relative !important;`, and `writing-mode: horizontal-tb !important;`. Confirmed zero fixed heights, zero overflow clipping, and zero character-by-character wrapping.
  3. **Input Box Font Size Enhancement:** Increased font size of complaint text inputs and text areas from `0.82rem` to `0.96rem` (`line-height: 1.5`) across BaseWeb input containers (`[data-testid="stTextInput"] input`, `[data-testid="stTextArea"] textarea`, `div[data-baseweb="input"] input`, `div[data-baseweb="textarea"] textarea`) for crisp readability during live demonstrations.
  4. **Strict CSS Selector Scoping:** Audited all CSS rules to eliminate unscoped global elements (`table`, `hr`, `div.stButton > button`). Scoped under `.stApp` and scoped utility classes to prevent cross-contamination with Streamlit native layout elements.
  5. **Streamlit Width API Migration Verified:** Maintained 0 occurrences of deprecated `use_container_width` across all application modules, confirmed with zero deprecation warnings on server startup.
  6. **Comprehensive UI Regression Tests:** Added all 12 required test cases (`test_page_title_is_horizontal`, `test_page_title_has_no_character_wrap_css`, `test_page_header_has_no_fixed_height`, `test_page_header_has_no_overflow_clipping`, `test_page_header_html_is_balanced`, `test_section_header_html_is_balanced`, `test_metric_card_html_is_balanced`, `test_status_card_html_is_balanced`, `test_no_use_container_width`, `test_no_forbidden_unscoped_css_selectors`, `test_no_orphan_html_closing_tags`, `test_page_subtitle_is_rendered_as_html_or_native_text_not_literal_markup`) alongside full AppTest module navigation, real Playwright Chromium browser geometry validation (`test_playwright_rendered_geometry_and_no_clipping`), live server HTTP 200 checks, and `compileall` validation. Tests inspect real UI helper implementations via live invocation and monkeypatched capture rather than mock strings.
  7. **Strict Preservation of NLP System:** Zero changes made to preprocessing, TF-IDF, classification, similarity retrieval, taxonomy, evaluation benchmarks, or model binaries.
- **Testing & Verification:**
  - Full test suite passing (`pytest -q`).
  - Python compilation validated: `compileall` passes cleanly on `src`, `app`, `scripts`, `tests`.
  - Headless Streamlit server verified: HTTP 200 on health check and main application with zero warnings.
  - Headless Chromium (Playwright) verified: exact bounding-box layout confirms title sits strictly below Streamlit toolbar (`y >= 60px`), horizontal orientation, and no clipping across all 9 modules.


