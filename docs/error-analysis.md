# CFPB Complaint Classification Error Analysis

**Project:** Customer Complaint NLP — CFPB Complaint Categorisation  
**Repository:** [https://github.com/Java-Mx/Customer-Complaint-NLP](https://github.com/Java-Mx/Customer-Complaint-NLP)  
**Milestone:** Diagnostic Error Analysis (No Model Modification)  
**Model evaluated:** `complaint_classifier.joblib` (LogisticRegression, combined TF-IDF, class_weight='balanced')

---

## 1. Purpose and Objectives

This document presents a comprehensive diagnostic analysis of the final trained CFPB complaint classification model. The purpose is to understand **where and why** the model is making classification errors — not to modify the model.

**Scope constraints:**
- Zero changes to the classifier, vectorizers, or train/test splits.
- The untouched 5,000-record holdout test set is used strictly for evaluation.
- No records from the test set were used for fitting, tuning, or feature selection.
- All measurements reported here are purely diagnostic observations.

---

## 2. Dataset and Split Methodology

| Partition | Records | Share |
|---|---:|---:|
| Full CFPB dataset | 25,000 | 100% |
| Training pool | 20,000 | 80% |
| — Train subset (internal) | 16,000 | 64% |
| — Validation subset (internal) | 4,000 | 16% |
| **Untouched holdout test set** | **5,000** | **20%** |

- **Stratification:** Yes, stratified by product category.
- **Random seed:** `random_state=42` throughout.
- **Categories:** 18 distinct CFPB financial product categories.
- **Zero data leakage:** TF-IDF vocabulary was fit exclusively on the 20,000-record training pool. The 5,000-record test set was never seen during model selection or training.

---

## 3. Final Model Configuration

| Parameter | Value |
|---|---|
| Classifier | LogisticRegression |
| Solver | lbfgs |
| C (regularization) | 1.0 |
| class_weight | balanced |
| max_iter | 1000 |
| random_state | 42 |
| Word TF-IDF | ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False |
| Char TF-IDF | analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True |
| Feature combination | `scipy.sparse.hstack` (CSR, no dense conversion) |
| **Total feature dimensions** | **237,148** |

---

## 4. Controlled Baseline vs. Improved Comparison

This section presents the only methodologically valid apples-to-apples comparison — both models evaluated on the **same** 5,000-record holdout test set.

> **Important:** The "historical baseline" referenced in earlier milestones (Accuracy: 61.83%, Macro F1: 24.78%) was measured on a different experiment with approximately 600 test records and 17 categories. It **cannot** be compared directly with the current 5,000-record results.

### Controlled Same-Split Baseline Model Configuration

- **Word TF-IDF:** ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False
- **LogisticRegression:** solver=lbfgs, C=1.0, class_weight=None (no balancing), max_iter=1000, random_state=42
- **Trained on:** Same 20,000-record training pool
- **Evaluated on:** Same 5,000-record test set

| Metric | Baseline (Word TF-IDF, no balancing) | Improved (Combined TF-IDF, balanced) | Absolute Diff | Relative Change |
|---|---:|---:|---:|---:|
| Accuracy | 69.14% | 69.56% | +0.42 pp | +0.61% |
| Weighted Precision | 66.53% | 70.03% | +3.50 pp | +5.26% |
| Weighted Recall | 69.14% | 69.56% | +0.42 pp | +0.61% |
| Weighted F1 | 65.73% | 69.55% | +3.82 pp | +5.81% |
| Macro Precision | 41.37% | 49.67% | +8.30 pp | +20.06% |
| Macro Recall | 34.08% | 51.97% | +17.89 pp | +52.49% |
| **Macro F1** | **34.15%** | **50.56%** | **+16.41 pp** | **+48.05%** |

**Key observation:** The primary improvement from the combined TF-IDF + class balancing strategy is on **Macro F1** (+16.41 pp, +48.1% relative), driven by substantially higher precision and recall across minority classes. Majority-class accuracy is similar between the two models.

---

## 5. Overall Final Test Set Performance

**Model:** `LogisticRegression` (Combined Word+Char TF-IDF, class_weight='balanced')  
**Test partition:** N = 5,000 (exact holdout, untouched during training)

| Metric | Value |
|---|---:|
| Overall Accuracy | 69.56% |
| Macro Precision | 49.67% |
| Macro Recall | 51.97% |
| Macro F1-Score | 50.56% |
| Weighted Precision | 70.03% |
| Weighted Recall | 69.56% |
| Weighted F1-Score | 69.55% |
| **Correctly classified** | **3,478 / 5,000** |
| **Misclassified** | **1,522 / 5,000 (30.44%)** |
| **Distinct confusion pairs** | **154** |

---

## 6. Confusion Matrix

The full 18×18 confusion matrix is stored at `results/confusion_matrix.png`.

Summary of diagonal (correctly classified per category) vs. off-diagonal (errors):

| Category | Support | Correct | Incorrect | Recall |
|---|---:|---:|---:|---:|
| Debt collection | 1,166 | 973 | 193 | 83.4% |
| Credit reporting, credit repair services, or other personal consumer reports | 850 | 452 | 398 | 53.2% |
| Mortgage | 790 | 735 | 55 | 93.0% |
| Credit reporting | 576 | 382 | 194 | 66.3% |
| Credit card | 336 | 202 | 134 | 60.1% |
| Student loan | 295 | 255 | 40 | 86.4% |
| Bank account or service | 269 | 147 | 122 | 54.6% |
| Credit card or prepaid card | 195 | 86 | 109 | 44.1% |
| Consumer Loan | 167 | 86 | 81 | 51.5% |
| Checking or savings account | 123 | 49 | 74 | 39.8% |
| Money transfer, virtual currency, or money service | 56 | 34 | 22 | 60.7% |
| Vehicle loan or lease | 44 | 13 | 31 | 29.6% |
| Payday loan, title loan, or personal loan | 38 | 8 | 30 | 21.1% |
| Payday loan | 34 | 18 | 16 | 52.9% |
| Money transfers | 29 | 19 | 10 | 65.5% |
| Prepaid card | 26 | 19 | 7 | 73.1% |
| Other financial service | 5 | 0 | 5 | 0.0% |
| Virtual currency | 1 | 0 | 1 | 0.0% |

---

## 7. Top 20 Confusion Pairs (Actual → Predicted)

| Actual | Predicted | Error Count | % of Actual Class |
|---|---|---:|---:|
| Credit reporting, credit repair services, or other personal consumer reports | Credit reporting | 207 | 24.4% |
| Credit reporting | Credit reporting, credit repair services, or other personal consumer reports | 107 | 18.6% |
| Credit card or prepaid card | Credit card | 73 | 37.4% |
| Credit card | Credit card or prepaid card | 66 | 19.6% |
| Credit reporting, credit repair services, or other personal consumer reports | Debt collection | 65 | 7.7% |
| Bank account or service | Checking or savings account | 63 | 23.4% |
| Checking or savings account | Bank account or service | 58 | 47.1% |
| Debt collection | Credit reporting, credit repair services, or other personal consumer reports | 43 | 3.7% |
| Debt collection | Credit reporting | 43 | 3.7% |
| Credit reporting | Debt collection | 41 | 7.1% |
| Vehicle loan or lease | Consumer Loan | 26 | 59.1% |
| Credit reporting, credit repair services, or other personal consumer reports | Consumer Loan | 26 | 3.1% |
| Debt collection | Consumer Loan | 21 | 1.8% |
| Debt collection | Student loan | 21 | 1.8% |
| Credit reporting, credit repair services, or other personal consumer reports | Credit card or prepaid card | 20 | 2.4% |
| Credit reporting, credit repair services, or other personal consumer reports | Credit card | 20 | 2.4% |
| Credit reporting, credit repair services, or other personal consumer reports | Mortgage | 19 | 2.2% |
| Credit card | Debt collection | 19 | 5.7% |
| Student loan | Debt collection | 19 | 6.4% |
| Debt collection | Credit card | 16 | 1.4% |

**The dominant confusion cluster** involves three category groups:
1. **"Credit reporting" / "Credit reporting, credit repair services, or other personal consumer reports"** — bidirectional, 207+107 = 314 errors total. These appear to be historical naming variants of the same CFPB category that exist in the dataset simultaneously.
2. **"Credit card" / "Credit card or prepaid card"** — bidirectional, 73+66 = 139 errors total. Again, apparent naming variants.
3. **"Bank account or service" / "Checking or savings account"** — bidirectional, 63+58 = 121 errors total. Same pattern.

---

## 8. Per-Category Performance

All 18 categories, sorted by F1-Score ascending (worst first):

| Category | Support | Precision | Recall | F1 | Correct | Incorrect | Primary Confusion Target | Confusion Count |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Virtual currency | 1 | 0.000 | 0.000 | 0.000 | 0 | 1 | Money transfer, virtual currency, or money service | 1 |
| Other financial service | 5 | 0.000 | 0.000 | 0.000 | 0 | 5 | Debt collection | 2 |
| Payday loan, title loan, or personal loan | 38 | 0.258 | 0.211 | 0.232 | 8 | 30 | Payday loan | 10 |
| Vehicle loan or lease | 44 | 0.295 | 0.295 | 0.295 | 13 | 31 | Consumer Loan | 26 |
| Checking or savings account | 123 | 0.366 | 0.398 | 0.381 | 49 | 74 | Bank account or service | 58 |
| Credit card or prepaid card | 195 | 0.418 | 0.441 | 0.429 | 86 | 109 | Credit card | 73 |
| Payday loan | 34 | 0.367 | 0.529 | 0.434 | 18 | 16 | Consumer Loan | 5 |
| Consumer Loan | 167 | 0.430 | 0.515 | 0.469 | 86 | 81 | Debt collection | 14 |
| Bank account or service | 269 | 0.540 | 0.546 | 0.543 | 147 | 122 | Checking or savings account | 63 |
| Money transfers | 29 | 0.500 | 0.655 | 0.567 | 19 | 10 | Money transfer, virtual currency, or money service | 4 |
| Money transfer, virtual currency, or money service | 56 | 0.576 | 0.607 | 0.591 | 34 | 22 | Money transfers | 7 |
| Credit reporting, credit repair services, or other personal consumer reports | 850 | 0.689 | 0.532 | 0.600 | 452 | 398 | Credit reporting | 207 |
| Credit reporting | 576 | 0.572 | 0.663 | 0.614 | 382 | 194 | Credit reporting, credit repair services, or other personal consumer reports | 107 |
| Credit card | 336 | 0.581 | 0.601 | 0.591 | 202 | 134 | Credit card or prepaid card | 66 |
| Prepaid card | 26 | 0.760 | 0.731 | 0.745 | 19 | 7 | Credit card or prepaid card | 4 |
| Student loan | 295 | 0.836 | 0.864 | 0.850 | 255 | 40 | Debt collection | 19 |
| Debt collection | 1,166 | 0.838 | 0.834 | 0.836 | 973 | 193 | Credit reporting | 43 |
| Mortgage | 790 | 0.914 | 0.930 | 0.922 | 735 | 55 | Debt collection | 11 |

---

## 9. Representative Error Cases

The following complaint texts are drawn from the actual 5,000-record holdout test set.

### Pair 1: "Credit reporting, credit repair services, or other personal consumer reports" → "Credit reporting"

This is the largest single confusion pair with **207 errors** (24.4% of the actual class). The complaints are structurally indistinguishable — both describe disputing items on credit reports, requesting credit bureau corrections, and referencing Equifax, Experian, and TransUnion. The CFPB changed the product name from "Credit reporting" to the longer "Credit reporting, credit repair services, or other personal consumer reports" — the data contains both label variants in the same corpus.

*Example:*
> "RE: File # XXXX I'm complaining because Transunion have failed to comply with, furthermore wilfully ignored, my request to provide me with the documents that their company have on file that was used to verify the account that I disputed..."
> 
> Actual: *Credit reporting, credit repair services, or other personal consumer reports* | Predicted: *Credit reporting*

*Example:*
> "I have been batting with Equifax for a year about this items on my credit file. I have disputed them sending documents and they have investigated and sent the results back to me..."
>
> Actual: *Credit reporting, credit repair services, or other personal consumer reports* | Predicted: *Credit reporting*

### Pair 2: "Credit reporting" → "Credit reporting, credit repair services, or other personal consumer reports"

The reverse direction with **107 errors** (18.6%). Same pattern — complaints describing credit bureau disputes, credit report corrections, and unauthorized credit inquiries that belong to the old label name but are predicted as the newer label.

*Example:*
> "I reviewed a copy of my credit report and the companies below ran an unauthorized credit inquiry on me on date provided. XXXX XXXX XXXX XXXX, XXXX XXXX XXXX XXXX..."
>
> Actual: *Credit reporting* | Predicted: *Credit reporting, credit repair services, or other personal consumer reports*

### Pair 3: "Credit card or prepaid card" → "Credit card"

**73 errors** (37.4%). The label "Credit card or prepaid card" is a CFPB re-labelling of the older "Credit card" product. Complaints about credit card billing disputes, unauthorized charges, and customer service issues appear under both labels.

### Pair 4: "Bank account or service" → "Checking or savings account"

**63 errors** (23.4% of Bank account or service). Both describe retail banking account complaints — overdraft fees, unauthorized debit transactions, account closures. The two labels are CFPB naming variants for the same product domain.

### Pair 5: "Vehicle loan or lease" → "Consumer Loan"

**26 errors** (59.1%). 59% of all Vehicle loan or lease complaints are predicted as Consumer Loan. Both involve loan payments, repossession, financing terms, and lender disputes. The lexical overlap is high: words like "loan", "payment", "lender", "vehicle" appear in both categories, but "vehicle" and "car" signals are insufficient when a complaint focuses mainly on loan terms.

---

## 10. Class Distribution

| Category | Total | % Total | Train | Val | Test | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Debt collection | 5,830 | 23.3% | 3,731 | 933 | 1,166 | 0.835 | 0.836 |
| Credit reporting, credit repair services, or other personal consumer reports | 4,250 | 17.0% | 2,720 | 680 | 850 | 0.532 | 0.600 |
| Mortgage | 3,950 | 15.8% | 2,528 | 632 | 790 | 0.930 | 0.922 |
| Credit reporting | 2,882 | 11.5% | 1,845 | 461 | 576 | 0.663 | 0.614 |
| Credit card | 1,680 | 6.7% | 1,075 | 269 | 336 | 0.601 | 0.591 |
| Student loan | 1,474 | 5.9% | 943 | 236 | 295 | 0.864 | 0.850 |
| Bank account or service | 1,346 | 5.4% | 862 | 215 | 269 | 0.546 | 0.543 |
| Credit card or prepaid card | 974 | 3.9% | 623 | 156 | 195 | 0.441 | 0.429 |
| Consumer Loan | 835 | 3.3% | 534 | 134 | 167 | 0.515 | 0.469 |
| Checking or savings account | 613 | 2.5% | 392 | 98 | 123 | 0.398 | 0.381 |
| Money transfer, virtual currency, or money service | 282 | 1.1% | 181 | 45 | 56 | 0.607 | 0.591 |
| Vehicle loan or lease | 221 | 0.9% | 142 | 35 | 44 | 0.295 | 0.295 |
| Payday loan, title loan, or personal loan | 192 | 0.8% | 123 | 31 | 38 | 0.211 | 0.232 |
| Payday loan | 169 | 0.7% | 108 | 27 | 34 | 0.529 | 0.434 |
| Money transfers | 144 | 0.6% | 92 | 23 | 29 | 0.655 | 0.567 |
| Prepaid card | 132 | 0.5% | 85 | 21 | 26 | 0.731 | 0.745 |
| Other financial service | 23 | 0.1% | 14 | 4 | 5 | 0.000 | 0.000 |
| Virtual currency | 3 | 0.01% | 2 | 1 | 1 | 0.000 | 0.000 |

**Correlation analysis:**
- Pearson(test_support, recall) = **0.568**
- Pearson(test_support, F1) = **0.622**

There is a moderate positive correlation between test set class support and both recall and F1. However, this is not a universal rule: several high-support categories (Credit reporting, Credit reporting, credit repair services, or other personal consumer reports) have relatively low F1 (0.60–0.61) despite large support. The dominant factor for those categories is label confusion rather than low support.

**Key support-performance observations:**
- "Mortgage" (support=790) achieves F1=0.922 — highest support, highest F1.
- "Virtual currency" (support=1) and "Other financial service" (support=5) have F1=0.000. With 1 and 5 test records respectively, a single misclassification yields 0% recall.
- "Payday loan, title loan, or personal loan" (support=38) and "Vehicle loan or lease" (support=44) have low F1 (0.23 and 0.30) not solely from low support, but also from confusions with near-identical legacy label variants ("Payday loan", "Consumer Loan").

---

## 11. Confidence / Probability Analysis

The final model is `LogisticRegression` and supports `predict_proba()`.

> **Note:** `predict_proba()` outputs from `LogisticRegression` with L-BFGS solver are **not calibrated probability estimates** unless a calibration procedure (e.g., `CalibratedClassifierCV`) has been applied. These values should be interpreted as **prediction confidence scores**, not as true probabilities.

**Confidence bands:**
- **High:** probability ≥ 0.70
- **Medium:** 0.40 ≤ probability < 0.70
- **Low:** probability < 0.40

### Correct Predictions (N = 3,478)

| Statistic | Predicted Class Probability |
|---|---:|
| Mean | 0.631 |
| Median | 0.637 |
| High (≥ 0.70) | 40.1% |
| Medium (0.40–0.69) | 42.3% |
| Low (< 0.40) | 17.6% |

### Incorrect Predictions (N = 1,522)

| Statistic | Predicted Class Probability |
|---|---:|
| Mean | 0.434 |
| Median | ~0.39 |
| High (≥ 0.70) | 8.3% |
| Medium (0.40–0.69) | 45.1% |
| Low (< 0.40) | 46.6% |

**Observations:**
- Correct predictions have substantially higher confidence (mean 0.631) than incorrect predictions (mean 0.434).
- 46.6% of incorrect predictions have low confidence (< 0.40), suggesting the model itself is uncertain about many of its errors.
- **127 high-confidence errors** (probability ≥ 0.70) exist — cases where the model was confident but wrong. These are the most significant failure cases and predominantly occur in the "Credit reporting" ↔ "Credit reporting, credit repair services, or other personal consumer reports" confusion pair.
- The wide gap in mean confidence between correct (0.631) and incorrect (0.434) predictions indicates the model's confidence score could serve as a useful uncertainty signal.

---

## 12. Text and N-gram Overlap Analysis

This section analyzes lexical overlap between the dominant confusion pairs using high-frequency unigram term analysis. This is **lexical analysis only** — no semantic similarity model was applied.

### Pair: "Credit reporting, credit repair services, or other personal consumer reports" ↔ "Credit reporting"

**Shared high-frequency terms:** `account`, `been`, `credit`, `from`, `have`, `information`, `report`, `reporting`, `that`, `their`, `them`, `they`

Both categories share identical top-frequency terms including "credit", "report", "reporting", and "information". The complaints are structurally and lexically near-identical, as both label names describe the same underlying consumer experience: disputing inaccurate items on credit bureau reports and requesting corrections. The distinction between these labels appears to be a CFPB administrative relabeling rather than a meaningful content difference in the complaints.

### Pair: "Credit card or prepaid card" ↔ "Credit card"

**Shared high-frequency terms:** `account`, `been`, `called`, `card`, `credit`, `from`, `have`, `payment`, `that`, `their`, `they`, `this`

Both categories share "credit", "card", "account", and "payment" as high-frequency terms. The two labels appear to be CFPB naming variants — "Credit card" is the older label and "Credit card or prepaid card" the newer one. The complaint texts contain identical types of financial disputes (unauthorized charges, billing errors, customer service issues) that do not systematically use prepaid-specific vocabulary to differentiate them.

### Pair: "Bank account or service" ↔ "Checking or savings account"

**Shared high-frequency terms:** `account`, `bank`, `been`, `charges`, `checking`, `from`, `have`, `that`, `their`, `they`, `this`, `with`

Both categories share "account", "bank", and "checking" as high-frequency terms. "Bank account or service" is the older CFPB label; "Checking or savings account" is the newer label. Complaints describe overdraft fees, unauthorized debits, account closures, and bank-customer service interactions — without using vocabulary that discriminates reliably between the two label versions.

### Pair: "Vehicle loan or lease" → "Consumer Loan" (59.1% error rate)

Vehicle loan complaints share high-frequency loan and payment vocabulary with Consumer Loan complaints. Terms like "loan", "payment", "lender", "financed" appear in both. The category-distinguishing terms ("vehicle", "car", "auto") appear in Vehicle loan complaints but not in sufficient frequency relative to generic loan vocabulary to consistently override the Consumer Loan prediction.

---

## 13. Evidence-Based Observations

Based on the measured data above:

1. **The dominant error source is CFPB administrative label versioning.** Three confusion cluster pairs account for a disproportionate share of all errors: Credit reporting (314 errors), Credit card (139 errors), and Bank account (121 errors). In each case, the two confused labels appear to be successive CFPB naming conventions for the same underlying product type applied across different time periods in the dataset.

2. **The model performs substantially better on "clean" categories.** Categories without a naming-variant counterpart in the dataset — Mortgage (F1=0.922), Student loan (F1=0.850), Debt collection (F1=0.836) — achieve much higher scores. These categories also have distinctive vocabularies (e.g., "foreclosure", "forbearance" for Mortgage; "repayment", "default", "servicer" for Student loan).

3. **High-confidence errors are concentrated in the label-ambiguity pairs.** Of 127 high-confidence errors (predicted probability ≥ 0.70), the majority are from the Credit reporting ↔ Credit reporting, credit repair services cluster, where both categories describe identical complaint types and share nearly identical top-frequency terms.

4. **Support correlates with F1, but is not the only driver (Pearson = 0.622).** Extremely low-support categories (Virtual currency: 1 record, Other financial service: 5 records) fail entirely, but intermediate-support categories with naming-variant confusion (Vehicle loan: 44 records, Payday loan, title loan, or personal loan: 38 records) also fail at recall < 30%. High-support categories with label ambiguity (Credit reporting, credit repair services: 850 records, F1=0.60) show that even abundant training data does not resolve the ambiguity when both labels describe identical content.

5. **The model assigns lower confidence to errors.** The average confidence of correct predictions (0.631) is substantially higher than that of incorrect predictions (0.434). This means model confidence could serve as a practical threshold for flagging uncertain predictions for human review.

---

## 14. Limitations

1. **Label ambiguity is architectural.** The largest source of errors (CFPB naming variant confusion) is inherent to the dataset labels. No model trained on this data with these labels can completely resolve this without label deduplication.

2. **Very low-support categories are statistically unreliable.** Results for Virtual currency (1 test record), Other financial service (5 records), Prepaid card (26 records), and Money transfers (29 records) are extremely sensitive to single prediction outcomes.

3. **Lexical overlap analysis is not semantic analysis.** Shared high-frequency terms do not prove semantic similarity. The analysis uses unigram frequency only — no word embeddings, topic models, or semantic similarity measures were applied.

4. **predict_proba values are not calibrated.** Without a calibration procedure, the absolute probability values should not be interpreted as true posterior class probabilities.

5. **The test set size for some categories is small.** Per-category F1 on 5–44 test records is not statistically stable.

---

## 15. Possible Directions for Future Improvement

> **These are directions for future consideration only. None of these are implemented in this milestone. The current model is not modified.**

1. **Label consolidation / deduplication:** Map all CFPB naming variants to a canonical label set (e.g., merge "Credit reporting" and "Credit reporting, credit repair services, or other personal consumer reports"; merge "Bank account or service" and "Checking or savings account"; merge "Credit card" and "Credit card or prepaid card"). This single change would eliminate the dominant error cluster.

2. **Confidence-based abstention or human routing:** Use the model confidence threshold (~0.40) to flag uncertain predictions for manual review or a secondary validation step, since errors are concentrated in the low-confidence band.

3. **Category-specific feature engineering:** For Vehicle loan vs. Consumer Loan, vocabulary features specific to vehicle finance (VIN, repossession, auto dealer, lease termination) could be up-weighted or given additional TF-IDF features.

4. **Feature importance analysis per category:** Examine the top TF-IDF feature weights per class in the LogisticRegression coefficients to identify whether distinguishing features are present in the vocabulary for the low-F1 categories.

5. **Temporal label split analysis:** If the dataset contains timestamps, separate complaints by date to assess whether specific label confusions are concentrated in particular time periods (i.e., when CFPB changed its product taxonomy).

---

## Output Files

| File | Description |
|---|---|
| `results/baseline_vs_improved.csv` | Controlled same-split baseline vs improved model comparison |
| `results/error_analysis.csv` | All 154 actual → predicted confusion pairs with error counts |
| `results/per_category_metrics.csv` | Per-category precision, recall, F1, support, and primary confusion target |
| `results/class_distribution.csv` | Class distribution across all splits with recall and F1 |
| `results/error_analysis_data.json` | Machine-readable analysis data for Streamlit and documentation |
| `results/confusion_matrix.png` | 18×18 confusion matrix heatmap |
| `scripts/generate_error_analysis.py` | Diagnostic analysis script (does not modify the model) |
| `tests/test_error_analysis.py` | Unit tests verifying analysis infrastructure |

---

*This document is a diagnostic-only milestone report. The trained classifier, vectorizers, feature configuration, and train/test splits are unchanged.*
