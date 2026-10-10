# Model Card: CFPB Customer Complaint Classification

## 1. Model Details

| Field | Description |
|---|---|
| Model name | CFPB Complaint Classifier |
| Task | Multi-class classification of consumer financial complaints into CFPB product categories |
| Classifier | Logistic Regression (`solver='lbfgs'`, `C=2.0`, `class_weight='balanced'`, `max_iter=1000`) |
| Features | Combined Word TF-IDF and Character TF-IDF |
| Word features | Unigrams (`ngram_range=(1, 1)`) |
| Character features | Character n-grams (`ngram_range=(3, 5)`, `analyzer='char'`) |
| Feature dimensions | 114,493 combined sparse features (14,493 word + 100,000 character features) |
| Dataset | CFPB Consumer Complaint Database; 25,000 complaint records and 18 original product categories |
| Data split | 20,000-record training pool (16,000 training and 4,000 validation records) and 5,000-record untouched holdout test set |
| Preprocessing | Text cleaning, lowercasing, URL/email and noise removal, CFPB redaction handling, whitespace normalization, tokenization, and stopword removal; numeric information is retained |
| Libraries | Python, scikit-learn, SciPy, joblib, Streamlit |
| Project | Customer Complaint Similarity & Categorisation, October 2026 |

## 2. Intended Use

- **Intended:** Educational demonstration of classical NLP; first-pass categorization of CFPB consumer complaints; exploration of complaint patterns; and assistance with initial complaint triage.
- **Out of scope:** Fully automated financial, legal, regulatory, or customer-impact decisions; replacing human review; or treating predictions as professional financial or legal advice.

## 3. Overall Performance

The current production configuration was evaluated on the 5,000-record untouched holdout test set. The model achieves 69.82% accuracy and 50.88% macro-F1. The difference between macro-F1 and weighted-F1 indicates that performance is less consistent across rare categories than across the dataset as a whole.

| Metric | Current production model |
|---|---:|
| Accuracy | **69.82%** |
| Macro F1 | **50.88%** |
| Weighted F1 | **69.78%** |
| Macro Precision | **50.41%** |
| Macro Recall | **51.69%** |
| Weighted Precision | **70.08%** |
| Weighted Recall | **69.82%** |

The data split was stratified with `random_state=42`. The 5,000-record holdout test set was not used for model fitting or model selection.

### Taxonomy experiment: 18, 11, and 10 categories

The project separately evaluated whether consolidating historically related CFPB product labels changes classification performance. The following scores are reported in [`docs/taxonomy-analysis.md`](https://github.com/Java-Mx/Customer-Complaint-NLP/blob/main/docs/taxonomy-analysis.md):

| Taxonomy formulation | Number of categories | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Original CFPB taxonomy | 18 | 69.56% | 50.56% | 69.55% |
| Conservative taxonomy (v1) | 11 | **81.50%** | **63.43%** | **81.61%** |
| Broad taxonomy (v2) | 10 | **82.32%** | **66.57%** | **82.42%** |

The **11-category Conservative taxonomy** merges documented label variants while keeping `Consumer Loan` separate. The **10-category Broad taxonomy** additionally merges `Consumer Loan` into `Consumer & Small Dollar Loans`. The mappings are defined in [`config/taxonomy_v1_conservative.json`](https://github.com/Java-Mx/Customer-Complaint-NLP/blob/main/config/taxonomy_v1_conservative.json) and [`config/taxonomy_v2_broad.json`](https://github.com/Java-Mx/Customer-Complaint-NLP/blob/main/config/taxonomy_v2_broad.json).

**Important comparison note:** These taxonomy experiment results are from a separate earlier experiment that used a 237,148-feature representation and `C=1.0`. The current production model uses 114,493 features and `C=2.0`, with 69.82% accuracy and 50.88% macro F1. Therefore, the taxonomy results should not be presented as measurements of the current production artifacts. The score increase in the taxonomy experiment is largely attributable to merging labels, which changes the classification task and loses some label granularity; it does not by itself show that the classifier learned substantially better distinctions.

## 4. Performance Across Text Subgroups

The project documents qualitative behavior by complaint style and quantitative errors between product categories. It does not provide verified accuracy and F1 scores for separately defined short, long, formal, or informal text subgroups, so no subgroup scores are claimed here.

| Subgroup / style | Observed behavior |
|---|---|
| Detailed complaints with clear product terminology | Generally easier to classify when distinctive product terms are present |
| Short, vague, or overlapping complaints | More difficult when the text contains few distinguishing terms |
| Complaints involving closely related financial products | More likely to be confused because their vocabulary overlaps |
| Complaints with historical CFPB label variants | May be confused because related labels from different taxonomy periods remain distinct in the original 18-category task |

**Key audit finding:** In the documented 18-category reference-model error analysis, the main bidirectional confusion groups were credit-reporting labels (314 errors), card/prepaid labels (148 errors), and banking-account labels (121 errors). These counts are from that reference-model audit, not from a separate current-production text-style subgroup evaluation. See [`docs/error-analysis.md`](https://github.com/Java-Mx/Customer-Complaint-NLP/blob/main/docs/error-analysis.md).

## 5. Explainability Check

The model uses TF-IDF features and Logistic Regression, a linear classifier. Its learned class coefficients and the non-zero TF-IDF features in an input can be inspected to help explain predictions.

For one example complaint shown in the live demo — “I noticed a payment on my credit card that I did not make, and the company has not resolved my dispute even after I contacted them.” — displayed active features included:

- `resolved dispute`
- `company resolved`
- `make company`
- `dispute even`
- `noticed payment`
- `payment credit`
- `credit card`

These are active input features, not a verified ranking of class-specific coefficient contributions. A feature being active does not alone establish whether it pushed the prediction toward or away from a category; that depends on the learned coefficients for the relevant class. The application's confidence score should not be interpreted as a calibrated probability or as prediction accuracy.

## 6. Limitations and Ethical Considerations

- **Minority categories:** Very small categories provide limited examples from which to learn. In the reference error audit, `Virtual currency` had one test example and `Other financial service` had five; neither example was correctly classified in that audit. These are reference-audit figures, not current-production per-category scores.
- **Related labels:** The classifier can confuse similar product categories, particularly historical CFPB label variants for credit reporting, cards/prepaid cards, and banking accounts.
- **Limited contextual understanding:** TF-IDF represents lexical patterns rather than full meaning and context. Complaints with overlapping vocabulary can be difficult to distinguish.
- **Confidence:** Logistic Regression scores are not necessarily calibrated probabilities.
- **Taxonomy trade-off:** Combining 18 labels into 11 or 10 categories raises aggregate scores in the taxonomy experiment, but it also merges distinctions and loses some product or regulatory detail.
- **Human oversight:** Predictions should assist reviewers, not replace them, especially for rare, ambiguous, or consequential cases.

## 7. Recommendations

- Send ambiguous or low-confidence predictions for human review rather than treating the predicted category as definitive.
- Monitor per-category precision, recall, F1, support, and confusion pairs; publish text-style subgroup metrics only after defining and evaluating those groups on held-out data.
- Use taxonomy normalization only when the intended use supports merging labels, and document the detail lost by doing so.
- Re-evaluate using a fresh, representative holdout sample when the dataset, label definitions, or deployment context changes.
- Keep the methods distinct: TF-IDF provides text features, cosine similarity retrieves similar historical complaints, and Logistic Regression predicts a complaint category.

**License:** The repository's original software is released under the MIT License. The CFPB dataset is a separate data source and remains subject to the CFPB's applicable data policies.

**Disclaimer:** This is an academic NLP system for educational and research purposes. Predictions may be wrong, especially for rare or closely related categories. Outputs are not financial, legal, or regulatory advice and should not be used for consequential decisions without human verification.

**Project repository:** [Customer Complaint Similarity & Categorisation](https://github.com/Java-Mx/Customer-Complaint-NLP)  
**Dataset source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
