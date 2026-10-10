# Model Card: CFPB Customer Complaint Classification

## 1. Model Details

| Field | Description |
|---|---|
| Model name | CFPB Complaint Classifier |
| Task | Multi-class classification of consumer financial complaints into CFPB product categories |
| Model | Word + character TF-IDF features combined with class-balanced Logistic Regression (`C=2.0`, `solver='lbfgs'`) |
| Training data | CFPB Consumer Complaint Database; 25,000 complaint records across 18 original product categories |
| Data split | 20,000-record training pool; internally split into 16,000 training and 4,000 validation records; 5,000-record untouched holdout test set |
| Preprocessing | Text cleaning, lowercasing, URL/email and noise removal, CFPB redaction handling, whitespace normalization, tokenization, and stopword removal; numeric information is retained |
| Feature configuration | Word TF-IDF unigrams `(1, 1)` plus character TF-IDF n-grams `(3, 5)`; 114,493 combined sparse features |
| Libraries | Python, scikit-learn, SciPy, joblib, Streamlit |
| Developed by | Customer Complaint NLP project team, NLP course, October 2026 |

## 2. Intended Use

- **Intended:** Educational demonstration of classical NLP; automatic first-pass categorization of CFPB consumer complaints; exploration of complaint patterns; and assistance with initial complaint triage.
- **Out of scope:** Fully automated financial, legal, regulatory, or customer-impact decisions; replacing a human reviewer; treating predicted category/confidence as authoritative; and using the model as professional financial or legal advice.

## 3. Overall Performance

On the 5,000-record untouched holdout test set, the current production configuration achieved **69.82% accuracy**, **50.88% macro-F1**, and **69.78% weighted-F1**. The lower macro-F1 compared with weighted-F1 indicates that performance is less consistent across rare categories than across the dataset as a whole.

| Metric | Score |
|---|---:|
| Accuracy | **69.82%** |
| Macro-F1 | **50.88%** |
| Weighted-F1 | **69.78%** |
| Macro Precision | **50.41%** |
| Macro Recall | **51.69%** |

Evaluation used a stratified split with `random_state=42`. The 5,000-record holdout test set was kept out of model fitting and model selection.

## 4. Performance Across Text Subgroups

The project’s error audit documents category-level and error-pattern differences. It does **not** report audited accuracy/F1 scores for the text-style subgroups below; these are qualitative observations, not measured subgroup metrics.

| Subgroup / style | Observed behavior |
|---|---|
| Detailed complaints with clear product terminology | Generally easier to classify when the text contains distinctive product terms |
| Short, vague, or overlapping complaints | More difficult when the narrative provides few distinguishing terms |
| Complaints involving closely related financial products | Frequent confusion between labels with overlapping vocabulary |
| Complaints involving historical CFPB label variants | Confusion occurs between related labels retained from different taxonomy periods |

**Key audit finding:** The leading confusion groups in the reference 18-category error analysis were credit-reporting labels (**314** bidirectional confusions), card/prepaid labels (**148**), and banking-account labels (**121**). These counts show that closely related and historically revised product labels are a material source of error. They are from the 18-category reference-model error analysis, not the newer production configuration’s reported per-style subgroup scores.

## 5. Explainability Check

**Method: Inspect active TF-IDF features and model outputs.** Because Logistic Regression is a linear classifier over TF-IDF features, its class-specific coefficients and the active TF-IDF features for a complaint can be inspected to understand which text patterns influence a prediction.

For one example complaint shown in the project’s live demo — “I noticed a payment on my credit card that I did not make, and the company has not resolved my dispute even after I contacted them.” — the displayed active features included:

- `resolved dispute`
- `company resolved`
- `make company`
- `dispute even`
- `noticed payment`
- `payment credit`
- `credit card`

These are examples of active features in the input representation, not a verified ranking of class-specific Logistic Regression coefficients. A feature being active does not, on its own, prove that it pushed the prediction toward a particular class; contribution direction depends on the predicted class’s learned coefficients. The app also displays a model confidence score, which should not be interpreted as calibrated probability or prediction accuracy.

## 6. Limitations and Ethical Considerations

- **Minority-category performance:** Very small classes have limited examples, making reliable patterns difficult to learn. In the reference error audit, `Virtual currency` had one test example and `Other financial service` had five; neither was correctly classified in that audit. These figures illustrate the sparse-class issue and are not claimed as current-production per-class results.
- **Closely related labels:** The model confuses categories with overlapping terminology and historical CFPB label variants, including credit reporting, credit card/prepaid card, and bank-account labels.
- **Bag-of-features limitation:** TF-IDF represents lexical patterns rather than deep contextual meaning. Similar wording across financial products can make categories hard to distinguish.
- **Confidence limitations:** Logistic Regression confidence scores are not necessarily calibrated probabilities.
- **Taxonomy trade-off:** Normalizing the 18 original categories into 11 conservative or 10 broad groups can improve aggregate scores, but merges distinctions and loses some regulatory/product detail. These taxonomy experiments are separate from the current 18-category production model.
- **Human review:** Predictions should support, not replace, human complaint review, especially for rare, ambiguous, or high-impact cases.

## 7. Recommendations

- Route ambiguous or low-confidence predictions to a human reviewer rather than treating the predicted category as definitive.
- Monitor per-category precision, recall, F1, support, and confusion pairs; report subgroup metrics only after defining subgroups and evaluating them on held-out data.
- Consider taxonomy normalization only when the intended use supports merging labels; document any product/regulatory distinctions lost by doing so.
- Re-evaluate the model on a fresh, representative holdout sample when the CFPB data, label definitions, or deployment context changes.
- Preserve the classical NLP constraint for the current project: TF-IDF feature extraction, cosine similarity for retrieval, and Logistic Regression for classification.

---

**License:** The repository’s original software is released under the MIT License. The CFPB dataset is a separate data source and is subject to the CFPB’s applicable data policies.

**Disclaimer:** This is an academic NLP system for educational and research purposes. Predictions can be incorrect, particularly for rare and closely related categories. Outputs are not financial, legal, or regulatory advice and should not be used for consequential decisions without human verification.

**Project:** [Customer Complaint Similarity & Categorisation](https://github.com/Java-Mx/Customer-Complaint-NLP)  
**Dataset source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
