# Model Card: CFPB Customer Complaint Classification

## Model Overview

This project uses classical Natural Language Processing (NLP) to automatically classify CFPB consumer complaints into financial product categories.

**Model:** Logistic Regression
**Features:** Combined Word + Character TF-IDF
**Dataset:** 25,000 CFPB consumer complaints
**Classes:** 18 original product categories
**Training/Test Split:** 80% training, 20% test
**Framework:** Python, scikit-learn

## Performance

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **69.56%** |
| Macro F1        | **50.56%** |
| Weighted F1     | **69.55%** |
| Macro Precision | **49.67%** |
| Macro Recall    | **51.97%** |

The model performs substantially better on common categories than on very small minority categories.

## Performance Across Text Subgroups

The error analysis showed different performance depending on the type of complaint:

| Text subgroup/style                                     | Observed behavior                     |
| ------------------------------------------------------- | ------------------------------------- |
| Detailed complaints with clear product terminology      | Generally easier to classify          |
| Short, vague, or overlapping complaints                 | More difficult to classify            |
| Complaints involving closely related financial products | Frequent confusion between categories |

For example, complaints involving credit reporting and credit cards can contain highly overlapping terms, making the distinction difficult for a TF-IDF-based classifier.

## Simple Explainability Check

The model uses TF-IDF features, so its predictions can be inspected through the features that receive high weights.

For an example complaint, important features included terms such as:

* `resolved dispute`
* `company resolved`
* `dispute even`
* `payment credit`
* `credit card`

These features provide an interpretable indication of which words and phrases contributed to the classification decision.

## Main Limitation

The model struggles with **minority categories and highly similar product categories**.

Some categories have very few examples in the dataset, making it difficult for the classifier to learn reliable patterns. The model can also confuse historically related CFPB categories, such as:

* `Credit card` vs. `Credit card or prepaid card`
* `Credit reporting` vs. `Credit reporting, credit repair services, or other personal consumer reports`
* `Bank account or service` vs. `Checking or savings account`

Therefore, the model should be treated as a **classification assistance system**, not as a replacement for human review.

## Intended Use

The model is intended for:

* Automatic complaint category prediction
* Demonstrating classical NLP classification
* Exploring patterns in CFPB consumer complaints
* Assisting with initial complaint routing

It is **not intended for making financial, legal, regulatory, or customer-impact decisions without human verification.**

## Technology Used

* **TF-IDF:** Converts complaint text into numerical features.
* **Word n-grams:** Capture important words and short phrases.
* **Character n-grams:** Capture subword patterns and spelling variations.
* **Logistic Regression:** Performs multi-class classification.
* **Cosine Similarity:** Retrieves historically similar complaints.

## License

This project is released under the **MIT License**.

## Disclaimer

This project is an academic NLP system developed for educational and research purposes. Predictions are generated automatically and may be incorrect, particularly for minority classes and closely related categories. Model outputs should not be considered professional financial, legal, or regulatory advice.
