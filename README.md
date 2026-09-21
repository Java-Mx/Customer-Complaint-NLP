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
│   └── .gitkeep               # Jupyter notebooks for exploratory analysis
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Text cleaning, normalization, and tokenization
│   ├── vectorization.py       # TF-IDF feature extraction
│   ├── similarity.py          # Cosine similarity calculations & retrieval
│   ├── classification.py      # Classifier training & inference
│   └── evaluation.py          # Metrics, classification reports, confusion matrices
│
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py  # Unit tests for preprocessing routines
│
├── app/
│   └── app.py                 # Streamlit web application interface
│
├── models/
│   └── .gitkeep               # Serialized models and vectorizers (gitignored)
│
├── results/
│   └── .gitkeep               # Generated plots, confusion matrices, evaluation metrics
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
Execute unit tests via pytest:
```bash
python -m pytest
```

### Running the Web Application
Launch the Streamlit web interface:
```bash
streamlit run app/app.py
```

---

## Model

The supervised classification component employs **Multinomial Logistic Regression** (with L2 regularization).

### Architectural Rationale:
1. **Convex Optimization**: Logistic Regression provides stable, globally optimal convergence on sparse, high-dimensional TF-IDF feature representations.
2. **Probability Calibration**: Logistic Regression outputs well-calibrated class probability distributions, allowing confidence thresholding for triage routing.
3. **Interpretability**: Linear feature weights allow straightforward inspection of top predictive n-grams associated with each product category.
4. **Computational Efficiency**: Extremely fast to train and evaluate compared to heavy ensemble or deep learning architectures, making it suitable for standard hardware.

---

## Evaluation

Model performance will be evaluated against unseen test partitions using standard classification metrics:
- **Accuracy**
- **Precision (Macro & Weighted)**
- **Recall (Macro & Weighted)**
- **F1-Score (Macro & Weighted)**
- **Confusion Matrix Visualization**

*(Official benchmark figures will be populated here once training and validation milestones are completed on the dataset).*

| Metric | Score |
|---|---|
| Accuracy | *[Pending model training]* |
| Macro Precision | *[Pending model training]* |
| Macro Recall | *[Pending model training]* |
| Macro F1-Score | *[Pending model training]* |

---

## Limitations

- **Syntactic Context**: Classical bag-of-words and TF-IDF representations do not capture complex long-range syntactic nuances or word re-ordering beyond the defined n-gram window.
- **Out-of-Vocabulary Terms**: Words not present in the training vocabulary will be ignored during inference.
- **Narrative Dependency**: The system requires complaints to contain narrative text; records where consumers opted out of narrative publication cannot be processed for textual similarity.

---

## Future Improvements

- Hyperparameter tuning via grid search across TF-IDF max features and regularization parameters.
- Incorporating domain-specific financial stopword lists.
- Support for hierarchical classification (predicting both `Product` and `Sub-product`).
- Exporting automated summary evaluation reports in PDF/Markdown.

---

## Dataset Source

- **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**:
  [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
