# Contributing Guidelines

Thank you for your interest in contributing to **Customer Complaint Similarity & Categorisation**! We welcome contributions to improve data preprocessing, vectorization, modeling, evaluation, and documentation.

Please follow these guidelines to keep development consistent and reliable.

---

## 1. Cloning the Repository

To get started, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Java-Mx/Customer-Complaint-NLP.git
cd Customer-Complaint-NLP
```

---

## 2. Creating a Branch

Always create a dedicated feature or fix branch from `main`. Do not work directly on `main`:

```bash
git checkout -b feat/your-feature-name
# or for bug fixes:
git checkout -b fix/issue-description
```

Use conventional branch prefixes:
- `feat/`: New features or pipeline components
- `fix/`: Bug fixes or correction in logic
- `docs/`: Documentation additions or revisions
- `test/`: New tests or test improvements

---

## 3. Setting Up the Environment & Installing Dependencies

It is recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Running Tests

Before making commits or proposing changes, verify that all unit tests pass:

```bash
python -m pytest
```

Ensure all tests pass and that any new functions include corresponding test cases in the `tests/` directory.

---

## 5. Making Changes

- **Incremental Progress**: Keep pull requests focused on one component or milestone at a time (e.g., preprocessing, vectorization, classification).
- **Clean Code**: Follow standard PEP 8 conventions. Use clear variable names, type hints, and docstrings.
- **No Unapproved Dependencies**: Stick to the approved classical ML/NLP stack (Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, Streamlit). Do not introduce heavy LLM libraries, pretrained transformers, or proprietary API dependencies.
- **Never Commit Raw Data**: Do not commit large CSV data files or model weights. Ensure `.gitignore` rules are respected.

---

## 6. Creating a Pull Request

1. Commit your changes with descriptive, conventional commit messages:
   ```bash
   git add .
   git commit -m "feat: add text tokenization and stopword removal"
   ```
2. Push your branch to GitHub:
   ```bash
   git push origin feat/your-feature-name
   ```
3. Open a Pull Request against the `main` branch on GitHub:
   - Provide a concise summary of changes made.
   - Reference any related issues.
   - Confirm that all tests pass locally.
