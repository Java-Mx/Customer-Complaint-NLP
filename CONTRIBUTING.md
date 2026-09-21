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

---

## 7. Documentation Synchronization

Documentation synchronization is a **mandatory** development workflow in this repository.

After **EVERY** completed feature, experiment, bug fix affecting behavior, or major analysis, contributors must execute the following 10-step synchronization checklist before proposing or committing changes:

1. **Inspect `README.md`**: Check if project architecture, benchmarks, dataset figures, or project structure require updating.
2. **Update `docs/project-updates.md`**: Record a structured chronological entry detailing the milestone, commit message, technical concepts, implementation details, evaluation results, tests, and artifacts.
3. **Update Relevant Technical Docs**: Update or create dedicated technical documentation (e.g., `docs/error-analysis.md`, `docs/taxonomy-analysis.md`) reflecting verified implementations.
4. **Update `docs/viva-preparation.md`**: Add or refine theoretical defense questions whenever a new NLP/ML concept, mathematical formulation, design decision, or experimental result is introduced.
5. **Verify Numerical Integrity**: Ensure that all numerical metrics, sample counts, feature counts, and percentages in documentation match generated result files (e.g., in `results/`).
6. **Run Full Test Suite**: Run `python -m pytest -v` to ensure 100% of tests pass.
7. **Execute Streamlit Smoke Test**: Verify that `app/app.py` runs cleanly and passes the health check (`HTTP 200` on `/_stcore/health`) whenever user-interface or visualization code changes.
8. **Inspect Git Diff**: Run `git diff` and `git diff --check` to review staged changes, whitespace issues, and file additions.
9. **Synchronized Commits**: Commit source code, configuration files, tests, and documentation together when they belong to the same milestone.
10. **Verify Before Push**: Push to `origin/main` only after complete local validation passes.

### Core Principles

- **Ground Truth Hierarchy**: *Generated experiment outputs and verified evaluation logs are authoritative over manually written README numbers.*
- **Zero Stale Documentation**: *Never leave README benchmark numbers, project trees, or test counts stale after a milestone.*
