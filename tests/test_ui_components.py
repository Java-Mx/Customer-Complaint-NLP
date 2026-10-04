"""Tests for UI components and demonstration helpers in app/ui_components.py."""

from pathlib import Path
import pytest

from app.ui_components import DEMO_COMPLAINT_EXAMPLES
from src.preprocessing import preprocess_text
from src.classification import load_classifier, predict_complaint_category
from src.vectorization import get_top_active_features
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_demo_complaint_examples_structure():
    """Verify demo complaint examples meet academic demonstration requirements."""
    assert len(DEMO_COMPLAINT_EXAMPLES) >= 9

    standard_count = 0
    ambiguous_count = 0

    for key, item in DEMO_COMPLAINT_EXAMPLES.items():
        assert "label" in item
        assert "category_hint" in item
        assert "is_ambiguous" in item
        assert "text" in item
        assert len(item["text"]) > 60
        assert item["label"].strip() != ""

        if item["is_ambiguous"]:
            ambiguous_count += 1
        else:
            standard_count += 1

    assert standard_count >= 6
    assert ambiguous_count >= 3


def test_demo_complaints_preprocessing():
    """Verify all demo complaints pass through the classical preprocessing pipeline cleanly."""
    for key, item in DEMO_COMPLAINT_EXAMPLES.items():
        clean = preprocess_text(item["text"])
        assert isinstance(clean, str)
        assert len(clean) > 0


def test_demo_complaints_end_to_end_inference():
    """Verify demo complaints produce valid predictions with trained model artifacts."""
    model_path = ROOT_DIR / "models" / "complaint_classifier.joblib"
    w_vec_path = ROOT_DIR / "models" / "tfidf_vectorizer.joblib"
    c_vec_path = ROOT_DIR / "models" / "char_vectorizer.joblib"

    if not (model_path.exists() and w_vec_path.exists()):
        pytest.skip("Model artifacts not present in models/")

    clf = load_classifier(model_path)
    w_vec = joblib.load(w_vec_path)
    c_vec = joblib.load(c_vec_path) if c_vec_path.exists() else None
    vec = (w_vec, c_vec) if c_vec is not None else w_vec

    for key, item in DEMO_COMPLAINT_EXAMPLES.items():
        pred_cat, conf = predict_complaint_category(
            model=clf,
            vectorizer=vec,
            narrative=item["text"],
            preprocess=True
        )
        assert pred_cat in clf.classes_
        assert 0.0 <= conf <= 1.0

        # Also test feature extraction
        clean_text = preprocess_text(item["text"])
        df_active = get_top_active_features(w_vec, c_vec, clean_text, top_n=5)
        assert not df_active.empty
        assert "feature" in df_active.columns
        assert "weight" in df_active.columns


def test_app_package_structure():
    """Verify app is a proper Python package with __init__.py and importable components."""
    init_file = ROOT_DIR / "app" / "__init__.py"
    assert init_file.exists(), "app directory must contain __init__.py to prevent namespace shadowing"

    import app
    assert hasattr(app, "__file__"), "app must be a package, not a namespace or shadowed module"

    import app.ui_components
    import app.charts
    assert hasattr(app.ui_components, "DEMO_COMPLAINT_EXAMPLES")
    assert hasattr(app.charts, "create_baseline_comparison_chart")


def test_app_script_execution_isolated():
    """Verify python app/app.py executes without ModuleNotFoundError across execution contexts."""
    import subprocess
    import sys

    # 1. Direct script execution from project root
    res_root = subprocess.run(
        [sys.executable, "app/app.py"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res_root.returncode == 0, f"app.py execution from root failed: {res_root.stderr}"
    assert "ModuleNotFoundError" not in res_root.stderr

    # 2. Direct script execution from app/ directory
    res_app_dir = subprocess.run(
        [sys.executable, "app.py"],
        cwd=str(ROOT_DIR / "app"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res_app_dir.returncode == 0, f"app.py execution from app/ failed: {res_app_dir.stderr}"
    assert "ModuleNotFoundError" not in res_app_dir.stderr

    # 3. Module execution (-m app.app) from project root
    res_module = subprocess.run(
        [sys.executable, "-m", "app.app"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res_module.returncode == 0, f"python -m app.app execution failed: {res_module.stderr}"
    assert "ModuleNotFoundError" not in res_module.stderr

