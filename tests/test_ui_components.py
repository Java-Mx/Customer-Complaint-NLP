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
