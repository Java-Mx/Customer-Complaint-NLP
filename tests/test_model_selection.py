"""Tests for leakage-safe model-selection helpers (src/model_selection.py)."""

import numpy as np
import pandas as pd
import pytest

from src.model_selection import (
    SELECTION_METRICS,
    FeatureSpec,
    fit_features,
    power_class_weights,
    rank_candidates,
    selection_key,
    text_statistics,
    transform_features,
    validation_metrics,
)

TRAIN = ["credit card late fee charged", "mortgage payment escrow issue", "debt collector calling daily"] * 4
VAL = ["totally unseenword zebra credit card", "mortgage escrow"]


def test_vectorizer_vocabulary_fit_on_train_only():
    spec = FeatureSpec.make(word=dict(ngram_range=(1, 1), min_df=1))
    blocks, X_tr = fit_features(spec, TRAIN)
    vocab = set(blocks["word"].vocabulary_)
    assert "zebra" not in vocab and "unseenword" not in vocab
    X_val = transform_features(spec, blocks, VAL)
    assert X_val.shape[1] == X_tr.shape[1]
    # transform must not mutate the vocabulary
    assert set(blocks["word"].vocabulary_) == vocab


def test_idf_not_influenced_by_validation_text():
    spec = FeatureSpec.make(word=dict(ngram_range=(1, 1), min_df=1))
    b1, _ = fit_features(spec, TRAIN)
    idf_before = b1["word"].idf_.copy()
    transform_features(spec, b1, VAL * 50)
    np.testing.assert_array_equal(idf_before, b1["word"].idf_)


def test_word_char_text_stats_combination_is_sparse():
    spec = FeatureSpec.make(word=dict(min_df=1), char=dict(ngram_range=(2, 3), min_df=1), text_stats=True)
    raw = ["Card XXXX fee", "no mask here"] * 3
    blocks, X = fit_features(spec, ["card fee", "no mask here"] * 3, raw)
    from scipy.sparse import issparse
    assert issparse(X)
    assert X.shape[1] == len(blocks["word"].vocabulary_) + len(blocks["char"].vocabulary_) + 3
    Xv = transform_features(spec, blocks, ["card"], ["card"])
    assert Xv.shape[1] == X.shape[1]


def test_text_stats_requires_raw_and_is_stateless():
    spec = FeatureSpec.make(word=dict(min_df=1), text_stats=True)
    with pytest.raises(ValueError):
        fit_features(spec, TRAIN)
    a = text_statistics(["hello XXXX world XX/XX/XXXX"]).toarray()
    b = text_statistics(["hello XXXX world XX/XX/XXXX"]).toarray()
    np.testing.assert_array_equal(a, b)
    assert a[0, 2] == 1.0 and text_statistics(["clean"]).toarray()[0, 2] == 0.0


def test_featurespec_requires_a_block():
    with pytest.raises(ValueError):
        FeatureSpec.make()


def test_power_class_weights_train_only_and_endpoints():
    y = ["a"] * 80 + ["b"] * 20
    uniform = power_class_weights(y, 0.0)
    assert uniform == {"a": 1.0, "b": 1.0}
    bal = power_class_weights(y, 1.0)
    assert bal["a"] == pytest.approx(100 / (2 * 80)) and bal["b"] == pytest.approx(100 / (2 * 20))
    soft = power_class_weights(y, 0.5)
    assert 1.0 < soft["b"] < bal["b"]
    # weights depend only on the labels given (training labels), so unseen classes get no weight
    assert set(soft) == {"a", "b"}
    with pytest.raises(ValueError):
        power_class_weights(y, -1)


def test_selection_order_macro_f1_then_recall_then_accuracy_then_weighted():
    assert SELECTION_METRICS == ("Val Macro F1", "Val Macro Rec", "Val Accuracy", "Val Weighted F1")
    base = dict(zip(SELECTION_METRICS, (0.5, 0.5, 0.7, 0.7)))
    rows = [
        {**base, "id": "tie_mf1_lower_rec", "Val Macro Rec": 0.4},
        {**base, "id": "tie_all_higher_wf1", "Val Weighted F1": 0.8},
        {**base, "id": "base"},
        {**base, "id": "higher_mf1", "Val Macro F1": 0.51, "Val Macro Rec": 0.1},
        {**base, "id": "higher_acc", "Val Accuracy": 0.9},
    ]
    ranked = rank_candidates(pd.DataFrame(rows))
    assert list(ranked["id"]) == ["higher_mf1", "higher_acc", "tie_all_higher_wf1", "base", "tie_mf1_lower_rec"]
    assert selection_key(rows[2]) > selection_key(rows[0])


def test_validation_metrics_keys_and_values():
    m = validation_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert m["Val Accuracy"] == pytest.approx(0.75)
    for k in ("Val Macro Prec", "Val Macro Rec", "Val Macro F1", "Val Weighted Prec", "Val Weighted Rec", "Val Weighted F1"):
        assert 0.0 <= m[k] <= 1.0


def test_experiment_script_split_protocol_is_disjoint():
    """Protocol: 16k/4k internal split of the pool never overlaps the held-out test indices."""
    from src.classification import train_test_split_data
    df = pd.DataFrame({"text": [f"t{i}" for i in range(500)], "category": ["a", "b", "c", "d", "e"] * 100})
    X_pool, X_test, y_pool, y_test = train_test_split_data(df, 0.2, 42, True)
    pool_df = pd.DataFrame({"text": X_pool, "category": y_pool})
    X_tr, X_val, _, _ = train_test_split_data(pool_df, 0.2, 42, True)
    assert not (set(X_tr.index) & set(X_val.index))
    assert not ((set(X_tr.index) | set(X_val.index)) & set(X_test.index))
    assert len(X_tr) == 320 and len(X_val) == 80 and len(X_test) == 100


def test_persisted_final_model_configuration():
    """Verify that persisted models match winning spec (C=2, Word 1,1, Char 3,5, 114,493 dims)."""
    import joblib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    clf = joblib.load(root / "models" / "complaint_classifier.joblib")
    w_vec = joblib.load(root / "models" / "tfidf_vectorizer.joblib")
    c_vec = joblib.load(root / "models" / "char_vectorizer.joblib")

    assert clf.C == 2.0
    assert clf.class_weight == "balanced"
    assert w_vec.ngram_range == (1, 1)
    assert c_vec.ngram_range == (3, 5)
    assert c_vec.analyzer == "char"
    assert len(w_vec.vocabulary_) + len(c_vec.vocabulary_) == 114493


def test_artifact_metrics_consistency_between_json_files():
    """Verify final_evaluation_metrics.json, error_analysis_data.json, and test_metrics.json agree."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    final_m = json.loads((root / "results" / "final_evaluation_metrics.json").read_text(encoding="utf-8"))
    ea_m = json.loads((root / "results" / "error_analysis_data.json").read_text(encoding="utf-8"))
    test_m = json.loads((root / "results" / "model_improvement_test_metrics.json").read_text(encoding="utf-8"))

    # Accuracy agreement
    assert final_m["final_model"]["accuracy"] == pytest.approx(0.6982, abs=1e-4)
    assert ea_m["improved_metrics"]["accuracy"] == pytest.approx(0.6982, abs=1e-4)
    assert test_m["new_test"]["accuracy"] == pytest.approx(0.6982, abs=1e-4)

    # Macro F1 agreement
    assert final_m["final_model"]["macro_f1"] == pytest.approx(0.5088, abs=1e-4)
    assert ea_m["improved_metrics"]["macro_f1"] == pytest.approx(0.5088, abs=1e-4)
    assert test_m["new_test"]["macro_f1"] == pytest.approx(0.5088, abs=1e-4)

