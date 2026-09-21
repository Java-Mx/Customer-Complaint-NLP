"""Classification Error Analysis for the CFPB Complaint Categorisation Model.

Performs a comprehensive diagnostic analysis of the final trained classification model
on the untouched 5,000-record holdout test set.

This script does NOT retrain or modify the existing model.
It does NOT modify the train/validation/test splits.
It uses the same random_state=42, stratified 80/20 partitioning as run_experiments.py.

Phases executed:
  Phase 0 - Controlled same-split baseline benchmark (Word TF-IDF, no class weighting)
  Phase 1 - Error analysis: confusion matrix, confusion pairs, per-category metrics
  Phase 2 - Confidence / probability analysis (LogisticRegression predict_proba)
  Phase 3 - Class distribution across all splits
  Phase 4 - Lexical overlap analysis for top confusion pairs

Output files:
  results/baseline_vs_improved.csv       -- Controlled baseline vs improved model
  results/error_analysis.csv             -- All actual->predicted confusion pairs
  results/per_category_metrics.csv       -- Per-category precision/recall/F1/support
  results/error_analysis_data.json       -- Machine-readable data for Streamlit & docs
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing import preprocess_series
from src.vectorization import (
    combine_sparse_matrices,
    create_char_vectorizer,
    create_vectorizer,
    fit_transform_tfidf,
    transform_tfidf,
    transform_word_char,
)
from src.classification import train_test_split_data, predict_categories


# Constants
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MED = 0.40

RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Helpers
def _confidence_band(prob: float) -> str:
    if prob >= CONFIDENCE_HIGH:
        return "high"
    if prob >= CONFIDENCE_MED:
        return "medium"
    return "low"


def _metrics_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "weighted_precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def _prob_stats(probs: np.ndarray) -> dict:
    if len(probs) == 0:
        return {}
    return {
        "count": int(len(probs)),
        "mean": round(float(np.mean(probs)), 4),
        "median": round(float(np.median(probs)), 4),
        "q1": round(float(np.percentile(probs, 25)), 4),
        "q3": round(float(np.percentile(probs, 75)), 4),
        "min": round(float(np.min(probs)), 4),
        "max": round(float(np.max(probs)), 4),
        "pct_high": round(float(np.mean(probs >= CONFIDENCE_HIGH)) * 100, 2),
        "pct_medium": round(float(np.mean((probs >= CONFIDENCE_MED) & (probs < CONFIDENCE_HIGH))) * 100, 2),
        "pct_low": round(float(np.mean(probs < CONFIDENCE_MED)) * 100, 2),
    }


# Phase 0 — Controlled Baseline Benchmark
def run_phase0_baseline(df: pd.DataFrame, X_test, y_test):
    """Fit the original baseline model on the exact same 20k training pool.

    Baseline model configuration:
      - Word TF-IDF: ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False
      - LogisticRegression: solver=lbfgs, C=1.0, class_weight=None, max_iter=1000, random_state=42

    Returns: baseline metrics dict and predictions on the test set.
    """
    from sklearn.linear_model import LogisticRegression

    print("\n[Phase 0] Training controlled same-split baseline model...")
    # Reproduce the exact 80/20 stratified split
    X_pool, _, y_pool, _ = train_test_split_data(df, test_size=0.20, random_state=42, stratify=True)

    print(f"  Training pool: {len(X_pool)} records.")
    clean_pool = preprocess_series(X_pool)
    clean_test = preprocess_series(X_test)

    vec_baseline = create_vectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False
    )
    _, X_pool_vec = fit_transform_tfidf(vec_baseline, clean_pool)
    X_test_vec = transform_tfidf(vec_baseline, clean_test)

    clf_baseline = LogisticRegression(
        solver="lbfgs", C=1.0, class_weight=None, max_iter=1000, random_state=42
    )
    t0 = time.time()
    clf_baseline.fit(X_pool_vec, y_pool)
    print(f"  Baseline trained in {time.time() - t0:.1f}s.")

    y_pred_baseline = clf_baseline.predict(X_test_vec)
    baseline_metrics = _metrics_dict(y_test, y_pred_baseline)
    print(f"  Baseline  Accuracy={baseline_metrics['accuracy']:.4f}  "
          f"Macro F1={baseline_metrics['macro_f1']:.4f}  "
          f"Weighted F1={baseline_metrics['weighted_f1']:.4f}")

    return baseline_metrics, y_pred_baseline


# Phase 1 — Full Error Analysis
def run_phase1_error_analysis(
    clf, X_test_vec, y_test, X_test_raw, n_error_examples: int = 5
) -> dict:
    """Full confusion matrix analysis, confusion pairs, per-category metrics, error examples."""

    print("\n[Phase 1] Generating error analysis...")
    classes = list(clf.classes_)
    y_true_arr = np.asarray(y_test)
    y_pred_arr = predict_categories(clf, X_test_vec)

    # ----- Confusion matrix -----
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    n_classes = len(classes)

    # ----- Per-category metrics -----
    report_dict = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
    per_cat_rows = []
    for cat in classes:
        cat_report = report_dict.get(cat, {})
        support = int(cat_report.get("support", 0))
        precision = round(float(cat_report.get("precision", 0.0)), 4)
        recall = round(float(cat_report.get("recall", 0.0)), 4)
        f1 = round(float(cat_report.get("f1-score", 0.0)), 4)
        correct = int(np.sum((y_true_arr == cat) & (y_pred_arr == cat)))
        incorrect = support - correct
        # Most common misclassification target
        misclassified_mask = (y_true_arr == cat) & (y_pred_arr != cat)
        if misclassified_mask.sum() > 0:
            wrong_preds = y_pred_arr[misclassified_mask]
            primary_confusion = Counter(wrong_preds).most_common(1)[0]
            primary_confusion_cat = primary_confusion[0]
            primary_confusion_count = int(primary_confusion[1])
        else:
            primary_confusion_cat = ""
            primary_confusion_count = 0
        per_cat_rows.append({
            "category": cat,
            "support": support,
            "correct": correct,
            "incorrect": incorrect,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "primary_confusion_category": primary_confusion_cat,
            "primary_confusion_count": primary_confusion_count,
        })
    per_cat_df = pd.DataFrame(per_cat_rows).sort_values("support", ascending=False).reset_index(drop=True)

    # ----- All confusion pairs -----
    pair_rows = []
    for i, actual_cat in enumerate(classes):
        actual_support = int(cm[i].sum())
        if actual_support == 0:
            continue
        for j, pred_cat in enumerate(classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count == 0:
                continue
            pct_of_actual = round(count / actual_support * 100, 2)
            pair_rows.append({
                "actual_category": actual_cat,
                "predicted_category": pred_cat,
                "error_count": count,
                "pct_of_actual": pct_of_actual,
            })
    pairs_df = pd.DataFrame(pair_rows).sort_values("error_count", ascending=False).reset_index(drop=True)

    print(f"  Total mislabeled samples: {len(y_pred_arr[y_pred_arr != y_true_arr])}")
    print(f"  Distinct confusion pairs: {len(pairs_df)}")

    # ----- Top-20 confusion pairs -----
    top20 = pairs_df.head(20).copy()

    # ----- Error examples for top-10 confusion pairs -----
    raw_text_arr = np.asarray(X_test_raw.reset_index(drop=True))
    error_examples = {}
    top10_pairs = pairs_df.head(10).to_dict("records")
    for pair_row in top10_pairs:
        actual_cat = pair_row["actual_category"]
        pred_cat = pair_row["predicted_category"]
        pair_key = f"{actual_cat} → {pred_cat}"
        mask = (y_true_arr == actual_cat) & (y_pred_arr == pred_cat)
        indices = np.where(mask)[0][:n_error_examples]
        examples = []
        for idx in indices:
            examples.append({
                "complaint_index": int(idx),
                "actual_category": actual_cat,
                "predicted_category": pred_cat,
                "complaint_text_truncated": str(raw_text_arr[idx])[:500].strip(),
            })
        error_examples[pair_key] = examples

    return {
        "classes": classes,
        "y_pred": y_pred_arr,
        "confusion_matrix": cm.tolist(),
        "per_category": per_cat_df,
        "all_pairs": pairs_df,
        "top20_pairs": top20,
        "error_examples": error_examples,
    }


# Phase 2 — Confidence / Probability Analysis
def run_phase2_confidence(clf, X_test_vec, y_true, y_pred) -> dict:
    """Analyze prediction probabilities from LogisticRegression.predict_proba()."""

    print("\n[Phase 2] Confidence / probability analysis...")
    if not hasattr(clf, "predict_proba"):
        print("  Skipping: model does not support predict_proba.")
        return {}

    proba_matrix = clf.predict_proba(X_test_vec)  # shape (N, n_classes)
    classes = list(clf.classes_)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Max predicted probability (confidence of predicted class)
    max_proba = proba_matrix.max(axis=1)

    # Probability assigned to the actual class
    class_idx_map = {c: i for i, c in enumerate(classes)}
    actual_proba = np.array([
        proba_matrix[i, class_idx_map[y_true_arr[i]]]
        for i in range(len(y_true_arr))
    ])

    correct_mask = (y_pred_arr == y_true_arr)
    incorrect_mask = ~correct_mask

    correct_probs = max_proba[correct_mask]
    incorrect_probs = max_proba[incorrect_mask]
    correct_actual_probs = actual_proba[correct_mask]
    incorrect_actual_probs = actual_proba[incorrect_mask]

    print(f"  Correct predictions:   {correct_mask.sum():,} "
          f"(mean prob={np.mean(correct_probs):.3f})")
    print(f"  Incorrect predictions: {incorrect_mask.sum():,} "
          f"(mean prob={np.mean(incorrect_probs):.3f})")

    # High-confidence errors (model was "sure" but wrong)
    high_conf_errors_mask = incorrect_mask & (max_proba >= CONFIDENCE_HIGH)
    high_conf_errors_count = int(high_conf_errors_mask.sum())
    print(f"  High-confidence errors (prob>={CONFIDENCE_HIGH}): {high_conf_errors_count}")

    return {
        "correct": {
            "predicted_class_prob": _prob_stats(correct_probs),
            "actual_class_prob": _prob_stats(correct_actual_probs),
        },
        "incorrect": {
            "predicted_class_prob": _prob_stats(incorrect_probs),
            "actual_class_prob": _prob_stats(incorrect_actual_probs),
        },
        "high_confidence_errors": high_conf_errors_count,
        "low_confidence_errors": int((incorrect_mask & (max_proba < CONFIDENCE_MED)).sum()),
    }


# Phase 3 — Class Distribution
def run_phase3_class_distribution(df: pd.DataFrame, y_test, per_cat_df: pd.DataFrame) -> dict:
    """Compile full class distribution across overall / train / val / test splits."""

    print("\n[Phase 3] Computing class distributions...")
    # Reproduce exact splits with same random_state=42
    X_pool, X_test_split, y_pool, y_test_split = train_test_split_data(
        df, test_size=0.20, random_state=42, stratify=True
    )
    df_pool = pd.DataFrame({"text": X_pool, "category": y_pool})
    X_tr, X_val, y_tr, y_val = train_test_split_data(
        df_pool, test_size=0.20, random_state=42, stratify=True
    )

    # Full dataset category distribution
    total_counts = df["category"].value_counts().to_dict()
    train_counts = y_tr.value_counts().to_dict()
    val_counts = y_val.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict() if hasattr(y_test, "value_counts") else Counter(y_test)

    # Build recall / F1 lookup from per_cat_df
    recall_lookup = dict(zip(per_cat_df["category"], per_cat_df["recall"]))
    f1_lookup = dict(zip(per_cat_df["category"], per_cat_df["f1"]))

    dist_rows = []
    for cat in sorted(total_counts.keys()):
        total = total_counts.get(cat, 0)
        dist_rows.append({
            "category": cat,
            "total_count": total,
            "total_pct": round(total / len(df) * 100, 2),
            "train_count": train_counts.get(cat, 0),
            "val_count": val_counts.get(cat, 0),
            "test_count": int(test_counts.get(cat, 0)),
            "test_recall": recall_lookup.get(cat, 0.0),
            "test_f1": f1_lookup.get(cat, 0.0),
        })
    dist_df = pd.DataFrame(dist_rows).sort_values("total_count", ascending=False).reset_index(drop=True)

    # Pearson correlation between test support and recall/F1
    test_supports = dist_df["test_count"].values.astype(float)
    recalls = dist_df["test_recall"].values.astype(float)
    f1s = dist_df["test_f1"].values.astype(float)

    corr_support_recall = float(np.corrcoef(test_supports, recalls)[0, 1])
    corr_support_f1 = float(np.corrcoef(test_supports, f1s)[0, 1])

    print(f"  Pearson corr (test support vs recall): {corr_support_recall:.3f}")
    print(f"  Pearson corr (test support vs F1):     {corr_support_f1:.3f}")

    return {
        "distribution": dist_df,
        "corr_support_recall": round(corr_support_recall, 4),
        "corr_support_f1": round(corr_support_f1, 4),
        "total_records": len(df),
        "train_records": len(y_tr),
        "val_records": len(y_val),
        "test_records": len(y_test_split),
    }


# Phase 4 — Lexical Overlap Analysis
def run_phase4_lexical_overlap(
    w_vec,
    top_pairs: list[dict],
    y_test,
    X_test_raw,
    n_terms: int = 20,
) -> dict:
    """Analyze shared TF-IDF vocabulary features between the top confusion pairs.

    Uses the fitted word TF-IDF vocabulary for lexical analysis.
    Does NOT perform semantic analysis.
    """

    print("\n[Phase 4] Lexical overlap analysis for top confusion pairs...")
    vocab = w_vec.vocabulary_
    feature_names = w_vec.get_feature_names_out()
    idf_values = w_vec.idf_

    # Build a mapping from category -> list of raw complaint texts
    y_true_arr = np.asarray(y_test)
    raw_texts = list(X_test_raw.reset_index(drop=True))
    cat_texts: dict[str, list[str]] = defaultdict(list)
    for i, (cat, text) in enumerate(zip(y_true_arr, raw_texts)):
        cat_texts[cat].append(str(text))

    results = {}
    for pair_row in top_pairs[:10]:
        actual_cat = pair_row["actual_category"]
        pred_cat = pair_row["predicted_category"]
        pair_key = f"{actual_cat} → {pred_cat}"

        # Collect texts for each category
        texts_actual = cat_texts.get(actual_cat, [])[:200]
        texts_pred = cat_texts.get(pred_cat, [])[:200]

        if not texts_actual or not texts_pred:
            continue

        # Get term-frequency word counts from raw text (lowercased, split)
        def top_tokens(texts, n=n_terms) -> list[tuple[str, int]]:
            counter: Counter = Counter()
            for t in texts:
                for tok in t.lower().split():
                    if len(tok) > 3:
                        counter[tok] += 1
            return counter.most_common(n)

        top_actual = set(tok for tok, _ in top_tokens(texts_actual))
        top_pred = set(tok for tok, _ in top_tokens(texts_pred))
        shared = sorted(top_actual & top_pred)

        results[pair_key] = {
            "actual_category": actual_cat,
            "predicted_category": pred_cat,
            "actual_sample_count": len(texts_actual),
            "predicted_category_sample_count": len(texts_pred),
            "shared_high_frequency_terms": shared[:30],
            "top_actual_terms": [tok for tok, _ in top_tokens(texts_actual)][:15],
            "top_predicted_terms": [tok for tok, _ in top_tokens(texts_pred)][:15],
        }
        print(f"  {actual_cat} -> {pred_cat}: {len(shared)} shared high-freq terms")

    return results


# Main Pipeline
def main():
    print("=" * 76)
    print("   CFPB COMPLAINT CLASSIFICATION — COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 76)

    # 1. Load full dataset and reproduce exact 5,000 test set
    data_path = ROOT_DIR / "data" / "complaints.csv"
    print(f"\nLoading CFPB dataset from {data_path}...")
    df = load_dataset(data_path, drop_invalid=True)
    print(f"  Total records: {len(df)}, Categories: {df['category'].nunique()}")

    # Reproduce exact test set with same split params as run_experiments.py
    _, X_test, _, y_test = train_test_split_data(
        df, test_size=0.20, random_state=42, stratify=True
    )
    print(f"  Test set reproduced: {len(X_test)} records.")
    assert len(X_test) == 5000, f"Expected 5000 test records, got {len(X_test)}"

    # 2. Load persisted improved model and vectorizers
    print("\nLoading persisted improved model and vectorizers...")
    clf = joblib.load(MODELS_DIR / "complaint_classifier.joblib")
    w_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    c_vec = joblib.load(MODELS_DIR / "char_vectorizer.joblib")
    print(f"  Classifier: {type(clf).__name__}")
    print(f"  Classes: {len(clf.classes_)}")
    print(f"  Word vocab size: {len(w_vec.vocabulary_):,}")
    print(f"  Char vocab size: {len(c_vec.vocabulary_):,}")

    # Transform test set
    print("\nTransforming test set...")
    clean_test = preprocess_series(X_test)
    X_test_vec = transform_word_char(w_vec, c_vec, clean_test)
    print(f"  Feature matrix shape: {X_test_vec.shape}")

    # Improved model metrics
    y_pred_improved = predict_categories(clf, X_test_vec)
    improved_metrics = _metrics_dict(y_test, y_pred_improved)
    print(f"\n  Improved Model: Accuracy={improved_metrics['accuracy']:.4f}  "
          f"Macro F1={improved_metrics['macro_f1']:.4f}  "
          f"Weighted F1={improved_metrics['weighted_f1']:.4f}")

    # 3. Phase 0: Baseline (skip if already generated)
    bvsi_cache = RESULTS_DIR / "baseline_vs_improved.csv"
    if bvsi_cache.exists():
        print("\n[Phase 0] Loading cached baseline_vs_improved.csv (skipping re-training)...")
        bvsi_cached_df = pd.read_csv(bvsi_cache)
        # Rebuild baseline metrics from cached CSV
        baseline_metrics = {}
        metric_key_map = {
            "Accuracy": "accuracy", "Weighted Precision": "weighted_precision",
            "Weighted Recall": "weighted_recall", "Weighted F1": "weighted_f1",
            "Macro Precision": "macro_precision", "Macro Recall": "macro_recall",
            "Macro F1": "macro_f1",
        }
        for _, row in bvsi_cached_df.iterrows():
            key = metric_key_map.get(row["metric"])
            if key:
                baseline_metrics[key] = float(row["baseline_word_tfidf_no_balancing"])
        print(f"  Baseline Accuracy={baseline_metrics['accuracy']:.4f}  "
              f"Macro F1={baseline_metrics['macro_f1']:.4f}  "
              f"Weighted F1={baseline_metrics['weighted_f1']:.4f}")
    else:
        baseline_metrics, _ = run_phase0_baseline(df, X_test, y_test)

    # Compute deltas
    baseline_vs_improved_rows = []
    metric_labels = [
        ("accuracy", "Accuracy"),
        ("weighted_precision", "Weighted Precision"),
        ("weighted_recall", "Weighted Recall"),
        ("weighted_f1", "Weighted F1"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
    ]
    for key, label in metric_labels:
        b_val = baseline_metrics[key]
        i_val = improved_metrics[key]
        abs_diff = round(i_val - b_val, 4)
        rel_change = round((i_val - b_val) / b_val * 100, 2) if b_val != 0 else 0.0
        baseline_vs_improved_rows.append({
            "metric": label,
            "baseline_word_tfidf_no_balancing": b_val,
            "improved_combined_tfidf_balanced": i_val,
            "absolute_difference_pp": abs_diff,
            "relative_change_pct": rel_change,
        })

    bvsi_df = pd.DataFrame(baseline_vs_improved_rows)
    bvsi_path = RESULTS_DIR / "baseline_vs_improved.csv"
    bvsi_df.to_csv(bvsi_path, index=False)
    print(f"\n  Saved: {bvsi_path}")

    # 4. Phase 1: Error analysis
    phase1 = run_phase1_error_analysis(
        clf, X_test_vec, y_test, X_test, n_error_examples=5
    )

    # Save results
    error_analysis_path = RESULTS_DIR / "error_analysis.csv"
    phase1["all_pairs"].to_csv(error_analysis_path, index=False)
    print(f"\n  Saved: {error_analysis_path}")

    per_cat_path = RESULTS_DIR / "per_category_metrics.csv"
    phase1["per_category"].to_csv(per_cat_path, index=False)
    print(f"  Saved: {per_cat_path}")

    # Print top 20 confusion pairs
    print("\n  Top 20 Confusion Pairs:")
    print(f"  {'Actual':<55} {'Predicted':<55} {'Count':>5} {'%Actual':>7}")
    for _, row in phase1["top20_pairs"].iterrows():
        print(f"  {row['actual_category']:<55} {row['predicted_category']:<55} "
              f"{row['error_count']:>5} {row['pct_of_actual']:>6.1f}%")

    # Print per-category metrics sorted by F1 ascending
    print("\n  Per-Category Metrics (worst F1 first):")
    worst = phase1["per_category"].sort_values("f1").head(10)
    for _, row in worst.iterrows():
        print(f"  {row['category']:<55} support={row['support']:>4}  "
              f"P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}  "
              f"primary_confusion->{row['primary_confusion_category']}")

    # 5. Phase 2: Confidence analysis
    phase2 = run_phase2_confidence(clf, X_test_vec, y_test, phase1["y_pred"])

    # 6. Phase 3: Class distribution
    phase3 = run_phase3_class_distribution(df, y_test, phase1["per_category"])

    dist_path = RESULTS_DIR / "class_distribution.csv"
    phase3["distribution"].to_csv(dist_path, index=False)
    print(f"\n  Saved: {dist_path}")

    # 7. Phase 4: Lexical overlap
    top_pairs_list = phase1["top20_pairs"].to_dict("records")
    phase4 = run_phase4_lexical_overlap(w_vec, top_pairs_list, y_test, X_test)

    # 8. Save comprehensive JSON for Streamlit and documentation
    json_data = {
        "analysis_metadata": {
            "total_records": 25000,
            "test_records": 5000,
            "training_pool": 20000,
            "train_subset": 16000,
            "val_subset": 4000,
            "n_categories": len(phase1["classes"]),
            "random_state": 42,
            "stratified": True,
        },
        "model_config": {
            "type": "LogisticRegression",
            "solver": "lbfgs",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "char_analyzer": "char_wb",
            "total_features": 237148,
        },
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "top20_confusion_pairs": phase1["top20_pairs"].to_dict("records"),
        "per_category_metrics": phase1["per_category"].to_dict("records"),
        "confidence_analysis": phase2,
        "class_distribution": {
            "total_records": phase3["total_records"],
            "train_records": phase3["train_records"],
            "val_records": phase3["val_records"],
            "test_records": phase3["test_records"],
            "corr_test_support_vs_recall": phase3["corr_support_recall"],
            "corr_test_support_vs_f1": phase3["corr_support_f1"],
            "by_category": phase3["distribution"].to_dict("records"),
        },
        "lexical_overlap": phase4,
        "error_examples": phase1["error_examples"],
    }

    json_path = RESULTS_DIR / "error_analysis_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    # 9. Summary printout
    print("\n" + "=" * 76)
    print("   SUMMARY")
    print("=" * 76)
    print(f"\n  TEST SET SIZE: {len(X_test):,} records")
    print(f"  CATEGORIES:    {len(phase1['classes'])}")
    print(f"\n  BASELINE  (Word TF-IDF, no class weighting): "
          f"Accuracy={baseline_metrics['accuracy']:.4f}  "
          f"Macro F1={baseline_metrics['macro_f1']:.4f}  "
          f"Weighted F1={baseline_metrics['weighted_f1']:.4f}")
    print(f"  IMPROVED  (Combined TF-IDF, balanced):        "
          f"Accuracy={improved_metrics['accuracy']:.4f}  "
          f"Macro F1={improved_metrics['macro_f1']:.4f}  "
          f"Weighted F1={improved_metrics['weighted_f1']:.4f}")

    print(f"\n  TOP 5 CONFUSION PAIRS:")
    for _, row in phase1["top20_pairs"].head(5).iterrows():
        print(f"    {row['actual_category']} -> {row['predicted_category']}: "
              f"{row['error_count']} errors ({row['pct_of_actual']}% of actual class)")

    if phase2:
        print(f"\n  CONFIDENCE ANALYSIS:")
        c = phase2["correct"]["predicted_class_prob"]
        w = phase2["incorrect"]["predicted_class_prob"]
        print(f"    Correct predictions:   mean prob={c['mean']:.3f}  "
              f"high={c['pct_high']:.1f}%  med={c['pct_medium']:.1f}%  low={c['pct_low']:.1f}%")
        print(f"    Incorrect predictions: mean prob={w['mean']:.3f}  "
              f"high={w['pct_high']:.1f}%  med={w['pct_medium']:.1f}%  low={w['pct_low']:.1f}%")
        print(f"    High-confidence errors (prob>={CONFIDENCE_HIGH}): "
              f"{phase2['high_confidence_errors']}")

    print(f"\n  CLASS DISTRIBUTION CORRELATIONS:")
    print(f"    Pearson(test_support, recall): {phase3['corr_support_recall']:.3f}")
    print(f"    Pearson(test_support, F1):     {phase3['corr_support_f1']:.3f}")

    print("\n  OUTPUT FILES:")
    for p in [bvsi_path, error_analysis_path, per_cat_path, dist_path, json_path]:
        print(f"    {p}")

    print("\n  Error analysis complete. Model was NOT modified.\n")
    return json_data


if __name__ == "__main__":
    main()
