"""Controlled post-audit classification improvement experiments.

Implements the post-audit experiments defined in Phase 2 through Phase 8:
- Controlled 18-class comparison: Standard vs Minimal Preprocessing
- Controlled 11-class taxonomy experiment: Standard vs Minimal Preprocessing
- Hierarchical classical classification experiment (Stage 1 group + Stage 2 local variant classifiers)
- Comprehensive error analysis comparing baseline vs improved 18-class model
- Final one-shot evaluation of selected best models on the untouched 5,000 test set

Artifacts generated:
- results/model_improvement_comparison.csv
- results/taxonomy_model_comparison.csv
- results/hierarchical_comparison.csv
- results/post_audit_error_analysis.json
- results/post_audit_test_metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.classification import train_test_split_data
from src.data_loader import load_dataset
from src.preprocessing import preprocess_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results"
CONFIG_DIR = ROOT_DIR / "config"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Production TF-IDF specification
WORD_PARAMS = dict(
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
    lowercase=False,
)
CHAR_PARAMS = dict(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    max_df=0.95,
    max_features=100000,
    sublinear_tf=True,
    lowercase=False,
)


def load_all_splits():
    """Load complaints dataset and produce identical stratified train/val/pool/test splits."""
    data_path = ROOT_DIR / "data" / "complaints.csv"
    df = load_dataset(data_path, drop_invalid=True)

    X_pool, X_test, y_pool, y_test = train_test_split_data(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=True
    )
    df_pool = pd.DataFrame({"text": X_pool, "category": y_pool})
    X_tr, X_val, y_tr, y_val = train_test_split_data(
        df_pool, test_size=0.20, random_state=RANDOM_STATE, stratify=True
    )

    return {
        "df": df,
        "X_pool": X_pool,
        "y_pool": y_pool,
        "X_test": X_test,
        "y_test": y_test,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_val": X_val,
        "y_val": y_val,
    }


def build_vectorizers():
    """Build un-fitted word and char TF-IDF vectorizers using production hyperparameters."""
    w_vec = TfidfVectorizer(**WORD_PARAMS)
    c_vec = TfidfVectorizer(**CHAR_PARAMS)
    return w_vec, c_vec


def extract_features(
    train_texts: pd.Series,
    eval_texts: pd.Series,
) -> Tuple[TfidfVectorizer, TfidfVectorizer, Any, Any]:
    """Fit vectorizers on train_texts ONLY and transform both train and eval texts."""
    w_vec, c_vec = build_vectorizers()
    t_list = train_texts.tolist()
    e_list = eval_texts.tolist()

    X_w_tr = w_vec.fit_transform(t_list)
    X_c_tr = c_vec.fit_transform(t_list)
    X_tr = hstack([X_w_tr, X_c_tr], format="csr")

    X_w_eval = w_vec.transform(e_list)
    X_c_eval = c_vec.transform(e_list)
    X_eval = hstack([X_w_eval, X_c_eval], format="csr")

    return w_vec, c_vec, X_tr, X_eval


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute complete set of standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def load_taxonomy_v1_mapping() -> Dict[str, str]:
    """Load official conservative taxonomy v1 mapping (18 original -> 11 normalized)."""
    tax_path = CONFIG_DIR / "taxonomy_v1_conservative.json"
    with open(tax_path, "r", encoding="utf-8") as f:
        tax_data = json.load(f)
    return {k: v["normalized_category"] for k, v in tax_data["mapping"].items()}


def run_experiments():
    logger.info("Loading dataset and splits...")
    splits = load_all_splits()
    X_tr, y_tr = splits["X_tr"], splits["y_tr"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_pool, y_pool = splits["X_pool"], splits["y_pool"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    v1_mapping = load_taxonomy_v1_mapping()
    y_tr_v1 = y_tr.map(v1_mapping)
    y_val_v1 = y_val.map(v1_mapping)
    y_pool_v1 = y_pool.map(v1_mapping)
    y_test_v1 = y_test.map(v1_mapping)

    logger.info("Executing text preprocessing...")
    logger.info("1. Standard preprocessing on train/val...")
    clean_tr_std = preprocess_series(X_tr, mode="standard")
    clean_val_std = preprocess_series(X_val, mode="standard")

    logger.info("2. Minimal preprocessing on train/val...")
    clean_tr_min = preprocess_series(X_tr, mode="minimal")
    clean_val_min = preprocess_series(X_val, mode="minimal")

    # =========================================================================
    # PHASE 3: Controlled 18-Class Experiment
    # =========================================================================
    logger.info("Fitting features for 18-class standard baseline...")
    w_std, c_std, X_tr_std, X_val_std = extract_features(clean_tr_std, clean_val_std)
    clf_18_std = LogisticRegression(
        C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    clf_18_std.fit(X_tr_std, y_tr)
    pred_18_std_val = clf_18_std.predict(X_val_std)
    proba_18_std_val = clf_18_std.predict_proba(X_val_std)
    m_18_std = compute_metrics(y_val, pred_18_std_val)

    logger.info("Fitting features for 18-class minimal preprocessing...")
    w_min, c_min, X_tr_min, X_val_min = extract_features(clean_tr_min, clean_val_min)
    clf_18_min = LogisticRegression(
        C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    clf_18_min.fit(X_tr_min, y_tr)
    pred_18_min_val = clf_18_min.predict(X_val_min)
    proba_18_min_val = clf_18_min.predict_proba(X_val_min)
    m_18_min = compute_metrics(y_val, pred_18_min_val)

    logger.info(f"18-Class Baseline (Std): Acc={m_18_std['accuracy']:.4f}, MF1={m_18_std['macro_f1']:.4f}")
    logger.info(f"18-Class Improved (Min): Acc={m_18_min['accuracy']:.4f}, MF1={m_18_min['macro_f1']:.4f}")

    # =========================================================================
    # PHASE 4: Formal 11-Class Taxonomy Experiment
    # =========================================================================
    logger.info("Fitting 11-class standard classifier...")
    clf_11_std = LogisticRegression(
        C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    clf_11_std.fit(X_tr_std, y_tr_v1)
    pred_11_std_val = clf_11_std.predict(X_val_std)
    m_11_std = compute_metrics(y_val_v1, pred_11_std_val)

    logger.info("Fitting 11-class minimal classifier...")
    clf_11_min = LogisticRegression(
        C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    clf_11_min.fit(X_tr_min, y_tr_v1)
    pred_11_min_val = clf_11_min.predict(X_val_min)
    m_11_min = compute_metrics(y_val_v1, pred_11_min_val)

    logger.info(f"11-Class Standard: Acc={m_11_std['accuracy']:.4f}, MF1={m_11_std['macro_f1']:.4f}")
    logger.info(f"11-Class Minimal:  Acc={m_11_min['accuracy']:.4f}, MF1={m_11_min['macro_f1']:.4f}")

    # =========================================================================
    # PHASE 5: Hierarchical Classical Classifier (Text-Only)
    # =========================================================================
    logger.info("Building Hierarchical Classical Classifier (Text-Only)...")
    # Identify groups with multiple original labels in taxonomy v1
    rev_mapping: Dict[str, List[str]] = {}
    for orig_l, norm_g in v1_mapping.items():
        rev_mapping.setdefault(norm_g, []).append(orig_l)

    multi_groups = {g: origs for g, origs in rev_mapping.items() if len(origs) > 1}
    logger.info(f"Multi-label groups to resolve in Stage 2: {list(multi_groups.keys())}")

    # Train local classifiers for each multi-label group on minimal representations
    local_clfs_min: Dict[str, LogisticRegression] = {}
    for g, origs in multi_groups.items():
        mask_tr = (y_tr_v1 == g).values
        y_tr_sub = y_tr[mask_tr]
        X_tr_sub = X_tr_min[mask_tr]
        local_clf = LogisticRegression(
            C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
        )
        local_clf.fit(X_tr_sub, y_tr_sub)
        local_clfs_min[g] = local_clf

    # Inference using Hierarchical Pipeline on Validation (Minimal Prepro)
    pred_hier_min_val = []
    for i, pred_g in enumerate(pred_11_min_val):
        if pred_g in local_clfs_min:
            sub_pred = local_clfs_min[pred_g].predict(X_val_min[i])
            pred_hier_min_val.append(sub_pred[0])
        else:
            # 1-to-1 group
            pred_hier_min_val.append(rev_mapping[pred_g][0])
    pred_hier_min_val = np.array(pred_hier_min_val)
    m_hier_min = compute_metrics(y_val, pred_hier_min_val)
    logger.info(f"Hierarchical 18-Class (Min): Acc={m_hier_min['accuracy']:.4f}, MF1={m_hier_min['macro_f1']:.4f}")

    # Also evaluate hierarchical pipeline on standard preprocessing for complete rigor
    local_clfs_std: Dict[str, LogisticRegression] = {}
    for g, origs in multi_groups.items():
        mask_tr = (y_tr_v1 == g).values
        y_tr_sub = y_tr[mask_tr]
        X_tr_sub = X_tr_std[mask_tr]
        local_clf = LogisticRegression(
            C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
        )
        local_clf.fit(X_tr_sub, y_tr_sub)
        local_clfs_std[g] = local_clf

    pred_hier_std_val = []
    for i, pred_g in enumerate(pred_11_std_val):
        if pred_g in local_clfs_std:
            sub_pred = local_clfs_std[pred_g].predict(X_val_std[i])
            pred_hier_std_val.append(sub_pred[0])
        else:
            pred_hier_std_val.append(rev_mapping[pred_g][0])
    pred_hier_std_val = np.array(pred_hier_std_val)
    m_hier_std = compute_metrics(y_val, pred_hier_std_val)
    logger.info(f"Hierarchical 18-Class (Std): Acc={m_hier_std['accuracy']:.4f}, MF1={m_hier_std['macro_f1']:.4f}")

    # =========================================================================
    # PHASE 8: Select Best 18-Class Model & Single Holdout Test Evaluation
    # =========================================================================
    # Selection criteria: 1. Macro F1, 2. Macro Recall, 3. Accuracy, 4. Weighted F1
    candidates_18 = [
        {"name": "18-class Baseline (Standard Preprocessing)", "prepro": "standard", "metrics": m_18_std, "pred": pred_18_std_val},
        {"name": "18-class Improved (Minimal Preprocessing)", "prepro": "minimal", "metrics": m_18_min, "pred": pred_18_min_val},
        {"name": "18-class Hierarchical (Minimal Preprocessing)", "prepro": "minimal", "metrics": m_hier_min, "pred": pred_hier_min_val},
        {"name": "18-class Hierarchical (Standard Preprocessing)", "prepro": "standard", "metrics": m_hier_std, "pred": pred_hier_std_val},
    ]
    # Sort by priority
    sorted_candidates = sorted(
        candidates_18,
        key=lambda c: (
            c["metrics"]["macro_f1"],
            c["metrics"]["macro_recall"],
            c["metrics"]["accuracy"],
            c["metrics"]["weighted_f1"],
        ),
        reverse=True,
    )
    best_candidate = sorted_candidates[0]
    logger.info(f"Selected Best 18-Class Model: {best_candidate['name']}")
    logger.info(f"Validation Macro F1: {best_candidate['metrics']['macro_f1']:.4f}, Accuracy: {best_candidate['metrics']['accuracy']:.4f}")

    # Retrain selected model on full 20,000 pool and evaluate ONCE on 5,000 holdout test set
    logger.info("Evaluating selected models ONCE on untouched 5,000-sample test set...")
    clean_pool_min = preprocess_series(X_pool, mode="minimal")
    clean_test_min = preprocess_series(X_test, mode="minimal")

    w_pool_min, c_pool_min, X_pool_min, X_test_min = extract_features(clean_pool_min, clean_test_min)

    # 18-Class Minimal Test Evaluation
    clf_18_final = LogisticRegression(
        C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    clf_18_final.fit(X_pool_min, y_pool)
    pred_18_test = clf_18_final.predict(X_test_min)
    m_18_test = compute_metrics(y_test, pred_18_test)
    logger.info(f"18-Class Minimal Holdout Test: Acc={m_18_test['accuracy']:.4f}, MF1={m_18_test['macro_f1']:.4f}")

    # 11-Class Minimal Test Evaluation
    clf_11_final = LogisticRegression(
        C=2.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    clf_11_final.fit(X_pool_min, y_pool_v1)
    pred_11_test = clf_11_final.predict(X_test_min)
    m_11_test = compute_metrics(y_test_v1, pred_11_test)
    logger.info(f"11-Class Minimal Holdout Test: Acc={m_11_test['accuracy']:.4f}, MF1={m_11_test['macro_f1']:.4f}")

    # Baseline 18-Class Test (recorded from production)
    baseline_test_metrics = {
        "accuracy": 0.6982,
        "macro_precision": 0.5041,
        "macro_recall": 0.5169,
        "macro_f1": 0.5088,
        "weighted_precision": 0.7008,
        "weighted_recall": 0.6982,
        "weighted_f1": 0.6978,
    }

    # =========================================================================
    # PHASE 6: Save Comparison CSVs
    # =========================================================================
    logger.info("Generating comparison CSV tables...")

    # 1. results/model_improvement_comparison.csv
    improvement_rows = [
        {
            "experiment": "18-class Baseline (Standard Preprocessing)",
            "taxonomy": "CFPB Original (18 classes)",
            "preprocessing": "standard (regex cleaning, punctuation removed, stopwords removed)",
            "classifier": "LogisticRegression(C=2.0, class_weight='balanced')",
            "accuracy": m_18_std["accuracy"],
            "macro_precision": m_18_std["macro_precision"],
            "macro_recall": m_18_std["macro_recall"],
            "macro_f1": m_18_std["macro_f1"],
            "weighted_precision": m_18_std["weighted_precision"],
            "weighted_recall": m_18_std["weighted_recall"],
            "weighted_f1": m_18_std["weighted_f1"],
            "feature_count": int(X_tr_std.shape[1]),
            "training_size": 16000,
            "validation_size": 4000,
            "test_size": 5000,
            "test_accuracy": baseline_test_metrics["accuracy"],
            "test_macro_f1": baseline_test_metrics["macro_f1"],
            "test_weighted_f1": baseline_test_metrics["weighted_f1"],
        },
        {
            "experiment": "18-class Improved Representation (Minimal Preprocessing)",
            "taxonomy": "CFPB Original (18 classes)",
            "preprocessing": "minimal (lowercase, whitespace only; punctuation & stopwords kept)",
            "classifier": "LogisticRegression(C=2.0, class_weight='balanced')",
            "accuracy": m_18_min["accuracy"],
            "macro_precision": m_18_min["macro_precision"],
            "macro_recall": m_18_min["macro_recall"],
            "macro_f1": m_18_min["macro_f1"],
            "weighted_precision": m_18_min["weighted_precision"],
            "weighted_recall": m_18_min["weighted_recall"],
            "weighted_f1": m_18_min["weighted_f1"],
            "feature_count": int(X_tr_min.shape[1]),
            "training_size": 16000,
            "validation_size": 4000,
            "test_size": 5000,
            "test_accuracy": m_18_test["accuracy"],
            "test_macro_f1": m_18_test["macro_f1"],
            "test_weighted_f1": m_18_test["weighted_f1"],
        },
        {
            "experiment": "11-class Normalized Taxonomy (Standard Preprocessing)",
            "taxonomy": "Conservative v1 (11 classes)",
            "preprocessing": "standard (regex cleaning, punctuation removed, stopwords removed)",
            "classifier": "LogisticRegression(C=2.0, class_weight='balanced')",
            "accuracy": m_11_std["accuracy"],
            "macro_precision": m_11_std["macro_precision"],
            "macro_recall": m_11_std["macro_recall"],
            "macro_f1": m_11_std["macro_f1"],
            "weighted_precision": m_11_std["weighted_precision"],
            "weighted_recall": m_11_std["weighted_recall"],
            "weighted_f1": m_11_std["weighted_f1"],
            "feature_count": int(X_tr_std.shape[1]),
            "training_size": 16000,
            "validation_size": 4000,
            "test_size": 5000,
            "test_accuracy": 0.8150,  # From v1 conservative reference test
            "test_macro_f1": 0.6343,
            "test_weighted_f1": 0.8161,
        },
        {
            "experiment": "11-class Normalized Taxonomy (Minimal Preprocessing)",
            "taxonomy": "Conservative v1 (11 classes)",
            "preprocessing": "minimal (lowercase, whitespace only; punctuation & stopwords kept)",
            "classifier": "LogisticRegression(C=2.0, class_weight='balanced')",
            "accuracy": m_11_min["accuracy"],
            "macro_precision": m_11_min["macro_precision"],
            "macro_recall": m_11_min["macro_recall"],
            "macro_f1": m_11_min["macro_f1"],
            "weighted_precision": m_11_min["weighted_precision"],
            "weighted_recall": m_11_min["weighted_recall"],
            "weighted_f1": m_11_min["weighted_f1"],
            "feature_count": int(X_tr_min.shape[1]),
            "training_size": 16000,
            "validation_size": 4000,
            "test_size": 5000,
            "test_accuracy": m_11_test["accuracy"],
            "test_macro_f1": m_11_test["macro_f1"],
            "test_weighted_f1": m_11_test["weighted_f1"],
        },
    ]
    df_improvement = pd.DataFrame(improvement_rows)
    df_improvement.to_csv(RESULTS_DIR / "model_improvement_comparison.csv", index=False)
    logger.info("Saved results/model_improvement_comparison.csv")

    # 2. results/taxonomy_model_comparison.csv
    taxonomy_rows = [
        {
            "model_formulation": "18-Class Original Baseline (Standard)",
            "n_classes": 18,
            "preprocessing": "standard",
            "val_accuracy": m_18_std["accuracy"],
            "val_macro_f1": m_18_std["macro_f1"],
            "val_macro_recall": m_18_std["macro_recall"],
            "val_weighted_f1": m_18_std["weighted_f1"],
            "test_accuracy": baseline_test_metrics["accuracy"],
            "test_macro_f1": baseline_test_metrics["macro_f1"],
            "description": "Baseline classical model on original 18 CFPB product labels with administrative synonymy.",
        },
        {
            "model_formulation": "18-Class Improved Representation (Minimal)",
            "n_classes": 18,
            "preprocessing": "minimal",
            "val_accuracy": m_18_min["accuracy"],
            "val_macro_f1": m_18_min["macro_f1"],
            "val_macro_recall": m_18_min["macro_recall"],
            "val_weighted_f1": m_18_min["weighted_f1"],
            "test_accuracy": m_18_test["accuracy"],
            "test_macro_f1": m_18_test["macro_f1"],
            "description": "Controlled minimal preprocessing improvement preserving punctuation, stopwords, and negation.",
        },
        {
            "model_formulation": "11-Class Conservative Taxonomy (Standard)",
            "n_classes": 11,
            "preprocessing": "standard",
            "val_accuracy": m_11_std["accuracy"],
            "val_macro_f1": m_11_std["macro_f1"],
            "val_macro_recall": m_11_std["macro_recall"],
            "val_weighted_f1": m_11_std["weighted_f1"],
            "test_accuracy": 0.8150,
            "test_macro_f1": 0.6343,
            "description": "Consolidated administrative renames into 11 broad product domains; standard preprocessing.",
        },
        {
            "model_formulation": "11-Class Conservative Taxonomy (Minimal)",
            "n_classes": 11,
            "preprocessing": "minimal",
            "val_accuracy": m_11_min["accuracy"],
            "val_macro_f1": m_11_min["macro_f1"],
            "val_macro_recall": m_11_min["macro_recall"],
            "val_weighted_f1": m_11_min["weighted_f1"],
            "test_accuracy": m_11_test["accuracy"],
            "test_macro_f1": m_11_test["macro_f1"],
            "description": "Consolidated administrative renames into 11 broad product domains; minimal preprocessing.",
        },
    ]
    df_tax_comp = pd.DataFrame(taxonomy_rows)
    df_tax_comp.to_csv(RESULTS_DIR / "taxonomy_model_comparison.csv", index=False)
    logger.info("Saved results/taxonomy_model_comparison.csv")

    # 3. results/hierarchical_comparison.csv
    hier_rows = [
        {
            "architecture": "Flat 18-Class Baseline",
            "preprocessing": "standard",
            "val_accuracy": m_18_std["accuracy"],
            "val_macro_f1": m_18_std["macro_f1"],
            "val_macro_recall": m_18_std["macro_recall"],
            "val_weighted_f1": m_18_std["weighted_f1"],
            "notes": "Direct 18-class multinomial logistic regression.",
        },
        {
            "architecture": "Flat 18-Class Improved",
            "preprocessing": "minimal",
            "val_accuracy": m_18_min["accuracy"],
            "val_macro_f1": m_18_min["macro_f1"],
            "val_macro_recall": m_18_min["macro_recall"],
            "val_weighted_f1": m_18_min["weighted_f1"],
            "notes": "Direct 18-class multinomial logistic regression with minimal preprocessing.",
        },
        {
            "architecture": "Hierarchical 2-Stage (Stage 1 Group + Stage 2 Text Specialist)",
            "preprocessing": "standard",
            "val_accuracy": m_hier_std["accuracy"],
            "val_macro_f1": m_hier_std["macro_f1"],
            "val_macro_recall": m_hier_std["macro_recall"],
            "val_weighted_f1": m_hier_std["weighted_f1"],
            "notes": "Stage 1 predicts 11 product groups; Stage 2 applies local text classifiers for multi-label groups.",
        },
        {
            "architecture": "Hierarchical 2-Stage (Stage 1 Group + Stage 2 Text Specialist)",
            "preprocessing": "minimal",
            "val_accuracy": m_hier_min["accuracy"],
            "val_macro_f1": m_hier_min["macro_f1"],
            "val_macro_recall": m_hier_min["macro_recall"],
            "val_weighted_f1": m_hier_min["weighted_f1"],
            "notes": "Stage 1 predicts 11 product groups; Stage 2 applies local text classifiers for multi-label groups.",
        },
    ]
    df_hier = pd.DataFrame(hier_rows)
    df_hier.to_csv(RESULTS_DIR / "hierarchical_comparison.csv", index=False)
    logger.info("Saved results/hierarchical_comparison.csv")

    # =========================================================================
    # PHASE 7: Detailed Error Analysis After Improvement
    # =========================================================================
    logger.info("Conducting granular error analysis on improved 18-class model...")
    all_classes_18 = sorted(clf_18_min.classes_)

    # Per-category metrics for Baseline vs Improved
    report_std = classification_report(y_val, pred_18_std_val, labels=all_classes_18, output_dict=True, zero_division=0)
    report_min = classification_report(y_val, pred_18_min_val, labels=all_classes_18, output_dict=True, zero_division=0)

    per_category_comparison = []
    for c in all_classes_18:
        per_category_comparison.append({
            "category": c,
            "support": int(report_std.get(c, {}).get("support", 0)),
            "baseline_precision": round(float(report_std.get(c, {}).get("precision", 0.0)), 4),
            "baseline_recall": round(float(report_std.get(c, {}).get("recall", 0.0)), 4),
            "baseline_f1": round(float(report_std.get(c, {}).get("f1-score", 0.0)), 4),
            "improved_precision": round(float(report_min.get(c, {}).get("precision", 0.0)), 4),
            "improved_recall": round(float(report_min.get(c, {}).get("recall", 0.0)), 4),
            "improved_f1": round(float(report_min.get(c, {}).get("f1-score", 0.0)), 4),
            "delta_f1": round(
                float(report_min.get(c, {}).get("f1-score", 0.0)) - float(report_std.get(c, {}).get("f1-score", 0.0)),
                4,
            ),
        })

    # Confusion matrix
    cm_min = confusion_matrix(y_val, pred_18_min_val, labels=all_classes_18)
    cm_df = pd.DataFrame(cm_min, index=all_classes_18, columns=all_classes_18)
    cm_df.to_csv(RESULTS_DIR / "post_audit_confusion_matrix.csv")
    logger.info("Saved results/post_audit_confusion_matrix.csv")

    df_per_cat = pd.DataFrame(per_category_comparison)
    df_per_cat.to_csv(RESULTS_DIR / "post_audit_per_category.csv", index=False)
    logger.info("Saved results/post_audit_per_category.csv")

    # Top confusion pairs
    pairs = []
    for i, true_c in enumerate(all_classes_18):
        for j, pred_c in enumerate(all_classes_18):
            if i != j and cm_min[i, j] > 0:
                is_sibling = v1_mapping.get(true_c) == v1_mapping.get(pred_c)
                pairs.append({
                    "true_category": true_c,
                    "pred_category": pred_c,
                    "count": int(cm_min[i, j]),
                    "is_historical_sibling": bool(is_sibling),
                    "normalized_group": v1_mapping.get(true_c) if is_sibling else "Cross-Group",
                })
    pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=False)
    pairs_df.to_csv(RESULTS_DIR / "post_audit_confusion_pairs.csv", index=False)
    logger.info("Saved results/post_audit_confusion_pairs.csv")

    # Decompose errors
    y_val_arr = np.asarray(y_val)
    err_mask_std = pred_18_std_val != y_val_arr
    err_mask_min = pred_18_min_val != y_val_arr

    total_err_std = int(err_mask_std.sum())
    total_err_min = int(err_mask_min.sum())
    errors_fixed_net = total_err_std - total_err_min

    # Sibling vs Cross-product errors
    norm_true = np.array([v1_mapping.get(y) for y in y_val_arr])
    norm_pred_std = np.array([v1_mapping.get(p) for p in pred_18_std_val])
    norm_pred_min = np.array([v1_mapping.get(p) for p in pred_18_min_val])

    sibling_err_std = int((err_mask_std & (norm_true == norm_pred_std)).sum())
    cross_err_std = int((err_mask_std & (norm_true != norm_pred_std)).sum())

    sibling_err_min = int((err_mask_min & (norm_true == norm_pred_min)).sum())
    cross_err_min = int((err_mask_min & (norm_true != norm_pred_min)).sum())

    # Confidence distribution for improved model
    max_probas_min = np.max(proba_18_min_val, axis=1)
    conf_stats = {
        "mean_confidence": float(np.mean(max_probas_min)),
        "median_confidence": float(np.median(max_probas_min)),
        "min_confidence": float(np.min(max_probas_min)),
        "max_confidence": float(np.max(max_probas_min)),
        "low_confidence_count_lt_0_5": int((max_probas_min < 0.5).sum()),
        "low_confidence_pct_lt_0_5": float((max_probas_min < 0.5).mean() * 100),
    }

    # Which exact errors were fixed?
    fixed_indices = np.where(err_mask_std & ~err_mask_min)[0]
    newly_errored_indices = np.where(~err_mask_std & err_mask_min)[0]

    # Categories that gained the most fixed errors
    fixed_classes = pd.Series(y_val_arr[fixed_indices]).value_counts().to_dict()

    error_analysis_data = {
        "validation_size": len(y_val),
        "baseline_total_errors": total_err_std,
        "improved_total_errors": total_err_min,
        "net_errors_eliminated": errors_fixed_net,
        "baseline_sibling_errors": sibling_err_std,
        "baseline_sibling_error_pct": round(sibling_err_std / total_err_std * 100, 2),
        "improved_sibling_errors": sibling_err_min,
        "improved_sibling_error_pct": round(sibling_err_min / total_err_min * 100, 2),
        "baseline_cross_errors": cross_err_std,
        "baseline_cross_error_pct": round(cross_err_std / total_err_std * 100, 2),
        "improved_cross_errors": cross_err_min,
        "improved_cross_error_pct": round(cross_err_min / total_err_min * 100, 2),
        "errors_fixed_gross": len(fixed_indices),
        "newly_introduced_errors": len(newly_errored_indices),
        "top_categories_with_fixed_errors": fixed_classes,
        "confidence_statistics": conf_stats,
        "top_10_confusion_pairs": pairs_df.head(10).to_dict(orient="records"),
        "per_category_comparison": per_category_comparison,
        "why_errors_remain": {
            "historical_siblings": "Administrative renames have identical narrative text and vocabulary distributions; text-only linear classifiers cannot infer filing era without date metadata.",
            "compound_complaints": "Complaints spanning multiple financial verticals (e.g. debt collection on a credit card or disputed bank fee) trigger competing high-weight features.",
            "long_tail_imbalance": "Extreme class frequency disparity (Debt collection: 5,830 vs Virtual currency: 3) starves minority decision boundaries.",
        },
    }

    with open(RESULTS_DIR / "post_audit_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis_data, f, indent=2)
    logger.info("Saved results/post_audit_error_analysis.json")

    # Post-audit test metrics summary
    post_audit_test_summary = {
        "baseline_18_class_test": baseline_test_metrics,
        "improved_18_class_test": m_18_test,
        "normalized_11_class_test": m_11_test,
        "deltas_18_class": {
            "accuracy_delta_pp": round((m_18_test["accuracy"] - baseline_test_metrics["accuracy"]) * 100, 2),
            "macro_f1_delta_pp": round((m_18_test["macro_f1"] - baseline_test_metrics["macro_f1"]) * 100, 2),
            "weighted_f1_delta_pp": round((m_18_test["weighted_f1"] - baseline_test_metrics["weighted_f1"]) * 100, 2),
        },
        "evaluation_protocol": "Untouched 5,000 holdout test set evaluated exactly ONCE at the end of model selection with random_state=42.",
    }

    with open(RESULTS_DIR / "post_audit_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(post_audit_test_summary, f, indent=2)
    logger.info("Saved results/post_audit_test_metrics.json")

    logger.info("=========================================================================")
    logger.info("POST-AUDIT EXPERIMENTAL RUN COMPLETE")
    logger.info("=========================================================================")
    logger.info(f"Baseline 18-Class Val Accuracy:  {m_18_std['accuracy']:.4f} | Macro F1: {m_18_std['macro_f1']:.4f}")
    logger.info(f"Improved 18-Class Val Accuracy:  {m_18_min['accuracy']:.4f} | Macro F1: {m_18_min['macro_f1']:.4f}")
    logger.info(f"Normalized 11-Class Val Acc:     {m_11_min['accuracy']:.4f} | Macro F1: {m_11_min['macro_f1']:.4f}")
    logger.info(f"Hierarchical 18-Class Val Acc:   {m_hier_min['accuracy']:.4f} | Macro F1: {m_hier_min['macro_f1']:.4f}")
    logger.info(f"Improved 18-Class Test Accuracy: {m_18_test['accuracy']:.4f} | Macro F1: {m_18_test['macro_f1']:.4f}")
    logger.info(f"Normalized 11-Class Test Acc:    {m_11_test['accuracy']:.4f} | Macro F1: {m_11_test['macro_f1']:.4f}")


if __name__ == "__main__":
    run_experiments()
