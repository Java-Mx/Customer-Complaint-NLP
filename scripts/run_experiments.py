"""Systematic classical NLP model improvement experiment runner.

Executes controlled validation experiments across multiple TF-IDF feature configurations,
class weighting strategies, and classical classifiers (Logistic Regression and LinearSVC)
on the official CFPB complaint dataset (25,000 records).

Workflow:
1. Stratified 80/20 partition: Training Pool (20,000) vs. Untouched Final Test Set (5,000).
2. Internal 80/20 split of Training Pool: Train Subset (16,000) vs. Validation Subset (4,000).
3. Compute sparse feature matrices ONCE per feature configuration and reuse across models.
4. Log validation metrics to results/model_comparison.csv.
5. Select best model by Validation Macro F1.
6. Retrain best model on the entire 20,000 Training Pool and evaluate ONCE on the Final Test Set.
7. Persist trained artifacts to models/ and save confusion matrix to results/confusion_matrix.png.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_loader import load_dataset
from src.preprocessing import preprocess_series
from src.vectorization import (
    combine_sparse_matrices,
    create_char_vectorizer,
    create_vectorizer,
    fit_transform_tfidf,
    transform_tfidf,
)
from src.classification import (
    create_classifier,
    create_linear_svc,
    fit_classifier,
    predict_categories,
    save_classifier,
    train_test_split_data,
)
from src.evaluation import (
    compute_per_category_metrics,
    evaluate_model,
    generate_classification_report,
    plot_confusion_matrix,
)


def run_full_experiment_pipeline():
    """Execute complete systematic experimentation and model selection."""
    print("=" * 78)
    print("   CFPB COMPLAINT CATEGORISATION: SYSTEMATIC MODEL IMPROVEMENT PIPELINE")
    print("=" * 78)

    # 1. Dataset Loading & Partitioning
    data_path = ROOT_DIR / "data" / "complaints.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"CFPB complaints dataset not found at {data_path}")

    t_start = time.time()
    print(f"\n[1/6] Loading CFPB complaint records from {data_path}...")
    df = load_dataset(data_path, drop_invalid=True)
    total_records = len(df)
    n_categories = df["category"].nunique()
    print(f"      Total usable records: {total_records:,} across {n_categories} categories.")

    print("\n[2/6] Performing stratified partitioning (Zero Data Leakage)...")
    # 80% Training Pool, 20% Untouched Final Test Set
    X_pool, X_test, y_pool, y_test = train_test_split_data(
        df, test_size=0.20, random_state=42, stratify=True
    )
    # Training Pool split: 80% Train Subset, 20% Validation Subset
    df_pool = pd.DataFrame({"text": X_pool, "category": y_pool})
    X_tr, X_val, y_tr, y_val = train_test_split_data(
        df_pool, test_size=0.20, random_state=42, stratify=True
    )

    print(f"      Full Dataset:        {total_records:,} records")
    print(f"      Training Pool:       {len(X_pool):,} records (80.0%)")
    print(f"      Untouched Test Set:  {len(X_test):,} records (20.0%)")
    print(f"      - Training Subset:   {len(X_tr):,} records (64.0% of total)")
    print(f"      - Validation Subset: {len(X_val):,} records (16.0% of total)")

    # 2. Text Preprocessing (Computed ONCE)
    print("\n[3/6] Preprocessing complaint text narratives...")
    t_prep = time.time()
    clean_tr = preprocess_series(X_tr)
    clean_val = preprocess_series(X_val)
    clean_pool = preprocess_series(X_pool)
    clean_test = preprocess_series(X_test)
    print(f"      Completed preprocessing in {time.time() - t_prep:.2f}s.")

    # 3. Feature Matrix Construction & Reuse
    print("\n[4/6] Building sparse TF-IDF feature representations...")
    feature_configs = {}

    # Config A: Word (1, 2), min_df=2
    t_feat = time.time()
    print("      -> Fitting Word TF-IDF Config A (ngram=(1,2), min_df=2)...")
    vec_word_a = create_vectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    vec_word_a, X_tr_wa = fit_transform_tfidf(vec_word_a, clean_tr)
    X_val_wa = transform_tfidf(vec_word_a, clean_val)
    feature_configs["Word (1,2) min_df=2"] = {
        "X_train": X_tr_wa, "X_val": X_val_wa, "vectorizer": vec_word_a, "type": "single"
    }

    # Config B: Word (1, 2), min_df=3
    print("      -> Fitting Word TF-IDF Config B (ngram=(1,2), min_df=3)...")
    vec_word_b = create_vectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    vec_word_b, X_tr_wb = fit_transform_tfidf(vec_word_b, clean_tr)
    X_val_wb = transform_tfidf(vec_word_b, clean_val)
    feature_configs["Word (1,2) min_df=3"] = {
        "X_train": X_tr_wb, "X_val": X_val_wb, "vectorizer": vec_word_b, "type": "single"
    }

    # Config C: Word (1, 3), min_df=2
    print("      -> Fitting Word TF-IDF Config C (ngram=(1,3), min_df=2)...")
    vec_word_c = create_vectorizer(ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True)
    vec_word_c, X_tr_wc = fit_transform_tfidf(vec_word_c, clean_tr)
    X_val_wc = transform_tfidf(vec_word_c, clean_val)
    feature_configs["Word (1,3) min_df=2"] = {
        "X_train": X_tr_wc, "X_val": X_val_wc, "vectorizer": vec_word_c, "type": "single"
    }

    # Config D: Character TF-IDF (char_wb, 3-5)
    print("      -> Fitting Char TF-IDF (analyzer=char_wb, ngram=(3,5), min_df=5)...")
    vec_char = create_char_vectorizer(ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True)
    vec_char, X_tr_char = fit_transform_tfidf(vec_char, clean_tr)
    X_val_char = transform_tfidf(vec_char, clean_val)
    feature_configs["Char (3,5) min_df=5"] = {
        "X_train": X_tr_char, "X_val": X_val_char, "vectorizer": vec_char, "type": "single"
    }

    # Config E: Combined Word A + Char
    print("      -> Combining Word Config A + Char TF-IDF via sparse hstack...")
    X_tr_comb = combine_sparse_matrices(X_tr_wa, X_tr_char)
    X_val_comb = combine_sparse_matrices(X_val_wa, X_val_char)
    feature_configs["Combined Word(1,2)+Char(3,5)"] = {
        "X_train": X_tr_comb,
        "X_val": X_val_comb,
        "word_vectorizer": vec_word_a,
        "char_vectorizer": vec_char,
        "type": "combined"
    }
    print(f"      Feature matrices ready in {time.time() - t_feat:.2f}s.")

    # 4. Systematic Model Exploration
    print("\n[5/6] Executing systematic classifier evaluations on validation subset...")
    results = []

    # Model exploration space (computationally optimized to complete quickly)
    models_to_test = [
        # LinearSVC models (3-8s per fit)
        ("LinearSVC", "balanced", 0.5),
        ("LinearSVC", "balanced", 1.0),
        ("LinearSVC", "balanced", 2.0),
        ("LinearSVC", None, 1.0),
        # Logistic Regression models (max_iter=100 for fast convergence)
        ("LogisticRegression", "balanced", 1.0),
        ("LogisticRegression", None, 1.0),
    ]

    for feat_name, feat_data in feature_configs.items():
        X_train_sp = feat_data["X_train"]
        X_val_sp = feat_data["X_val"]
        dim = X_train_sp.shape[1]

        for m_name, cw, c_val in models_to_test:
            t_m = time.time()
            if m_name == "LinearSVC":
                clf = create_linear_svc(C=c_val, class_weight=cw, random_state=42, max_iter=2000)
            else:
                clf = create_classifier(C=c_val, class_weight=cw, random_state=42, max_iter=100)

            fit_classifier(clf, X_train_sp, y_tr)
            y_pred_val = predict_categories(clf, X_val_sp)
            fit_time = time.time() - t_m

            acc = float(accuracy_score(y_val, y_pred_val))
            mf1 = float(f1_score(y_val, y_pred_val, average="macro", zero_division=0))
            wf1 = float(f1_score(y_val, y_pred_val, average="weighted", zero_division=0))
            m_prec = float(precision_score(y_val, y_pred_val, average="macro", zero_division=0))
            m_rec = float(recall_score(y_val, y_pred_val, average="macro", zero_division=0))

            results.append({
                "Model": m_name,
                "Features": feat_name,
                "Dimensions": dim,
                "Class Weight": str(cw),
                "C": c_val,
                "Val Accuracy": round(acc, 4),
                "Val Macro F1": round(mf1, 4),
                "Val Weighted F1": round(wf1, 4),
                "Val Macro Prec": round(m_prec, 4),
                "Val Macro Rec": round(m_rec, 4),
                "Fit Time (s)": round(fit_time, 2)
            })
            print(
                f"      {m_name:18} | {feat_name:28} | CW={str(cw):8} | C={c_val:<4} "
                f"-> Acc: {acc:.4f} | Macro F1: {mf1:.4f} | Weighted F1: {wf1:.4f} ({fit_time:.2f}s)",
                flush=True
            )

    # Save results to DataFrame and CSV
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Val Macro F1", ascending=False).reset_index(drop=True)
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = results_dir / "model_comparison.csv"
    df_results.to_csv(comparison_csv_path, index=False)
    print(f"\n      Saved full experiment comparison to {comparison_csv_path}")

    # Display Top 5 validation configurations
    print("\n" + "=" * 78)
    print("                     TOP 5 VALIDATION CONFIGURATIONS")
    print("=" * 78)
    print(df_results.head(5)[["Model", "Features", "Class Weight", "C", "Val Accuracy", "Val Macro F1", "Val Weighted F1"]].to_string(index=False))

    # 5. Winning Configuration Selection
    best_row = df_results.iloc[0]
    print("\n" + "=" * 78)
    print(f" WINNING CONFIGURATION: {best_row['Model']} with {best_row['Features']}")
    print(f" (Class Weight: {best_row['Class Weight']}, C: {best_row['C']} | Val Macro F1: {best_row['Val Macro F1']:.4f})")
    print("=" * 78)

    # 6. Retraining Winning Model on Full Training Pool & Final Test Evaluation
    print("\n[6/6] Retraining winning model on full Training Pool (20,000 records)...")
    best_model_name = best_row["Model"]
    best_features = best_row["Features"]
    best_cw = None if best_row["Class Weight"] == "None" else best_row["Class Weight"]
    best_c = float(best_row["C"])

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if "Combined" in best_features:
        # Combined Word + Char
        w_vec_final = create_vectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
        c_vec_final = create_char_vectorizer(ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True)
        w_vec_final, X_pool_w = fit_transform_tfidf(w_vec_final, clean_pool)
        c_vec_final, X_pool_c = fit_transform_tfidf(c_vec_final, clean_pool)
        X_pool_final = combine_sparse_matrices(X_pool_w, X_pool_c)

        X_test_w = transform_tfidf(w_vec_final, clean_test)
        X_test_c = transform_tfidf(c_vec_final, clean_test)
        X_test_final = combine_sparse_matrices(X_test_w, X_test_c)

        # Save both vectorizers
        joblib.dump(w_vec_final, models_dir / "tfidf_vectorizer.joblib")
        joblib.dump(c_vec_final, models_dir / "char_vectorizer.joblib")
    else:
        # Word only
        w_vec_final = create_vectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
        w_vec_final, X_pool_final = fit_transform_tfidf(w_vec_final, clean_pool)
        X_test_final = transform_tfidf(w_vec_final, clean_test)
        joblib.dump(w_vec_final, models_dir / "tfidf_vectorizer.joblib")

    # Retrain classifier
    if best_model_name == "LinearSVC":
        final_clf = create_linear_svc(C=best_c, class_weight=best_cw, random_state=42, max_iter=3000)
    else:
        final_clf = create_classifier(C=best_c, class_weight=best_cw, random_state=42, max_iter=1000)

    t_retrain = time.time()
    fit_classifier(final_clf, X_pool_final, y_pool)
    print(f"      Model retrained on {X_pool_final.shape[0]:,} records ({X_pool_final.shape[1]:,} features) in {time.time() - t_retrain:.2f}s.")

    # Save trained classifier
    save_classifier(final_clf, models_dir / "complaint_classifier.joblib")
    print(f"      Saved final classifier to {models_dir / 'complaint_classifier.joblib'}")

    # Evaluate ONCE on Untouched Final Test Set
    print("\n" + "=" * 78)
    print("           FINAL EVALUATION ON UNTOUCHED TEST SET (N = 5,000)")
    print("=" * 78)
    y_test_pred = predict_categories(final_clf, X_test_final)

    eval_bundle = evaluate_model(y_test, y_test_pred, labels=final_clf.classes_)
    test_metrics = eval_bundle["metrics"]

    # Baseline metrics comparison
    baseline_acc = 0.6183
    baseline_mf1 = 0.2478
    baseline_wf1 = 0.5496

    print(f"\n      Overall Test Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)  [Baseline: {baseline_acc:.4f}, Delta: {test_metrics['accuracy'] - baseline_acc:+.4f}]")
    print(f"      Test Macro F1-Score:     {test_metrics['macro_f1']:.4f} ({test_metrics['macro_f1']*100:.2f}%)  [Baseline: {baseline_mf1:.4f}, Delta: {test_metrics['macro_f1'] - baseline_mf1:+.4f}]")
    print(f"      Test Weighted F1-Score:  {test_metrics['weighted_f1']:.4f} ({test_metrics['weighted_f1']*100:.2f}%)  [Baseline: {baseline_wf1:.4f}, Delta: {test_metrics['weighted_f1'] - baseline_wf1:+.4f}]")
    print(f"      Test Macro Precision:    {test_metrics['macro_precision']:.4f}")
    print(f"      Test Macro Recall:       {test_metrics['macro_recall']:.4f}")
    print(f"      Test Weighted Precision: {test_metrics['weighted_precision']:.4f}")
    print(f"      Test Weighted Recall:    {test_metrics['weighted_recall']:.4f}")

    # Save metrics JSON
    metrics_summary = {
        "baseline": {
            "accuracy": baseline_acc,
            "macro_f1": baseline_mf1,
            "weighted_f1": baseline_wf1,
            "samples": 600
        },
        "improved_final_model": {
            "model_type": best_model_name,
            "features": best_features,
            "class_weight": str(best_cw),
            "C": best_c,
            "test_samples": len(y_test),
            "training_samples": len(y_pool),
            "accuracy": round(test_metrics["accuracy"], 4),
            "macro_f1": round(test_metrics["macro_f1"], 4),
            "weighted_f1": round(test_metrics["weighted_f1"], 4),
            "macro_precision": round(test_metrics["macro_precision"], 4),
            "macro_recall": round(test_metrics["macro_recall"], 4),
            "weighted_precision": round(test_metrics["weighted_precision"], 4),
            "weighted_recall": round(test_metrics["weighted_recall"], 4),
            "total_vocabulary_dimensions": int(X_pool_final.shape[1]),
        }
    }
    with open(results_dir / "final_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save Confusion Matrix Plot
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        labels=final_clf.classes_,
        save_path=str(cm_path),
        title=f"CFPB Complaint Categorisation - {best_model_name} (Holdout Test Set N=5,000)"
    )
    print(f"      Updated confusion matrix plot saved to {cm_path}")

    # Print top categories table
    print("\nPer-Category Test Set Performance (Top Categories by Support):")
    print(eval_bundle["per_category"].head(10).to_string(index=False))

    print(f"\nTotal Pipeline Execution Time: {time.time() - t_start:.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    run_full_experiment_pipeline()
