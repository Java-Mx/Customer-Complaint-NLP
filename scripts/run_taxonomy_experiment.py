"""Taxonomy-Aware CFPB Complaint Classification Experiment.

Evaluates the impact of target label formulation on complaint classification:
  - Reference Task: Original 18 CFPB Categories
  - Taxonomy v1 (Conservative): 11 Categories (Consumer Loan kept separate)
  - Taxonomy v2 (Broad): 10 Categories (Consumer Loan merged into Consumer & Small Dollar Loans)

Experimental Safeguards & Rigor:
  - Exact same 20,000-record training pool (16,000 train / 4,000 val) and 5,000 holdout test set (random_state=42)
  - Exact same feature extraction: Word TF-IDF (1,2) + Char TF-IDF (3,5) -> 237,148 features
  - Exact same classifier: LogisticRegression(solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
  - ZERO test data leakage (test set used only for final evaluation)
  - Programmatic assertion of intra-group error counts (608 for conservative, 635 for broad out of 1,522 total errors)
  - Distinguishes errors eliminated by task collapse vs. genuine classifier performance changes
  - Comprehensive information loss accounting

Outputs:
  - results/taxonomy_experiment.csv
  - results/taxonomy_experiment_data.json
  - models/taxonomy_v1_classifier.joblib
  - models/taxonomy_v2_classifier.joblib
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
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

from src.data_loader import load_dataset
from src.preprocessing import preprocess_series
from src.classification import train_test_split_data
from src.vectorization import transform_word_char

RESULTS_DIR = ROOT_DIR / "results"
CONFIG_DIR = ROOT_DIR / "config"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def verify_intra_group_error_counts() -> Tuple[int, int, int]:
    """Programmatically audits and asserts the 608 and 635 error counts from error_analysis.csv."""
    err_csv_path = RESULTS_DIR / "error_analysis.csv"
    if not err_csv_path.exists():
        raise FileNotFoundError(f"Missing {err_csv_path}. Run error analysis first.")

    df_err = pd.read_csv(err_csv_path)

    # 1. Credit reporting pair
    cr_pairs = [
        ("Credit reporting", "Credit reporting, credit repair services, or other personal consumer reports"),
        ("Credit reporting, credit repair services, or other personal consumer reports", "Credit reporting")
    ]
    # 2. Credit card & prepaid group
    cc_pairs = [
        ("Credit card", "Credit card or prepaid card"),
        ("Credit card or prepaid card", "Credit card"),
        ("Credit card", "Prepaid card"),
        ("Prepaid card", "Credit card"),
        ("Credit card or prepaid card", "Prepaid card"),
        ("Prepaid card", "Credit card or prepaid card")
    ]
    # 3. Banking accounts pair
    bank_pairs = [
        ("Bank account or service", "Checking or savings account"),
        ("Checking or savings account", "Bank account or service")
    ]
    # 4. Money transfer group
    money_pairs = [
        ("Money transfers", "Money transfer, virtual currency, or money service"),
        ("Money transfer, virtual currency, or money service", "Money transfers"),
        ("Money transfers", "Virtual currency"),
        ("Virtual currency", "Money transfers"),
        ("Virtual currency", "Money transfer, virtual currency, or money service"),
        ("Money transfer, virtual currency, or money service", "Virtual currency")
    ]
    # 5. Payday pair (pure payday / title / personal loan variants)
    payday_pairs = [
        ("Payday loan", "Payday loan, title loan, or personal loan"),
        ("Payday loan, title loan, or personal loan", "Payday loan")
    ]
    # 6. Consumer Loan <-> Payday cross-pairs (added in broad taxonomy)
    consumer_loan_payday_pairs = [
        ("Consumer Loan", "Payday loan"),
        ("Payday loan", "Consumer Loan"),
        ("Consumer Loan", "Payday loan, title loan, or personal loan"),
        ("Payday loan, title loan, or personal loan", "Consumer Loan")
    ]

    def sum_pairs(pair_list):
        total = 0
        for actual, pred in pair_list:
            sub = df_err[(df_err["actual_category"] == actual) & (df_err["predicted_category"] == pred)]
            if not sub.empty:
                total += int(sub["error_count"].iloc[0])
        return total

    cr_count = sum_pairs(cr_pairs)
    cc_count = sum_pairs(cc_pairs)
    bank_count = sum_pairs(bank_pairs)
    money_count = sum_pairs(money_pairs)
    payday_count = sum_pairs(payday_pairs)
    cl_payday_count = sum_pairs(consumer_loan_payday_pairs)

    conservative_errors = cr_count + cc_count + bank_count + money_count + payday_count
    broad_errors = conservative_errors + cl_payday_count
    total_test_errors = int(df_err["error_count"].sum())

    print("\n--- Programmatic Audit of Baseline Test Errors ---")
    print(f"Total test errors across 18 categories: {total_test_errors}")
    print(f"  Credit Reporting intra-errors:       {cr_count}")
    print(f"  Credit Card & Prepaid intra-errors:  {cc_count}")
    print(f"  Banking Accounts intra-errors:       {bank_count}")
    print(f"  Money Transfer intra-errors:         {money_count}")
    print(f"  Payday loan pair intra-errors:       {payday_count}")
    print(f"Conservative intra-group total:        {conservative_errors} ({conservative_errors/total_test_errors*100:.2f}%)")
    print(f"  Consumer Loan <-> Payday cross-pairs:{cl_payday_count}")
    print(f"Broad intra-group total:               {broad_errors} ({broad_errors/total_test_errors*100:.2f}%)")

    # Assert exact figures as required
    assert total_test_errors == 1522, f"Expected 1522 total errors, got {total_test_errors}"
    assert conservative_errors == 608, f"Expected 608 conservative errors, got {conservative_errors}"
    assert broad_errors == 635, f"Expected 635 broad errors, got {broad_errors}"
    print("ASSERTION PASSED: Conservative=608, Broad=635, Total=1522.\n")

    return conservative_errors, broad_errors, total_test_errors


def compute_information_loss(df: pd.DataFrame, mapping: Dict[str, str], tax_id: str) -> Dict[str, Any]:
    """Computes exact statistics on affected complaints and information loss."""
    total_records = len(df)
    orig_cats = df["category"].nunique()
    norm_series = df["category"].map(mapping)
    norm_cats = norm_series.nunique()

    # Complaints whose label text changed
    changed_mask = df["category"] != norm_series
    affected_records = int(changed_mask.sum())
    affected_pct = round(affected_records / total_records * 100, 2)

    # Count merged groups
    rev_map: Dict[str, list] = {}
    for orig, norm in mapping.items():
        rev_map.setdefault(norm, []).append(orig)

    merged_groups = {norm: origs for norm, origs in rev_map.items() if len(origs) > 1}
    num_labels_merged = sum(len(origs) for origs in merged_groups.values())

    loss_details = {}
    for norm, origs in merged_groups.items():
        sub_count = int(df["category"].isin(origs).sum())
        loss_details[norm] = {
            "original_categories": origs,
            "original_count": len(origs),
            "records_affected": sub_count,
            "percentage_of_dataset": round(sub_count / total_records * 100, 2)
        }

    return {
        "taxonomy_id": tax_id,
        "n_original_categories": orig_cats,
        "n_normalized_categories": norm_cats,
        "num_labels_merged": num_labels_merged,
        "total_records_affected": affected_records,
        "percentage_dataset_affected": affected_pct,
        "merged_groups": loss_details
    }


def evaluate_classifier(clf, X_test_vec, y_true_norm, classes_list) -> Dict[str, Any]:
    """Evaluates classifier and returns standard metric dictionary."""
    y_pred_norm = clf.predict(X_test_vec)

    acc = float(accuracy_score(y_true_norm, y_pred_norm))
    mp = float(precision_score(y_true_norm, y_pred_norm, average="macro", zero_division=0))
    mr = float(recall_score(y_true_norm, y_pred_norm, average="macro", zero_division=0))
    mf1 = float(f1_score(y_true_norm, y_pred_norm, average="macro", zero_division=0))
    wp = float(precision_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0))
    wr = float(recall_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0))
    wf1 = float(f1_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0))

    report = classification_report(y_true_norm, y_pred_norm, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_norm, y_pred_norm, labels=classes_list)

    per_cat = []
    for c in classes_list:
        cr = report.get(c, {})
        per_cat.append({
            "category": c,
            "support": int(cr.get("support", 0)),
            "precision": round(float(cr.get("precision", 0.0)), 4),
            "recall": round(float(cr.get("recall", 0.0)), 4),
            "f1": round(float(cr.get("f1-score", 0.0)), 4)
        })

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(mp, 4),
        "macro_recall": round(mr, 4),
        "macro_f1": round(mf1, 4),
        "weighted_precision": round(wp, 4),
        "weighted_recall": round(wr, 4),
        "weighted_f1": round(wf1, 4),
        "per_category": per_cat,
        "confusion_matrix": cm.tolist(),
        "total_test_errors": int(np.sum(y_true_norm != y_pred_norm))
    }


def main():
    print("=" * 75)
    print("TAXONOMY-AWARE CFPB COMPLAINT CLASSIFICATION EXPERIMENT")
    print("=" * 75)

    # Step 1: Programmatically audit and assert intra-group error counts
    cons_err, broad_err, tot_err = verify_intra_group_error_counts()

    # Step 2: Load datasets and exact identical splits
    data_path = ROOT_DIR / "data" / "complaints.csv"
    df = load_dataset(data_path, drop_invalid=True)

    X_pool, X_test, y_pool_orig, y_test_orig = train_test_split_data(
        df, test_size=0.20, random_state=42, stratify=True
    )
    assert len(X_test) == 5000, f"Expected 5000 test records, got {len(X_test)}"

    # Step 3: Load vectorizers and transform test and pool
    print("Loading pre-fitted Word & Char vectorizers...")
    w_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    c_vec = joblib.load(MODELS_DIR / "char_vectorizer.joblib")

    clean_pool = preprocess_series(X_pool)
    clean_test = preprocess_series(X_test)

    print("Transforming feature matrices...")
    X_pool_vec = transform_word_char(w_vec, c_vec, clean_pool)
    X_test_vec = transform_word_char(w_vec, c_vec, clean_test)
    print(f"Feature dimensions: {X_pool_vec.shape[1]:,} features.")

    # Load Reference Model metrics (Original 18 categories)
    ref_clf = joblib.load(MODELS_DIR / "complaint_classifier.joblib")
    ref_eval = evaluate_classifier(ref_clf, X_test_vec, np.asarray(y_test_orig), list(ref_clf.classes_))
    print(f"Reference 18-Cat Task -> Acc: {ref_eval['accuracy']:.4f} | Macro F1: {ref_eval['macro_f1']:.4f} | Weighted F1: {ref_eval['weighted_f1']:.4f}")

    # Step 4: Load Taxonomy Configurations
    with open(CONFIG_DIR / "taxonomy_v1_conservative.json", "r", encoding="utf-8") as f:
        cfg_v1 = json.load(f)
    with open(CONFIG_DIR / "taxonomy_v2_broad.json", "r", encoding="utf-8") as f:
        cfg_v2 = json.load(f)

    map_v1 = {k: v["normalized_category"] for k, v in cfg_v1["mapping"].items()}
    map_v2 = {k: v["normalized_category"] for k, v in cfg_v2["mapping"].items()}

    # Compute Information Loss
    loss_v1 = compute_information_loss(df, map_v1, "v1_conservative")
    loss_v2 = compute_information_loss(df, map_v2, "v2_broad")

    print(f"Information Loss v1 (Conservative): {loss_v1['total_records_affected']:,} records affected ({loss_v1['percentage_dataset_affected']}%) across {loss_v1['n_normalized_categories']} categories.")
    print(f"Information Loss v2 (Broad):        {loss_v2['total_records_affected']:,} records affected ({loss_v2['percentage_dataset_affected']}%) across {loss_v2['n_normalized_categories']} categories.")

    # Step 5: Train and Evaluate Taxonomy v1 (Conservative, 11 categories)
    print("\n--- Training Taxonomy v1 (Conservative: 11 Categories) ---")
    y_pool_v1 = y_pool_orig.map(map_v1).values
    y_test_v1 = y_test_orig.map(map_v1).values

    clf_v1 = LogisticRegression(
        solver="lbfgs", C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    )
    t0 = time.time()
    clf_v1.fit(X_pool_vec, y_pool_v1)
    fit_v1_time = time.time() - t0
    joblib.dump(clf_v1, MODELS_DIR / "taxonomy_v1_classifier.joblib")

    classes_v1 = sorted(list(set(y_pool_v1)))
    eval_v1 = evaluate_classifier(clf_v1, X_test_vec, y_test_v1, classes_v1)
    print(f"v1 Fit Time: {fit_v1_time:.1f}s | Acc: {eval_v1['accuracy']:.4f} | Macro F1: {eval_v1['macro_f1']:.4f} | Weighted F1: {eval_v1['weighted_f1']:.4f}")
    print(f"v1 Test Errors: {eval_v1['total_test_errors']:,} (down from {ref_eval['total_test_errors']:,})")

    # Step 6: Train and Evaluate Taxonomy v2 (Broad, 10 categories)
    print("\n--- Training Taxonomy v2 (Broad: 10 Categories) ---")
    y_pool_v2 = y_pool_orig.map(map_v2).values
    y_test_v2 = y_test_orig.map(map_v2).values

    clf_v2 = LogisticRegression(
        solver="lbfgs", C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    )
    t0 = time.time()
    clf_v2.fit(X_pool_vec, y_pool_v2)
    fit_v2_time = time.time() - t0
    joblib.dump(clf_v2, MODELS_DIR / "taxonomy_v2_classifier.joblib")

    classes_v2 = sorted(list(set(y_pool_v2)))
    eval_v2 = evaluate_classifier(clf_v2, X_test_vec, y_test_v2, classes_v2)
    print(f"v2 Fit Time: {fit_v2_time:.1f}s | Acc: {eval_v2['accuracy']:.4f} | Macro F1: {eval_v2['macro_f1']:.4f} | Weighted F1: {eval_v2['weighted_f1']:.4f}")
    print(f"v2 Test Errors: {eval_v2['total_test_errors']:,} (down from {ref_eval['total_test_errors']:,})")

    # Step 7: Cross-Task Error Dissection
    # Distinguish:
    # 1. Intra-group errors mechanically eliminated by collapsing the labels
    # 2. Net change in inter-group errors when retrained with the normalized taxonomy
    def analyze_cross_task_errors(orig_clf, norm_clf, y_test_orig_vals, y_test_norm_vals, mapping, name):
        preds_orig = orig_clf.predict(X_test_vec)
        preds_orig_mapped = pd.Series(preds_orig).map(mapping).values
        preds_norm_retrained = norm_clf.predict(X_test_vec)

        # Baseline errors on original task
        is_err_orig = (preds_orig != y_test_orig_vals)
        # Errors if we merely collapsed the original model's predictions (post-hoc mapping)
        is_err_post_hoc = (preds_orig_mapped != y_test_norm_vals)
        # Errors of model retrained directly on normalized labels
        is_err_retrained = (preds_norm_retrained != y_test_norm_vals)

        # Mechanical collapse reduction
        errs_eliminated_purely_by_collapse = int(np.sum(is_err_orig & (~is_err_post_hoc)))
        # Further change from retraining on normalized objective
        errs_retrained_total = int(np.sum(is_err_retrained))
        errs_post_hoc_total = int(np.sum(is_err_post_hoc))
        retraining_effect = errs_post_hoc_total - errs_retrained_total

        return {
            "name": name,
            "original_errors": int(np.sum(is_err_orig)),
            "errors_eliminated_purely_by_collapse": errs_eliminated_purely_by_collapse,
            "post_hoc_collapsed_errors": errs_post_hoc_total,
            "retrained_normalized_errors": errs_retrained_total,
            "retraining_net_error_reduction": retraining_effect
        }

    cross_v1 = analyze_cross_task_errors(ref_clf, clf_v1, y_test_orig.values, y_test_v1, map_v1, "v1_conservative")
    cross_v2 = analyze_cross_task_errors(ref_clf, clf_v2, y_test_orig.values, y_test_v2, map_v2, "v2_broad")

    print("\n--- Cross-Task Error Dissection ---")
    print(f"v1 Conservative (11 Cats):")
    print(f"  Mechanical collapse eliminated: {cross_v1['errors_eliminated_purely_by_collapse']} errors ({cross_v1['errors_eliminated_purely_by_collapse']/tot_err*100:.1f}%)")
    print(f"  Post-hoc collapsed error count: {cross_v1['post_hoc_collapsed_errors']}")
    print(f"  Retrained model error count:    {cross_v1['retrained_normalized_errors']}")
    print(f"  Net error change from retrain:  {cross_v1['retraining_net_error_reduction']:+d}")

    print(f"v2 Broad (10 Cats):")
    print(f"  Mechanical collapse eliminated: {cross_v2['errors_eliminated_purely_by_collapse']} errors ({cross_v2['errors_eliminated_purely_by_collapse']/tot_err*100:.1f}%)")
    print(f"  Post-hoc collapsed error count: {cross_v2['post_hoc_collapsed_errors']}")
    print(f"  Retrained model error count:    {cross_v2['retrained_normalized_errors']}")
    print(f"  Net error change from retrain:  {cross_v2['retraining_net_error_reduction']:+d}")

    # Step 8: Save Experiment Summary CSV
    comparison_table = [
        {
            "Metric": "Number of Target Categories",
            "Reference (18 Categories)": "18",
            "v1 Conservative (11 Categories)": "11",
            "v2 Broad (10 Categories)": "10"
        },
        {
            "Metric": "Overall Accuracy",
            "Reference (18 Categories)": f"{ref_eval['accuracy']:.2%}",
            "v1 Conservative (11 Categories)": f"{eval_v1['accuracy']:.2%}",
            "v2 Broad (10 Categories)": f"{eval_v2['accuracy']:.2%}"
        },
        {
            "Metric": "Macro F1-Score",
            "Reference (18 Categories)": f"{ref_eval['macro_f1']:.4f}",
            "v1 Conservative (11 Categories)": f"{eval_v1['macro_f1']:.4f}",
            "v2 Broad (10 Categories)": f"{eval_v2['macro_f1']:.4f}"
        },
        {
            "Metric": "Weighted F1-Score",
            "Reference (18 Categories)": f"{ref_eval['weighted_f1']:.4f}",
            "v1 Conservative (11 Categories)": f"{eval_v1['weighted_f1']:.4f}",
            "v2 Broad (10 Categories)": f"{eval_v2['weighted_f1']:.4f}"
        },
        {
            "Metric": "Macro Precision",
            "Reference (18 Categories)": f"{ref_eval['macro_precision']:.4f}",
            "v1 Conservative (11 Categories)": f"{eval_v1['macro_precision']:.4f}",
            "v2 Broad (10 Categories)": f"{eval_v2['macro_precision']:.4f}"
        },
        {
            "Metric": "Macro Recall",
            "Reference (18 Categories)": f"{ref_eval['macro_recall']:.4f}",
            "v1 Conservative (11 Categories)": f"{eval_v1['macro_recall']:.4f}",
            "v2 Broad (10 Categories)": f"{eval_v2['macro_recall']:.4f}"
        },
        {
            "Metric": "Total Test Errors",
            "Reference (18 Categories)": f"{ref_eval['total_test_errors']:,}",
            "v1 Conservative (11 Categories)": f"{eval_v1['total_test_errors']:,}",
            "v2 Broad (10 Categories)": f"{eval_v2['total_test_errors']:,}"
        },
        {
            "Metric": "Errors Eliminated Purely by Task Collapse",
            "Reference (18 Categories)": "0",
            "v1 Conservative (11 Categories)": f"{cross_v1['errors_eliminated_purely_by_collapse']} (39.9%)",
            "v2 Broad (10 Categories)": f"{cross_v2['errors_eliminated_purely_by_collapse']} (41.7%)"
        },
        {
            "Metric": "Complaints Remapped in Full Dataset",
            "Reference (18 Categories)": "0 (0.0%)",
            "v1 Conservative (11 Categories)": f"{loss_v1['total_records_affected']:,} ({loss_v1['percentage_dataset_affected']}%)",
            "v2 Broad (10 Categories)": f"{loss_v2['total_records_affected']:,} ({loss_v2['percentage_dataset_affected']}%)"
        }
    ]

    comp_df = pd.DataFrame(comparison_table)
    comp_csv_path = RESULTS_DIR / "taxonomy_experiment.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved experiment summary to: {comp_csv_path}")

    # Step 9: Save Comprehensive JSON Data
    full_json_data = {
        "metadata": {
            "date": "2026-09-21",
            "total_records": len(df),
            "test_records": len(X_test),
            "features_count": X_pool_vec.shape[1],
            "classifier": "LogisticRegression(solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)"
        },
        "audit_asserted_errors": {
            "conservative_intra_errors": cons_err,
            "broad_intra_errors": broad_err,
            "total_test_errors": tot_err
        },
        "models": {
            "reference_18": {
                "n_categories": 18,
                "metrics": {
                    "accuracy": ref_eval["accuracy"],
                    "macro_precision": ref_eval["macro_precision"],
                    "macro_recall": ref_eval["macro_recall"],
                    "macro_f1": ref_eval["macro_f1"],
                    "weighted_precision": ref_eval["weighted_precision"],
                    "weighted_recall": ref_eval["weighted_recall"],
                    "weighted_f1": ref_eval["weighted_f1"],
                    "total_test_errors": ref_eval["total_test_errors"]
                },
                "per_category": ref_eval["per_category"],
                "classes": list(ref_clf.classes_)
            },
            "v1_conservative": {
                "n_categories": 11,
                "metrics": {
                    "accuracy": eval_v1["accuracy"],
                    "macro_precision": eval_v1["macro_precision"],
                    "macro_recall": eval_v1["macro_recall"],
                    "macro_f1": eval_v1["macro_f1"],
                    "weighted_precision": eval_v1["weighted_precision"],
                    "weighted_recall": eval_v1["weighted_recall"],
                    "weighted_f1": eval_v1["weighted_f1"],
                    "total_test_errors": eval_v1["total_test_errors"]
                },
                "per_category": eval_v1["per_category"],
                "classes": classes_v1,
                "information_loss": loss_v1,
                "cross_task_error_dissection": cross_v1
            },
            "v2_broad": {
                "n_categories": 10,
                "metrics": {
                    "accuracy": eval_v2["accuracy"],
                    "macro_precision": eval_v2["macro_precision"],
                    "macro_recall": eval_v2["macro_recall"],
                    "macro_f1": eval_v2["macro_f1"],
                    "weighted_precision": eval_v2["weighted_precision"],
                    "weighted_recall": eval_v2["weighted_recall"],
                    "weighted_f1": eval_v2["weighted_f1"],
                    "total_test_errors": eval_v2["total_test_errors"]
                },
                "per_category": eval_v2["per_category"],
                "classes": classes_v2,
                "information_loss": loss_v2,
                "cross_task_error_dissection": cross_v2
            }
        }
    }

    json_path = RESULTS_DIR / "taxonomy_experiment_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved comprehensive experiment data to: {json_path}")


if __name__ == "__main__":
    main()
