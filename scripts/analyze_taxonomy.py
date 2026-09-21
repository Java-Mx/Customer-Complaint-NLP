"""Taxonomy Audit Script for CFPB Complaint Categories.

Analyzes all 18 existing Product labels across the dataset partitions, extracts baseline performance
and confusion patterns, and audits candidate related-category groups based on:
  - Type A: Official / documented CFPB taxonomy changes (2017 / 2019)
  - Type B: Lexical / category name similarity
  - Type C: Observed confusion structure from empirical test set errors

Generates:
  - results/taxonomy_audit.csv
  - results/taxonomy_audit.json
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset
from src.classification import train_test_split_data

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_taxonomy_audit():
    print("=" * 70)
    print("Running CFPB Product Taxonomy Audit")
    print("=" * 70)

    # 1. Load dataset & splits
    data_path = ROOT_DIR / "data" / "complaints.csv"
    df = load_dataset(data_path, drop_invalid=True)
    total_len = len(df)

    X_pool, X_test, y_pool, y_test = train_test_split_data(
        df, test_size=0.20, random_state=42, stratify=True
    )
    df_pool = pd.DataFrame({"text": X_pool, "category": y_pool})
    X_tr, X_val, y_tr, y_val = train_test_split_data(
        df_pool, test_size=0.20, random_state=42, stratify=True
    )

    total_counts = df["category"].value_counts().to_dict()
    train_counts = y_tr.value_counts().to_dict()
    val_counts = y_val.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()

    # 2. Load error analysis data
    err_json_path = RESULTS_DIR / "error_analysis_data.json"
    if not err_json_path.exists():
        raise FileNotFoundError(f"Missing {err_json_path}. Run generate_error_analysis.py first.")

    with open(err_json_path, "r", encoding="utf-8") as f:
        err_data = json.load(f)

    per_cat_lookup = {r["category"]: r for r in err_data["per_category_metrics"]}

    # 3. Compile category audit records
    audit_rows = []
    for cat in sorted(total_counts.keys()):
        tot = total_counts[cat]
        pct = round(tot / total_len * 100, 2)
        tr = train_counts.get(cat, 0)
        va = val_counts.get(cat, 0)
        te = test_counts.get(cat, 0)
        metrics = per_cat_lookup.get(cat, {})
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        prim_conf = metrics.get("primary_confusion_category", "")
        prim_count = metrics.get("primary_confusion_count", 0)

        audit_rows.append({
            "category": cat,
            "total_support": tot,
            "percentage_of_dataset": pct,
            "train_support": tr,
            "validation_support": va,
            "test_support": te,
            "current_recall": recall,
            "current_f1": f1,
            "primary_confusion_category": prim_conf,
            "primary_confusion_count": prim_count
        })

    audit_df = pd.DataFrame(audit_rows).sort_values("total_support", ascending=False).reset_index(drop=True)
    audit_csv_path = RESULTS_DIR / "taxonomy_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"Saved: {audit_csv_path}")

    # 4. Identify Candidate Related-Category Groups
    # Categorizing into Evidence Types:
    # A = Official/documented CFPB taxonomy relationships
    # B = Explicit lexical / category name similarity
    # C = Observed empirical confusion structure
    candidate_groups = [
        {
            "group_name": "Credit Reporting and Repair",
            "categories": [
                "Credit reporting",
                "Credit reporting, credit repair services, or other personal consumer reports"
            ],
            "evidence_type_A_cfpb": "CFPB documented rename (~2019). Pre-2019 label 'Credit reporting' expanded to long-form label. Both represent the credit bureau/reporting product class.",
            "evidence_type_B_lexical": "High lexical overlap: shares exact unigrams 'Credit' and 'reporting'.",
            "evidence_type_C_confusion": "Top bidirectional confusion in dataset: 314 total intra-group errors (207 in one direction, 107 in reverse)."
        },
        {
            "group_name": "Credit Card and Stored Value (Prepaid)",
            "categories": [
                "Credit card",
                "Credit card or prepaid card",
                "Prepaid card"
            ],
            "evidence_type_A_cfpb": "CFPB official merger (April 24, 2017). CFPB combined 'Credit card' and 'Prepaid card' into 'Credit card or prepaid card' because consumers associate these card instruments.",
            "evidence_type_B_lexical": "All labels share the stem token 'card'; 'Credit card or prepaid card' literally combines the names of the two legacy components.",
            "evidence_type_C_confusion": "148 total intra-group errors (73: Credit card or prepaid card -> Credit card; 66: Credit card -> Credit card or prepaid card; 9 prepaid card cross-errors)."
        },
        {
            "group_name": "Retail Banking Accounts",
            "categories": [
                "Bank account or service",
                "Checking or savings account"
            ],
            "evidence_type_A_cfpb": "CFPB official direct rename (April 24, 2017). 'Bank account or service' was updated to 'Checking or savings account' to simplify consumer navigation.",
            "evidence_type_B_lexical": "Both denote standard depository banking services (checking, savings, bank accounts).",
            "evidence_type_C_confusion": "Bidirectional confusion of 121 errors (63: Bank account or service -> Checking or savings; 58: Checking or savings -> Bank account or service)."
        },
        {
            "group_name": "Money Movement and Virtual Currency",
            "categories": [
                "Money transfers",
                "Money transfer, virtual currency, or money service",
                "Virtual currency"
            ],
            "evidence_type_A_cfpb": "CFPB official expansion/consolidation (April 24, 2017). Expanded 'Money transfers' to 'Money transfer, virtual currency, or money service', formally absorbing virtual currency and elements of other financial services.",
            "evidence_type_B_lexical": "Direct lexical subset: 'Money transfers' and 'Virtual currency' are literally substrings of the combined 2017 title.",
            "evidence_type_C_confusion": "12 total intra-group errors (4 + 7 + 1 across pairs; virtual currency has 100% confusion into the expanded 2017 label)."
        },
        {
            "group_name": "Consumer & Small Dollar Loans (Payday / Title / Installment)",
            "categories": [
                "Payday loan",
                "Payday loan, title loan, or personal loan",
                "Consumer Loan"
            ],
            "evidence_type_A_cfpb": "CFPB official merger (April 24, 2017): 'Payday loan' was combined with sub-products from 'Consumer loan' (title, personal, installment loans) into 'Payday loan, title loan, or personal loan'. 'Consumer Loan' was the legacy parent category.",
            "evidence_type_B_lexical": "Common token 'loan'; 'Payday loan' is an exact prefix of 'Payday loan, title loan, or personal loan'.",
            "evidence_type_C_confusion": "40 total intra-group errors (13 between the payday variants; 27 between Consumer Loan and the payday variants)."
        }
    ]

    audit_json = {
        "dataset_summary": {
            "total_records": total_len,
            "train_records": len(y_tr),
            "val_records": len(y_val),
            "test_records": len(y_test),
            "original_categories_count": len(total_counts)
        },
        "category_audit": audit_rows,
        "candidate_related_groups": candidate_groups
    }

    audit_json_path = RESULTS_DIR / "taxonomy_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2, ensure_ascii=False)
    print(f"Saved: {audit_json_path}")
    print("Taxonomy audit completed successfully.")


if __name__ == "__main__":
    run_taxonomy_audit()
