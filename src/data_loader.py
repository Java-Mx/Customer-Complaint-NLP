"""CFPB Consumer Complaint Dataset Loading and Validation Module.

Provides robust dataset loading, schema verification, missing-value auditing,
and column standardization for consumer complaint data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# Candidate column names across different CFPB exports and historical archives
TEXT_COLUMN_CANDIDATES: List[str] = [
    "Consumer Complaint",
    "Consumer complaint narrative",
    "complaint_what_happened",
    "complaint_text",
    "text",
]

CATEGORY_COLUMN_CANDIDATES: List[str] = [
    "Product",
    "product",
    "category",
    "sub_product",
]

ID_COLUMN_CANDIDATES: List[str] = [
    "Complaint ID",
    "complaint_id",
    "id",
]


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Identify and map dataset columns to canonical roles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw CFPB complaints.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping canonical roles ('text', 'category', 'id') to actual DataFrame column names.

    Raises
    ------
    ValueError
        If required text or category columns cannot be identified.
    """
    column_set = set(df.columns)
    resolved: Dict[str, str] = {}

    # Locate complaint text column
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in column_set:
            resolved["text"] = candidate
            break

    if "text" not in resolved:
        raise ValueError(
            f"Missing required complaint text column. Looked for candidates: {TEXT_COLUMN_CANDIDATES}. "
            f"Available columns in dataset: {list(df.columns)}"
        )

    # Locate product/category column
    for candidate in CATEGORY_COLUMN_CANDIDATES:
        if candidate in column_set:
            resolved["category"] = candidate
            break

    if "category" not in resolved:
        raise ValueError(
            f"Missing required category/product column. Looked for candidates: {CATEGORY_COLUMN_CANDIDATES}. "
            f"Available columns in dataset: {list(df.columns)}"
        )

    # Locate identifier column (optional but tracked if present)
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in column_set:
            resolved["id"] = candidate
            break

    return resolved


def validate_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Validate that the DataFrame contains required text and category columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.

    Returns
    -------
    Dict[str, str]
        Resolved column mapping.
    """
    if df.empty:
        raise ValueError("Cannot validate columns: provided DataFrame is empty.")
    return resolve_columns(df)


def get_dataset_summary(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Generate a structural and statistical summary of the complaints dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded complaints DataFrame.
    column_mapping : Optional[Dict[str, str]], optional
        Resolved column map. If None, resolve_columns() is called.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing row count, column count, column names,
        missing value statistics, and category distributions.
    """
    mapping = column_mapping or resolve_columns(df)
    text_col = mapping["text"]
    cat_col = mapping["category"]
    id_col = mapping.get("id")

    missing_counts = df.isnull().sum().to_dict()
    category_counts = df[cat_col].value_counts().to_dict()

    summary: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "text_column": text_col,
        "category_column": cat_col,
        "id_column": id_col,
        "missing_text_count": int(missing_counts.get(text_col, 0)),
        "missing_category_count": int(missing_counts.get(cat_col, 0)),
        "missing_id_count": int(missing_counts.get(id_col, 0)) if id_col else 0,
        "all_missing_counts": missing_counts,
        "num_unique_categories": int(df[cat_col].nunique(dropna=True)),
        "top_categories": dict(list(category_counts.items())[:5]),
    }
    return summary


def load_dataset(
    filepath: str | Path = "data/complaints.csv",
    nrows: Optional[int] = None,
    drop_empty: bool = True,
    standardize_columns: bool = True
) -> pd.DataFrame:
    """Load and validate the CFPB Consumer Complaint dataset from a local CSV file.

    Parameters
    ----------
    filepath : str | Path, default='data/complaints.csv'
        Path to the local CSV dataset.
    nrows : Optional[int], default=None
        Number of rows to read for sampling or exploratory verification.
    drop_empty : bool, default=True
        Whether to drop records with null or blank complaint narratives or missing categories.
    standardize_columns : bool, default=True
        If True, renames resolved columns to canonical names:
        ('complaint_text', 'product_category', 'complaint_id').

    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame ready for downstream NLP processing.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the loaded file is empty or missing required columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: '{path.resolve()}'. "
            f"Please place the CFPB complaints CSV in data/ as documented in data/README.md."
        )

    # Load CSV with pandas
    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    if df.empty:
        raise ValueError(f"Dataset at '{path}' was loaded but contains 0 rows.")

    # Validate schema & detect columns
    mapping = validate_columns(df)
    text_col = mapping["text"]
    cat_col = mapping["category"]
    id_col = mapping.get("id")

    # Filter empty or whitespace-only texts / categories if requested
    if drop_empty:
        initial_len = len(df)
        # Drop null values in core columns
        df = df.dropna(subset=[text_col, cat_col])
        # Drop rows where narrative text is pure whitespace
        df = df[df[text_col].astype(str).str.strip().str.len() > 0]
        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            logger.info("Dropped %d records with missing or empty complaint narratives/categories.", dropped_count)

    if df.empty:
        raise ValueError("All records in the dataset had missing or empty complaint text/categories.")

    # Optionally standardize column names
    if standardize_columns:
        rename_map = {
            text_col: "complaint_text",
            cat_col: "product_category",
        }
        if id_col:
            rename_map[id_col] = "complaint_id"
        df = df.rename(columns=rename_map)

    return df
