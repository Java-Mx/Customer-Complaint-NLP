"""CFPB Consumer Complaint Dataset Loading and Validation Module.

Provides robust dataset loading, schema verification, missing-value auditing,
and column standardization for the CFPB consumer complaint dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Known CFPB naming variants across official exports, APIs, and historical archives
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
]

ID_COLUMN_CANDIDATES: List[str] = [
    "Complaint ID",
    "complaint_id",
    "id",
]


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Identify and map dataset columns to canonical roles ('text', 'category', 'id').

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
    """Validate that the DataFrame is non-empty and contains required text and category columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.

    Returns
    -------
    Dict[str, str]
        Resolved column mapping.

    Raises
    ------
    ValueError
        If DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("Cannot validate columns: provided DataFrame is empty (0 rows).")
    return resolve_columns(df)


def get_dataset_summary(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Generate a structural, validation, and missing-value summary of the complaints dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded complaints DataFrame.
    column_mapping : Optional[Dict[str, str]], optional
        Resolved column mapping. If None, resolve_columns() is called.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing total rows, column count, column names,
        missing text count, missing category count, usable rows count,
        and category distribution.
    """
    mapping = column_mapping or resolve_columns(df)
    text_col = mapping["text"]
    cat_col = mapping["category"]
    id_col = mapping.get("id")

    total_rows = int(len(df))
    missing_counts = df.isnull().sum().to_dict()

    missing_text = int(missing_counts.get(text_col, 0))
    # Also count empty / whitespace-only string narratives
    non_null_text = df[text_col].dropna()
    empty_text_strings = int((non_null_text.astype(str).str.strip().str.len() == 0).sum())
    total_invalid_text = missing_text + empty_text_strings

    missing_cat = int(missing_counts.get(cat_col, 0))

    # Usable rows: both text and category are non-null and text is not empty
    usable_mask = (
        df[text_col].notnull()
        & (df[text_col].astype(str).str.strip().str.len() > 0)
        & df[cat_col].notnull()
    )
    usable_rows = int(usable_mask.sum())

    category_counts = df[cat_col].dropna().value_counts().to_dict()

    summary: Dict[str, Any] = {
        "total_rows": total_rows,
        "total_columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "text_column": text_col,
        "category_column": cat_col,
        "id_column": id_col,
        "missing_text_count": total_invalid_text,
        "missing_category_count": missing_cat,
        "missing_id_count": int(missing_counts.get(id_col, 0)) if id_col else 0,
        "usable_rows_count": usable_rows,
        "all_missing_counts": missing_counts,
        "num_unique_categories": int(df[cat_col].nunique(dropna=True)),
        "top_categories": dict(list(category_counts.items())[:8]),
    }
    return summary


def load_dataset(
    filepath: str | Path = "data/complaints.csv",
    nrows: Optional[int] = None,
    drop_invalid: bool = False,
    standardize_columns: bool = True
) -> pd.DataFrame:
    """Load and validate the CFPB Consumer Complaint dataset from a local CSV file.

    Parameters
    ----------
    filepath : str | Path, default='data/complaints.csv'
        Relative or configurable path to the local CSV dataset.
    nrows : Optional[int], default=None
        Number of rows to read for sampling or quick verification.
    drop_invalid : bool, default=False
        If True, filters out records with missing/blank complaint narratives or missing categories.
        If False (default), retains all records without altering the dataset permanently.
    standardize_columns : bool, default=True
        If True, adds standardized columns ('text', 'category', 'complaint_id')
        while preserving all original DataFrame columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing original useful columns plus standardized fields ('text', 'category', 'complaint_id').

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the loaded file is empty or missing required text/category columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: '{path.resolve()}'. "
            f"Please ensure the CFPB complaints CSV is placed at the specified location."
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

    if drop_invalid:
        initial_count = len(df)
        valid_mask = (
            df[text_col].notnull()
            & (df[text_col].astype(str).str.strip().str.len() > 0)
            & df[cat_col].notnull()
        )
        df = df[valid_mask].copy()
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info("Filtered %d records with missing or empty text/category.", dropped_count)

    # Add standardized fields without discarding original columns
    if standardize_columns:
        df["text"] = df[text_col]
        df["category"] = df[cat_col]
        if id_col and id_col in df.columns:
            df["complaint_id"] = df[id_col]

    return df


def load_dataset_from_api(
    max_records: int = 50,
    search_term: Optional[str] = None,
    drop_invalid: bool = False,
    timeout: float = 25.0
) -> pd.DataFrame:
    """Fetch and normalize real complaint data from the official CFPB API.

    Parameters
    ----------
    max_records : int, default=50
        Maximum number of records to retrieve.
    search_term : Optional[str], optional
        Search keyword or phrase.
    drop_invalid : bool, default=False
        If True, drops records with empty or whitespace-only narratives.
    timeout : float, default=25.0
        Network timeout in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized fields ('text', 'category', 'complaint_id').
    """
    from src.cfpb_api import fetch_cfpb_data

    df = fetch_cfpb_data(
        max_records=max_records,
        search_term=search_term,
        drop_empty_narratives=drop_invalid,
        timeout=timeout
    )

    if drop_invalid and not df.empty:
        valid_mask = (
            df["text"].notnull()
            & (df["text"].astype(str).str.strip().str.len() > 0)
            & df["category"].notnull()
        )
        df = df[valid_mask].copy().reset_index(drop=True)

    return df


def get_complaints_data(
    source: str = "csv",
    filepath: str | Path = "data/complaints.csv",
    max_records: Optional[int] = None,
    drop_invalid: bool = False,
    search_term: Optional[str] = None
) -> pd.DataFrame:
    """Unified interface to retrieve complaints from either local CSV or live CFPB API.

    Parameters
    ----------
    source : str, default='csv'
        Data source: 'csv' for local file, 'api' for live CFPB API.
    filepath : str | Path, default='data/complaints.csv'
        Local CSV path (used when source='csv').
    max_records : Optional[int], default=None
        Maximum records to retrieve (used as nrows for CSV or record limit for API).
    drop_invalid : bool, default=False
        Whether to filter out records missing text or category.
    search_term : Optional[str], optional
        Search term (used when source='api').

    Returns
    -------
    pd.DataFrame
        Normalized complaints DataFrame.

    Raises
    ------
    ValueError
        If source is not 'csv' or 'api'.
    """
    normalized_source = str(source).strip().lower()
    if normalized_source == "csv":
        return load_dataset(
            filepath=filepath,
            nrows=max_records,
            drop_invalid=drop_invalid,
            standardize_columns=True
        )
    elif normalized_source == "api":
        limit = max_records if max_records is not None else 50
        return load_dataset_from_api(
            max_records=limit,
            search_term=search_term,
            drop_invalid=drop_invalid
        )
    else:
        raise ValueError(f"Unknown data source '{source}'. Expected 'csv' or 'api'.")

