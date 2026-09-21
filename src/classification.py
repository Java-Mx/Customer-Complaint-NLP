"""Supervised machine learning classification module for complaints.

Implements supervised training and inference using classical linear models
(Multinomial Logistic Regression) to categorize customer complaints into
financial product classes based on sparse TF-IDF feature representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse, spmatrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)


def create_classifier(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    solver: str = "lbfgs",
    **kwargs: Any
) -> LogisticRegression:
    """Initialize and configure a classical Logistic Regression classifier.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength. Must be a positive float.
    max_iter : int, default=1000
        Maximum iterations for solver convergence.
    random_state : int, default=42
        Seed for reproducibility.
    solver : str, default='lbfgs'
        Optimization algorithm. 'lbfgs' natively supports multinomial loss
        and sparse input matrices.
    **kwargs : Any
        Additional keyword arguments forwarded to Scikit-learn's LogisticRegression.

    Returns
    -------
    LogisticRegression
        Configured un-fitted LogisticRegression instance.

    Raises
    ------
    ValueError
        If C <= 0 or max_iter <= 0.
    """
    if C <= 0:
        raise ValueError(f"Inverse regularization strength C must be positive, got {C}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}.")

    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
        **kwargs
    )


def fit_classifier(
    classifier: LogisticRegression,
    X_train: Any,
    y_train: Any
) -> LogisticRegression:
    """Fit a Logistic Regression classifier on training feature matrix and labels.

    Operates natively on sparse matrices (e.g. scipy.sparse.csr_matrix) to
    maintain scalability and prevent dense memory allocation on large corpora.

    Parameters
    ----------
    classifier : LogisticRegression
        Configured LogisticRegression instance.
    X_train : spmatrix | np.ndarray
        Sparse NxD feature matrix (or dense ndarray) of complaint representations.
    y_train : Sequence[Any] | np.ndarray | pd.Series
        Target category labels corresponding to rows of X_train.

    Returns
    -------
    LogisticRegression
        Fitted LogisticRegression classifier.

    Raises
    ------
    TypeError
        If classifier, X_train, or y_train is None or of invalid type.
    ValueError
        If training data is empty, sample lengths mismatch, or fewer than 2 classes exist.
    """
    if classifier is None:
        raise TypeError("classifier cannot be None.")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression classifier, got {type(classifier).__name__}.")

    if X_train is None:
        raise TypeError("X_train cannot be None.")
    if y_train is None:
        raise TypeError("y_train cannot be None.")

    if not (issparse(X_train) or isinstance(X_train, np.ndarray)):
        raise TypeError(f"Expected sparse matrix or ndarray for X_train, got {type(X_train).__name__}.")

    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty. Cannot fit classifier with 0 samples.")

    # Convert y_train to 1D numpy array
    if isinstance(y_train, pd.Series):
        y_arr = y_train.values
    else:
        y_arr = np.asarray(y_train)

    if len(y_arr) == 0:
        raise ValueError("y_train is empty. Cannot fit classifier with 0 labels.")

    if X_train.shape[0] != len(y_arr):
        raise ValueError(
            f"Sample count mismatch: X_train has {X_train.shape[0]} rows, "
            f"but y_train has {len(y_arr)} labels."
        )

    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training data must contain at least 2 distinct classes to fit classifier. "
            f"Found {len(unique_classes)} class: {unique_classes}."
        )

    classifier.fit(X_train, y_arr)
    return classifier


def train_logistic_regression(
    X_train: Any,
    y_train: Any,
    c_param: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any
) -> LogisticRegression:
    """Train a Multinomial Logistic Regression model on TF-IDF features.

    Convenience wrapper combining model creation and fitting, preserving the
    scaffold function signature.

    Parameters
    ----------
    X_train : Any
        Training feature matrix (sparse TF-IDF).
    y_train : Any
        Target category labels.
    c_param : float, default=1.0
        Inverse regularization strength.
    max_iter : int, default=1000
        Maximum iterations for solver convergence.
    random_state : int, default=42
        Seed for reproducibility.
    **kwargs : Any
        Additional keyword arguments forwarded to create_classifier.

    Returns
    -------
    LogisticRegression
        Fitted Scikit-learn LogisticRegression classifier.
    """
    clf = create_classifier(
        C=c_param,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    return fit_classifier(clf, X_train, y_train)


def predict_categories(
    classifier: LogisticRegression,
    X: Any
) -> np.ndarray:
    """Predict product category labels for a feature matrix.

    Parameters
    ----------
    classifier : LogisticRegression
        Fitted LogisticRegression model.
    X : spmatrix | np.ndarray
        Sparse NxD feature matrix (or dense ndarray) to predict.

    Returns
    -------
    np.ndarray
        1D array of predicted category labels of length N.

    Raises
    ------
    TypeError
        If classifier or X is None or invalid type.
    NotFittedError
        If classifier is not fitted prior to prediction.
    ValueError
        If X is empty or feature dimensions do not match training data.
    """
    if classifier is None:
        raise TypeError("classifier cannot be None.")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression classifier, got {type(classifier).__name__}.")

    check_is_fitted(classifier)

    if X is None:
        raise TypeError("X cannot be None.")
    if not (issparse(X) or isinstance(X, np.ndarray)):
        raise TypeError(f"Expected sparse matrix or ndarray for X, got {type(X).__name__}.")

    if X.shape[0] == 0:
        raise ValueError("X is empty. Cannot generate predictions for 0 samples.")

    if hasattr(classifier, "n_features_in_") and X.shape[1] != classifier.n_features_in_:
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]} features, "
            f"but classifier was trained on {classifier.n_features_in_} features."
        )

    return classifier.predict(X)


def predict_category_proba(
    classifier: LogisticRegression,
    X: Any
) -> np.ndarray:
    """Predict class probability distributions for a feature matrix.

    Parameters
    ----------
    classifier : LogisticRegression
        Fitted LogisticRegression model.
    X : spmatrix | np.ndarray
        Sparse NxD feature matrix (or dense ndarray).

    Returns
    -------
    np.ndarray
        2D array of class probabilities of shape (N, num_classes).

    Raises
    ------
    TypeError
        If classifier or X is None.
    NotFittedError
        If classifier is not fitted.
    ValueError
        If X is empty or feature dimensions mismatch.
    """
    if classifier is None:
        raise TypeError("classifier cannot be None.")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression classifier, got {type(classifier).__name__}.")

    check_is_fitted(classifier)

    if X is None:
        raise TypeError("X cannot be None.")
    if not (issparse(X) or isinstance(X, np.ndarray)):
        raise TypeError(f"Expected sparse matrix or ndarray for X, got {type(X).__name__}.")

    if X.shape[0] == 0:
        raise ValueError("X is empty. Cannot generate probabilities for 0 samples.")

    if hasattr(classifier, "n_features_in_") and X.shape[1] != classifier.n_features_in_:
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]} features, "
            f"but classifier was trained on {classifier.n_features_in_} features."
        )

    return classifier.predict_proba(X)


def predict_complaint_category(
    model: Any,
    vectorizer: Any,
    narrative: str,
    preprocess: bool = True
) -> Tuple[str, float]:
    """Predict the complaint product category along with prediction confidence.

    Executes the single-complaint inference pipeline:
    Raw Text -> [Preprocessing] -> TF-IDF Vectorization -> Classifier Inference -> (Category, Confidence).

    Parameters
    ----------
    model : Any
        Trained classification model (LogisticRegression).
    vectorizer : Any
        Fitted TfidfVectorizer.
    narrative : str
        Customer complaint text narrative.
    preprocess : bool, default=True
        Whether to run src.preprocessing.preprocess_text on the narrative.

    Returns
    -------
    Tuple[str, float]
        (predicted_category, confidence_score) where confidence is the maximum
        predicted class probability in range [0.0, 1.0].

    Raises
    ------
    TypeError
        If model, vectorizer, or narrative is None.
    ValueError
        If narrative is empty or whitespace-only.
    """
    if narrative is None:
        raise TypeError("Complaint narrative cannot be None.")
    if not isinstance(narrative, str):
        raise TypeError(f"Expected string narrative, got {type(narrative).__name__}.")
    if not narrative.strip():
        raise ValueError("Complaint narrative cannot be empty.")

    if model is None:
        raise TypeError("Classification model cannot be None.")
    if vectorizer is None:
        raise TypeError("TF-IDF vectorizer cannot be None.")

    cleaned_text = preprocess_text(narrative) if preprocess else narrative
    # Transform using fitted TF-IDF vectorizer
    X_vec = vectorizer.transform([cleaned_text])

    predicted_label = model.predict(X_vec)[0]

    # Compute prediction confidence if predict_proba is available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_vec)[0]
        confidence = float(np.max(probabilities))
    else:
        confidence = 1.0

    return str(predicted_label), confidence


# Alias preserving suggested naming
predict_category = predict_complaint_category


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split complaint text and categories into training and testing sets.

    Ensures train/test partition occurs strictly BEFORE TF-IDF fitting to
    prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing complaint records.
    test_size : float, default=0.20
        Proportion of dataset to reserve for test split.
    random_state : int, default=42
        Reproducibility seed.
    stratify : bool, default=True
        Whether to stratify split by category label distribution.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        (X_train_text, X_test_text, y_train, y_test)

    Raises
    ------
    TypeError
        If df is None or not a DataFrame.
    ValueError
        If df is empty or missing required columns.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a non-null pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot split empty DataFrame.")

    # Locate text column
    text_col = None
    for candidate in ["text", "Consumer Complaint", "complaint_text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(f"Could not find text column in DataFrame. Available columns: {list(df.columns)}")

    # Locate category column
    cat_col = None
    for candidate in ["category", "Product", "product"]:
        if candidate in df.columns:
            cat_col = candidate
            break
    if cat_col is None:
        raise ValueError(f"Could not find category column in DataFrame. Available columns: {list(df.columns)}")

    texts = df[text_col]
    categories = df[cat_col]

    strat_target = None
    if stratify:
        counts = categories.value_counts()
        min_count = counts.min()
        n_classes = len(counts)
        n_samples = len(df)
        n_test = int(np.ceil(test_size * n_samples)) if isinstance(test_size, float) else int(test_size)
        n_train = n_samples - n_test
        if min_count < 2 or n_test < n_classes or n_train < n_classes:
            logger.warning(
                f"Stratification conditions not met (n_classes={n_classes}, n_test={n_test}, min_samples={min_count}). "
                f"Falling back to unstratified split."
            )
            strat_target = None
        else:
            strat_target = categories

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        categories,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_target
    )

    return X_train, X_test, y_train, y_test


def save_classifier(
    classifier: Any,
    filepath: Union[str, Path]
) -> Path:
    """Serialize and save a trained classifier model to disk using joblib.

    Parameters
    ----------
    classifier : Any
        Trained classifier instance.
    filepath : str | Path
        Target destination path for the serialized artifact.

    Returns
    -------
    Path
        Absolute path to the saved artifact file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, path)
    return path.resolve()


def load_classifier(
    filepath: Union[str, Path]
) -> Any:
    """Deserialize and load a trained classifier model from disk.

    Parameters
    ----------
    filepath : str | Path
        Path to the serialized artifact.

    Returns
    -------
    Any
        Loaded classifier instance.

    Raises
    ------
    FileNotFoundError
        If the specified model artifact does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(path)
