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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)


def create_classifier(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    solver: str = "lbfgs",
    classifier_type: str = "logistic_regression",
    class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    **kwargs: Any
) -> Union[LogisticRegression, LinearSVC]:
    """Initialize and configure a classical classifier (Logistic Regression or LinearSVC).

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength. Must be a positive float.
    max_iter : int, default=1000
        Maximum iterations for solver convergence.
    random_state : int, default=42
        Seed for reproducibility.
    solver : str, default='lbfgs'
        Optimization algorithm for LogisticRegression.
    classifier_type : str, default='logistic_regression'
        Model architecture: 'logistic_regression' or 'linear_svc'.
    class_weight : str | dict | None, optional
        Weights associated with classes (e.g. 'balanced' to mitigate class imbalance).
    **kwargs : Any
        Additional keyword arguments forwarded to the classifier constructor.

    Returns
    -------
    LogisticRegression | LinearSVC
        Configured un-fitted classifier instance.

    Raises
    ------
    ValueError
        If C <= 0, max_iter <= 0, or unsupported classifier_type.
    """
    if C <= 0:
        raise ValueError(f"Inverse regularization strength C must be positive, got {C}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}.")

    c_type = classifier_type.strip().lower()
    if c_type in ("logistic_regression", "lr", "logistic"):
        return LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver=solver,
            class_weight=class_weight,
            **kwargs
        )
    elif c_type in ("linear_svc", "svc", "linear_svm", "svm"):
        return LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'. Choose 'logistic_regression' or 'linear_svc'."
        )


def create_linear_svc(
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
    class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    **kwargs: Any
) -> LinearSVC:
    """Initialize and configure a classical Linear Support Vector Classifier (LinearSVC).

    LinearSVC uses a linear kernel optimized via liblinear, operating with high
    efficiency on large-vocabulary sparse text matrices.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    max_iter : int, default=2000
        Maximum iterations for convergence.
    random_state : int, default=42
        Reproducibility seed.
    class_weight : str | dict | None, optional
        Set to 'balanced' to automatically adjust weights inversely proportional
        to class frequencies.
    **kwargs : Any
        Additional arguments forwarded to LinearSVC.

    Returns
    -------
    LinearSVC
        Configured un-fitted LinearSVC instance.
    """
    return create_classifier(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        classifier_type="linear_svc",
        class_weight=class_weight,
        **kwargs
    )


def fit_classifier(
    classifier: Union[LogisticRegression, LinearSVC],
    X_train: Any,
    y_train: Any
) -> Union[LogisticRegression, LinearSVC]:
    """Fit a Logistic Regression or LinearSVC classifier on training feature matrix and labels.

    Operates natively on sparse matrices (e.g. scipy.sparse.csr_matrix) to
    maintain scalability and prevent dense memory allocation on large corpora.

    Parameters
    ----------
    classifier : LogisticRegression | LinearSVC
        Configured classifier instance.
    X_train : spmatrix | np.ndarray
        Sparse NxD feature matrix (or dense ndarray) of complaint representations.
    y_train : Sequence[Any] | np.ndarray | pd.Series
        Target category labels corresponding to rows of X_train.

    Returns
    -------
    LogisticRegression | LinearSVC
        Fitted classifier.

    Raises
    ------
    TypeError
        If classifier, X_train, or y_train is None or of invalid type.
    ValueError
        If training data is empty, sample lengths mismatch, or fewer than 2 classes exist.
    """
    if classifier is None:
        raise TypeError("classifier cannot be None.")
    if not isinstance(classifier, (LogisticRegression, LinearSVC)):
        raise TypeError(f"Expected LogisticRegression or LinearSVC classifier, got {type(classifier).__name__}.")

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
    if not isinstance(classifier, (LogisticRegression, LinearSVC)):
        raise TypeError(f"Expected LogisticRegression or LinearSVC classifier, got {type(classifier).__name__}.")

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
    classifier: Union[LogisticRegression, LinearSVC],
    X: Any
) -> np.ndarray:
    """Predict class probability distributions or normalized confidence scores for a feature matrix.

    For Logistic Regression, returns true calibrated class probabilities from predict_proba().
    For LinearSVC, computes normalized confidence scores via numerically stable softmax
    over decision_function() margins. Note that for LinearSVC, these are normalized
    confidence scores rather than calibrated probabilities.

    Parameters
    ----------
    classifier : LogisticRegression | LinearSVC
        Fitted classifier model.
    X : spmatrix | np.ndarray
        Sparse NxD feature matrix (or dense ndarray).

    Returns
    -------
    np.ndarray
        2D array of class scores of shape (N, num_classes).

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
    if not isinstance(classifier, (LogisticRegression, LinearSVC)):
        raise TypeError(f"Expected LogisticRegression or LinearSVC classifier, got {type(classifier).__name__}.")

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

    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(X)
    elif hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    else:
        raise AttributeError(f"Classifier {type(classifier).__name__} does not have predict_proba or decision_function.")


def predict_complaint_category(
    model: Any,
    vectorizer: Any,
    narrative: str,
    preprocess: bool = True
) -> Tuple[str, float]:
    """Predict the complaint product category along with prediction confidence.

    Executes the single-complaint inference pipeline:
    Raw Text -> [Preprocessing] -> TF-IDF Vectorization -> Classifier Inference -> (Category, Confidence).

    Supports single TfidfVectorizer or composite tuple/list of (word_vectorizer, char_vectorizer).

    Parameters
    ----------
    model : Any
        Trained classification model (LogisticRegression or LinearSVC).
    vectorizer : Any
        Fitted TfidfVectorizer or tuple/list of (word_vec, char_vec).
    narrative : str
        Customer complaint text narrative.
    preprocess : bool, default=True
        Whether to run src.preprocessing.preprocess_text on the narrative.

    Returns
    -------
    Tuple[str, float]
        (predicted_category, confidence_score) where confidence is the maximum
        predicted class probability or normalized decision confidence in range [0.0, 1.0].

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

    # Handle composite (word, char) vectorizers or single vectorizer
    if isinstance(vectorizer, (tuple, list)):
        from src.vectorization import transform_word_char
        X_vec = transform_word_char(vectorizer[0], vectorizer[1], [cleaned_text])
    else:
        X_vec = vectorizer.transform([cleaned_text])

    predicted_label = model.predict(X_vec)[0]

    # Compute prediction confidence
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_vec)[0]
        confidence = float(np.max(probabilities))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_vec)[0]
        if scores.ndim == 0:
            scores = np.array([-float(scores), float(scores)])
        exp_s = np.exp(scores - np.max(scores))
        norm_scores = exp_s / np.sum(exp_s)
        confidence = float(np.max(norm_scores))
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

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            categories,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_target
        )
    except ValueError as err:
        logger.warning(
            f"Stratified train_test_split failed ({err}). Falling back to unstratified split."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            categories,
            test_size=test_size,
            random_state=random_state,
            stratify=None
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


def validate_model_artifacts(
    models_dir: Optional[Union[str, Path]] = None,
    expected_features: int = 114493,
    expected_n_classes: int = 18,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Validate presence, integrity, and compatibility of production model artifacts.

    Ensures that serialized model artifacts exist in the target models/ directory,
    can be properly deserialized with joblib, conform to the expected combined
    feature representation (Word TF-IDF + Character TF-IDF), and match the
    trained classifier classes and feature dimension.

    Parameters
    ----------
    models_dir : str | Path | None, optional
        Path to the models directory. Defaults to ROOT_DIR / "models" if None.
    expected_features : int, default=114493
        Expected combined input feature dimension (14,493 word + 100,000 char).
    expected_n_classes : int, default=18
        Expected number of target category classes in the classifier.
    raise_on_error : bool, default=False
        If True, raises FileNotFoundError or ValueError on validation failure.

    Returns
    -------
    dict
        Dictionary containing:
        - 'is_valid': bool, True if all artifacts exist and are compatible
        - 'errors': list of str, error descriptions if invalid
        - 'classifier': deserialized classifier or None
        - 'word_vectorizer': deserialized word vectorizer or None
        - 'char_vectorizer': deserialized char vectorizer or None
        - 'feature_dim': int, total feature dimension or None
        - 'classes': np.ndarray, classifier classes or None
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parents[1] / "models"
    else:
        models_dir = Path(models_dir)

    clf_path = models_dir / "complaint_classifier.joblib"
    w_vec_path = models_dir / "tfidf_vectorizer.joblib"
    c_vec_path = models_dir / "char_vectorizer.joblib"

    errors: List[str] = []
    missing_files: List[str] = []

    for name, p in [
        ("complaint_classifier.joblib", clf_path),
        ("tfidf_vectorizer.joblib", w_vec_path),
        ("char_vectorizer.joblib", c_vec_path),
    ]:
        if not p.exists():
            missing_files.append(name)
            errors.append(f"Missing required model artifact: {name} at {p}")

    if missing_files:
        if raise_on_error:
            raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing_files)}")
        return {
            "is_valid": False,
            "errors": errors,
            "classifier": None,
            "word_vectorizer": None,
            "char_vectorizer": None,
            "feature_dim": None,
            "classes": None,
        }

    clf, w_vec, c_vec = None, None, None
    try:
        clf = joblib.load(clf_path)
    except Exception as e:
        errors.append(f"Failed to deserialize complaint_classifier.joblib: {e}")

    try:
        w_vec = joblib.load(w_vec_path)
    except Exception as e:
        errors.append(f"Failed to deserialize tfidf_vectorizer.joblib: {e}")

    try:
        c_vec = joblib.load(c_vec_path)
    except Exception as e:
        errors.append(f"Failed to deserialize char_vectorizer.joblib: {e}")

    if errors:
        if raise_on_error:
            raise ValueError("; ".join(errors))
        return {
            "is_valid": False,
            "errors": errors,
            "classifier": clf,
            "word_vectorizer": w_vec,
            "char_vectorizer": c_vec,
            "feature_dim": None,
            "classes": None,
        }

    # Verify vectorizer fitted state
    if not hasattr(w_vec, "vocabulary_"):
        errors.append("Word vectorizer (tfidf_vectorizer.joblib) is not fitted.")
    if not hasattr(c_vec, "vocabulary_"):
        errors.append("Character vectorizer (char_vectorizer.joblib) is not fitted.")

    # Verify classifier fitted state and classes
    if not hasattr(clf, "classes_"):
        errors.append("Classifier (complaint_classifier.joblib) has no classes_ attribute.")
    elif len(clf.classes_) != expected_n_classes:
        errors.append(
            f"Classifier has {len(clf.classes_)} classes, expected {expected_n_classes}."
        )

    # Verify feature dimension compatibility
    w_dim = len(w_vec.vocabulary_) if hasattr(w_vec, "vocabulary_") else 0
    c_dim = len(c_vec.vocabulary_) if hasattr(c_vec, "vocabulary_") else 0
    total_dim = w_dim + c_dim

    if total_dim != expected_features:
        errors.append(
            f"Combined vectorizer features ({total_dim}) does not match expected ({expected_features})."
        )

    clf_features = getattr(clf, "n_features_in_", None)
    if clf_features is None and hasattr(clf, "coef_"):
        clf_features = clf.coef_.shape[1]

    if clf_features is not None and clf_features != total_dim:
        errors.append(
            f"Classifier expected {clf_features} features, but combined vectorizers have {total_dim}."
        )

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise ValueError("; ".join(errors))

    return {
        "is_valid": is_valid,
        "errors": errors,
        "classifier": clf if is_valid else None,
        "word_vectorizer": w_vec if is_valid else None,
        "char_vectorizer": c_vec if is_valid else None,
        "feature_dim": total_dim if is_valid else None,
        "classes": clf.classes_ if (clf is not None and hasattr(clf, "classes_")) else None,
    }
