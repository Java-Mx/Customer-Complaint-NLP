"""Leakage-safe helpers for classical (TF-IDF + linear/NB) model-selection experiments.

Everything here is *fit on the training subset only*: TF-IDF vocabularies / IDF weights,
class weights and (stateless) text statistics. Validation / test texts are only ever
*transformed* with objects fitted on training data.

Used by ``scripts/run_model_improvement.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RANDOM_STATE = 42
_REDACTION_RE = re.compile(r"\bx{2,}\b")

# Validation selection order (task requirement): Macro F1, Macro Recall, Accuracy, Weighted F1.
SELECTION_METRICS: Tuple[str, ...] = ("Val Macro F1", "Val Macro Rec", "Val Accuracy", "Val Weighted F1")


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative description of a sparse feature representation.

    ``word`` / ``char`` hold keyword arguments for ``TfidfVectorizer`` (None = block unused).
    ``text_stats`` appends a few stateless, text-derived scalar features.
    """

    word: Optional[Tuple[Tuple[str, Any], ...]] = None
    char: Optional[Tuple[Tuple[str, Any], ...]] = None
    text_stats: bool = False

    @staticmethod
    def make(word: Optional[Mapping[str, Any]] = None, char: Optional[Mapping[str, Any]] = None,
             text_stats: bool = False) -> "FeatureSpec":
        def freeze(d):
            return None if d is None else tuple(sorted(d.items()))
        if word is None and char is None:
            raise ValueError("FeatureSpec needs at least one of word / char blocks.")
        return FeatureSpec(freeze(word), freeze(char), text_stats)

    def describe(self) -> str:
        parts = []
        for label, block in (("word", self.word), ("char", self.char)):
            if block is not None:
                d = dict(block)
                parts.append(label + "(" + ",".join(f"{k}={v}" for k, v in d.items()) + ")")
        if self.text_stats:
            parts.append("text_stats")
        return " + ".join(parts)


def _make_vectorizer(kind: str, params: Mapping[str, Any]) -> TfidfVectorizer:
    cfg = dict(params)
    cfg.setdefault("lowercase", False)  # lowercasing already done in src.preprocessing
    if kind == "char":
        cfg.setdefault("analyzer", "char_wb")
    return TfidfVectorizer(**cfg)


def text_statistics(raw_texts: Sequence[str]) -> csr_matrix:
    """Stateless, text-only scalar features computed from RAW narratives.

    The standard preprocessing deletes CFPB redaction masks ("XXXX"), discarding a stylistic
    signal. These features retain it with fixed (non-fitted) scaling:
    log-length, log redaction-mask count, redaction-present flag.
    """
    rows = []
    for t in raw_texts:
        s = t.lower() if isinstance(t, str) else ""
        n_mask = len(_REDACTION_RE.findall(s))
        rows.append((np.log1p(len(s)) / 10.0, min(np.log1p(n_mask), 5.0) / 5.0, 1.0 if n_mask else 0.0))
    return csr_matrix(np.asarray(rows, dtype=np.float64))


def fit_features(spec: FeatureSpec, train_clean: Sequence[str], train_raw: Optional[Sequence[str]] = None):
    """Fit all vectorizers on TRAINING text only. Returns (fitted_blocks, X_train)."""
    blocks: Dict[str, TfidfVectorizer] = {}
    mats = []
    if spec.word is not None:
        blocks["word"] = _make_vectorizer("word", dict(spec.word))
        mats.append(blocks["word"].fit_transform(list(train_clean)))
    if spec.char is not None:
        blocks["char"] = _make_vectorizer("char", dict(spec.char))
        mats.append(blocks["char"].fit_transform(list(train_clean)))
    if spec.text_stats:
        if train_raw is None:
            raise ValueError("text_stats requires raw texts.")
        mats.append(text_statistics(list(train_raw)))
    X = mats[0] if len(mats) == 1 else hstack(mats, format="csr")
    return blocks, csr_matrix(X)


def transform_features(spec: FeatureSpec, blocks: Mapping[str, TfidfVectorizer],
                       clean: Sequence[str], raw: Optional[Sequence[str]] = None) -> csr_matrix:
    """Transform held-out text with already-fitted (training-only) vectorizers."""
    mats = []
    if spec.word is not None:
        mats.append(blocks["word"].transform(list(clean)))
    if spec.char is not None:
        mats.append(blocks["char"].transform(list(clean)))
    if spec.text_stats:
        if raw is None:
            raise ValueError("text_stats requires raw texts.")
        mats.append(text_statistics(list(raw)))
    X = mats[0] if len(mats) == 1 else hstack(mats, format="csr")
    return csr_matrix(X)


def power_class_weights(y_train: Sequence[Any], power: float = 1.0) -> Dict[Any, float]:
    """Training-only class weights ``(n / (k * n_c)) ** power``.

    power=0 -> uniform, power=1 -> sklearn 'balanced', 0<power<1 -> softened balancing.
    """
    if power < 0:
        raise ValueError("power must be non-negative")
    y = np.asarray(y_train)
    classes, counts = np.unique(y, return_counts=True)
    n, k = len(y), len(classes)
    return {c: float((n / (k * cnt)) ** power) for c, cnt in zip(classes, counts)}


def validation_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, float]:
    """Accuracy and macro / weighted precision-recall-F1 (zero_division=0)."""
    out = {"Val Accuracy": float(accuracy_score(y_true, y_pred))}
    for avg, tag in (("macro", "Macro"), ("weighted", "Weighted")):
        out[f"Val {tag} Prec"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        out[f"Val {tag} Rec"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        out[f"Val {tag} F1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    return out


def selection_key(row: Mapping[str, float]) -> Tuple[float, ...]:
    """Sort key (higher is better) following Macro F1 > Macro Recall > Accuracy > Weighted F1."""
    return tuple(round(float(row[m]), 6) for m in SELECTION_METRICS)


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return candidates sorted best-first using the validation selection order."""
    return df.sort_values(list(SELECTION_METRICS), ascending=False, kind="mergesort").reset_index(drop=True)
