"""Model training, baselines, and evaluation."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_EXCLUDE = {"label", "participant", "set", "window_start", "category"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "majority": DummyClassifier(strategy="most_frequent", random_state=random_state),
        "stratified": DummyClassifier(strategy="stratified", random_state=random_state),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state, n_jobs=-1)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, random_state=random_state
        ),
    }


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def cross_validate_subjects(
    df: pd.DataFrame,
    models: dict[str, object] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Leave-one-subject-out CV. Returns a long DataFrame of per-fold metrics."""
    if models is None:
        models = build_models(random_state)
    feats = feature_columns(df)
    records = []
    from barbell.cv.splits import leave_one_subject_out

    for tr, te, held in leave_one_subject_out(df):
        Xtr, Xte = df.iloc[tr][feats], df.iloc[te][feats]
        ytr, yte = df.iloc[tr]["label"], df.iloc[te]["label"]
        for name, model in models.items():
            m = _clone(model)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            metrics = evaluate_predictions(yte, pred)
            metrics.update({"model": name, "held_out_subject": held, "n_test": len(yte)})
            records.append(metrics)
    return pd.DataFrame(records)


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def fit_final(
    train_df: pd.DataFrame, model_name: str = "hist_gradient_boosting", random_state: int = 42
):
    models = build_models(random_state)
    if model_name not in models:
        raise KeyError(model_name)
    feats = feature_columns(train_df)
    m = _clone(models[model_name])
    m.fit(train_df[feats], train_df["label"])
    return m, feats


def confusion(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    return confusion_matrix(y_true, y_pred, labels=labels), labels