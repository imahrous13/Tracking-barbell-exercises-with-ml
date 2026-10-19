"""Leakage-safe cross-validation splits for subject-held-out evaluation."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def subject_groups(df: pd.DataFrame, subject_col: str = "participant") -> np.ndarray:
    return df[subject_col].to_numpy()


def leave_one_subject_out(
    df: pd.DataFrame, subject_col: str = "participant"
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Yield (train_idx, test_idx, held_out_subject) for each subject."""
    subjects = sorted(df[subject_col].unique())
    for s in subjects:
        test_idx = np.where(df[subject_col].to_numpy() == s)[0]
        train_idx = np.where(df[subject_col].to_numpy() != s)[0]
        yield train_idx, test_idx, s


def group_kfold_indices(
    df: pd.DataFrame, n_splits: int = 4, subject_col: str = "participant"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[subject_col].to_numpy()
    X = np.zeros((len(df), 1))
    yield from gkf.split(X, groups=groups)