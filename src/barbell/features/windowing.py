"""Windowing helpers for time-series classification."""
from __future__ import annotations

import pandas as pd


def make_windows(
    df: pd.DataFrame,
    window_size: int = 14,
    step_size: int = 7,
    feature_cols: list | None = None,
    label_col: str = "label",
    subject_col: str = "participant",
    set_col: str = "set",
) -> pd.DataFrame:
    """Slice a (possibly feature-enriched) frame into fixed-length windows.

    Each window inherits the majority label and the subject/set of its rows.
    Returns one row per window with aggregated (mean) features plus metadata.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in (label_col, subject_col, set_col, "category", "window_start")
                        and not pd.api.types.is_datetime64_any_dtype(df[c])]

    rows = []
    # Window within each set to avoid crossing set boundaries.
    for set_id, g in df.groupby(set_col, sort=False):
        g = g.sort_index()
        vals = g[feature_cols].to_numpy()
        labels = g[label_col].to_numpy()
        subjects = g[subject_col].to_numpy()
        n = len(g)
        if n < window_size:
            continue
        for start in range(0, n - window_size + 1, step_size):
            sl = slice(start, start + window_size)
            window_features = vals[sl].mean(axis=0)
            # majority label
            wlabels = labels[sl]
            maj = pd.Series(wlabels).mode().iloc[0]
            row = dict(zip(feature_cols, window_features))
            row[label_col] = maj
            row[subject_col] = subjects[0]
            row[set_col] = set_id
            row["window_start"] = g.index[start]
            rows.append(row)
    return pd.DataFrame(rows)