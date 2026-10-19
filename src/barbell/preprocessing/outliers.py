"""Outlier removal (Chauvenet + IQR + LOF) applied per (participant, label)."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import scipy.special

logger = logging.getLogger(__name__)


def mark_chauvenet(series: pd.Series, c: int = 2) -> pd.Series:
    """Return boolean mask (True = outlier) for a single series."""
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(False, index=series.index)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    n = len(s)
    criterion = 1.0 / (c * n)
    deviation = (series - mean).abs() / std
    # probability of observing a deviation this large under N(0,1)
    prob = 1.0 - scipy.special.erf(deviation / math.sqrt(2))
    mask = prob < criterion
    return mask.fillna(False)


def remove_outliers_chauvenet(
    df: pd.DataFrame, columns: list[str], c: int = 2, group_cols: list[str] = None
) -> pd.DataFrame:
    """Mark outliers as NaN (not drop) so downstream imputation handles them.

    Applied per group (default participant+label) to avoid flagging legitimate
    exercise-specific extremes.
    """
    if group_cols is None:
        group_cols = ["participant", "label"]
    out = df.copy()
    for col in columns:
        mask = pd.Series(False, index=df.index)
        for _, g in df.groupby(group_cols):
            m = mark_chauvenet(g[col], c=c)
            mask.loc[g.index] = m.loc[g.index]
        out.loc[mask, col] = np.nan
    logger.info("Chauvenet (c=%d): marked %d values across %s", c, int(out[columns].isna().sum().sum()) - int(df[columns].isna().sum().sum()), columns)
    return out


def impute_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        med = out[col].median()
        out[col] = out[col].fillna(med)
    return out
