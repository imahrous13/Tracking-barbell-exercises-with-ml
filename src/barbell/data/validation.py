"""Data validation helpers (schema, ranges, duplicates)."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
    "participant", "label", "category", "set",
]

SENSOR_RANGES = {
    "acc_x": (-8.0, 8.0), "acc_y": (-8.0, 8.0), "acc_z": (-8.0, 8.0),
    "gyr_x": (-50.0, 50.0), "gyr_y": (-50.0, 50.0), "gyr_z": (-50.0, 50.0),
}


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Empty dataframe")


def check_ranges(df: pd.DataFrame) -> dict[str, int]:
    """Return count of out-of-range values per sensor column."""
    counts: dict[str, int] = {}
    for col, (lo, hi) in SENSOR_RANGES.items():
        if col in df.columns:
            counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    return counts


def summarize(df: pd.DataFrame) -> dict[str, object]:
    return {
        "shape": df.shape,
        "participants": sorted(df["participant"].unique().tolist()),
        "labels": df["label"].value_counts().to_dict(),
        "categories": df["category"].value_counts().to_dict(),
        "n_sets": int(df["set"].nunique()),
        "date_start": str(df.index.min()),
        "date_end": str(df.index.max()),
        "missing": int(df.isna().sum().sum()),
        "range_violations": check_ranges(df),
    }