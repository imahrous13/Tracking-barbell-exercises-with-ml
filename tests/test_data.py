"""Tests for data validation and ingestion."""
import numpy as np
import pandas as pd
import pytest

from barbell.data.validation import check_ranges, summarize, validate_schema


def make_frame():
    idx = pd.date_range("2020-01-01", periods=50, freq="200ms")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "acc_x": rng.normal(0, 1, 50), "acc_y": rng.normal(0, 1, 50),
        "acc_z": rng.normal(0, 1, 50), "gyr_x": rng.normal(0, 1, 50),
        "gyr_y": rng.normal(0, 1, 50), "gyr_z": rng.normal(0, 1, 50),
        "participant": ["A"] * 25 + ["B"] * 25,
        "label": ["bench"] * 50,
        "category": ["heavy"] * 50,
        "set": [1] * 50,
    }, index=idx)
    df.index.name = "epoch (ms)"
    return df


def test_validate_schema_ok():
    df = make_frame()
    validate_schema(df)


def test_validate_schema_missing():
    df = make_frame().drop(columns=["acc_z"])
    with pytest.raises(ValueError):
        validate_schema(df)


def test_check_ranges():
    df = make_frame()
    counts = check_ranges(df)
    assert all(v >= 0 for v in counts.values())


def test_summarize():
    df = make_frame()
    s = summarize(df)
    assert s["shape"] == (50, 10)
    assert set(s["participants"]) == {"A", "B"}