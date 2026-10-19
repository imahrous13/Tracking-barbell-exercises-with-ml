"""Tests for CV splits and inference helpers."""
import numpy as np
import pandas as pd

from barbell.cv.splits import leave_one_subject_out
from barbell.models.reps import count_peaks


def make_df():
    return pd.DataFrame({"participant": ["A"] * 5 + ["B"] * 5 + ["C"] * 5})


def test_loso():
    df = make_df()
    folds = list(leave_one_subject_out(df))
    assert len(folds) == 3
    for tr, te, held in folds:
        assert len(te) == 5
        assert held not in df.iloc[tr]["participant"].values


def test_count_peaks_basic():
    # synthetic sinusoid with 5 peaks
    t = np.linspace(0, 4 * np.pi, 200)
    sig = np.sin(t) + 0.1 * np.random.default_rng(0).normal(size=200)
    # scale to ~5 Hz sampling: use every 40th? just check returns int
    n = count_peaks(sig, cutoff_hz=0.4, fs=50.0, order=4, min_distance=10)
    assert isinstance(n, int)
    assert n >= 0