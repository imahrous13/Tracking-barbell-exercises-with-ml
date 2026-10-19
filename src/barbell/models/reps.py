"""Repetition counting from a filtered accelerometer magnitude signal."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, butter, filtfilt

logger = logging.getLogger(__name__)


def count_peaks(signal: np.ndarray, cutoff_hz: float = 0.4, fs: float = 5.0, order: int = 10,
                min_distance: int = 5) -> int:
    """Low-pass filter then count local maxima with a minimum distance."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    filt = filtfilt(b, a, signal)
    peaks = argrelextrema(filt, np.greater)[0]
    # enforce minimum distance
    if len(peaks) == 0:
        return 0
    kept = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] >= min_distance:
            kept.append(p)
    return len(kept)


def evaluate_reps(
    df: pd.DataFrame, cutoffs: dict[str, float] | None = None, fs: float = 5.0
) -> pd.DataFrame:
    """Count reps per set and compare to the known count (5 heavy-ish -> 5, medium -> 10).

    Ground truth in this dataset: 'heavy' sets have 5 reps, 'medium' sets 10 reps.
    Returns a per-set DataFrame with predicted vs. true reps.
    """
    if cutoffs is None:
        cutoffs = {"bench": 0.4, "squat": 0.35, "ohp": 0.35, "dead": 0.5, "row": 0.65}
    df = df.copy()
    df["acc_r"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    records = []
    for set_id, g in df.groupby("set"):
        g = g.sort_index()
        label = g["label"].iloc[0]
        if label == "rest":
            continue
        category = g["category"].iloc[0]
        true_reps = 5 if category == "heavy" else 10
        # signal choice: row uses gyr_x magnitude proxy
        if label == "row":
            sig = g["gyr_x"].to_numpy() if "gyr_x" in g.columns else g["acc_r"].to_numpy()
        else:
            sig = g["acc_r"].to_numpy()
        cutoff = cutoffs.get(label, 0.4)
        pred = count_peaks(sig, cutoff_hz=cutoff, fs=fs)
        records.append({
            "set": set_id, "label": label, "category": category,
            "participant": g["participant"].iloc[0],
            "true_reps": true_reps, "pred_reps": pred,
            "abs_error": abs(true_reps - pred),
        })
    return pd.DataFrame(records)