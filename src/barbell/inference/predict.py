"""Inference: predict exercise + count reps from a raw or processed frame."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from barbell.features.engineering import FeaturePipeline
from barbell.features.windowing import make_windows
from barbell.models.reps import count_peaks

logger = logging.getLogger(__name__)


def load_model(path: str | Path):
    return joblib.load(path)


def predict_exercise(model, feature_pipeline: FeaturePipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Given a processed (resampled) frame, return per-window predictions."""
    featured = feature_pipeline.transform(frame)
    windows = make_windows(featured, window_size=14, step_size=7)
    feats = [c for c in windows.columns if c not in {"label", "participant", "set", "window_start", "category"}]
    preds = model.predict(windows[feats])
    windows = windows.copy()
    windows["predicted"] = preds
    return windows


def predict_reps(frame: pd.DataFrame, label: str, cutoffs: dict[str, float] | None = None) -> int:
    if cutoffs is None:
        cutoffs = {"bench": 0.4, "squat": 0.35, "ohp": 0.35, "dead": 0.5, "row": 0.65}
    frame = frame.copy()
    frame["acc_r"] = np.sqrt(frame["acc_x"] ** 2 + frame["acc_y"] ** 2 + frame["acc_z"] ** 2)
    cutoff = cutoffs.get(label, 0.4)
    if label == "row":
        sig = frame["gyr_x"].to_numpy()
    else:
        sig = frame["acc_r"].to_numpy()
    return count_peaks(sig, cutoff_hz=cutoff, fs=5.0)