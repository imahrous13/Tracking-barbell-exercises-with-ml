"""Data ingestion: parse raw MetaMotion CSVs into a single resampled DataFrame."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ACCEL_SUFFIX = "Accelerometer"
GYRO_SUFFIX = "Gyroscope"


def _parse_filename(name: str) -> tuple[str, str, str]:
    """Return (participant, label, category) from a MetaMotion filename."""
    stem = Path(name).name
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {name}")
    participant = parts[0]
    label = parts[1]
    category = parts[2].split("_")[0]
    # Collapse set variants (heavy2, medium3, ...) to their base category.
    category = category.rstrip("0123456789")
    return participant, label, category


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    participant, label, category = _parse_filename(path.name)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    return df


def load_raw_files(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all accelerometer and gyroscope CSVs.

    Returns two DataFrames (accel, gyro) with metadata columns attached.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    accel_files = [f for f in files if ACCEL_SUFFIX in f.name]
    gyro_files = [f for f in files if GYRO_SUFFIX in f.name]
    logger.info("Found %d accelerometer and %d gyroscope files", len(accel_files), len(gyro_files))

    acc_df = pd.concat((_read_one(f) for f in accel_files), ignore_index=True)
    gyr_df = pd.concat((_read_one(f) for f in gyro_files), ignore_index=True)

    # Assign a per-sensor set id so paired recordings can be traced.
    acc_df = _assign_set_ids(acc_df, accel_files)
    gyr_df = _assign_set_ids(gyr_df, gyro_files)
    return acc_df, gyr_df


def _assign_set_ids(df: pd.DataFrame, files: list[Path]) -> pd.DataFrame:
    # Set ids are assigned in file order; each file is one recording session.
    # We recompute from the concatenated frame length per file is not available
    # after concat, so instead number rows by file index using a helper column.
    return df


def resample_to_common_rate(
    acc_df: pd.DataFrame, gyr_df: pd.DataFrame, resample_ms: int = 200
) -> pd.DataFrame:
    """Merge accel + gyro and resample each calendar day to a common rate.

    Accelerometer is 12.5 Hz, gyroscope 25 Hz. Both are downsampled to 1/resample_ms
    Hz (default 5 Hz) using mean aggregation, which acts as a light anti-aliasing
    filter and aligns the two streams on a shared time index.
    """
    # Convert epoch ms to datetime index.
    acc = acc_df.copy()
    gyr = gyr_df.copy()
    acc.index = pd.to_datetime(acc["epoch (ms)"], unit="ms")
    gyr.index = pd.to_datetime(gyr["epoch (ms)"], unit="ms")

    for frame in (acc, gyr):
        for col in ("epoch (ms)", "time (01:00)", "elapsed (s)"):
            if col in frame.columns:
                del frame[col]

    # Standardise column names.
    acc = acc.rename(columns={"x-axis (g)": "acc_x", "y-axis (g)": "acc_y", "z-axis (g)": "acc_z"})
    gyr = gyr.rename(columns={"x-axis (deg/s)": "gyr_x", "y-axis (deg/s)": "gyr_y", "z-axis (deg/s)": "gyr_z"})

    # Keep only the six sensor channels + metadata.
    acc_cols = ["acc_x", "acc_y", "acc_z", "participant", "label", "category", "set"]
    gyr_cols = ["gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"]
    # 'set' may not exist yet; add a placeholder grouping by file order is handled upstream.
    # For safety, ensure 'set' exists.
    for frame, cols in ((acc, acc_cols), (gyr, gyr_cols)):
        for c in cols:
            if c not in frame.columns and c != "set":
                raise KeyError(f"Missing column {c}")

    if "set" not in acc.columns:
        acc["set"] = 0
    if "set" not in gyr.columns:
        gyr["set"] = 0

    acc = acc[["acc_x", "acc_y", "acc_z", "participant", "label", "category", "set"]]
    gyr = gyr[["gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"]]

    merged = pd.concat([acc.iloc[:, :3], gyr], axis=1)
    merged.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
                      "participant", "label", "category", "set"]

    sampling = {
        "acc_x": "mean", "acc_y": "mean", "acc_z": "mean",
        "gyr_x": "mean", "gyr_y": "mean", "gyr_z": "mean",
        "participant": "last", "label": "last", "category": "last", "set": "last",
    }

    rule = f"{resample_ms}ms"
    days = [g for _, g in merged.groupby(pd.Grouper(freq="D"))]
    resampled = pd.concat([d.resample(rule=rule).agg(sampling).dropna() for d in days])
    resampled["set"] = resampled["set"].astype(int)
    resampled.index.name = "epoch (ms)"
    return resampled


def build_dataset(raw_dir: str | Path, resample_ms: int = 200) -> pd.DataFrame:
    """End-to-end ingestion. Returns the resampled DataFrame."""
    acc_df, gyr_df = load_raw_files(raw_dir)
    # Assign set ids by file order: each accel/gyro file is one set.
    # Re-read to attach set ids deterministically.
    acc_df, gyr_df = _attach_set_ids(raw_dir)
    df = resample_to_common_rate(acc_df, gyr_df, resample_ms=resample_ms)
    logger.info("Built dataset: %s", df.shape)
    return df


def _attach_set_ids(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    accel_files = [f for f in files if ACCEL_SUFFIX in f.name]
    gyro_files = [f for f in files if GYRO_SUFFIX in f.name]

    acc_frames = []
    for i, f in enumerate(accel_files, start=1):
        d = pd.read_csv(f)
        participant, label, category = _parse_filename(f.name)
        d["participant"] = participant
        d["label"] = label
        d["category"] = category
        d["set"] = i
        acc_frames.append(d)
    gyr_frames = []
    for i, f in enumerate(gyro_files, start=1):
        d = pd.read_csv(f)
        participant, label, category = _parse_filename(f.name)
        d["participant"] = participant
        d["label"] = label
        d["category"] = category
        d["set"] = i
        gyr_frames.append(d)

    acc_df = pd.concat(acc_frames, ignore_index=True)
    gyr_df = pd.concat(gyr_frames, ignore_index=True)
    return acc_df, gyr_df


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise if the dataset violates basic invariants."""
    required = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
                "participant", "label", "category", "set"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.isna().any().any():
        n = int(df.isna().sum().sum())
        raise ValueError(f"Dataset contains {n} missing values")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df["label"].nunique() < 2:
        raise ValueError("Need at least 2 labels")
    if df["participant"].nunique() < 2:
        raise ValueError("Need at least 2 participants for subject-held-out eval")
    logger.info("Dataset validation passed: %s", df.shape)
