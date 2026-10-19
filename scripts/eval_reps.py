"""Evaluate repetition counting."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import mean_absolute_error

from barbell.data.ingestion import validate_dataset
from barbell.models.reps import evaluate_reps
from barbell.utils.config import load_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

cfg = load_config("configs/default.yaml")
root = project_root(cfg)
interim = root / cfg["paths"]["interim_dir"]
reports = root / cfg["paths"]["reports_dir"]
reports.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(interim / "01_data_processed.pkl")
validate_dataset(df)

res = evaluate_reps(df)
res.to_csv(reports / "rep_counts.csv", index=False)
print(res.groupby(["label", "category"]).agg(
    n_sets=("set", "nunique"),
    mae=("abs_error", "mean"),
    mean_true=("true_reps", "mean"),
    mean_pred=("pred_reps", "mean"),
).round(3).to_string())
overall_mae = mean_absolute_error(res["true_reps"], res["pred_reps"])
print(f"\nOverall MAE: {overall_mae:.3f} over {len(res)} sets")
print(f"Exact-match accuracy: {(res['abs_error'] == 0).mean():.3f}")

# Per-exercise MAE
print("\nPer exercise MAE:")
print(res.groupby("label")["abs_error"].mean().round(3).to_string())