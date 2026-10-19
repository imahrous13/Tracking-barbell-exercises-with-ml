"""Final held-out evaluation on subject A."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from barbell.data.ingestion import validate_dataset
from barbell.features.engineering import FeaturePipeline
from barbell.features.windowing import make_windows
from barbell.models.training import build_models, feature_columns, fit_final, evaluate_predictions, confusion
from barbell.preprocessing.outliers import remove_outliers_chauvenet, impute_median
from barbell.utils.config import load_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("final")

cfg = load_config("configs/default.yaml")
root = project_root(cfg)
interim = root / cfg["paths"]["interim_dir"]
models_dir = root / cfg["paths"]["models_dir"]
reports_dir = root / cfg["paths"]["reports_dir"]
figures_dir = root / cfg["paths"]["figures_dir"]
for d in (models_dir, reports_dir, figures_dir):
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(interim / "01_data_processed.pkl")
validate_dataset(df)
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
df = remove_outliers_chauvenet(df, SENSOR_COLS, c=cfg["preprocessing"]["chauvenet_c"],
                               group_cols=["participant", "label"])
df = impute_median(df, SENSOR_COLS)

fp_params = dict(
    lowpass_cutoff=cfg["features"]["lowpass_cutoff_hz"],
    lowpass_order=cfg["features"]["lowpass_order"],
    pca_components=cfg["features"]["pca_components"],
    temporal_window=cfg["features"]["temporal_window"],
    fft_window=cfg["features"]["fft_window"],
    kmeans_clusters=cfg["features"]["kmeans_clusters"],
    fs=1000.0 / cfg["data"]["resample_ms"],
)

held_out = cfg["modeling"]["held_out_subject"]
train_mask = df["participant"] != held_out
test_mask = df["participant"] == held_out

fp = FeaturePipeline(**fp_params)
fp.fit(df[train_mask])
train_featured = fp.transform(df[train_mask])
test_featured = fp.transform(df[test_mask])
train_w = make_windows(train_featured, window_size=cfg["windowing"]["window_size"],
                       step_size=cfg["windowing"]["step_size"])
test_w = make_windows(test_featured, window_size=cfg["windowing"]["window_size"],
                      step_size=cfg["windowing"]["step_size"])
logger.info("train windows %d, test windows %d", len(train_w), len(test_w))

feats = feature_columns(train_w)
# best model from CV
cv_summary = pd.read_csv(reports_dir / "cv_summary.csv", index_col=0)
best_model_name = cv_summary["macro_f1"].idxmax()
logger.info("Best model: %s", best_model_name)

model, _ = fit_final(train_w, model_name=best_model_name, random_state=cfg["modeling"]["random_state"])
model.fit(train_w[feats], train_w["label"])
pred = model.predict(test_w[feats])

final_metrics = evaluate_predictions(test_w["label"], pred)
logger.info("Final held-out %s: %s", held_out, final_metrics)
print("\n=== Final held-out subject", held_out, "===")
print(final_metrics)
print("\nClassification report:")
print(classification_report(test_w["label"], pred, digits=4))

labels = sorted(df["label"].unique())
cm, lab = confusion(test_w["label"], pred, labels=labels)
cm_df = pd.DataFrame(cm, index=lab, columns=lab)
cm_df.to_csv(reports_dir / "confusion_matrix.csv")
print("\nConfusion matrix:")
print(cm_df.to_string())

test_w = test_w.copy()
test_w["pred"] = pred
per_set = test_w.groupby("set").agg(
    true=("label", "first"),
    pred=("pred", lambda s: s.mode().iloc[0]),
    n_windows=("label", "size"),
).reset_index()
set_acc = float((per_set["true"] == per_set["pred"]).mean())
logger.info("Set-level accuracy: %.4f (%d sets)", set_acc, len(per_set))
print(f"\nSet-level accuracy: {set_acc:.4f} over {len(per_set)} sets")
per_set.to_csv(reports_dir / "per_set_predictions.csv", index=False)

joblib.dump({"model": model, "feature_pipeline": fp, "features": feats,
             "labels": labels, "config": cfg}, models_dir / "model.joblib")
import json
cv_summary_dict = cv_summary.round(4).to_dict()
metrics = {"cv_summary": cv_summary_dict, "final_metrics": final_metrics,
           "best_model": best_model_name, "held_out_subject": held_out,
           "set_level_accuracy": set_acc, "n_train_windows": len(train_w),
           "n_test_windows": len(test_w)}
with open(reports_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
logger.info("Saved model and metrics")
print("\nSaved: models/model.joblib, reports/metrics.json")