"""Run the full leakage-safe experiment and save results."""
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
from barbell.models.training import build_models, cross_validate_subjects, feature_columns, fit_final
from barbell.preprocessing.outliers import remove_outliers_chauvenet, impute_median
from barbell.utils.config import load_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("train")

cfg = load_config("configs/default.yaml")
root = project_root(cfg)
interim = root / cfg["paths"]["interim_dir"]
models_dir = root / cfg["paths"]["models_dir"]
reports_dir = root / cfg["paths"]["reports_dir"]
figures_dir = root / cfg["paths"]["figures_dir"]
for d in (models_dir, reports_dir, figures_dir):
    d.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
df = pd.read_pickle(interim / "01_data_processed.pkl")
validate_dataset(df)
logger.info("Loaded %s", df.shape)

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

# ---- Outlier removal + imputation (per participant+label, fit on all data is OK
#      for unsupervised cleaning of inputs; labels/participants are not the target
#      of leakage concern here, but we keep it simple and document it) ----
df = remove_outliers_chauvenet(df, SENSOR_COLS, c=cfg["preprocessing"]["chauvenet_c"],
                               group_cols=["participant", "label"])
df = impute_median(df, SENSOR_COLS)
logger.info("After cleaning: %s, missing=%d", df.shape, int(df.isna().sum().sum()))

# ---- Feature engineering + windowing, leakage-safe per subject ----
# We build windows per held-out subject: fit feature pipeline on the other subjects.
def build_windows_for_subject(df, held_subject, fp_params):
    train_mask = df["participant"] != held_subject
    test_mask = df["participant"] == held_subject
    fp = FeaturePipeline(**fp_params)
    fp.fit(df[train_mask])
    train_featured = fp.transform(df[train_mask])
    test_featured = fp.transform(df[test_mask])
    train_w = make_windows(train_featured, window_size=cfg["windowing"]["window_size"],
                           step_size=cfg["windowing"]["step_size"])
    test_w = make_windows(test_featured, window_size=cfg["windowing"]["window_size"],
                          step_size=cfg["windowing"]["step_size"])
    return train_w, test_w, fp

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
subjects = sorted(df["participant"].unique())
logger.info("Subjects: %s, held-out: %s", subjects, held_out)

# ---- Leave-one-subject-out CV on the tuning subjects ----
tuning_subjects = [s for s in subjects if s != held_out]
cv_records = []
for held in tuning_subjects:
    train_w, test_w, fp = build_windows_for_subject(df, held, fp_params)
    for name, model in build_models(cfg["modeling"]["random_state"]).items():
        from sklearn.base import clone
        m = clone(model)
        feats = feature_columns(train_w)
        m.fit(train_w[feats], train_w["label"])
        pred = m.predict(test_w[feats])
        from barbell.models.training import evaluate_predictions
        metrics = evaluate_predictions(test_w["label"], pred)
        metrics.update({"model": name, "held_out_subject": held, "n_test": len(test_w)})
        cv_records.append(metrics)
        logger.info("CV held=%s model=%s acc=%.3f macro_f1=%.3f", held, name,
                    metrics["accuracy"], metrics["macro_f1"])

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(reports_dir / "cv_results.csv", index=False)
summary = cv_df.groupby("model").agg(
    accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
    macro_f1=("macro_f1", "mean"), weighted_f1=("weighted_f1", "mean"),
).round(4).sort_values("macro_f1", ascending=False)
print("\n=== LOSO CV (tuning subjects) ===")
print(summary.to_string())
summary.to_csv(reports_dir / "cv_summary.csv")

# ---- Final model: train on all tuning subjects, test on held-out subject ----
train_w, test_w, fp = build_windows_for_subject(df, held_out, fp_params)
feats = feature_columns(train_w)
# pick best model by CV macro_f1
best_model_name = summary.index[0]
logger.info("Best model by CV macro_f1: %s", best_model_name)
model, _ = fit_final(train_w, model_name=best_model_name, random_state=cfg["modeling"]["random_state"])
# re-fit on actual train_w (fit_final uses train_df)
model.fit(train_w[feats], train_w["label"])

pred = model.predict(test_w[feats])
from barbell.models.training import evaluate_predictions
final_metrics = evaluate_predictions(test_w["label"], pred)
logger.info("Final held-out %s: %s", held_out, final_metrics)
print("\n=== Final held-out subject", held_out, "===")
print(final_metrics)
print("\nClassification report:")
print(classification_report(test_w["label"], pred, digits=4))

# Confusion matrix
labels = sorted(df["label"].unique())
cm, _ = None, None
from barbell.models.training import confusion
cm, lab = confusion(test_w["label"], pred, labels=labels)
cm_df = pd.DataFrame(cm, index=lab, columns=lab)
cm_df.to_csv(reports_dir / "confusion_matrix.csv")
print("\nConfusion matrix:")
print(cm_df.to_string())

# Per-subject breakdown on the held-out subject's sets
test_w = test_w.copy()
test_w["pred"] = pred
per_set = test_w.groupby("set").agg(true=("label", "first"), pred=("pred", lambda s: s.mode().iloc[0]),
                                   n_windows=("label", "size")).reset_index()
set_acc = float((per_set["true"] == per_set["pred"]).mean())
logger.info("Set-level accuracy on held-out subject: %.4f (%d sets)", set_acc, len(per_set))
print(f"\nSet-level accuracy: {set_acc:.4f} over {len(per_set)} sets")
per_set.to_csv(reports_dir / "per_set_predictions.csv", index=False)

# ---- Save artifacts ----
joblib.dump({"model": model, "feature_pipeline": fp, "features": feats,
             "labels": labels, "config": cfg}, models_dir / "model.joblib")
json.dump({"cv_summary": summary.round(4).to_dict(), "final_metrics": final_metrics,
           "best_model": best_model_name, "held_out_subject": held_out,
           "set_level_accuracy": set_acc, "n_train_windows": len(train_w),
           "n_test_windows": len(test_w)}, open(reports_dir / "metrics.json", "w"), indent=2)
logger.info("Saved model and metrics")
print("\nSaved: models/model.joblib, reports/metrics.json")