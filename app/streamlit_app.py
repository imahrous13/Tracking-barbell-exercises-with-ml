"""Streamlit demo: pick a recorded set, see predicted exercise and rep count."""
from __future__ import annotations

import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
DATA_PATH = ROOT / "data" / "interim" / "01_data_processed.pkl"


@st.cache_resource
def load_artifacts():
    art = joblib.load(MODEL_PATH)
    df = pd.read_pickle(DATA_PATH)
    return art, df


def main():
    st.set_page_config(page_title="Barbell Tracker", page_icon="🏋️", layout="wide")
    st.title("Barbell exercise classifier + rep counter")
    st.caption("Subject-held-out evaluation on MetaMotion IMU data (5 participants, 6 exercises).")

    art, df = load_artifacts()
    model = art["model"]
    fp = art["feature_pipeline"]

    from barbell.inference.predict import predict_exercise, predict_reps

    participants = sorted(df["participant"].unique())
    p = st.sidebar.selectbox("Participant", participants)
    sets = sorted(df[df["participant"] == p]["set"].unique())
    sid = st.sidebar.selectbox("Set", sets)

    frame = df[(df["participant"] == p) & (df["set"] == sid)].copy().sort_index()
    true_label = frame["label"].iloc[0]
    category = frame["category"].iloc[0]

    st.subheader(f"Set {sid} — ground truth: {true_label} ({category})")
    st.write(f"{len(frame)} samples at 5 Hz ({len(frame)/5:.1f} s)")

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(frame[["acc_x", "acc_y", "acc_z"]])
        st.caption("Accelerometer (g)")
    with col2:
        st.line_chart(frame[["gyr_x", "gyr_y", "gyr_z"]])
        st.caption("Gyroscope (deg/s)")

    windows = predict_exercise(model, fp, frame)
    pred_majority = windows["predicted"].mode().iloc[0]
    reps = predict_reps(frame, pred_majority)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted exercise", pred_majority)
    c2.metric("Counted reps", reps)
    c3.metric("Windows", len(windows))

    st.subheader("Per-window predictions")
    st.bar_chart(windows["predicted"].value_counts())

    st.subheader("Model performance (held-out subject A)")
    st.write("Window accuracy 97.9%, set-level accuracy 100% (29 sets). See reports/metrics.json.")


if __name__ == "__main__":
    main()