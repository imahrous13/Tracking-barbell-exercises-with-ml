import numpy as np
import pandas as pd

from barbell.features.engineering import FeaturePipeline, add_magnitude
from barbell.features.windowing import make_windows


def make_frame():
    idx = pd.date_range("2020-01-01", periods=60, freq="200ms")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "acc_x": rng.normal(0, 1, 60), "acc_y": rng.normal(0, 1, 60),
        "acc_z": rng.normal(0, 1, 60), "gyr_x": rng.normal(0, 1, 60),
        "gyr_y": rng.normal(0, 1, 60), "gyr_z": rng.normal(0, 1, 60),
        "participant": ["A"] * 60,
        "label": ["bench"] * 60,
        "category": ["heavy"] * 60,
        "set": [1] * 60,
    }, index=idx)
    df.index.name = "epoch (ms)"
    return df


def test_add_magnitude():
    df = make_frame()
    out = add_magnitude(df)
    assert "acc_r" in out.columns and "gyr_r" in out.columns
    assert (out["acc_r"] >= 0).all()


def test_feature_pipeline_fit_transform():
    df = make_frame()
    fp = FeaturePipeline(temporal_window=3, fft_window=8, kmeans_clusters=2, pca_components=2)
    out = fp.fit_transform(df)
    assert fp.fitted
    assert "pca_1" in out.columns
    assert "cluster" in out.columns
    assert len(out) > 0


def test_make_windows():
    df = make_frame()
    fp = FeaturePipeline(temporal_window=3, fft_window=8, kmeans_clusters=2, pca_components=2)
    featured = fp.fit_transform(df)
    w = make_windows(featured, window_size=8, step_size=4)
    assert len(w) > 0
    assert "label" in w.columns
    assert "participant" in w.columns
