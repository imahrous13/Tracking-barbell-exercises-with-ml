"""Feature engineering for IMU time series.

All transformers are fit on training data only and applied to test data,
to avoid leakage. Stateful objects (scalers, PCA, KMeans) are returned so
the caller can persist and reuse them for inference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["acc_r"] = np.sqrt(out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2)
    out["gyr_r"] = np.sqrt(out["gyr_x"] ** 2 + out["gyr_y"] ** 2 + out["gyr_z"] ** 2)
    return out


def lowpass_filter_df(df: pd.DataFrame, cols: list[str], fs: float, cutoff: float, order: int = 5) -> pd.DataFrame:
    out = df.copy()
    nyq = 0.5 * fs
    cut = cutoff / nyq
    b, a = butter(order, cut, btype="low")
    for col in cols:
        out[col] = filtfilt(b, a, out[col].values)
    return out


@dataclass
class FeaturePipeline:
    lowpass_cutoff: float = 1.3
    lowpass_order: int = 5
    pca_components: int = 3
    temporal_window: int = 5
    fft_window: int = 14
    kmeans_clusters: int = 5
    fs: float = 5.0
    fitted: bool = False
    pca_: PCA | None = None
    kmeans_: KMeans | None = None
    kmeans_cols_: list[str] = field(default_factory=lambda: ["acc_x", "acc_y", "acc_z"])

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        d = add_magnitude(df)
        # PCA fit on the six IMU axes (normalised by range, as in original).
        cols = SENSOR_COLS
        norm = d[cols].copy()
        for c in cols:
            rng = norm[c].max() - norm[c].min()
            if rng > 0:
                norm[c] = (norm[c] - norm[c].mean()) / rng
        self.pca_ = PCA(n_components=self.pca_components)
        self.pca_.fit(norm[cols])
        # KMeans on acc axes.
        self.kmeans_ = KMeans(n_clusters=self.kmeans_clusters, n_init=20, random_state=42)
        self.kmeans_.fit(d[self.kmeans_cols_])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline must be fit first")
        d = add_magnitude(df.copy())
        # Low-pass filter on the six axes + magnitudes.
        filter_cols = SENSOR_COLS + ["acc_r", "gyr_r"]
        d = lowpass_filter_df(d, filter_cols, fs=self.fs, cutoff=self.lowpass_cutoff, order=self.lowpass_order)

        # PCA
        cols = SENSOR_COLS
        norm = d[cols].copy()
        for c in cols:
            rng = norm[c].max() - norm[c].min()
            if rng > 0:
                norm[c] = (norm[c] - norm[c].mean()) / rng
        pca_vals = self.pca_.transform(norm[cols])
        for i in range(self.pca_components):
            d[f"pca_{i+1}"] = pca_vals[:, i]

        # Sum of squares (already have magnitudes); keep acc_r/gyr_r.

        # Temporal abstraction: rolling mean/std over window.
        predictor_cols = SENSOR_COLS + ["acc_r", "gyr_r"]
        for col in predictor_cols:
            d[f"{col}_temp_mean_ws_{self.temporal_window}"] = (
                d[col].rolling(self.temporal_window).mean()
            )
            d[f"{col}_temp_std_ws_{self.temporal_window}"] = (
                d[col].rolling(self.temporal_window).std()
            )

        # Frequency features via FFT over windows.
        d = self._add_frequency(d, predictor_cols)

        # Clustering.
        d["cluster"] = self.kmeans_.predict(d[self.kmeans_cols_])

        # Duration per set (seconds span).
        if "set" in d.columns:
            dur = d.groupby("set").apply(lambda g: (g.index[-1] - g.index[0]).total_seconds(), include_groups=False)
            d["duration"] = d["set"].map(dur)

        d = d.dropna().reset_index(drop=False)
        # keep index as column if it was datetime
        if "index" in d.columns and "epoch (ms)" not in d.columns:
            d = d.rename(columns={"index": "epoch (ms)"})
        return d

    def _add_frequency(self, d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        fs = self.fs
        ws = self.fft_window
        freqs = np.round((np.fft.rfftfreq(int(ws)) * fs), 3)
        out = d.copy().reset_index(drop=True)
        for col in cols:
            out[f"{col}_max_freq"] = np.nan
            out[f"{col}_freq_weighted"] = np.nan
            out[f"{col}_pse"] = np.nan
            for freq in freqs:
                out[f"{col}_freq_{freq}_Hz_ws_{ws}"] = np.nan

        for i in range(ws, len(out)):
            for col in cols:
                seg = out[col].iloc[i - ws: i + 1].values
                real = np.fft.rfft(seg, len(seg)).real
                for j, freq in enumerate(freqs):
                    out.loc[i, f"{col}_freq_{freq}_Hz_ws_{ws}"] = real[j]
                out.loc[i, f"{col}_max_freq"] = freqs[np.argmax(real)]
                denom = np.sum(real)
                out.loc[i, f"{col}_freq_weighted"] = (
                    float(np.sum(freqs * real) / denom) if denom != 0 else 0.0
                )
                psd = np.square(real) / float(len(real))
                psd_pdf = psd / np.sum(psd) if np.sum(psd) != 0 else psd
                pse = -np.sum(np.log(np.where(psd_pdf > 0, psd_pdf, 1.0)) * psd_pdf)
                out.loc[i, f"{col}_pse"] = pse
        # restore datetime index
        if isinstance(d.index, pd.DatetimeIndex):
            out.index = d.index[: len(out)] if len(out) == len(d) else out.index
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
