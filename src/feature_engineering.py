"""
feature_engineering.py

Converts raw OBD-II readings into the feature set the model is trained on.
Two usage modes:

  - batch:  build_features_df(df)      -> used during offline training on a
            full drive-cycle DataFrame (vectorized, uses pandas rolling/diff).
  - online: FeatureBuffer.push(reading) -> used during live inference, where
            we only have a short rolling history (deque) rather than a full
            DataFrame, and must compute the same derived features
            incrementally and consistently with the batch path.

Keeping both paths producing IDENTICAL feature definitions is the most
common source of train/serve skew bugs in projects like this, so the
derived-feature formulas live in one place (`_derived_from_window`) and
both paths call it.
"""

from __future__ import annotations
from collections import deque
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

RAW_FEATURES = ["RPM", "SPEED", "THROTTLE_POS", "ENGINE_LOAD", "MAF", "MAP", "IAT", "COOLANT_TEMP"]

ENGINEERED_FEATURES = [
    "RPM_x_LOAD", "MAP_over_IAT", "SPEED_ACCEL", "RPM_DELTA",
    "THROTTLE_DELTA", "MAF_ROLL_MEAN", "MAF_ROLL_STD", "LOAD_ROLL_MEAN",
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

ROLL_WINDOW = 5  # samples (~2.5s at 2Hz) used for rolling stats / derivatives


def _safe_div(a, b, default=0.0):
    b = np.where(np.asarray(b) == 0, np.nan, b)
    out = np.asarray(a) / b
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default)


def build_features_df(df: pd.DataFrame, dt_seconds: float = 0.5) -> pd.DataFrame:
    """Vectorized feature engineering for a full (already-ordered-in-time) DataFrame."""
    out = df.copy()

    out["RPM_x_LOAD"] = out["RPM"] * out["ENGINE_LOAD"] / 1000.0
    out["MAP_over_IAT"] = _safe_div(out["MAP"], out["IAT"] + 273.15)

    speed_diff = out["SPEED"].diff().fillna(0.0)
    out["SPEED_ACCEL"] = speed_diff / dt_seconds

    out["RPM_DELTA"] = out["RPM"].diff().fillna(0.0)
    out["THROTTLE_DELTA"] = out["THROTTLE_POS"].diff().fillna(0.0)

    out["MAF_ROLL_MEAN"] = out["MAF"].rolling(ROLL_WINDOW, min_periods=1).mean()
    out["MAF_ROLL_STD"] = out["MAF"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0.0)
    out["LOAD_ROLL_MEAN"] = out["ENGINE_LOAD"].rolling(ROLL_WINDOW, min_periods=1).mean()

    return out[ALL_FEATURES]


class FeatureBuffer:
    """
    Maintains a short rolling history of raw readings for live inference and
    produces a single feature row (dict) per push(), matching build_features_df.
    """

    def __init__(self, window: int = ROLL_WINDOW, dt_seconds: float = 0.5):
        self.window = window
        self.dt_seconds = dt_seconds
        self._hist: deque = deque(maxlen=window)
        self._prev: Optional[Dict[str, float]] = None

    def push(self, reading: Dict[str, Optional[float]]) -> Optional[Dict[str, float]]:
        """
        reading: dict with keys RAW_FEATURES (missing/None values are filled
        with the last known good value, or 0 if none yet available).
        Returns a feature dict ready for model_inference, or None only if
        every field so far has been missing (can't impute anything).
        """
        clean = {}
        for k in RAW_FEATURES:
            v = reading.get(k)
            if v is None:
                v = self._prev[k] if self._prev is not None and self._prev.get(k) is not None else 0.0
            clean[k] = float(v)

        self._hist.append(clean)
        prev = self._prev or clean

        maf_vals = [h["MAF"] for h in self._hist]
        load_vals = [h["ENGINE_LOAD"] for h in self._hist]

        features = dict(clean)  # raw features pass through
        features["RPM_x_LOAD"] = clean["RPM"] * clean["ENGINE_LOAD"] / 1000.0
        features["MAP_over_IAT"] = clean["MAP"] / (clean["IAT"] + 273.15) if (clean["IAT"] + 273.15) != 0 else 0.0
        features["SPEED_ACCEL"] = (clean["SPEED"] - prev["SPEED"]) / self.dt_seconds
        features["RPM_DELTA"] = clean["RPM"] - prev["RPM"]
        features["THROTTLE_DELTA"] = clean["THROTTLE_POS"] - prev["THROTTLE_POS"]
        features["MAF_ROLL_MEAN"] = float(np.mean(maf_vals))
        features["MAF_ROLL_STD"] = float(np.std(maf_vals)) if len(maf_vals) > 1 else 0.0
        features["LOAD_ROLL_MEAN"] = float(np.mean(load_vals))

        self._prev = clean
        return {k: features[k] for k in ALL_FEATURES}
