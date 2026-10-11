"""
model_inference.py

Loads the trained pipeline (scaler + GradientBoostingRegressor) and exposes
a simple predict(reading) API that internally maintains the rolling feature
buffer needed for the engineered features (so callers just hand it raw OBD
readings one at a time, as they arrive).
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict

import joblib
import pandas as pd

from feature_engineering import FeatureBuffer, ALL_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class FuelRatePredictor:
    def __init__(self, model_path: Optional[str] = None, dt_seconds: float = 0.5):
        model_path = model_path or os.path.join(MODELS_DIR, "fuel_rate_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run `python train_model.py` first."
            )
        self.model = joblib.load(model_path)
        self.buffer = FeatureBuffer(dt_seconds=dt_seconds)

        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def predict(self, reading: Dict[str, Optional[float]]) -> float:
        """reading: dict with raw PID keys (RPM, SPEED, THROTTLE_POS, ENGINE_LOAD,
        MAF, MAP, IAT, COOLANT_TEMP). Missing values are imputed from history.
        Returns predicted fuel rate in L/h."""
        features = self.buffer.push(reading)
        row = pd.DataFrame([features])[ALL_FEATURES]
        pred = float(self.model.predict(row)[0])
        return max(pred, 0.0)  # fuel rate can't be negative

    def reset(self):
        self.buffer = FeatureBuffer(dt_seconds=self.buffer.dt_seconds)
