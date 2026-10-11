"""
train_model.py

Trains a gradient-boosted regression model to predict instantaneous fuel
rate (L/h) from OBD-II engine parameters + engineered features.

Usage:
    python train_model.py                      # trains on simulated drive cycles (demo)
    python train_model.py --csv my_log.csv      # trains on a real logged CSV instead

Expected CSV columns (if --csv is used): RPM, SPEED, THROTTLE_POS,
ENGINE_LOAD, MAF, MAP, IAT, COOLANT_TEMP, FUEL_RATE_REF_LPH, plus a
`timestamp` column or evenly-spaced rows at a known --hz.

Outputs:
    models/fuel_rate_model.joblib   - trained sklearn pipeline (scaler + model)
    models/model_metadata.json      - feature list, metrics, training config
    models/training_report.png      - predicted vs measured scatter + residuals
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import build_features_df, ALL_FEATURES
from data_simulator import generate_drive_cycle

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_training_data(csv_path: str | None, hz: float, n_cycles: int) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
        missing = [c for c in ["RPM", "SPEED", "THROTTLE_POS", "ENGINE_LOAD",
                                "MAF", "MAP", "IAT", "COOLANT_TEMP", "FUEL_RATE_REF_LPH"]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        return df

    # Demo mode: stitch together several distinct synthetic drive cycles
    # (different seeds = different "trips") so the model sees varied conditions.
    frames = []
    for seed in range(n_cycles):
        frames.append(generate_drive_cycle(duration_s=900, hz=hz, seed=100 + seed))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="path to a real logged CSV (optional)")
    ap.add_argument("--hz", type=float, default=2.0, help="sample rate of the data")
    ap.add_argument("--n-cycles", type=int, default=6, help="number of synthetic trips (demo mode only)")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[1/5] Loading training data (csv={args.csv!r})...")
    raw_df = load_training_data(args.csv, args.hz, args.n_cycles)
    print(f"      {len(raw_df)} raw samples loaded")

    print("[2/5] Engineering features...")
    X = build_features_df(raw_df, dt_seconds=1.0 / args.hz)
    y = raw_df["FUEL_RATE_REF_LPH"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Drop the (few) rows with NaNs from rolling/diff warmup
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=0, shuffle=True
    )
    print(f"      train={len(X_train)}  test={len(X_test)}")

    print("[3/5] Training GradientBoostingRegressor...")
    t0 = time.time()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            random_state=0,
        )),
    ])
    model.fit(X_train, y_train)
    print(f"      trained in {time.time() - t0:.1f}s")

    print("[4/5] Evaluating...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 0.1, None))) * 100)
    print(f"      MAE={mae:.3f} L/h   RMSE={rmse:.3f} L/h   R2={r2:.4f}   MAPE={mape:.1f}%")

    # Baseline comparison: naive MAF-only linear estimate (what you'd get with
    # NO machine learning, just the physics formula) -- shows the model's lift.
    baseline_pred = raw_df.loc[X_test.index, "MAF"] / 14.7 * 3600.0 / 750.0
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print(f"      [baseline physics-only MAE for reference: {baseline_mae:.3f} L/h]")

    importances = dict(zip(ALL_FEATURES, model.named_steps["gbr"].feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    print("[5/5] Saving model + report...")
    model_path = os.path.join(MODELS_DIR, "fuel_rate_model.joblib")
    joblib.dump(model, model_path)

    metadata = {
        "features": ALL_FEATURES,
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sample_rate_hz": args.hz,
        "metrics": {"mae_lph": mae, "rmse_lph": rmse, "r2": r2, "mape_pct": mape,
                     "baseline_physics_only_mae_lph": float(baseline_mae)},
        "feature_importances": importances,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_test, y_pred, s=6, alpha=0.35)
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Measured fuel rate (L/h)")
    axes[0].set_ylabel("Predicted fuel rate (L/h)")
    axes[0].set_title(f"Predicted vs Measured  (R2={r2:.3f})")

    residuals = y_pred - y_test.values
    axes[1].hist(residuals, bins=50)
    axes[1].set_xlabel("Residual (Predicted - Measured), L/h")
    axes[1].set_title(f"Residuals  (MAE={mae:.3f}, RMSE={rmse:.3f})")

    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "training_report.png"), dpi=140)
    print(f"Done. Model saved to {model_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
