# Live OBD-II Fuel Rate Predictor

An ML system that reads real-time OBD-II engine data (RPM, speed, throttle
position, engine load, MAF, MAP, intake-air temperature, coolant
temperature) and predicts instantaneous fuel consumption, displaying
predicted vs. measured fuel rate on a live web dashboard.

## Why a model at all? (the actual problem this solves)

Most vehicles don't expose OBD PID `0x5E` (true ECU fuel rate). The usual
DIY workaround is a single-sensor physics formula from MAF:

```
fuel_rate (L/h) = (MAF [g/s] / 14.7) * 3600 / 750
```

That formula is only as good as the MAF sensor reading, which drifts with
a dirty air filter, has measurement noise, and occasionally glitches. This
project instead trains a gradient-boosted regressor on **all 8 PIDs**, so
it can cross-check a noisy/biased MAF reading against RPM, engine load,
MAP, and throttle position — and get closer to the vehicle's true fuel
burn than the single-sensor formula does. In the bundled demo (synthetic
data with realistic MAF sensor noise/drift/dropouts):

| Approach | MAE (L/h) |
|---|---|
| Naive MAF-only physics formula | 0.502 |
| ML model (all 8 PIDs) | **0.081** |

(Numbers from the included `models/model_metadata.json` / `training_report.png`.)

## Architecture

```
src/
  physics_baseline.py   Physics fuel-rate formulas (MAF-based + speed-density
                         fallback). Used both as training labels and as the
                         "measured/reference" trace on the dashboard.
  data_simulator.py     Generates synthetic drive cycles (idle/accel/cruise/
                         brake/stop-go) with realistic sensor noise, for
                         training and demoing without a real vehicle.
  obd_interface.py      Talks to a real ELM327 OBD-II adapter via python-OBD,
                         or falls back to the simulator. Same interface either way.
  feature_engineering.py Raw PIDs -> model features (RPM*load, MAP/IAT,
                         acceleration, rolling MAF stats, etc). One shared
                         implementation for both batch training and live
                         streaming inference, to avoid train/serve skew.
  train_model.py        Trains + evaluates a GradientBoostingRegressor,
                         saves the model, metrics, and a diagnostic plot.
  model_inference.py    Loads the trained model; predict() takes one raw
                         OBD reading at a time and returns fuel rate (L/h).
  dashboard_app.py       Background thread polls OBD -> model -> physics
                         reference, pushes into a thread-safe ring buffer;
                         a Dash web app re-renders the live chart + gauges
                         every second and logs every sample to CSV.
models/                 Trained model, metadata, and training report (included,
                         pre-trained on synthetic data so the dashboard runs out of the box).
logs/                   CSV session logs written by the dashboard (one per run).
```

## Setup

```bash
cd obd_fuel_predictor
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`python-OBD` is only needed if you'll connect to a real adapter; the
dashboard runs fine without it in `--simulate` mode.

## Run the live dashboard

**No car needed (synthetic drive cycle):**
```bash
cd src
python dashboard_app.py --simulate
```
Open http://127.0.0.1:8050

**With a real vehicle**, plug in an ELM327-compatible OBD-II adapter
(USB, Bluetooth, or WiFi-to-serial), with the ignition/engine on:
```bash
python dashboard_app.py --port /dev/ttyUSB0       # Linux/USB example
python dashboard_app.py --port /dev/rfcomm0        # Linux/Bluetooth example
python dashboard_app.py --port COM3                 # Windows example
# or just: python dashboard_app.py   (auto-detect)
```
If the adapter can't be reached, it automatically falls back to simulated
data so the dashboard still runs.

Useful flags: `--poll-hz 2` (sampling rate), `--window-s 180` (seconds of
history shown on the chart), `--dash-port 8050`.

**Safety note:** mount the adapter/device securely and only interact with
the dashboard as a passenger or while parked — don't operate it while
driving.

## Retrain on your own vehicle's data

Every dashboard session writes a CSV to `logs/` with all raw PIDs plus
predicted/measured fuel rate. After collecting a reasonably varied driving
session (idle, city, highway, stop-and-go):

```bash
cd src
python train_model.py --csv ../logs/session_YYYYMMDD_HHMMSS.csv --hz 2
```

This overwrites `models/fuel_rate_model.joblib` and
`models/model_metadata.json`/`training_report.png` with a model trained on
your real vehicle instead of the synthetic demo data. If your vehicle
*does* support true PID `0x5E` fuel rate, `physics_baseline.reference_fuel_rate()`
will automatically prefer it as the training label over the MAF-based estimate.

## Notes & limitations

- The physics-based "measured" fuel rate is itself an estimate (unless your
  vehicle reports true PID `0x5E`), so treat it as a strong reference, not
  perfect ground truth — the included demo quantifies this by injecting
  realistic MAF sensor noise independent of the true value it's being
  compared against.
- The bundled model is trained entirely on synthetic data for an out-of-
  the-box demo; retrain on real logged data (above) before relying on
  predictions for a specific vehicle.
- `GradientBoostingRegressor` was chosen for a good accuracy/interpretability/
  dependency-footprint tradeoff; swapping in `xgboost` or a small neural net
  is straightforward since `train_model.py` and `model_inference.py` only
  depend on the scikit-learn `Pipeline`/`.predict()` interface.
