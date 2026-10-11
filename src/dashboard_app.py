"""
dashboard_app.py

Live dashboard: polls OBD-II data (real adapter or simulated drive cycle) in
a background thread, runs it through the trained ML model, computes the
physics-based reference fuel rate for comparison, and streams both onto a
real-time Dash chart alongside gauges for the raw PIDs and running error
stats. Also logs every sample to a CSV for later retraining.

Run:
    python dashboard_app.py                 # simulated data (no hardware needed)
    python dashboard_app.py --port /dev/ttyUSB0   # real ELM327 adapter
    python dashboard_app.py --poll-hz 2 --window-s 120

Then open http://127.0.0.1:8050 in a browser.
"""

from __future__ import annotations
import argparse
import csv
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

from obd_interface import get_source, PID_NAMES
from model_inference import FuelRatePredictor
from physics_baseline import reference_fuel_rate

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Shared, thread-safe state between the polling thread and the Dash callbacks
# --------------------------------------------------------------------------
class LiveState:
    def __init__(self, maxlen: int):
        self.lock = threading.Lock()
        self.t = deque(maxlen=maxlen)
        self.predicted = deque(maxlen=maxlen)
        self.measured = deque(maxlen=maxlen)
        self.raw = {name: deque(maxlen=maxlen) for name in
                    ["RPM", "SPEED", "THROTTLE_POS", "ENGINE_LOAD", "MAF", "MAP", "IAT", "COOLANT_TEMP"]}
        self.latest_raw = {}
        self.source_label = "initializing..."
        self.n_samples = 0
        self.start_time = time.time()
        self.errors = deque(maxlen=maxlen)

    def push(self, t, pred, meas, raw_values):
        with self.lock:
            self.t.append(t)
            self.predicted.append(pred)
            self.measured.append(meas)
            self.errors.append(pred - meas)
            for name in self.raw:
                self.raw[name].append(raw_values.get(name))
            self.latest_raw = dict(raw_values)
            self.n_samples += 1

    def snapshot(self):
        with self.lock:
            return {
                "t": list(self.t),
                "predicted": list(self.predicted),
                "measured": list(self.measured),
                "raw": {k: list(v) for k, v in self.raw.items()},
                "latest_raw": dict(self.latest_raw),
                "source_label": self.source_label,
                "n_samples": self.n_samples,
                "errors": list(self.errors),
            }


def polling_worker(state: LiveState, source, predictor: FuelRatePredictor,
                    interval_s: float, csv_path: str):
    state.source_label = "hardware" if source.__class__.__name__ == "RealOBDSource" else "simulated"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "elapsed_s"] + PID_NAMES + ["predicted_fuel_rate_lph", "measured_fuel_rate_lph"])

        t0 = time.time()
        while True:
            loop_start = time.time()
            reading = source.read()
            values = reading.values

            # "Measured" reference: use the simulator's clean ground-truth signal if present
            # (demo mode), otherwise compute the standard physics reference from live sensors.
            if "_REFERENCE_FUEL_RATE_LPH" in values:
                measured = float(values["_REFERENCE_FUEL_RATE_LPH"])
            else:
                measured = reference_fuel_rate(values)

            predicted = predictor.predict(values)

            elapsed = time.time() - t0
            state.push(elapsed, predicted, measured, values)

            writer.writerow([datetime.now().isoformat(), f"{elapsed:.2f}"] +
                             [values.get(name) for name in PID_NAMES] +
                             [predicted, measured])
            f.flush()

            dt = time.time() - loop_start
            time.sleep(max(0.0, interval_s - dt))


# --------------------------------------------------------------------------
# Dash app
# --------------------------------------------------------------------------
def build_app(state: LiveState) -> Dash:
    app = Dash(__name__)
    app.title = "Live Fuel Rate: Predicted vs Measured"

    gauge_spec = [
        ("RPM", "rpm"), ("SPEED", "km/h"), ("THROTTLE_POS", "%"),
        ("ENGINE_LOAD", "%"), ("MAF", "g/s"), ("MAP", "kPa"),
        ("IAT", "\u00b0C"), ("COOLANT_TEMP", "\u00b0C"),
    ]

    app.layout = html.Div(style={"fontFamily": "Inter, Arial, sans-serif", "padding": "18px",
                                  "backgroundColor": "#0f1117", "color": "#e8e8e8", "minHeight": "100vh"}, children=[
        html.Div([
            html.H2("Live Engine Fuel-Rate Predictor", style={"marginBottom": "2px"}),
            html.Div(id="status-line", style={"color": "#9aa0a6", "fontSize": "14px", "marginBottom": "14px"}),
        ]),

        html.Div([
            html.Div([dcc.Graph(id="fuel-rate-graph", style={"height": "420px"})],
                     style={"flex": "3", "minWidth": "520px"}),
            html.Div([
                html.Div(id="metric-cards", style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}),
            ], style={"flex": "2", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

        html.H4("Live Sensor PIDs", style={"marginTop": "22px"}),
        html.Div(id="pid-cards", style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}),

        dcc.Interval(id="tick", interval=1000, n_intervals=0),
    ])

    def card(title, value, unit="", color="#4fc3f7"):
        return html.Div([
            html.Div(title, style={"fontSize": "12px", "color": "#9aa0a6", "textTransform": "uppercase"}),
            html.Div(f"{value}", style={"fontSize": "24px", "fontWeight": "700", "color": color}),
            html.Div(unit, style={"fontSize": "11px", "color": "#6b7280"}),
        ], style={"background": "#171a23", "borderRadius": "10px", "padding": "12px 16px",
                   "minWidth": "130px", "border": "1px solid #262a36"})

    @app.callback(
        Output("fuel-rate-graph", "figure"),
        Output("metric-cards", "children"),
        Output("pid-cards", "children"),
        Output("status-line", "children"),
        Input("tick", "n_intervals"),
    )
    def update(_n):
        snap = state.snapshot()
        t = snap["t"]
        pred = snap["predicted"]
        meas = snap["measured"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=meas, mode="lines", name="Measured (reference)",
                                  line=dict(color="#ef5350", width=2)))
        fig.add_trace(go.Scatter(x=t, y=pred, mode="lines", name="Predicted (ML model)",
                                  line=dict(color="#4fc3f7", width=2)))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Elapsed time (s)", yaxis_title="Fuel rate (L/h)",
            legend=dict(orientation="h", y=1.08),
            title="Predicted vs Measured Instantaneous Fuel Rate",
        )

        errors = snap["errors"]
        if errors:
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(np.square(errors))))
        else:
            mae = rmse = 0.0
        cur_pred = pred[-1] if pred else 0.0
        cur_meas = meas[-1] if meas else 0.0

        metric_cards = [
            card("Predicted now", f"{cur_pred:.2f}", "L/h", "#4fc3f7"),
            card("Measured now", f"{cur_meas:.2f}", "L/h", "#ef5350"),
            card("Live MAE", f"{mae:.3f}", "L/h (rolling window)", "#a5d6a7"),
            card("Live RMSE", f"{rmse:.3f}", "L/h (rolling window)", "#a5d6a7"),
            card("Samples", f"{snap['n_samples']}", "collected this session", "#ffcc80"),
        ]

        latest = snap["latest_raw"]
        pid_cards = []
        for name, unit in gauge_spec:
            v = latest.get(name)
            v_str = f"{v:.1f}" if isinstance(v, (int, float)) else "—"
            pid_cards.append(card(name.replace("_", " ").title(), v_str, unit))

        status = (f"Source: {snap['source_label']}  |  "
                  f"Logging to CSV  |  Updated {datetime.now().strftime('%H:%M:%S')}")
        return fig, metric_cards, pid_cards, status

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, default=None, help="serial port of ELM327 adapter, e.g. /dev/ttyUSB0 or COM3")
    ap.add_argument("--simulate", action="store_true", help="force simulated data even if hardware is available")
    ap.add_argument("--poll-hz", type=float, default=2.0, help="sampling rate (Hz)")
    ap.add_argument("--window-s", type=int, default=180, help="seconds of history kept on the live chart")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--dash-port", type=int, default=8050)
    args = ap.parse_args()

    source = get_source(port=args.port, simulate=args.simulate)
    predictor = FuelRatePredictor(dt_seconds=1.0 / args.poll_hz)

    maxlen = int(args.window_s * args.poll_hz)
    state = LiveState(maxlen=maxlen)

    csv_path = os.path.join(LOGS_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    worker = threading.Thread(
        target=polling_worker,
        args=(state, source, predictor, 1.0 / args.poll_hz, csv_path),
        daemon=True,
    )
    worker.start()

    app = build_app(state)
    print(f"Logging live session to {csv_path}")
    print(f"Dashboard running at http://{args.host}:{args.dash_port}")
    app.run(host=args.host, port=args.dash_port, debug=False)


if __name__ == "__main__":
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(__file__))
    main()
