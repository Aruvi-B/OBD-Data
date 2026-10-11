"""
data_simulator.py

Generates synthetic but physically-plausible OBD-II driving data so the
whole pipeline (training + live dashboard) can be demoed and unit-tested
without a physical vehicle or ELM327 adapter.

Approach: stitch together archetypal driving "phases" (idle, launch,
acceleration, cruise, deceleration, hard braking, stop-and-go) similar in
spirit to standard chassis-dynamometer drive cycles (e.g. UDDS/FTP-75
style shapes, not an exact reproduction). Speed and RPM are generated with
smooth, correlated dynamics; throttle/load/MAF/MAP are derived from speed,
acceleration, and RPM with realistic noise so the dataset has genuine
non-linear structure for the ML model to learn (not just a close-form
restatement of the MAF fuel-rate formula).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from physics_baseline import reference_fuel_rate


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _phase_speed_profile(phase: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n)
    if phase == "idle":
        return np.zeros(n)
    if phase == "launch":
        return 35 * t ** 1.3
    if phase == "accel":
        start = rng.uniform(10, 40)
        end = rng.uniform(55, 95)
        return start + (end - start) * t ** 0.9
    if phase == "cruise":
        base = rng.uniform(45, 85)
        return base + rng.normal(0, 1.2, n).cumsum() * 0.02  # gentle drift
    if phase == "decel":
        start = rng.uniform(40, 90)
        return start * (1 - t) ** 1.1
    if phase == "hard_brake":
        start = rng.uniform(40, 70)
        return np.maximum(start * (1 - t) ** 0.5, 0)
    if phase == "stopgo":
        return np.abs(20 * np.sin(2 * np.pi * 2 * t)) * (1 - 0.3 * t)
    raise ValueError(phase)


def generate_drive_cycle(duration_s: int = 1800, hz: float = 2.0, seed: int = 42,
                          engine_displacement_l: float = 2.0) -> pd.DataFrame:
    """
    Returns a DataFrame of synthetic OBD-II readings at `hz` samples/sec for
    `duration_s` seconds, including the physics-based reference fuel rate
    used as the training label.
    """
    rng = np.random.default_rng(seed)
    n_total = int(duration_s * hz)
    phases_cycle = ["idle", "launch", "accel", "cruise", "decel",
                     "stopgo", "accel", "cruise", "hard_brake", "idle"]

    speed_segments = []
    remaining = n_total
    i = 0
    while remaining > 0:
        phase = phases_cycle[i % len(phases_cycle)]
        seg_len = min(int(rng.uniform(8, 40) * hz), remaining)
        seg_len = max(seg_len, 2)
        speed_segments.append(_phase_speed_profile(phase, seg_len, rng))
        remaining -= seg_len
        i += 1

    speed = np.concatenate(speed_segments)[:n_total]
    speed = np.clip(_smooth(speed, window=max(int(hz * 1.5), 1)), 0, 130)

    dt = 1.0 / hz
    accel = np.gradient(speed, dt) / 3.6  # km/h -> m/s^2 approx (speed in km/h-ish units)

    # RPM: idle ~750, scales with speed and transient throttle demand, with gear-ish steps
    gear_ratio_effect = 1.0 + 0.6 * np.clip(accel, 0, None)  # higher accel -> lower gear -> relatively higher RPM
    rpm = 750 + speed * 38 * (0.6 + 0.4 * gear_ratio_effect)
    rpm += rng.normal(0, 40, n_total)
    rpm = np.clip(_smooth(rpm, window=max(int(hz), 1)), 650, 6500)

    # Throttle position (%): driven by acceleration demand + baseline to hold speed
    throttle = 8 + 2.2 * np.clip(accel, -5, 10) * 10 + 0.15 * speed
    throttle += rng.normal(0, 2.0, n_total)
    throttle = np.clip(_smooth(throttle, window=max(int(hz), 1)), 0, 100)

    # Engine load (%): correlates with throttle & RPM, with some lag/noise
    engine_load = 0.55 * throttle + 0.01 * rpm + rng.normal(0, 3, n_total)
    engine_load = np.clip(_smooth(engine_load, window=max(int(hz), 1)), 5, 100)

    # MAP (kPa): inversely related to throttle closure at low load (vacuum), rises toward ~100 kPa (WOT)
    map_kpa = 30 + 0.68 * throttle + 0.05 * (rpm / 100)
    map_kpa += rng.normal(0, 1.5, n_total)
    map_kpa = np.clip(map_kpa, 20, 105)

    # Intake air temperature (C): slow-moving, mild correlation with engine load (under-hood heat soak)
    iat = 25 + 6 * (engine_load / 100) + rng.normal(0, 0.5, n_total).cumsum() * 0.01
    iat = np.clip(iat, 5, 55)

    # Coolant temperature (C): warms up over first ~5 min then thermostatically regulated ~90C
    warm_up = 20 + 70 * (1 - np.exp(-np.arange(n_total) / (hz * 180)))
    coolant = warm_up + rng.normal(0, 0.8, n_total)
    coolant = np.clip(coolant, 15, 105)

    # --- "Clean" air mass flow (g/s), computed from speed-density physics. This
    # is treated as the vehicle's TRUE internal air/fuel state and is what the
    # ground-truth fuel-rate label is derived from. It is NOT what ends up in
    # the MAF column the model trains on -- see sensor model below.
    from physics_baseline import estimate_maf_from_speed_density
    maf_true = np.array([
        estimate_maf_from_speed_density(map_kpa[j], iat[j], max(rpm[j], 1),
                                         engine_displacement_l, volumetric_efficiency=0.78)
        for j in range(n_total)
    ])
    idle_mask = speed < 1
    maf_true[idle_mask] = np.clip(maf_true[idle_mask], 1.5, 4.0)
    maf_true = np.clip(maf_true, 0, 400)

    # --- MAF SENSOR reading (what actually gets logged as the "MAF" PID / feature).
    # Real MAF sensors drift, get dirty air filters/contamination (multiplicative
    # bias that wanders slowly), have white measurement noise, and occasionally
    # glitch/dropout. This is deliberately noisier/more biased than maf_true so
    # that a naive "fuel_rate = f(MAF_sensor)" baseline is measurably worse than
    # an ML model that can cross-check MAF against RPM/MAP/load/throttle.
    slow_bias = 1.0 + 0.15 * np.sin(np.linspace(0, rng.uniform(1, 4) * np.pi, n_total))  # +/-15% slow drift (dirty filter / recalibration drift)
    white_noise = rng.normal(1.0, 0.07, n_total)  # +/-7% measurement noise
    maf_sensor = maf_true * slow_bias * white_noise
    # occasional sensor glitches / brief dropouts (flat-lines or spikes), ~1% of samples
    glitch_mask = rng.random(n_total) < 0.01
    maf_sensor[glitch_mask] *= rng.uniform(0.3, 2.0, glitch_mask.sum())
    maf_sensor = np.clip(_smooth(maf_sensor, window=2), 0, 450)

    df = pd.DataFrame({
        "RPM": rpm,
        "SPEED": speed,
        "THROTTLE_POS": throttle,
        "ENGINE_LOAD": engine_load,
        "MAF": maf_sensor,       # noisy sensor reading -> what the model/baseline see
        "MAP": map_kpa,
        "IAT": iat,
        "COOLANT_TEMP": coolant,
    })
    df.index = pd.date_range("2026-01-01", periods=n_total, freq=pd.Timedelta(seconds=dt))
    df.index.name = "timestamp"

    # Ground-truth label: physics-based reference fuel rate (L/h), computed from the
    # CLEAN air-mass-flow estimate (maf_true), not the noisy sensor column. This is
    # the realistic target -- actual fuel burned -- which neither the model nor the
    # naive baseline get to see directly; both must infer it from noisy sensors.
    df["FUEL_RATE_REF_LPH"] = [
        reference_fuel_rate({"MAF": maf_true[j], "RPM": None, "MAP": None, "IAT": None})
        for j in range(n_total)
    ]
    return df


def stream_rows(df: pd.DataFrame):
    """Generator that yields rows of a pre-generated drive cycle one at a time,
    mimicking a live OBD polling loop. Used by obd_interface.py's simulated mode."""
    for _, row in df.iterrows():
        yield row.to_dict()


if __name__ == "__main__":
    df = generate_drive_cycle(duration_s=600, hz=2.0)
    print(df.describe())
    print(df.head())
