"""
physics_baseline.py

Physically-derived fuel-rate estimators. These serve two purposes in this
project:

1. GROUND TRUTH / LABELS for training when the vehicle's ECU does not expose
   OBD-II PID 0x5E (Engine Fuel Rate) directly -- many vehicles don't.
   The MAF-based formula below is the industry-standard approximation used
   by most DIY fuel-economy loggers (e.g. ScanGauge, Torque Pro) and is
   accurate to within a few percent for port-injected gasoline engines at
   steady state.

2. "MEASURED" trace on the live dashboard -- i.e. what you'd compute today
   with a pocket calculator and a MAF sensor, with NO machine learning.
   The ML model is judged against this (or against true PID 0x5E when the
   vehicle supports it) so you can see how much the learned model adds.

Units:
    MAF            g/s   (grams of air per second)
    MAP            kPa
    IAT            deg C
    fuel_rate_lph  liters / hour
    fuel_rate_gph  gallons / hour
"""

from __future__ import annotations

# --- Constants (gasoline, port/direct injection, closed-loop stoichiometric) ---
AFR_STOICH_GASOLINE = 14.7      # air-fuel ratio by mass, stoichiometric gasoline
FUEL_DENSITY_G_PER_L = 750.0    # ~0.72-0.78 kg/L; 750 g/L is a common average
R_SPECIFIC_AIR = 287.05         # J/(kg*K), specific gas constant for dry air
L_PER_GALLON = 3.78541


def fuel_rate_from_maf(maf_g_s: float, afr: float = AFR_STOICH_GASOLINE,
                        fuel_density_g_per_l: float = FUEL_DENSITY_G_PER_L) -> float:
    """
    Standard MAF-based fuel rate estimate (L/h).

    fuel_mass_flow (g/s) = MAF / AFR
    fuel_volume_flow (L/h) = fuel_mass_flow * 3600 / density
    """
    if maf_g_s is None or maf_g_s < 0:
        return 0.0
    fuel_mass_g_s = maf_g_s / afr
    return fuel_mass_g_s * 3600.0 / fuel_density_g_per_l


def estimate_maf_from_speed_density(map_kpa: float, iat_c: float, rpm: float,
                                     engine_displacement_l: float = 2.0,
                                     volumetric_efficiency: float = 0.80) -> float:
    """
    Speed-Density fallback: estimate MAF (g/s) from MAP, IAT and RPM when a
    MAF sensor reading is missing or implausible. Useful because several of
    our target vehicles report MAP but not MAF, or vice versa.

    Ideal gas law: air_mass_per_cycle = (MAP * V_cyl) / (R_specific * T) * VE
    Airflow (g/s) = air_mass_per_intake_stroke * (RPM / 2) / 60   [4-stroke: 1 intake per 2 revs]
    """
    if map_kpa is None or iat_c is None or rpm is None or rpm <= 0:
        return 0.0
    T_kelvin = iat_c + 273.15
    map_pa = map_kpa * 1000.0
    displacement_m3 = engine_displacement_l / 1000.0
    # mass of air inducted per full engine cycle (kg)
    mass_per_cycle_kg = (map_pa * displacement_m3 * volumetric_efficiency) / (R_SPECIFIC_AIR * T_kelvin)
    intakes_per_sec = (rpm / 2.0) / 60.0
    mass_flow_kg_s = mass_per_cycle_kg * intakes_per_sec
    return max(mass_flow_kg_s * 1000.0, 0.0)  # kg/s -> g/s


def lph_to_gph(lph: float) -> float:
    return lph / L_PER_GALLON


def reference_fuel_rate(reading: dict, engine_displacement_l: float = 2.0) -> float:
    """
    Best-effort "measured" fuel rate (L/h) from a single OBD reading dict.
    Prefers a true PID 0x5E fuel-rate value if present; otherwise prefers
    MAF; falls back to speed-density if MAF is missing/zero while the
    engine is clearly running.
    """
    if reading.get("FUEL_RATE") is not None:
        return float(reading["FUEL_RATE"])

    maf = reading.get("MAF")
    if maf is not None and maf > 0:
        return fuel_rate_from_maf(maf)

    rpm = reading.get("RPM")
    if rpm and rpm > 300:  # engine running but MAF missing -> speed-density fallback
        est_maf = estimate_maf_from_speed_density(
            reading.get("MAP"), reading.get("IAT"), rpm, engine_displacement_l
        )
        return fuel_rate_from_maf(est_maf)

    return 0.0
