"""
obd_interface.py

Thin wrapper around python-OBD (https://python-obd.readthedocs.io/) that
polls the 8 PIDs this project cares about and returns a flat dict per
sample. Falls back to a SimulatedOBD source (data_simulator.py) when no
adapter is connected or SIMULATE=True, so the rest of the system (model
inference + dashboard) never needs to know which source it's reading from.

Hardware notes:
  - Requires an ELM327-compatible OBD-II adapter (USB, Bluetooth SPP, or
    WiFi-to-serial bridge) plugged into the vehicle's OBD-II port.
  - USB: port is typically /dev/ttyUSB0 (Linux) or COM3 (Windows).
  - Bluetooth: pair first, then the adapter exposes a serial port (e.g.
    /dev/rfcomm0) -- python-obd can usually auto-detect with obd.OBD().
  - Not every vehicle/PID combination is supported; obd_interface checks
    `connection.supports(cmd)` and reports missing PIDs rather than
    crashing, so the model/dashboard degrade gracefully.
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Iterator

logger = logging.getLogger("obd_interface")

PID_NAMES = ["RPM", "SPEED", "THROTTLE_POS", "ENGINE_LOAD",
             "MAF", "MAP", "IAT", "COOLANT_TEMP", "FUEL_RATE"]

# Maps our internal names -> python-obd command names
_OBD_COMMAND_MAP = {
    "RPM": "RPM",
    "SPEED": "SPEED",
    "THROTTLE_POS": "THROTTLE_POS",
    "ENGINE_LOAD": "ENGINE_LOAD",
    "MAF": "MAF",
    "MAP": "INTAKE_PRESSURE",
    "IAT": "INTAKE_TEMP",
    "COOLANT_TEMP": "COOLANT_TEMP",
    "FUEL_RATE": "FUEL_RATE",   # PID 0x5E, not supported on all vehicles
}


@dataclass
class OBDReading:
    timestamp: float
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    source: str = "unknown"  # "hardware" or "simulated"


class RealOBDSource:
    """Live connection to a physical ELM327 adapter via python-OBD."""

    def __init__(self, port: Optional[str] = None, fast: bool = True):
        import obd  # imported lazily so the project runs without the dep in simulate mode
        self._obd = obd
        self.connection = obd.OBD(portstr=port, fast=fast)
        if not self.connection.is_connected():
            raise ConnectionError(
                "Could not connect to an OBD-II adapter. Check that it's plugged in, "
                "paired (if Bluetooth), and the ignition/engine is on."
            )
        self._commands = {}
        for name, obd_cmd_name in _OBD_COMMAND_MAP.items():
            cmd = getattr(obd.commands, obd_cmd_name, None)
            if cmd is not None and self.connection.supports(cmd):
                self._commands[name] = cmd
            else:
                logger.warning("PID not supported by this vehicle: %s", name)

    def read(self) -> OBDReading:
        values = {}
        for name, cmd in self._commands.items():
            resp = self.connection.query(cmd)
            if resp.is_null() or resp.value is None:
                values[name] = None
            else:
                try:
                    values[name] = float(resp.value.magnitude)  # pint Quantity -> float
                except AttributeError:
                    values[name] = float(resp.value)
        for name in PID_NAMES:
            values.setdefault(name, None)
        return OBDReading(timestamp=time.time(), values=values, source="hardware")


class SimulatedOBDSource:
    """Replays a generated drive cycle as if it were live hardware."""

    def __init__(self, duration_s: int = 1800, hz: float = 2.0, seed: int = 7):
        from data_simulator import generate_drive_cycle
        self.hz = hz
        self._df = generate_drive_cycle(duration_s=duration_s, hz=hz, seed=seed)
        self._iter = self._df.iterrows()

    def read(self) -> OBDReading:
        try:
            _, row = next(self._iter)
        except StopIteration:
            self._iter = self._df.iterrows()
            _, row = next(self._iter)
        values = {
            "RPM": row["RPM"], "SPEED": row["SPEED"], "THROTTLE_POS": row["THROTTLE_POS"],
            "ENGINE_LOAD": row["ENGINE_LOAD"], "MAF": row["MAF"], "MAP": row["MAP"],
            "IAT": row["IAT"], "COOLANT_TEMP": row["COOLANT_TEMP"], "FUEL_RATE": None,
        }
        values["_REFERENCE_FUEL_RATE_LPH"] = row["FUEL_RATE_REF_LPH"]
        return OBDReading(timestamp=time.time(), values=values, source="simulated")


def get_source(port: Optional[str] = None, simulate: bool = False):
    """Factory: tries real hardware first unless simulate=True or hardware init fails."""
    if not simulate:
        try:
            return RealOBDSource(port=port)
        except Exception as e:
            logger.warning("Falling back to simulated OBD source: %s", e)
    return SimulatedOBDSource()


def poll_loop(source, interval_s: float = 0.5) -> Iterator[OBDReading]:
    """Infinite generator yielding readings at a fixed cadence."""
    while True:
        t0 = time.time()
        yield source.read()
        elapsed = time.time() - t0
        time.sleep(max(0.0, interval_s - elapsed))
