"""Shared fixtures and helpers for the telemetry-resilience test suite."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telemetry_resilience.faults import FAULT_TYPES
from telemetry_resilience.models import FaultSpec, Scenario


@pytest.fixture
def telemetry_df():
    """Synthetic 10 Hz / 60 s telemetry frame (600 rows), built with seed 0.

    NOTE: the timestamp column is forced to ``datetime64[ns, UTC]`` (pandas 3
    defaults to ``us``). This matters for ``timestamp_jitter``: its ``apply``
    adds a ``timedelta64[ns]`` delta, and under pandas 3 the result can only
    be assigned back into an ns-unit column (see tests/test_faults.py).
    """
    rng = np.random.default_rng(0)
    n = 600
    timestamp = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC").as_unit(
        "ns"
    )
    motor_temperature = (65.0 + 0.5 * rng.normal(0, 1, n)).astype(np.float32)
    pressure = 101.3 + 0.05 * rng.normal(0, 1, n)
    rpm = pd.Series(
        np.round(
            1500.0
            + 50.0 * np.sin(2 * np.pi * np.arange(n) / 600.0)
            + 10.0 * rng.normal(0, 1, n)
        ).astype("int64"),
        dtype="Int64",
    )
    rpm[rng.random(n) < 0.05] = pd.NA
    gps_latitude = 37.7749 + 1e-6 * np.cumsum(rng.normal(0, 1, n))
    imu_z = rng.normal(0, 0.5, n).astype(np.float32)
    count_int = np.arange(n, dtype=np.int64)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "motor_temperature": motor_temperature,
            "pressure": pressure,
            "rpm": rpm,
            "gps_latitude": gps_latitude,
            "imu_z": imu_z,
            "count_int": count_int,
        }
    )


def make_scenario(faults, seed=123, time_column="timestamp",
                  input_file="data.parquet", expect=None):
    """Build a :class:`Scenario` directly, bypassing YAML.

    ``faults`` is a list of dicts with keys: ``type`` (required), ``channel``,
    ``start`` / ``duration`` (seconds; default 10 / 5), ``params`` (raw fault
    params) and ``cast``. Raw params are normalized through the fault
    module's ``validate()``, exactly as ``scenario.load_scenario`` does.
    """
    specs = []
    for f in faults:
        ftype = f["type"]
        params = FAULT_TYPES[ftype].validate(dict(f.get("params") or {}))
        specs.append(
            FaultSpec(
                channel=f.get("channel"),
                type=ftype,
                start_seconds=float(f.get("start", 10)),
                duration_seconds=float(f.get("duration", 5)),
                params=params,
                cast=f.get("cast"),
            )
        )
    return Scenario(
        version=1,
        seed=seed,
        input_file=input_file,
        time_column=time_column,
        faults=specs,
        expect=expect,
    )


def write_yaml(tmp_path, name, text):
    """Write YAML ``text`` to ``tmp_path / name`` and return the Path."""
    path = Path(tmp_path) / name
    path.write_text(text)
    return path


def window_mask(df, time_col="timestamp", start=10.0, duration=5.0):
    """Boolean mask of the ``[start, start+duration)`` fault window (seconds).

    Mirrors the engine's window computation (``t0 = df[time_col].min()``).
    With the 10 Hz fixture, ``start=10, duration=5`` selects rows 100..149.
    """
    t0 = df[time_col].min()
    return (df[time_col] >= t0 + pd.to_timedelta(start, "s")) & (
        df[time_col] < t0 + pd.to_timedelta(start + duration, "s")
    )
