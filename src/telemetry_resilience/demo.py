"""Synthetic demo telemetry generator and demo-project templates.

The generated signals are synthetic and non-authoritative: they exist for
fault-injection demos only and must not be treated as measurements from any
real vehicle, motor, or sensor.

Template files (``demo_app.py``, ``navigation_campaign.yaml``,
``degraded_navigation.yaml``, ``README.md``)
live in the ``templates/`` package directory and are loaded with
``importlib.resources`` so they resolve from an installed wheel, not just
the source tree.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .io import write_file

# Kept identical to examples/make_demo.py so the generator reproduces the
# historical demo file byte-for-byte (same seed, same RNG stream).
DEMO_SEED = 7
DEFAULT_SECONDS = 300.0
DEFAULT_HZ = 10.0

#: Template files bundled under ``templates/`` and copied by ``demo init``.
TEMPLATE_FILES = (
    "demo_app.py",
    "navigation_campaign.yaml",
    "degraded_navigation.yaml",
    "README.md",
)

PathLike = Union[str, Path]


def generate_demo_frame(
    seconds: float = DEFAULT_SECONDS,
    hz: float = DEFAULT_HZ,
    seed: int = DEMO_SEED,
) -> pd.DataFrame:
    """Build the synthetic demo telemetry frame.

    ``n = seconds * hz`` rows with columns ``timestamp``,
    ``motor_temperature``, ``pressure``, ``rpm``, ``gps_latitude`` and
    ``imu_z``. The baseline is clean (no missing values), so the demo app
    reports NOMINAL on unmutated data. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * hz)
    t = np.arange(n) / hz

    timestamp = pd.date_range(
        "2026-01-01",
        periods=n,
        freq=pd.to_timedelta(1 / hz, unit="s"),
        tz="UTC",
    )

    motor_temperature = (
        65.0 + 3.0 * np.sin(2 * np.pi * t / 60.0) + rng.normal(0, 0.3, n)
    ).astype(np.float32)
    motor_temperature[rng.random(n) < 0.0] = np.nan  # no missing values injected (clean demo baseline)

    pressure = 101.3 + rng.normal(0, 0.05, n)

    rpm = pd.Series(
        np.round(
            1500.0 + 50.0 * np.sin(2 * np.pi * t / 120.0) + rng.normal(0, 10, n)
        )
    ).astype("Int64")
    rpm[rng.random(n) < 0.0] = pd.NA  # no missing values injected (clean demo baseline)

    gps_latitude = 37.7749 + 1e-6 * np.cumsum(rng.normal(0, 1, n))

    imu_z = rng.normal(0, 0.5, n).astype(np.float32)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "motor_temperature": motor_temperature,
            "pressure": pressure,
            "rpm": rpm,
            "gps_latitude": gps_latitude,
            "imu_z": imu_z,
        }
    )


def write_demo_parquet(df: pd.DataFrame, output: PathLike) -> Path:
    """Write the demo frame to ``output`` as Parquet (creating parents)."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_file(df, out)
    return out


def templates_dir():
    """Return the bundled ``templates/`` directory (works from a wheel)."""
    return importlib_resources.files("telemetry_resilience") / "templates"


def read_template(name: str) -> bytes:
    """Read a bundled template file's bytes via importlib.resources."""
    data = (templates_dir() / name).read_bytes()
    if not data:
        raise ValueError(f"template is empty: {name!r}")
    return data


def list_templates() -> list:
    """Names of the bundled template files."""
    return list(TEMPLATE_FILES)
