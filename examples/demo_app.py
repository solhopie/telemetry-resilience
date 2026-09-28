#!/usr/bin/env python3
"""Demo consumer app: reads a telemetry file and reports degraded conditions.

Usage: python demo_app.py <data-file>

Prints one line per detected condition (or NOMINAL) and always exits 0,
unless the file cannot be read (exit 2).
"""

import sys

import pandas as pd


def read_any(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, engine="pyarrow")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported file format: '{path}'")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <data-file>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        df = read_any(path)
    except Exception as exc:
        print(f"Error: could not read {path}: {exc}", file=sys.stderr)
        return 2

    degraded = False

    if "motor_temperature" in df.columns and df["motor_temperature"].isna().any():
        print("DEGRADED_MODE: motor_temperature dropout detected")
        degraded = True
    if "imu_z" in df.columns and df["imu_z"].isna().any():
        print("DEGRADED_MODE: imu_z sample-rate degradation detected")
        degraded = True
    if "gps_latitude" in df.columns and df["gps_latitude"].isna().any():
        print("DEGRADED_MODE: gps dropout detected")
        degraded = True
    if "rpm" in df.columns:
        jumps = pd.to_numeric(df["rpm"], errors="coerce").diff().abs()
        if bool((jumps > 300).any()):
            print("SPIKE_DETECTED: rpm")
            degraded = True

    if not degraded:
        print("NOMINAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
