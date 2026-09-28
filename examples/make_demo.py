#!/usr/bin/env python3
"""Generate synthetic demo telemetry for the telemetry-resilience examples.

Thin wrapper around :mod:`telemetry_resilience.demo`.

NOTE: these are synthetic signals for fault-injection demos only. They are
not physically authoritative models of any real vehicle, motor, or sensor.

Generates ``n = seconds * hz`` rows with columns:
  timestamp, motor_temperature, pressure, rpm, gps_latitude, imu_z
"""

import argparse

from telemetry_resilience.demo import (
    DEFAULT_HZ,
    DEFAULT_SECONDS,
    generate_demo_frame,
    write_demo_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo telemetry.")
    parser.add_argument("--output", default="examples/demo_drive.parquet",
                        help="Output file (default: examples/demo_drive.parquet)")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="Duration in seconds (default: 300)")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ,
                        help="Sample rate in Hz (default: 10)")
    args = parser.parse_args()

    df = generate_demo_frame(seconds=args.seconds, hz=args.hz)
    out = write_demo_parquet(df, args.output)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
