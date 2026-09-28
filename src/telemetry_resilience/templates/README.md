# Telemetry Resilience demo project

A self-contained fault-injection demo: synthetic telemetry, a navigation
mutation campaign, and a small consumer app that reports degraded
conditions.

**The data is synthetic and non-authoritative** -- it is for
fault-injection demos only and is not physically accurate.

## Contents

- `demo_drive.parquet` -- synthetic demo telemetry (300 s at 10 Hz, clean
  baseline: the demo app prints `NOMINAL` on unmutated data).
- `navigation_campaign.yaml` -- 17 deterministic single-fault mutation
  cases over the demo telemetry.
- `degraded_navigation.yaml` -- a single 5-fault inject scenario
  (gps_latitude dropout, motor_temperature drift, imu_z sample-rate,
  pressure noise, rpm spike) for `telemetry-resilience inject`; the demo
  app reports `DEGRADED_MODE` on its corrupted output.
- `demo_app.py` -- demo consumer app: reads a telemetry file and prints
  `NOMINAL` or one `DEGRADED_MODE: ...` line per detected condition.

## Quickstart

1. Look at the data:

   ```sh
   telemetry-resilience inspect demo_drive.parquet
   ```

2. Validate the campaign without executing anything:

   ```sh
   telemetry-resilience campaign navigation_campaign.yaml --plan
   ```

3. Inject the single-scenario demo and check the app notices:

   ```sh
   telemetry-resilience inject demo_drive.parquet --scenario degraded_navigation.yaml
   python demo_app.py demo_drive.corrupted.parquet
   ```

   The scenario applies 5 faults and writes `demo_drive.corrupted.parquet`
   plus the fault manifest and run report; the app should print a
   `DEGRADED_MODE` line.

4. Run the full campaign against the demo app:

   ```sh
   telemetry-resilience campaign navigation_campaign.yaml --artifacts resilience-results -- python demo_app.py "{data}"
   ```

   The target command comes only from the arguments after `--`; the
   campaign YAML holds data and expectations and is never executed.
   Artifacts (expanded suite, coverage, per-case results) land in
   `resilience-results/`.

## Expected behaviour

- Clean baseline: `python demo_app.py demo_drive.parquet` prints `NOMINAL`.
- GPS dropout case: the app prints `DEGRADED_MODE: gps dropout detected`.
- IMU sample-rate case: the app prints
  `DEGRADED_MODE: imu_z sample-rate degradation detected`.
