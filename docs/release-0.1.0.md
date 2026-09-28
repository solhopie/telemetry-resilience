# Telemetry Resilience 0.1.0 — Mutation testing for telemetry-driven software.

This is an early developer release. It does not certify physical-system safety.

## What it is

Telemetry Resilience tests how telemetry-driven software behaves when sensor
data goes bad. You describe sensor failures once in YAML — dropouts, freezes,
spikes, bias, drift, noise, clipping, timestamp gaps, sample-rate degradation,
and more — and the CLI injects them into a copy of your telemetry, then runs
your program against the corrupted data to see whether it degrades gracefully
or crashes.

Instead of hand-writing individual fault scenarios, mutation campaigns expand
a compact declaration into deterministic single-fault cases, report per-case
PASS/FAIL/NOT EXECUTED coverage, and fingerprint both the campaign and the
input so results are traceable.

## Tiny demo

```bash
pip install telemetry-resilience
python examples/make_demo.py
telemetry-resilience inspect examples/demo_drive.parquet
telemetry-resilience campaign examples/navigation_campaign.yaml --plan
telemetry-resilience campaign examples/navigation_campaign.yaml \
  --artifacts resilience-results \
  -- python examples/demo_app.py "{data}"
```

## Highlights

- 11 deterministic fault operators with a strict dtype safety rule: the
  engine never silently changes a column's dtype; it raises an error telling
  you exactly which `cast:` to request.
- Fault manifests record ground truth — what actually happened — not intent.
- Baseline-first suites and campaigns: the target runs on unmodified input
  before any faults, so an already-broken app can't be blamed on injection.
- Original telemetry is never modified; artifact directories refuse to
  overwrite unrelated data.
- Everything runs locally: no account, no API key, no cloud, no analytics.

## Known limitations

- File-based telemetry only (Parquet/CSV/JSONL); no live hardware or
  streaming adapters (no ROS/CAN/MQTT/Kafka/OPC-UA).
- Campaign cases are single-fault by default; multi-fault combinations are
  never generated.
- Physical meaning and severity of faults must be provided by the engineer —
  the tool injects what you configure and reports what happened.
- The tool does not certify safety or reliability. Corrupted outputs are
  synthetic test data — never feed them back as real telemetry.

## CI status

Cross-platform CI matrix prepared for: Python 3.11 / 3.12 / 3.13,
Ubuntu / macOS / Windows. Status: NOT YET EXECUTED IN GITHUB ACTIONS.

**This is an early developer release. It does not certify physical-system safety.**
