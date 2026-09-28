# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0

Initial developer release.

- Local telemetry inspection (`telemetry-resilience inspect`) for Parquet,
  CSV, and JSONL files.
- 11 fault operators: `dropout`, `freeze`, `spike`, `bias`, `drift`, `noise`,
  `clipping`, `timestamp_gap`, `timestamp_jitter`, `sample_rate`, `scale`.
- `telemetry-resilience inject`: applies a YAML scenario to telemetry and
  writes the corrupted dataset plus a deterministic fault manifest and run
  report.
- Deterministic fault manifests: record what actually happened (channels,
  windows, affected counts), not just what was requested; identical values
  reproduce with the same input, scenario, seed, and tool version.
- CSV/Parquet/JSONL support, including CSV schema sidecars that preserve
  dtypes across round trips.
- Target process testing (`telemetry-resilience test`): runs a target
  program against corrupted telemetry in a temp copy and checks the
  scenario's `expect:` block (default expected exit code 0).
- Resilience suites (`telemetry-resilience suite`): baseline-first execution,
  isolated single-fault cases, per-case timeouts, strict suite validation,
  safe artifact directories, Ctrl+C handling.
- Safe baselines: the target runs against unmodified input first; a
  baseline failure blocks the suite instead of blaming injected faults.
- Fault applicability preflight before any target executes, with a
  temporary byte-for-byte baseline copy protecting the original input.
- JUnit/JSON/Markdown reports: `summary.json`, `summary.md`, `junit.xml`
  (XML 1.0-safe), per-case manifests, stdout/stderr logs.
- Mutation campaigns (`telemetry-resilience campaign`): expand a compact
  YAML declaration into deterministic single-fault cases, with `--plan` and
  `--list-cases` previews, channel/fault filtering, and a max-cases guard
  (default 100).
- Campaign coverage (`coverage.json`/`.md`/`.csv`): per-case PASS/FAIL/NOT
  EXECUTED statuses, aggregate cells, and `configured_pass_rate`.
- Deterministic fingerprints: campaign and input hashes recorded in
  `summary.json`, plus an input SHA-256, so a result traces back to exactly
  what ran.
- CI example: `.github/workflows/telemetry-resilience.yml.example`.
