# Telemetry Resilience

**Mutation testing for telemetry-driven software.** Point the CLI at real
telemetry, describe sensor failures once in YAML, and get back corrupted
datasets plus exact fault manifests — everything you need to measure how your
software responds when sensors degrade instead of crashing. Fully local: no account,
no API key, no AI, no cloud, no network calls.

## Quick start

Under five minutes, end to end — from a completely empty directory, with
only the installed wheel (no repository clone needed):

```bash
pip install telemetry-resilience

# 1. Check the environment can run the CLI
telemetry-resilience doctor

# 2. Scaffold a self-contained demo project (generates demo_drive.parquet
#    and copies demo_app.py, navigation_campaign.yaml,
#    degraded_navigation.yaml, and a README into ./telemetry-demo)
telemetry-resilience demo init telemetry-demo
cd telemetry-demo

# 3. Inspect the demo telemetry
telemetry-resilience inspect demo_drive.parquet

# 4. Inject one ready-made scenario (writes
#    demo_drive.corrupted.parquet, demo_drive.faults.json, and
#    demo_drive.report.json next to the input)
telemetry-resilience inject demo_drive.parquet --scenario degraded_navigation.yaml

# 5. Preview a mutation campaign without executing anything
telemetry-resilience campaign navigation_campaign.yaml --plan

# 6. Run the campaign against the demo app; artifacts land in
#    resilience-results/
telemetry-resilience campaign navigation_campaign.yaml \
  --artifacts resilience-results \
  -- python demo_app.py "{data}"
```

The installed package bundles the demo assets, so `demo init` works from
the wheel alone — no source checkout required. The scaffolded project
includes `degraded_navigation.yaml`, a ready-made single scenario so you
can try `inject` without writing YAML first.

*The sections below illustrate commands against a source checkout — the
`examples/` paths there require cloning the repository.*

## Local by design

- Telemetry files stay on your machine. Nothing is uploaded anywhere.
- No account required. No API key required.
- No AI service is contacted.
- No analytics or usage telemetry is uploaded.
- Target programs are executed only when explicitly requested — via
  `test`, `suite`, or `campaign` with `-- <command>` arguments. Nothing
  runs your program silently in the background.
- Original telemetry is not modified by default. `inject` writes new files
  alongside the input; `test`/`suite`/`campaign` inject into temporary
  copies and leave the original untouched.

## Known limitations

- **File-based telemetry only.** Inputs are read from Parquet, CSV, or
  JSONL files into memory (pandas). No live hardware, no streaming.
- **No protocol integrations.** No ROS/CAN/MQTT/Kafka/OPC-UA adapters; the
  tool reads files, not streams.
- **No live hardware.** There is no hardware-in-the-loop mode.
- **Campaign cases are single-fault by default.** Every case isolates
  exactly one mutation; multi-fault combinations are never generated.
- **Physical meaning and severity must be provided by the engineer.** You
  choose the channels, fault types, and parameters — the tool injects what
  you configure and reports what actually happened.
- **The tool does not certify safety or reliability.** Coverage measures how
  your application responded to the configured campaign, not system safety.

## What this is not

- No dashboard or visualization UI.
- No protocol integrations (MQTT, Kafka, ROS, …): it reads files, not
  streams.
- Corrupted outputs are synthetic test data — never feed them back as real
  telemetry.

## Test one scenario against one target

```bash
telemetry-resilience test examples/degraded_navigation.yaml \
  python examples/demo_app.py {data}
```

`test` injects the scenario into a temp copy, runs your command with
`{data}` replaced by the corrupted file's path, and checks the scenario's
`expect:` block. The default expected exit code is always `0` unless the
scenario sets `expect.exit_code` explicitly — so no `expect` block + target
exit 9 ⇒ FAIL. A crashing target can never silently pass.

## Resilience suites

A suite runs a **collection** of fault scenarios against a target app — the
practical CI shape of this tool:

```bash
telemetry-resilience suite examples/resilience_suite.yaml \
  --artifacts ./resilience-results --keep-data --overwrite-artifacts \
  -- python examples/demo_app.py {data}
```

A suite YAML names the input once and lists independent cases, each with its
own seed and faults:

```yaml
version: 1
name: navigation-resilience
input:
  file: drive.parquet
  time_column: timestamp
timeout_seconds: 60          # suite default; cases may override
baseline:
  enabled: true
  expect:
    exit_code: 0
    stdout_contains: [READY]
cases:
  - name: gps-dropout
    seed: 1001
    faults:
      - channel: gps_latitude
        type: dropout
        start: 30s
        duration: 10s
    expect:
      exit_code: 0
      stdout_contains: [DEGRADED_MODE]
```

How it runs:

- **Baseline first.** With `baseline.enabled: true`, the target runs against
  the *unmodified* input before any faults. A baseline failure prints
  `BASELINE FAILED`, marks the suite `BLOCKED`, and stops — you never get to
  blame injected faults for an app that was already broken.
- **Isolated cases.** Every case injects from the *original* input
  (original→A, original→B, …), never chained. Each case has its own seed.
- **Timeouts.** `timeout_seconds` at suite level, overridable per case. A
  timeout is a *test failure* (`status: fail`, reason `timeout`), not a crash.
- **Assertions.** `exit_code` (default expected: 0 — a crash never silently
  passes), `stdout_contains`, `stdout_not_contains`, `stderr_contains`,
  `stderr_not_contains`, `max_duration_seconds`. Results report
  `baseline_exit_code`/`case_exit_code` and both durations as measurements
  only; no automatic regression claims.
- **`--fail-fast`** stops after the first failing case (default runs all).
- **`--dry-run`** validates the suite, input file, channels, and fault configs,
  then *applies every case's faults in memory* against the real input to verify
  each fault is actually applicable (dtype incompatibility, unsafe casts, and
  invalid windows fail dry-run with exit 2). Prints cases/seeds/fault counts
  and the target command — and executes nothing, writes nothing.
- **Ctrl+C** aborts cleanly during baseline, case execution, and reporting:
  temporary telemetry is removed and a message is printed (exit code 3).
- **Target launch errors** (missing executable, permission denied) produce a
  clean `TARGET COULD NOT START` message and exit code 2 — never a traceback,
  and never counted as a resilience failure.

Artifacts (`--artifacts <dir>`) are CI-friendly:

```
resilience-results/
  summary.json   # machine-readable: baseline, per-case status/seed/durations,
                 # failures, passed/failed/total (no temp machine paths)
  summary.md     # Markdown table + failure explanations (PR comments, reports)
  junit.xml      # each case as a JUnit testcase (properly escaped XML)
  baseline/      # stdout.txt, stderr.txt, result.json
  gps-dropout/   # faults.json (manifest), stdout.txt, stderr.txt, result.json
                 # + corrupted.parquet when --keep-data is given
```

Without `--keep-data`, temporary corrupted datasets are deleted after each
case (manifests and logs remain). The original input is never modified or
deleted. Case names are validated — `../`, absolute paths, separators,
control characters, Windows-unsafe characters (`<>:"|?*`), and reserved
names are rejected so artifacts can never escape the directory. Cases whose
names normalize to the same artifact directory (compared case-insensitively)
are rejected, as are names colliding with the suite's own `baseline/`,
`summary.json`, `summary.md`, and `junit.xml`.

Artifact-directory safety: by default the tool *refuses* a non-empty
`--artifacts` directory (it may hold a previous run or unrelated data) and
tells you to pick a clean directory. `--overwrite-artifacts` only replaces a
directory that carries the telemetry-resilience ownership marker
(`.telemetry-resilience-artifacts.json`); directories with unrelated data are
never deleted. On an owned rerun, stale case artifacts, corrupted data, and
summaries are removed first, so an old `--keep-data` output can never masquerade
as a fresh result. Symlinked or escaping case/baseline paths are refused
before anything executes.

Suite YAML is strictly validated (unknown keys, duplicate/empty/unsafe case
names, missing seeds, empty fault lists, bad channels/timeouts all fail
loudly), and it holds **data and expectations only**: the target command comes
exclusively from the CLI's `-- <command>` arguments and is never read from
YAML. Execution uses subprocess argument arrays, never `shell=True`.

## Mutation campaigns

**Mutation testing for telemetry-driven software.** Instead of hand-writing
individual fault scenarios, you describe families of telemetry faults once —
channels, fault types, parameter grids — and the CLI expands them into
deterministic single-fault cases.

Measure how your application responds to configured telemetry faults.
Run reproducible physical-data failure campaigns in CI.

```bash
# validate, expand and preflight the campaign without executing the target
telemetry-resilience campaign examples/navigation_campaign.yaml --plan

# list the generated case IDs (with parameters and seeds) and exit
telemetry-resilience campaign examples/navigation_campaign.yaml --list-cases

# run the full campaign against the demo app
telemetry-resilience campaign examples/navigation_campaign.yaml \
  --artifacts resilience-results \
  -- python examples/demo_app.py "{data}"
```

A campaign YAML names the input once and declares fault *families* per
channel (plus timeline faults):

```yaml
version: 1
name: navigation-mutations
seed: 5000
input:
  file: demo_drive.parquet      # relative paths resolve against the campaign file
  time_column: timestamp
timeout_seconds: 60             # campaign default; blocks below may override
baseline:
  enabled: true
  expect:
    exit_code: 0
window:
  start: 30s                    # campaign default window
  duration: 10s
expect:                         # campaign default expectations
  exit_code: 0
  stderr_not_contains: [Traceback]
channels:
  gps_latitude:
    expect:
      exit_code: 0
      stdout_contains: [DEGRADED_MODE]
    dropout:
      durations: [2s, 10s]      # one case per duration
    noise:
      std: [0.0001, 0.001]
  motor_temperature:
    drift:
      rates: [0.1, 0.5]
      duration: 20s
    bias:
      offsets: [-5, 5]
timeline:
  timestamp_gap:
    durations: [1s, 5s]
```

Each parameter variant expands into exactly ONE isolated single-fault case;
multi-fault combinations are never generated. Cases are named deterministically
from the channel, fault type, and parameters. The campaign YAML holds data and
expectations only: the target command comes exclusively from the CLI's
`-- <command>` arguments and is never read from YAML.

Case selection:

- `--plan` validates the campaign, expands every case, and preflights each
  fault against the real input — without executing any target or writing data.
- `--list-cases` prints the generated case IDs with their parameters and
  seeds, then exits without executing.
- `--channel NAME` (repeatable) only generates cases for that channel;
  `--fault TYPE` (repeatable) only generates cases for that fault type.
- `--max-cases N` overrides the campaign's `max_cases` limit (default 100);
  N must be a positive integer.

**Expectation inheritance** (campaign < channel < fault): a case's
expectations start from the campaign-level `expect:`, channel-level keys
refine them, and fault-level keys refine further. A more specific assertion
key REPLACES the less-specific value — lists are not merged. So if the
campaign expects `stdout_contains: [READY]` and a channel sets
`stdout_contains: [DEGRADED_MODE]`, cases for that channel check
`DEGRADED_MODE` only.

**Window inheritance** (fault > channel > campaign > full-input-duration
default): the fault block's window wins first, then the channel's `window:`,
then the campaign's `window:`, and finally the full input duration as the
default. Every generated case gets an explicit start and duration.

**Deterministic seeds:** each case's seed is the SHA-256 of the root seed,
the target (channel or timeline section), the fault type, and the normalized
parameters — so reordering keys in the YAML never changes seeds, and the same
campaign always generates the same cases.

**Coverage:** `coverage.json`/`.md`/`.csv` report one status per generated
case — `PASS`, `FAIL`, or `NOT EXECUTED` — plus
`configured_pass_rate` (passes over configured cases). Aggregate
channel/fault cells in the matrix can additionally be `PARTIAL` (a mix of
pass and fail), and matrix cells with no configured case are `NOT
CONFIGURED`. `PARTIAL` and `NOT CONFIGURED` never describe an individual
case. `summary.json` records
the campaign and input fingerprints (hashes identifying the campaign
definition and the input dataset) so a result can be traced back to exactly
what ran.

Artifacts (`--artifacts <dir>`) reuse the suite layout, plus campaign files:

```
resilience-results/
  expanded-suite.yaml  # the campaign expanded into an explicit suite
  coverage.json        # per-case coverage statuses + configured_pass_rate
  coverage.md          # Markdown coverage table
  coverage.csv         # CSV coverage table
  summary.json         # machine-readable: fingerprints, baseline, per-case status
  summary.md           # Markdown table + failure explanations
  junit.xml            # each case as a JUnit testcase
  baseline/            # stdout.txt, stderr.txt, result.json
  <case-name>/         # faults.json, stdout.txt, stderr.txt, result.json
                       # + corrupted.parquet when --keep-data is given
```

The same artifact-directory safety rules as suites apply: a non-empty
`--artifacts` directory is refused unless it carries the
telemetry-resilience ownership marker (`--overwrite-artifacts`).

**Limitations:** campaigns generate single-fault cases only — every case
isolates exactly one mutation. Coverage measures how your application
responded to the configured campaign, not system safety.

## Deterministic seeds

Every scenario carries a `seed`. With the same input file, scenario file,
seed, Telemetry Resilience version, and compatible dependency versions,
fault *selection* and the generated *corruption values* are deterministic:
re-running the injection reproduces the same corrupted dataset values and
an identical manifest.

Note the distinction between **deterministic values** and **byte-for-byte
file serialization**: Parquet/JSON/CSV writers may embed library-version
metadata, so identical values are not guaranteed to serialize to
byte-identical files on every machine or library version. Share the
scenario file and anyone can regenerate the same corrupted values you
tested against.

## Manifests: ground truth, not intent

The fault manifest records **what actually happened** — the concrete channels,
time windows, parameters, and affected observation counts produced by the run —
not just what the scenario asked for. If a fault window falls outside the data
or a channel is missing, the manifest says so. Treat the manifest as the test's
source of truth when asserting on behavior.

## File formats

- **Parquet (recommended).** Dtypes round-trip exactly: nullable integers,
  timezones, `float32`. Use it whenever you can.
- **CSV.** Plain text, but dtypes don't survive a bare round trip, so every CSV
  we write gets its own schema sidecar: `demo_drive.schema.json` sitting next
  to `demo_drive.csv`, and a corrupted `demo_drive.corrupted.csv` gets its own
  `demo_drive.corrupted.schema.json` (the input's sidecar is never modified).
  The sidecar records each column's dtype and the time column; on read we
  restore `datetime64`, nullable `Int64`/`Int32`/`boolean`, floats, and strings,
  and raise a clear error on mismatch. Reading a corrupted CSV prefers its own
  sidecar, falling back to the input's sidecar only for files written by
  older versions.
- **JSONL.** One JSON object per line, ISO-formatted timestamps.

## Fault semantics that matter

- **`dropout` vs `timestamp_gap`.** `dropout` nulls a channel's observations but
  keeps every row. `timestamp_gap` removes the rows entirely. Code that indexes
  by row position sees the difference — choose deliberately.
- **`sample_rate` vs row deletion.** `sample_rate` degrades one channel
  (observations nulled, timeline intact); it never deletes rows. If you want
  rows gone, that's `timestamp_gap`. Thinning is deterministic and supports
  non-integer ratios: for window positions `i = 0..n-1`, position `i` is kept
  when `floor(i·to/from) > floor((i-1)·to/from)` (position 0 always kept), so
  the retained count is exactly `floor((n-1)·to/from) + 1` — approximately
  `n·to/from`, evenly spread. The manifest records `requested_from_hz`,
  `requested_to_hz`, `window_samples`, `retained_samples`,
  `observations_made_unavailable`, and `effective_ratio` (the achieved
  thinning ratio, not a measured physical sample rate).

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success — or, for `test`/`suite`/`campaign`, all expectations passed |
| 1 | `test`/`suite`/`campaign` expectations failed, baseline failed, or the target command timed out |
| 2 | Invalid input: bad scenario/suite, missing file, unknown fault, unsupported format, refusing to overwrite, non-empty unowned artifact directory, or the target executable could not be launched (missing executable, permission denied) |
| 3 | Internal error (unexpected exception) or user interrupt (Ctrl+C) |

## Scenario format

```yaml
version: 1
seed: 3811
input:
  file: demo_drive.parquet      # relative paths resolve against the scenario file
  time_column: timestamp
faults:
  - channel: gps_latitude
    type: dropout
    start: 90s
    duration: 12s
  - channel: motor_temperature
    type: drift
    start: 120s
    duration: 30s
    rate: 0.4                   # units per second
  - channel: imu_z
    type: sample_rate
    start: 180s
    duration: 20s
    from: 10Hz
    to: 2Hz
expect:
  exit_code: 0
  stdout_contains: [DEGRADED_MODE]
  stderr_not_contains: [Traceback]
```

`start`/`duration` are given in seconds (`90s`). Every scenario is validated on
load — anything malformed raises a clear error instead of silently doing the
wrong thing. Only the keys `version`, `seed`, `input`, `faults`, and `expect`
are allowed at the top level (and only `file`/`time_column` under `input`);
anything else is rejected as a likely typo.

The `expect:` block is optional, but a crashing target can never silently
pass: the default expected exit code is always `0` unless the scenario sets
`expect.exit_code` explicitly. So no `expect` block + target exit 0 ⇒ PASS,
while no `expect` block + target exit 9 ⇒ FAIL.

Supported assertion keys: `exit_code`, `stdout_contains`, `stdout_not_contains`,
`stderr_contains`, `stderr_not_contains`, and `max_duration_seconds` (the target
must finish within that many seconds).

## Fault operators

| Operator | Semantics |
|----------|-----------|
| `dropout` | Null a channel's observations over the window; rows kept |
| `freeze` | Hold the last valid value through the window (flatline) |
| `spike` | Inject short excursions of `magnitude` in `direction` |
| `bias` | Add a constant offset to the channel in the window |
| `drift` | Add a linear drift of `rate` per second over the window |
| `noise` | Add Gaussian noise with standard deviation `std` over the window |
| `clipping` | Clamp channel values to [`min`, `max`] in the window |
| `timestamp_gap` | Delete rows whose timestamps fall in the window (timeline fault: no `channel`) |
| `timestamp_jitter` | Shift each timestamp in the window by a uniform random delta in [`-max_jitter_ms`, `+max_jitter_ms`] (timeline fault: no `channel`; rows are not re-sorted) |
| `sample_rate` | Degrade a channel from `from` to `to`; observations nulled, timeline intact |
| `scale` | Affine transform `value * factor + offset` in the window |

## Type safety / cast rule

The engine never silently changes a channel's dtype. Instead of guessing, it
raises a `FaultError` telling you exactly which `cast:` to request:

- `spike`, `drift`, `noise` are float-only ops: the channel must already be
  `float32`/`float64`, or you opt in explicitly with `cast: float32` (or
  `float64`) — the column then keeps that float dtype.
- `dropout` and `sample_rate` need a null-capable dtype (float, or nullable
  `Int64`/`Int32`/`boolean`); a numpy `int64`/`bool` column needs an explicit
  cast to a nullable dtype first.
- `bias`, `scale`, `clipping` apply integer-safe coercions automatically when
  the parameters are integral (e.g. an integral `bias` offset on an `Int64`
  column); otherwise they ask for an explicit `cast`.
- After each numeric op, an untouched `float32` column is restored to
  `float32`, so dtypes never widen behind your back.

Example: a `spike` on the nullable-`Int64` `rpm` channel needs
`cast: float32`, as in `examples/degraded_navigation.yaml`.

## CI status

Cross-platform CI matrix (Python 3.11 / 3.12 / 3.13 × Ubuntu / macOS / Windows,
see `.github/workflows/ci.yml`) is verified green: 9/9 jobs pass. Each job
runs the full test suite (377 passed, 4 skipped — the skips are pre-existing
packaging tests that require a `dist/` build to be present), the standalone
packaging regression tests (5 passed), a wheel/sdist build from a separate
source copy, a wheel-only install with `doctor` and CLI smoke tests, and a
checkout debris check.

## Dependencies and licenses

Runtime dependencies (declared in `pyproject.toml`):

| Dependency | License |
|------------|---------|
| typer | BSD-3-Clause |
| pandas | BSD-3-Clause |
| numpy | BSD-3-Clause |
| pyarrow | Apache-2.0 |
| PyYAML | MIT |

Test dependency: pytest (MIT). Telemetry Resilience itself is licensed under
the Apache License 2.0 (see `LICENSE`).

## Not a safety certification

Telemetry Resilience tells you how your software responded to the faults you
configured. It does not certify that a physical system is safe or reliable,
and its coverage numbers measure test outcomes, not system safety.
Corrupted outputs are synthetic test data — never feed them back as real
telemetry.

## Commercial support

Telemetry Resilience is free and open source (Apache 2.0). If your team
needs help putting it to work — integrating it into CI, designing fault
scenarios for your telemetry, or support adopting it — commercial support
and integration engagements are available.

Contact: Hopiedavis1@icloud.com
