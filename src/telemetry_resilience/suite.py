"""Resilience suite definitions: multi-case fault collections for CI.

A suite YAML looks like::

    version: 1
    name: navigation-resilience
    input:
      file: drive.parquet
      time_column: timestamp
    timeout_seconds: 60
    baseline:
      enabled: true
      expect:
        exit_code: 0
        stdout_contains: [READY]
    cases:
      - name: gps-dropout
        seed: 1001
        timeout_seconds: 10
        faults:
          - channel: gps_latitude
            type: dropout
            start: 30s
            duration: 10s
        expect:
          exit_code: 0
          stdout_contains: [DEGRADED_MODE]

Validation is strict: unknown keys, duplicate/empty/unsafe case names,
missing seeds, empty fault lists, invalid timeouts, bad expectations and
bad fault definitions all raise ScenarioError. Nothing is silently
ignored. Suite YAML holds DATA and expectations only -- it can never
specify a shell command; the target command comes exclusively from the
CLI's ``-- <command>`` arguments.
"""
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .models import FaultSpec, Scenario, ScenarioError
from .scenario import _parse_expect, _parse_fault

_SUITE_TOP_KEYS = (
    "version",
    "name",
    "input",
    "baseline",
    "cases",
    "timeout_seconds",
)
_SUITE_INPUT_KEYS = ("file", "time_column")
_BASELINE_KEYS = ("enabled", "expect")
_CASE_KEYS = ("name", "seed", "faults", "expect", "timeout_seconds")
_DEFAULT_TIMEOUT_SECONDS = 60.0
_MISSING = object()

# Artifact-directory names reserved for the suite's own reports. A case may
# never normalize to one of these (compared case-insensitively), otherwise
# the case could overwrite the suite's internal artifacts.
_RESERVED_ARTIFACT_NAMES = frozenset(
    {"baseline", "summary.json", "summary.md", "junit.xml"}
)
# Ownership marker written into every artifacts directory we create, so we
# can tell our own artifact directories apart from unrelated user data.
# Format version 2 records exactly which root entries the run created
# (``owned_entries``); --overwrite-artifacts may only delete/replace those
# recorded entries, never untracked user files.
ARTIFACT_MARKER = ".telemetry-resilience-artifacts.json"
ARTIFACT_FORMAT_VERSION = 2

# Characters that are unsafe on common Windows/macOS/Linux filesystems.
_UNSAFE_NAME_CHARS = frozenset('<>:"|?*')
# Reserved Windows device names (compared against the name stem, lowercase).
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


@dataclass
class SuiteCase:
    """One named fault scenario inside a suite."""

    name: str
    seed: int
    faults: list
    expect: dict | None
    timeout_seconds: float | None  # None -> use the suite-level default

    def to_scenario(self, input_file: str, time_column: str) -> Scenario:
        """Build a regular Scenario so cases reuse the fault engine."""
        return Scenario(
            version=1,
            seed=self.seed,
            input_file=input_file,
            time_column=time_column,
            faults=self.faults,
            expect=self.expect,
        )


@dataclass
class Suite:
    """A validated resilience suite."""

    name: str
    input_file: str
    time_column: str
    baseline_enabled: bool
    baseline_expect: dict | None
    timeout_seconds: float
    cases: list


def _parse_timeout(value, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ScenarioError(
            f"{where}: 'timeout_seconds' must be a positive number "
            f"(got {value!r})"
        )
    return float(value)


def _validate_case_name(name, tag: str) -> str:
    """Reject empty names and anything that could escape an artifact dir."""
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{tag}: 'name' must be a non-empty string")
    if name != name.strip():
        raise ScenarioError(
            f"{tag}: case name {name!r} must not have leading/trailing whitespace"
        )
    if os.path.isabs(name):
        raise ScenarioError(
            f"{tag}: case name {name!r} must not be an absolute path"
        )
    if "/" in name or "\\" in name:
        raise ScenarioError(
            f"{tag}: case name {name!r} must not contain path separators"
        )
    if name in (".", ".."):
        raise ScenarioError(f"{tag}: case name {name!r} is not allowed")
    bad = sorted({c for c in name if c in _UNSAFE_NAME_CHARS})
    if bad:
        raise ScenarioError(
            f"{tag}: case name {name!r} contains characters unsafe for "
            f"common filesystems: {''.join(bad)!r}"
        )
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ScenarioError(
            f"{tag}: case name {name!r} must not contain control characters "
            "(including NUL and newlines)"
        )
    if name.endswith("."):
        raise ScenarioError(
            f"{tag}: case name {name!r} must not end with a dot"
        )
    stem = name.split(".")[0].lower()
    if stem in _WINDOWS_DEVICE_NAMES:
        raise ScenarioError(
            f"{tag}: case name {name!r} is a reserved device name"
        )
    return name


def safe_case_dirname(name: str) -> str:
    """Normalize an already-validated case name for disk use.

    Raises ScenarioError if the name normalizes to an empty directory name.
    """
    dirname = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not dirname:
        raise ScenarioError(
            f"case name {name!r} normalizes to an empty artifact "
            "directory name"
        )
    return dirname


def _parse_case(index: int, raw) -> SuiteCase:
    tag = f"case #{index}"
    if not isinstance(raw, dict):
        raise ScenarioError(f"{tag}: must be a mapping")
    for key in raw:
        if key not in _CASE_KEYS:
            raise ScenarioError(
                f"{tag}: unknown key {key!r}; allowed keys are "
                f"{sorted(_CASE_KEYS)}"
            )
    name = _validate_case_name(raw.get("name"), tag)

    seed = raw.get("seed", _MISSING)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ScenarioError(
            f"case '{name}': 'seed' is required and must be an integer "
            f"(got {seed!r})"
        )

    faults_raw = raw.get("faults", _MISSING)
    if not isinstance(faults_raw, list) or not faults_raw:
        raise ScenarioError(
            f"case '{name}': 'faults' must be a non-empty list"
        )
    faults = []
    for i, item in enumerate(faults_raw, start=1):
        try:
            faults.append(_parse_fault(i, item))
        except ScenarioError as exc:
            raise ScenarioError(f"case '{name}': {exc}") from exc

    expect_raw = raw.get("expect")
    try:
        expect = _parse_expect(expect_raw) if expect_raw is not None else None
    except ScenarioError as exc:
        raise ScenarioError(f"case '{name}': {exc}") from exc

    timeout_raw = raw.get("timeout_seconds")
    timeout = (
        _parse_timeout(timeout_raw, f"case '{name}'")
        if timeout_raw is not None
        else None
    )

    return SuiteCase(
        name=name, seed=seed, faults=faults, expect=expect,
        timeout_seconds=timeout,
    )


def load_suite(path) -> Suite:
    """Load and fully validate a suite YAML file."""
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError:
        raise ScenarioError(f"suite file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"suite file {path}: top level must be a mapping")

    for key in raw:
        if key not in _SUITE_TOP_KEYS:
            raise ScenarioError(
                f"suite: unknown top-level key {key!r}; allowed keys are "
                f"{sorted(_SUITE_TOP_KEYS)}"
            )

    version = raw.get("version", _MISSING)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ScenarioError(
            f"suite: 'version' must be 1 (got {raw.get('version', _MISSING)!r})"
        )

    name = raw.get("name", Path(path).stem)
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError("suite: 'name' must be a non-empty string")

    input_cfg = raw.get("input", _MISSING)
    if not isinstance(input_cfg, dict):
        raise ScenarioError("suite: 'input' must be a mapping")
    for key in input_cfg:
        if key not in _SUITE_INPUT_KEYS:
            raise ScenarioError(
                f"suite: unknown 'input' key {key!r}; allowed keys are "
                f"{sorted(_SUITE_INPUT_KEYS)}"
            )
    input_file = input_cfg.get("file")
    if not isinstance(input_file, str) or not input_file:
        raise ScenarioError("suite: 'input.file' must be a non-empty string")
    time_column = input_cfg.get("time_column")
    if not isinstance(time_column, str) or not time_column:
        raise ScenarioError(
            "suite: 'input.time_column' must be a non-empty string"
        )

    timeout_seconds = _parse_timeout(
        raw.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS), "suite"
    )

    baseline_cfg = raw.get("baseline")
    baseline_enabled = False
    baseline_expect = None
    if baseline_cfg is not None:
        if not isinstance(baseline_cfg, dict):
            raise ScenarioError("suite: 'baseline' must be a mapping")
        for key in baseline_cfg:
            if key not in _BASELINE_KEYS:
                raise ScenarioError(
                    f"suite: unknown 'baseline' key {key!r}; allowed keys are "
                    f"{sorted(_BASELINE_KEYS)}"
                )
        enabled = baseline_cfg.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ScenarioError("suite: 'baseline.enabled' must be true/false")
        baseline_enabled = enabled
        expect_raw = baseline_cfg.get("expect")
        try:
            baseline_expect = (
                _parse_expect(expect_raw) if expect_raw is not None else None
            )
        except ScenarioError as exc:
            raise ScenarioError(f"suite baseline: {exc}") from exc

    cases_raw = raw.get("cases", _MISSING)
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ScenarioError("suite: 'cases' must be a non-empty list")
    cases = [_parse_case(i, item) for i, item in enumerate(cases_raw, start=1)]

    seen = set()
    dirnames = {}
    for case in cases:
        if case.name in seen:
            raise ScenarioError(
                f"suite: duplicate case name {case.name!r}"
            )
        seen.add(case.name)
        # Derive every case's artifact directory name up front and reject
        # collisions before anything executes. Comparison is
        # case-insensitive so the suite stays safe on case-insensitive
        # filesystems; otherwise two cases could silently overwrite each
        # other's artifacts.
        dirname = safe_case_dirname(case.name)
        key = dirname.lower()
        if key in dirnames:
            raise ScenarioError(
                f"suite: case names {dirnames[key]!r} and {case.name!r} map "
                f"to the same artifact directory {dirname!r}"
            )
        dirnames[key] = case.name
        if key in _RESERVED_ARTIFACT_NAMES or key == ARTIFACT_MARKER:
            raise ScenarioError(
                f"suite: case name {case.name!r} is reserved (it would "
                f"collide with the suite's own artifact {dirname!r})"
            )

    return Suite(
        name=name,
        input_file=input_file,
        time_column=time_column,
        baseline_enabled=baseline_enabled,
        baseline_expect=baseline_expect,
        timeout_seconds=timeout_seconds,
        cases=cases,
    )


def verify_suite_channels(df: pd.DataFrame, suite: Suite) -> None:
    """Check time column and every case's channels against the input data.

    Used by --dry-run (and as a pre-flight for real runs) so an invalid
    channel is reported as a configuration error before anything executes.
    The fault engine re-validates again when actually applying faults.
    """
    time_col = suite.time_column
    if time_col not in df.columns:
        raise ScenarioError(
            f"suite: time column '{time_col}' not found in input data"
        )
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise ScenarioError(
            f"suite: time column '{time_col}' must be datetime64 "
            f"(got {df[time_col].dtype})"
        )
    for case in suite.cases:
        for fspec in case.faults:
            if fspec.type in ("timestamp_gap", "timestamp_jitter"):
                continue  # timeline faults target no data channel
            if fspec.channel not in df.columns:
                raise ScenarioError(
                    f"case '{case.name}': channel '{fspec.channel}' not "
                    "found in input data"
                )
