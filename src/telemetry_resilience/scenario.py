"""Scenario file parsing and validation.

A scenario is a YAML mapping like::

    version: 1
    seed: 42
    input:
      file: data.csv
      time_column: timestamp
    faults:
      - type: dropout
        channel: imu_z
        start: 10s
        duration: 5s
      - type: spike
        channel: imu_z
        start: 30s
        duration: 2s
        magnitude: 5.0
        direction: positive
        count: 3

Every validation failure raises ScenarioError with a message identifying
the offending fault, e.g. "fault #2 (spike on 'imu_z'): magnitude must be
numeric". Nothing is ever silently ignored.
"""
import math
import re

import yaml

from .faults import FAULT_TYPES
from .models import FaultError, FaultSpec, Scenario, ScenarioError

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)\s*$", re.IGNORECASE)
_FREQUENCY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*hz\s*$", re.IGNORECASE)
_DURATION_FACTORS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_CASTS = ("float32", "float64", "Int64", "Int32")
_FAULT_KEYS = ("type", "channel", "start", "duration", "cast")
# Timeline faults operate on the global time axis, not on a data channel;
# a scenario must not (and cannot) assign them to a sensor channel.
_TIMELINE_FAULTS = ("timestamp_gap", "timestamp_jitter")
_TOP_LEVEL_KEYS = ("version", "seed", "input", "faults", "expect")
_INPUT_KEYS = ("file", "time_column")
_MISSING = object()


def parse_duration(value) -> float:
    """Parse a duration to seconds.

    Accepts a plain int/float (seconds) or a string like '500ms', '10s',
    '2m', '1.5h' (case-insensitive, surrounding whitespace allowed).
    Negative or non-finite values raise ScenarioError.
    """
    if isinstance(value, bool):
        raise ScenarioError(
            f"invalid duration {value!r}: expected a number of seconds "
            "or a string like '500ms'"
        )
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        match = _DURATION_RE.match(value)
        if not match:
            raise ScenarioError(
                f"invalid duration {value!r}: expected a number of seconds "
                "or a string like '500ms', '10s', '2m', '1.5h'"
            )
        seconds = float(match.group(1)) * _DURATION_FACTORS[match.group(2).lower()]
    else:
        raise ScenarioError(
            f"invalid duration {value!r}: expected a number of seconds "
            "or a duration string"
        )
    if not math.isfinite(seconds):
        raise ScenarioError(f"invalid duration {value!r}: must be finite")
    if seconds < 0:
        raise ScenarioError(f"invalid duration {value!r}: must not be negative")
    return seconds


def parse_frequency(value) -> float:
    """Parse a frequency to Hz.

    Accepts a plain int/float (Hz) or a string like '100Hz' / '25 hz'
    (case-insensitive, optional space). Must be > 0, else ScenarioError.
    """
    if isinstance(value, bool):
        raise ScenarioError(
            f"invalid frequency {value!r}: expected a positive number of Hz "
            "or a string like '100Hz'"
        )
    if isinstance(value, (int, float)):
        hz = float(value)
    elif isinstance(value, str):
        match = _FREQUENCY_RE.match(value)
        if not match:
            raise ScenarioError(
                f"invalid frequency {value!r}: expected a positive number "
                "or a string like '100Hz'"
            )
        hz = float(match.group(1))
    else:
        raise ScenarioError(
            f"invalid frequency {value!r}: expected a positive number of Hz"
        )
    if not math.isfinite(hz) or hz <= 0:
        raise ScenarioError(f"invalid frequency {value!r}: must be > 0")
    return hz


def _parse_fault(index, raw):
    tag = f"fault #{index}"
    if not isinstance(raw, dict):
        raise ScenarioError(f"{tag}: must be a mapping")
    ftype = raw.get("type")
    if not isinstance(ftype, str) or ftype not in FAULT_TYPES:
        raise ScenarioError(
            f"{tag}: unknown fault type {ftype!r}; "
            f"must be one of {sorted(FAULT_TYPES)}"
        )
    channel = raw.get("channel")
    if ftype in _TIMELINE_FAULTS:
        if channel is not None:
            raise ScenarioError(
                f"{tag} ({ftype}): 'channel' must be absent; {ftype} is a "
                "global timeline fault and cannot target a sensor channel"
            )
        label = f"{tag} ({ftype})"
    else:
        if not isinstance(channel, str) or not channel:
            raise ScenarioError(
                f"{tag} ({ftype}): 'channel' is required and must be "
                "a non-empty string"
            )
        label = f"{tag} ({ftype} on '{channel}')"

    if raw.get("start", _MISSING) is _MISSING:
        raise ScenarioError(f"{label}: 'start' is required")
    if raw.get("duration", _MISSING) is _MISSING:
        raise ScenarioError(f"{label}: 'duration' is required")
    try:
        start_seconds = parse_duration(raw["start"])
    except ScenarioError as exc:
        raise ScenarioError(f"{label}: invalid 'start': {exc}") from exc
    try:
        duration_seconds = parse_duration(raw["duration"])
    except ScenarioError as exc:
        raise ScenarioError(f"{label}: invalid 'duration': {exc}") from exc
    if duration_seconds <= 0:
        raise ScenarioError(f"{label}: 'duration' must be > 0")

    cast = raw.get("cast")
    if cast is not None and cast not in _CASTS:
        raise ScenarioError(
            f"{label}: 'cast' must be one of {list(_CASTS)} (got {cast!r})"
        )

    params = {k: v for k, v in raw.items() if k not in _FAULT_KEYS}
    try:
        normalized = FAULT_TYPES[ftype].validate(params)
    except FaultError as exc:
        raise ScenarioError(f"{label}: {exc}") from exc

    return FaultSpec(
        channel=channel,
        type=ftype,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        params=normalized,
        cast=cast,
    )


def _parse_expect(raw):
    if not isinstance(raw, dict):
        raise ScenarioError("scenario: 'expect' must be a mapping")
    for key in raw:
        if key not in (
            "exit_code",
            "stdout_contains",
            "stdout_not_contains",
            "stderr_contains",
            "stderr_not_contains",
            "max_duration_seconds",
        ):
            raise ScenarioError(f"scenario: unknown 'expect' key {key!r}")
    expect = {}
    if "exit_code" in raw:
        value = raw["exit_code"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScenarioError("scenario: 'expect.exit_code' must be an integer")
        expect["exit_code"] = value
    for key in (
        "stdout_contains",
        "stdout_not_contains",
        "stderr_contains",
        "stderr_not_contains",
    ):
        if key in raw:
            value = raw[key]
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise ScenarioError(
                    f"scenario: 'expect.{key}' must be a list of strings"
                )
            expect[key] = list(value)
    if "max_duration_seconds" in raw:
        value = raw["max_duration_seconds"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ScenarioError(
                "scenario: 'expect.max_duration_seconds' must be a "
                "positive number"
            )
        expect["max_duration_seconds"] = float(value)
    return expect


def load_scenario(path) -> Scenario:
    """Load and fully validate a scenario YAML file."""
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError:
        raise ScenarioError(f"scenario file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"scenario file {path}: top level must be a mapping")

    for key in raw:
        if key not in _TOP_LEVEL_KEYS:
            raise ScenarioError(
                f"scenario: unknown top-level key {key!r}; "
                f"allowed keys are {sorted(_TOP_LEVEL_KEYS)}"
            )

    version = raw.get("version", _MISSING)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ScenarioError(
            f"scenario: 'version' must be 1 (got {raw.get('version', _MISSING)!r})"
        )
    seed = raw.get("seed", _MISSING)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ScenarioError(f"scenario: 'seed' must be an integer (got {seed!r})")

    input_cfg = raw.get("input", _MISSING)
    if not isinstance(input_cfg, dict):
        raise ScenarioError("scenario: 'input' must be a mapping")
    for key in input_cfg:
        if key not in _INPUT_KEYS:
            raise ScenarioError(
                f"scenario: unknown 'input' key {key!r}; "
                f"allowed keys are {sorted(_INPUT_KEYS)}"
            )
    input_file = input_cfg.get("file")
    if not isinstance(input_file, str) or not input_file:
        raise ScenarioError("scenario: 'input.file' must be a non-empty string")
    time_column = input_cfg.get("time_column")
    if not isinstance(time_column, str) or not time_column:
        raise ScenarioError("scenario: 'input.time_column' must be a non-empty string")

    faults_raw = raw.get("faults", _MISSING)
    if not isinstance(faults_raw, list) or not faults_raw:
        raise ScenarioError("scenario: 'faults' must be a non-empty list")
    faults = [_parse_fault(i, item) for i, item in enumerate(faults_raw, start=1)]

    expect_raw = raw.get("expect")
    expect = _parse_expect(expect_raw) if expect_raw is not None else None

    return Scenario(
        version=1,
        seed=seed,
        input_file=input_file,
        time_column=time_column,
        faults=faults,
        expect=expect,
    )
