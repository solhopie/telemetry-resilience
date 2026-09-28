"""Telemetry mutation campaign expansion.

A *campaign* is a compact YAML description of a whole family of
single-fault mutation cases. Each parameter variant listed under a fault
expands into exactly ONE isolated single-fault case; multi-fault
combinations are never generated. The expanded cases reuse the regular
fault engine (:mod:`telemetry_resilience.engine`) and suite machinery
(:mod:`telemetry_resilience.suite`) unchanged.

Campaign YAML sketch::

    version: 1
    name: navigation-mutations
    seed: 5000
    input: {file: drive.parquet, time_column: timestamp}
    channels:
      gps_latitude:
        dropout: {durations: [2s, 10s], start: 90s}
        spike: {magnitudes: [5, 10], directions: [positive]}
    timeline:
      timestamp_jitter: {max_jitter_ms: [20, 100], duration: 15s}

Validation is strict at every level: any unknown key raises
:class:`~telemetry_resilience.models.ScenarioError`. Nothing is ever
silently ignored.

Inheritance rules
-----------------
*Window* (fault window override > channel ``window`` > campaign
``window`` > default). The effective start defaults to ``0s``; the
effective duration defaults to the full input time span (derived from
the loaded dataframe's time column, and an error if it cannot be
determined or is not > 0). Every generated case carries explicit
``start_seconds`` / ``duration_seconds`` floats, and the resolved
duration must be > 0.

*Expectations* (campaign < channel < fault). The merged expectation is::

    merged = {**campaign_expect, **channel_expect, **fault_expect}

A more specific assertion KEY replaces the less-specific value for that
key; lists are NOT concatenated. For example, if the campaign sets
``stdout_contains: [A]`` and a fault sets ``stdout_contains: [B]``, the
case expects exactly ``[B]``.

Case IDs look like ``{target}__{fault}__{param-parts}`` where ``target``
is the channel name or ``timeline`` and the param parts are sorted by
key, e.g. ``gps_latitude__dropout__duration-2s`` or
``pressure__spike__direction-positive_magnitude-10``.

Seeds are deterministic content hashes (see :func:`case_seed`): the
same campaign definition always yields the same case seeds, and YAML
reordering never changes existing case seeds. Seeds are never derived
from list ordering.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .faults import FAULT_TYPES
from .faults._common import is_number, parse_hz
from .models import FaultError, FaultSpec, ScenarioError
from .scenario import (
    _CASTS,
    _TIMELINE_FAULTS,
    _parse_expect,
    _parse_fault,
    parse_duration,
)
from .suite import (
    ARTIFACT_MARKER,
    _RESERVED_ARTIFACT_NAMES,
    _parse_timeout,
    _validate_case_name,
    safe_case_dirname,
    Suite,
    SuiteCase,
)

#: Default cap on the number of cases a campaign may expand to.
DEFAULT_MAX_CASES = 100

_DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_ID_LEN = 120
_MISSING = object()

_CAMPAIGN_TOP_KEYS = (
    "version",
    "name",
    "seed",
    "input",
    "timeout_seconds",
    "max_cases",
    "baseline",
    "window",
    "expect",
    "channels",
    "timeline",
)
_CAMPAIGN_INPUT_KEYS = ("file", "time_column")
_BASELINE_KEYS = ("enabled", "expect")
_WINDOW_KEYS = ("start", "duration")
_CHANNEL_RESERVED_KEYS = ("expect", "window", "cast")
_FAULT_COMMON_KEYS = ("start", "duration", "expect", "cast")

# Expansion-axis keys per fault type. Every other key in a fault config
# must be one of _FAULT_COMMON_KEYS.
_FAULT_AXIS_KEYS = {
    "dropout": ("durations",),
    "freeze": ("durations",),
    "spike": ("magnitudes", "directions"),
    "bias": ("offsets",),
    "drift": ("rates",),
    "noise": ("std",),
    "clipping": ("bounds",),
    "sample_rate": ("transitions",),
    "scale": ("transforms",),
    "timestamp_gap": ("durations",),
    "timestamp_jitter": ("max_jitter_ms",),
}
_CHANNEL_FAULTS = tuple(f for f in FAULT_TYPES if f not in _TIMELINE_FAULTS)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a canonical duration string.

    Integral seconds become ``"Ns"`` (``90`` -> ``"90s"``), sub-second
    durations with an integral millisecond value become ``"Nms"``
    (``0.5`` -> ``"500ms"``), and anything else uses the shortest ``%g``
    representation with an ``"s"`` suffix (``1.5`` -> ``"1.5s"``).
    """
    seconds = float(seconds)
    if not math.isfinite(seconds):
        raise ScenarioError(f"invalid duration {seconds!r}: must be finite")
    if seconds < 0:
        raise ScenarioError(f"invalid duration {seconds!r}: must not be negative")
    if seconds.is_integer():
        return f"{int(seconds)}s"
    if seconds < 1:
        ms = seconds * 1000.0
        if abs(ms - round(ms)) < 1e-6:
            return f"{int(round(ms))}ms"
    return f"{seconds:g}s"


def _fmt_number(value) -> str:
    """Shortest human representation of a number: 2.0 -> '2', 0.1 -> '0.1'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fmt_id_value(key: str, value) -> str:
    """Format one id-param value for a case-ID part."""
    if key == "duration":
        return format_duration(value)
    if key in ("from", "to"):
        return _fmt_number(value) + "hz"
    if isinstance(value, str):
        return value
    return _fmt_number(value)


def _build_case_name(target: str, fault_type: str, id_params: dict) -> str:
    """Build ``{target}__{fault}__{param-parts}`` with parts sorted by key.

    Clipping keeps its natural ``min``-then-``max`` order (matching the
    spec's ``min-0_max-100`` example). Over-long IDs are truncated to 100
    chars plus a short sha256 digest so the name stays unique and
    filesystem-safe.
    """
    fixed_order = {"clipping": ("min", "max")}.get(fault_type)
    if fixed_order is not None:
        keys = [k for k in fixed_order if k in id_params]
    else:
        keys = sorted(id_params)
    parts = [f"{key}-{_fmt_id_value(key, id_params[key])}" for key in keys]
    human = f"{target}__{fault_type}__" + "_".join(parts)
    if len(human) > _MAX_ID_LEN:
        digest = hashlib.sha256(human.encode("utf-8")).hexdigest()[:8]
        human = human[:100] + "-" + digest
    return human


def case_seed(
    root_seed: int, target_label: str, fault_type: str, seed_material: dict
) -> int:
    """Derive a deterministic per-case seed from content, never ordering.

    ``seed_material`` is ``{"start", "duration", "cast"}`` plus the
    engine-normalized fault params (the dict returned by
    ``scenario._parse_fault``). The same inputs always produce the same
    seed; YAML reordering never changes existing case seeds.

    The seed is the first 7 bytes of the SHA-256 of the canonical JSON
    payload masked to 53 bits, so it always satisfies
    ``0 <= seed <= 9007199254740991``: exactly representable as a JSON
    number (a seed -> float -> int round-trip is lossless) and valid for
    NumPy's ``SeedSequence`` (which accepts small ints).
    """
    payload = json.dumps(
        {
            "root": root_seed,
            "target": target_label,
            "fault": fault_type,
            "params": seed_material,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        int.from_bytes(hashlib.sha256(payload).digest()[:7], "big")
        & 0x1FFFFFFFFFFFFF
    )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CampaignCase:
    """One expanded single-fault case of a campaign."""

    name: str
    target: str  # channel name, or "timeline"
    target_kind: str  # "channel" | "timeline"
    channel: str | None
    fault_type: str
    spec: FaultSpec  # built via scenario._parse_fault
    seed: int
    expect: dict | None
    id_params: dict  # normalized display params used for the case ID

    def to_suite_case(self) -> SuiteCase:
        """Convert to a regular single-fault SuiteCase."""
        return SuiteCase(
            name=self.name,
            seed=self.seed,
            faults=[self.spec],
            expect=self.expect,
            timeout_seconds=None,
        )


@dataclass
class CampaignConfig:
    """A parsed and validated campaign definition (no dataframe needed)."""

    name: str
    version: int
    seed: int
    input_file: str
    time_column: str
    timeout_seconds: float
    max_cases: int
    baseline_enabled: bool
    baseline_expect: dict | None
    campaign_window: dict | None  # {"start": float|None, "duration": float|None}
    campaign_expect: dict | None
    raw_channels: dict  # channel -> {"expect","window","cast","faults"}
    raw_timeline: dict  # timeline fault -> fault config


# ---------------------------------------------------------------------------
# Campaign YAML parsing (no dataframe required)
# ---------------------------------------------------------------------------


def _parse_window(raw, where: str) -> dict:
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: 'window' must be a mapping")
    for key in raw:
        if key not in _WINDOW_KEYS:
            raise ScenarioError(
                f"{where}: unknown 'window' key {key!r}; "
                f"allowed keys are {sorted(_WINDOW_KEYS)}"
            )
    window: dict = {}
    if "start" in raw:
        try:
            window["start"] = parse_duration(raw["start"])
        except ScenarioError as exc:
            raise ScenarioError(
                f"{where}: invalid window 'start': {exc}"
            ) from exc
    if "duration" in raw:
        try:
            duration = parse_duration(raw["duration"])
        except ScenarioError as exc:
            raise ScenarioError(
                f"{where}: invalid window 'duration': {exc}"
            ) from exc
        if duration <= 0:
            raise ScenarioError(f"{where}: window 'duration' must be > 0")
        window["duration"] = duration
    return window


def _require_list(cfg: dict, key: str, tag: str) -> list:
    values = cfg.get(key, _MISSING)
    if values is _MISSING:
        raise ScenarioError(f"{tag}: requires a non-empty '{key}' list")
    if not isinstance(values, list) or not values:
        raise ScenarioError(f"{tag}: '{key}' must be a non-empty list")
    return values


def _dup_error(tag: str, key: str, value, other) -> ScenarioError:
    return ScenarioError(
        f"{tag}: duplicate values in '{key}' "
        f"({value!r} normalizes to the same value as {other!r})"
    )


def _parse_durations(values: list, tag: str, key: str = "durations") -> list:
    seen: dict = {}
    out = []
    for value in values:
        try:
            seconds = parse_duration(value)
        except ScenarioError as exc:
            raise ScenarioError(
                f"{tag}: invalid duration in '{key}': {exc}"
            ) from exc
        if seconds <= 0:
            raise ScenarioError(
                f"{tag}: '{key}' values must be > 0 (got {value!r})"
            )
        if seconds in seen:
            raise _dup_error(tag, key, value, seen[seconds])
        seen[seconds] = value
        out.append(seconds)
    return out


def _parse_numbers(values: list, tag: str, key: str) -> list:
    seen: dict = {}
    out = []
    for value in values:
        if not is_number(value):
            raise ScenarioError(
                f"{tag}: '{key}' values must be numbers (got {value!r})"
            )
        norm = float(value)
        if norm in seen:
            raise _dup_error(tag, key, value, seen[norm])
        seen[norm] = value
        out.append(value)
    return out


def _parse_directions(cfg: dict, tag: str) -> list:
    if "directions" not in cfg:
        return [None]
    values = cfg["directions"]
    if not isinstance(values, list) or not values:
        raise ScenarioError(f"{tag}: 'directions' must be a non-empty list")
    seen = set()
    out = []
    for value in values:
        if value not in ("positive", "negative"):
            raise ScenarioError(
                f"{tag}: 'directions' values must be 'positive' or "
                f"'negative' (got {value!r})"
            )
        if value in seen:
            raise _dup_error(tag, "directions", value, value)
        seen.add(value)
        out.append(value)
    return out


def _parse_bounds(values: list, tag: str) -> list:
    seen: dict = {}
    out = []
    for item in values:
        if not isinstance(item, dict):
            raise ScenarioError(
                f"{tag}: 'bounds' entries must be mappings with "
                f"'min'/'max' (got {item!r})"
            )
        for key in item:
            if key not in ("min", "max"):
                raise ScenarioError(
                    f"{tag}: unknown 'bounds' key {key!r}; "
                    "allowed keys are ['max', 'min']"
                )
        lo = item.get("min")
        hi = item.get("max")
        if lo is None and hi is None:
            raise ScenarioError(
                f"{tag}: 'bounds' entries need at least one of 'min'/'max'"
            )
        for label, val in (("min", lo), ("max", hi)):
            if val is not None and not is_number(val):
                raise ScenarioError(
                    f"{tag}: 'bounds' {label} must be numeric (got {val!r})"
                )
        if lo is not None and hi is not None and lo > hi:
            raise ScenarioError(
                f"{tag}: 'bounds' min must be <= max (got {item!r})"
            )
        dupkey = (
            None if lo is None else float(lo),
            None if hi is None else float(hi),
        )
        if dupkey in seen:
            raise _dup_error(tag, "bounds", item, seen[dupkey])
        seen[dupkey] = item
        engine = {}
        if lo is not None:
            engine["min"] = lo
        if hi is not None:
            engine["max"] = hi
        out.append((engine, dict(engine)))
    return out


def _parse_transitions(values: list, tag: str) -> list:
    seen: dict = {}
    out = []
    for item in values:
        if not isinstance(item, dict):
            raise ScenarioError(
                f"{tag}: 'transitions' entries must be mappings with "
                f"'from'/'to' (got {item!r})"
            )
        for key in item:
            if key not in ("from", "to"):
                raise ScenarioError(
                    f"{tag}: unknown 'transitions' key {key!r}; "
                    "allowed keys are ['from', 'to']"
                )
        if "from" not in item or "to" not in item:
            raise ScenarioError(
                f"{tag}: 'transitions' entries need both 'from' and 'to' "
                f"(got {item!r})"
            )
        try:
            from_hz = parse_hz(item["from"], "'from'")
            to_hz = parse_hz(item["to"], "'to'")
        except FaultError as exc:
            raise ScenarioError(f"{tag}: {exc}") from exc
        dupkey = (from_hz, to_hz)
        if dupkey in seen:
            raise _dup_error(tag, "transitions", item, seen[dupkey])
        seen[dupkey] = item
        # Raw values pass through to the engine; the engine's own
        # parse_hz validation runs again inside scenario._parse_fault.
        out.append(
            (
                {"from": item["from"], "to": item["to"]},
                {"from": from_hz, "to": to_hz},
            )
        )
    return out


def _parse_transforms(values: list, tag: str) -> list:
    seen: dict = {}
    out = []
    for item in values:
        if not isinstance(item, dict):
            raise ScenarioError(
                f"{tag}: 'transforms' entries must be mappings with "
                f"'factor'/'offset' (got {item!r})"
            )
        for key in item:
            if key not in ("factor", "offset"):
                raise ScenarioError(
                    f"{tag}: unknown 'transforms' key {key!r}; "
                    "allowed keys are ['factor', 'offset']"
                )
        if "factor" not in item:
            raise ScenarioError(
                f"{tag}: 'transforms' entries need 'factor' (got {item!r})"
            )
        factor = item["factor"]
        offset = item.get("offset", 0.0)
        if not is_number(factor):
            raise ScenarioError(
                f"{tag}: 'transforms' factor must be numeric "
                f"(got {factor!r})"
            )
        if not is_number(offset):
            raise ScenarioError(
                f"{tag}: 'transforms' offset must be numeric "
                f"(got {offset!r})"
            )
        dupkey = (float(factor), float(offset))
        if dupkey in seen:
            raise _dup_error(tag, "transforms", item, seen[dupkey])
        seen[dupkey] = item
        engine = {"factor": factor}
        if "offset" in item:
            engine["offset"] = offset
        out.append((engine, {"factor": factor, "offset": offset}))
    return out


def _parse_jitter(values: list, tag: str) -> list:
    seen: dict = {}
    out = []
    for value in values:
        if not is_number(value):
            raise ScenarioError(
                f"{tag}: 'max_jitter_ms' values must be numbers "
                f"(got {value!r})"
            )
        if value <= 0:
            raise ScenarioError(
                f"{tag}: 'max_jitter_ms' values must be > 0 (got {value!r})"
            )
        norm = float(value)
        if norm in seen:
            raise _dup_error(tag, "max_jitter_ms", value, seen[norm])
        seen[norm] = value
        out.append(({"max_jitter_ms": value}, {"max_jitter_ms": value}))
    return out


def _parse_axis(ftype: str, cfg: dict, tag: str) -> list:
    """Expand one fault config into [(raw engine params, id params)].

    Each pair becomes exactly one isolated single-fault case.
    """
    if ftype in ("dropout", "freeze", "timestamp_gap"):
        durations = _parse_durations(_require_list(cfg, "durations", tag), tag)
        return [({}, {"duration": d}) for d in durations]
    if ftype == "spike":
        magnitudes = _parse_numbers(
            _require_list(cfg, "magnitudes", tag), tag, "magnitudes"
        )
        directions = _parse_directions(cfg, tag)
        variants = []
        for magnitude in magnitudes:
            for direction in directions:
                engine = {"magnitude": magnitude}
                id_params = {"magnitude": magnitude}
                if direction is not None:
                    engine["direction"] = direction
                    id_params["direction"] = direction
                variants.append((engine, id_params))
        return variants
    if ftype == "bias":
        offsets = _parse_numbers(
            _require_list(cfg, "offsets", tag), tag, "offsets"
        )
        return [({"offset": o}, {"offset": o}) for o in offsets]
    if ftype == "drift":
        rates = _parse_numbers(_require_list(cfg, "rates", tag), tag, "rates")
        return [({"rate": r}, {"rate": r}) for r in rates]
    if ftype == "noise":
        stds = _parse_numbers(_require_list(cfg, "std", tag), tag, "std")
        return [({"std": s}, {"std": s}) for s in stds]
    if ftype == "clipping":
        return _parse_bounds(_require_list(cfg, "bounds", tag), tag)
    if ftype == "sample_rate":
        return _parse_transitions(_require_list(cfg, "transitions", tag), tag)
    if ftype == "scale":
        return _parse_transforms(_require_list(cfg, "transforms", tag), tag)
    if ftype == "timestamp_jitter":
        return _parse_jitter(_require_list(cfg, "max_jitter_ms", tag), tag)
    raise AssertionError(f"unreachable fault type: {ftype}")  # pragma: no cover


def _parse_fault_config(channel: str | None, ftype: str, cfg) -> dict:
    """Validate one fault's campaign config; return the normalized config."""
    where = f"campaign fault '{ftype}'"
    where += f" on channel {channel!r}" if channel else " (timeline)"
    if not isinstance(cfg, dict):
        raise ScenarioError(f"{where}: must be a mapping")
    allowed = set(_FAULT_AXIS_KEYS[ftype]) | set(_FAULT_COMMON_KEYS)
    for key in cfg:
        if key not in allowed:
            raise ScenarioError(
                f"{where}: unknown key {key!r}; "
                f"allowed keys are {sorted(allowed)}"
            )

    start = None
    if "start" in cfg:
        try:
            start = parse_duration(cfg["start"])
        except ScenarioError as exc:
            raise ScenarioError(f"{where}: invalid 'start': {exc}") from exc
    duration = None
    if "duration" in cfg:
        try:
            duration = parse_duration(cfg["duration"])
        except ScenarioError as exc:
            raise ScenarioError(f"{where}: invalid 'duration': {exc}") from exc
        if duration <= 0:
            raise ScenarioError(f"{where}: 'duration' must be > 0")
    if "durations" in _FAULT_AXIS_KEYS[ftype] and "duration" in cfg:
        raise ScenarioError(
            f"{where}: 'duration' conflicts with the 'durations' expansion "
            "axis; use one or the other"
        )

    expect = None
    if "expect" in cfg:
        try:
            expect = _parse_expect(cfg["expect"])
        except ScenarioError as exc:
            raise ScenarioError(f"{where}: {exc}") from exc
    cast = cfg.get("cast")
    if cast is not None and cast not in _CASTS:
        raise ScenarioError(
            f"{where}: 'cast' must be one of {list(_CASTS)} (got {cast!r})"
        )

    variants = _parse_axis(ftype, cfg, where)
    return {
        "variants": variants,
        "start": start,
        "duration": duration,
        "expect": expect,
        "cast": cast,
    }


def _parse_channels(raw) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise ScenarioError(
            "campaign: 'channels' must be a non-empty mapping of "
            "channel name -> fault mapping"
        )
    parsed = {}
    for channel, cfg in raw.items():
        tag = f"campaign channel {channel!r}"
        if not isinstance(channel, str) or not channel:
            raise ScenarioError(
                "campaign: channel names must be non-empty strings "
                f"(got {channel!r})"
            )
        if not isinstance(cfg, dict):
            raise ScenarioError(f"{tag}: must be a mapping")
        channel_expect = None
        channel_window = None
        channel_cast = None
        faults = {}
        for key, value in cfg.items():
            if key == "expect":
                try:
                    channel_expect = _parse_expect(value)
                except ScenarioError as exc:
                    raise ScenarioError(f"{tag}: {exc}") from exc
            elif key == "window":
                channel_window = _parse_window(value, tag)
            elif key == "cast":
                if value not in _CASTS:
                    raise ScenarioError(
                        f"{tag}: 'cast' must be one of {list(_CASTS)} "
                        f"(got {value!r})"
                    )
                channel_cast = value
            elif key in _TIMELINE_FAULTS:
                raise ScenarioError(
                    f"{tag}: {key!r} is a timeline fault and cannot be "
                    "configured under a channel; move it to top-level "
                    "'timeline'"
                )
            elif key in FAULT_TYPES:
                faults[key] = _parse_fault_config(channel, key, value)
            else:
                raise ScenarioError(
                    f"{tag}: unknown fault {key!r}; must be one of "
                    f"{sorted(_CHANNEL_FAULTS)} "
                    f"(or one of {sorted(_CHANNEL_RESERVED_KEYS)})"
                )
        if not faults:
            raise ScenarioError(f"{tag}: defines no faults")
        parsed[channel] = {
            "expect": channel_expect,
            "window": channel_window,
            "cast": channel_cast,
            "faults": faults,
        }
    return parsed


def _parse_timeline(raw) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise ScenarioError(
            "campaign: 'timeline' must be a non-empty mapping of "
            "timeline fault -> config"
        )
    parsed = {}
    for key, value in raw.items():
        if key not in _TIMELINE_FAULTS:
            raise ScenarioError(
                f"campaign timeline: {key!r} is not a timeline fault; only "
                f"{sorted(_TIMELINE_FAULTS)} are allowed here (channel "
                "faults belong under 'channels')"
            )
        parsed[key] = _parse_fault_config(None, key, value)
    return parsed


def parse_campaign_config(path) -> CampaignConfig:
    """Load and strictly validate a campaign YAML file.

    No dataframe is needed: everything that can be checked without data
    (keys, types, windows, expectations, expansion axes, duplicates) is
    checked here. Raises ScenarioError on any problem.
    """
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError:
        raise ScenarioError(f"campaign file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(
            f"campaign file {path}: top level must be a mapping"
        )

    for key in raw:
        if key not in _CAMPAIGN_TOP_KEYS:
            raise ScenarioError(
                f"campaign: unknown top-level key {key!r}; allowed keys are "
                f"{sorted(_CAMPAIGN_TOP_KEYS)}"
            )

    version = raw.get("version", _MISSING)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ScenarioError(
            f"campaign: 'version' must be 1 (got {version!r})"
        )

    name = raw.get("name", _MISSING)
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError("campaign: 'name' must be a non-empty string")

    seed = raw.get("seed", _MISSING)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ScenarioError(
            f"campaign: 'seed' is required and must be an integer "
            f"(got {seed!r})"
        )

    input_cfg = raw.get("input", _MISSING)
    if not isinstance(input_cfg, dict):
        raise ScenarioError("campaign: 'input' must be a mapping")
    for key in input_cfg:
        if key not in _CAMPAIGN_INPUT_KEYS:
            raise ScenarioError(
                f"campaign: unknown 'input' key {key!r}; allowed keys are "
                f"{sorted(_CAMPAIGN_INPUT_KEYS)}"
            )
    input_file = input_cfg.get("file")
    if not isinstance(input_file, str) or not input_file:
        raise ScenarioError(
            "campaign: 'input.file' must be a non-empty string"
        )
    time_column = input_cfg.get("time_column")
    if not isinstance(time_column, str) or not time_column:
        raise ScenarioError(
            "campaign: 'input.time_column' must be a non-empty string"
        )

    timeout_seconds = _parse_timeout(
        raw.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS), "campaign"
    )

    max_cases_raw = raw.get("max_cases", DEFAULT_MAX_CASES)
    if (
        isinstance(max_cases_raw, bool)
        or not isinstance(max_cases_raw, int)
        or max_cases_raw <= 0
    ):
        raise ScenarioError(
            "campaign: 'max_cases' must be an integer > 0 "
            f"(got {max_cases_raw!r})"
        )

    baseline_cfg = raw.get("baseline")
    baseline_enabled = False
    baseline_expect = None
    if baseline_cfg is not None:
        if not isinstance(baseline_cfg, dict):
            raise ScenarioError("campaign: 'baseline' must be a mapping")
        for key in baseline_cfg:
            if key not in _BASELINE_KEYS:
                raise ScenarioError(
                    f"campaign: unknown 'baseline' key {key!r}; allowed "
                    f"keys are {sorted(_BASELINE_KEYS)}"
                )
        enabled = baseline_cfg.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ScenarioError(
                "campaign: 'baseline.enabled' must be true/false"
            )
        baseline_enabled = enabled
        expect_raw = baseline_cfg.get("expect")
        try:
            baseline_expect = (
                _parse_expect(expect_raw) if expect_raw is not None else None
            )
        except ScenarioError as exc:
            raise ScenarioError(f"campaign baseline: {exc}") from exc

    window_raw = raw.get("window")
    campaign_window = (
        _parse_window(window_raw, "campaign")
        if window_raw is not None
        else None
    )

    expect_raw = raw.get("expect")
    try:
        campaign_expect = (
            _parse_expect(expect_raw) if expect_raw is not None else None
        )
    except ScenarioError as exc:
        raise ScenarioError(f"campaign: {exc}") from exc

    channels_raw = raw.get("channels")
    timeline_raw = raw.get("timeline")
    if channels_raw is None and timeline_raw is None:
        raise ScenarioError(
            "campaign: at least one of 'channels' or 'timeline' must be "
            "present"
        )
    raw_channels = (
        _parse_channels(channels_raw) if channels_raw is not None else {}
    )
    raw_timeline = (
        _parse_timeline(timeline_raw) if timeline_raw is not None else {}
    )
    if not raw_channels and not raw_timeline:
        raise ScenarioError(
            "campaign: 'channels' and 'timeline' are both empty; the "
            "campaign must define at least one fault"
        )

    return CampaignConfig(
        name=name,
        version=1,
        seed=seed,
        input_file=input_file,
        time_column=time_column,
        timeout_seconds=timeout_seconds,
        max_cases=max_cases_raw,
        baseline_enabled=baseline_enabled,
        baseline_expect=baseline_expect,
        campaign_window=campaign_window,
        campaign_expect=campaign_expect,
        raw_channels=raw_channels,
        raw_timeline=raw_timeline,
    )


# ---------------------------------------------------------------------------
# Channel validation against data
# ---------------------------------------------------------------------------


def validate_campaign_channels(config: CampaignConfig, df: pd.DataFrame) -> None:
    """Check the time column and every configured channel against the data.

    Raises ScenarioError if the time column is missing or not datetime64,
    or if any channel in ``config.raw_channels`` is not a dataframe column.
    """
    time_col = config.time_column
    if time_col not in df.columns:
        raise ScenarioError(
            f"campaign: time column '{time_col}' not found in input data"
        )
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise ScenarioError(
            f"campaign: time column '{time_col}' must be datetime64 "
            f"(got {df[time_col].dtype})"
        )
    for channel in config.raw_channels:
        if channel not in df.columns:
            raise ScenarioError(
                f"campaign: channel '{channel}' not found in input data"
            )


def _input_duration_seconds(df: pd.DataFrame, time_column: str) -> float:
    """Full input time span in seconds; the default fault-window duration."""
    if time_column not in df.columns:
        raise ScenarioError(
            f"campaign: time column '{time_column}' not found in input data"
        )
    col = df[time_column]
    if not pd.api.types.is_datetime64_any_dtype(col):
        raise ScenarioError(
            f"campaign: time column '{time_column}' must be datetime64 "
            f"(got {col.dtype})"
        )
    if len(col) == 0:
        raise ScenarioError(
            "campaign: input data is empty; cannot determine the default "
            "window duration"
        )
    span = (col.max() - col.min()).total_seconds()
    if not math.isfinite(span) or span <= 0:
        raise ScenarioError(
            "campaign: input time span must be > 0 seconds "
            f"(got {span}); cannot determine the default window duration"
        )
    return float(span)


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _merge_expect(*levels) -> dict | None:
    """Merge expectations campaign < channel < fault.

    A more specific assertion KEY replaces the less-specific value for
    that key; lists are NOT concatenated.
    """
    merged: dict = {}
    for level in levels:
        if level:
            merged.update(level)
    return merged or None


def _resolve_window(
    campaign_window: dict | None,
    owner_window: dict | None,
    fcfg: dict,
    id_params: dict,
    full_seconds: float,
    where: str,
) -> tuple:
    """Resolve (start_seconds, duration_seconds) with fault > owner > campaign > default."""
    if fcfg["start"] is not None:
        start = fcfg["start"]
    elif owner_window and owner_window.get("start") is not None:
        start = owner_window["start"]
    elif campaign_window and campaign_window.get("start") is not None:
        start = campaign_window["start"]
    else:
        start = 0.0

    if "duration" in id_params:
        # durations-axis faults (dropout/freeze/timestamp_gap): the axis
        # value IS the case's window duration.
        duration = id_params["duration"]
    elif fcfg["duration"] is not None:
        duration = fcfg["duration"]
    elif owner_window and owner_window.get("duration") is not None:
        duration = owner_window["duration"]
    elif campaign_window and campaign_window.get("duration") is not None:
        duration = campaign_window["duration"]
    else:
        duration = full_seconds

    if duration <= 0:
        raise ScenarioError(
            f"{where}: resolved window duration must be > 0 "
            f"(got {duration})"
        )
    return float(start), float(duration)


def _expand_one_fault(
    config: CampaignConfig,
    target: str,
    target_kind: str,
    channel: str | None,
    owner_expect: dict | None,
    owner_window: dict | None,
    owner_cast: str | None,
    ftype: str,
    fcfg: dict,
    full_seconds: float,
) -> list:
    """Expand one fault config into its CampaignCase variants."""
    where = f"campaign fault '{ftype}'"
    where += f" on channel {channel!r}" if channel else " (timeline)"
    cases = []
    for index, (engine_raw, id_params) in enumerate(fcfg["variants"], start=1):
        start_s, duration_s = _resolve_window(
            config.campaign_window,
            owner_window,
            fcfg,
            id_params,
            full_seconds,
            where,
        )
        cast = fcfg["cast"] if fcfg["cast"] is not None else owner_cast
        expect = _merge_expect(
            config.campaign_expect, owner_expect, fcfg["expect"]
        )
        raw_fault = {
            "type": ftype,
            "start": start_s,
            "duration": duration_s,
            **engine_raw,
        }
        if channel is not None:
            raw_fault["channel"] = channel
        if cast is not None:
            raw_fault["cast"] = cast
        spec = _parse_fault(index, raw_fault)
        name = _build_case_name(target, ftype, id_params)
        _validate_case_name(name, "campaign case")
        seed_material = {
            "start": spec.start_seconds,
            "duration": spec.duration_seconds,
            "cast": spec.cast,
        }
        seed_material.update(spec.params)
        seed = case_seed(config.seed, target, ftype, seed_material)
        cases.append(
            CampaignCase(
                name=name,
                target=target,
                target_kind=target_kind,
                channel=channel,
                fault_type=ftype,
                spec=spec,
                seed=seed,
                expect=expect,
                id_params=dict(id_params),
            )
        )
    return cases


def _verify_case_names(cases: list) -> None:
    """Reject duplicate case names, artifact-dirname collisions, reserved names.

    Mirrors the checks in suite.load_suite so expanded campaigns are safe
    on case-insensitive filesystems and can never clobber the suite's own
    artifacts. Raises ScenarioError on any problem.
    """
    seen = set()
    dirnames = {}
    for case in cases:
        if case.name in seen:
            raise ScenarioError(
                f"campaign: duplicate case name {case.name!r}"
            )
        seen.add(case.name)
        dirname = safe_case_dirname(case.name)
        key = dirname.lower()
        if key in dirnames:
            raise ScenarioError(
                f"campaign: case names {dirnames[key]!r} and {case.name!r} "
                f"map to the same artifact directory {dirname!r}"
            )
        dirnames[key] = case.name
        if key in _RESERVED_ARTIFACT_NAMES or key == ARTIFACT_MARKER:
            raise ScenarioError(
                f"campaign: case name {case.name!r} is reserved (it would "
                f"collide with the suite's own artifact {dirname!r})"
            )


def effective_max_cases(config: CampaignConfig, override=None) -> int:
    """Resolve the effective cap on expanded campaign cases.

    Priority: explicit ``override`` > ``config.max_cases``
    (``config.max_cases`` already defaults to
    :data:`DEFAULT_MAX_CASES` at parse time). An invalid override
    (non-int, bool, or <= 0) raises ScenarioError.
    """
    if override is None:
        return config.max_cases
    if (
        isinstance(override, bool)
        or not isinstance(override, int)
        or override <= 0
    ):
        raise ScenarioError(
            "campaign: 'max_cases' override must be an integer > 0 "
            f"(got {override!r})"
        )
    return override


def validate_filters(config: CampaignConfig, channels, faults) -> tuple:
    """Strictly validate ``--channel``/``--fault`` filter values.

    Exact duplicate entries are deduplicated first (harmless). Then:

    * every requested channel must exist in ``config.raw_channels``;
    * every requested fault must be a supported fault type (in
      ``FAULT_TYPES``);
    * every requested fault must be configured somewhere relevant: in at
      least one of the selected channels when channels were requested,
      or in at least one channel or the timeline otherwise.

    Returns the deduplicated ``(channels, faults)`` tuples. Raises
    ScenarioError naming the bad filter value on any violation (mixed
    valid+invalid input fails the same way; a requested filter is never
    silently ignored).
    """
    channels = tuple(dict.fromkeys(channels))
    faults = tuple(dict.fromkeys(faults))
    for channel in channels:
        if channel not in config.raw_channels:
            raise ScenarioError(
                f"campaign: unknown channel {channel!r} in '--channel' "
                "filter; available channels are "
                f"{sorted(config.raw_channels)}"
            )
    for fault in faults:
        if fault not in FAULT_TYPES:
            raise ScenarioError(
                f"campaign: unknown fault {fault!r} in '--fault' filter; "
                f"supported fault types are {sorted(FAULT_TYPES)}"
            )
    if faults:
        if channels:
            for fault in faults:
                if not any(
                    fault in config.raw_channels[channel]["faults"]
                    for channel in channels
                ):
                    raise ScenarioError(
                        f"campaign: fault {fault!r} in '--fault' filter is "
                        "not configured in any of the selected channels "
                        f"{list(channels)}"
                    )
        else:
            configured = set()
            for cdata in config.raw_channels.values():
                configured.update(cdata["faults"])
            configured.update(config.raw_timeline)
            for fault in faults:
                if fault not in configured:
                    raise ScenarioError(
                        f"campaign: fault {fault!r} in '--fault' filter is "
                        "not configured in any channel or the timeline"
                    )
    return channels, faults


def expand_campaign(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    channels: tuple = (),
    faults: tuple = (),
    max_cases: int | None = None,
) -> list:
    """Expand a campaign into its deterministic list of CampaignCase.

    ``channels`` / ``faults`` filter which channel names and fault types
    are expanded (empty tuple = no filter). A non-empty ``channels``
    filter restricts expansion to those channels: NO timeline cases are
    generated. When no channel filter is given, channels and the
    timeline both expand normally. The ``faults`` filter applies to
    whatever is generated. Filters are strictly validated first (see
    :func:`validate_filters`): any invalid filter raises ScenarioError.

    Expansion order is deterministic: channels in YAML order, faults in
    YAML order within each channel, variants in listed order, then
    timeline faults in YAML order.

    Raises ScenarioError if the expansion is empty, if it exceeds the
    effective max (never silently truncated), or if any case names
    collide.
    """
    channel_filter, fault_filter = validate_filters(config, channels, faults)
    effective_max = effective_max_cases(config, max_cases)

    full_seconds = _input_duration_seconds(df, config.time_column)

    cases = []
    for channel, cdata in config.raw_channels.items():
        if channel_filter and channel not in channel_filter:
            continue
        if channel not in df.columns:
            raise ScenarioError(
                f"campaign: channel '{channel}' not found in input data"
            )
        for ftype, fcfg in cdata["faults"].items():
            if fault_filter and ftype not in fault_filter:
                continue
            cases.extend(
                _expand_one_fault(
                    config,
                    channel,
                    "channel",
                    channel,
                    cdata["expect"],
                    cdata["window"],
                    cdata["cast"],
                    ftype,
                    fcfg,
                    full_seconds,
                )
            )
    if not channel_filter:
        for ftype, fcfg in config.raw_timeline.items():
            if fault_filter and ftype not in fault_filter:
                continue
            cases.extend(
                _expand_one_fault(
                    config,
                    "timeline",
                    "timeline",
                    None,
                    None,
                    None,
                    None,
                    ftype,
                    fcfg,
                    full_seconds,
                )
            )

    if not cases:
        raise ScenarioError(
            "campaign: expansion produced 0 cases; the campaign must "
            "generate at least 1 case (check the channels/faults filters)"
        )
    if len(cases) > effective_max:
        raise ScenarioError(
            f"Campaign expands to {len(cases)} cases. Configured maximum "
            f"is {effective_max}. Reduce the parameter combinations or "
            "raise 'max_cases'."
        )
    _verify_case_names(cases)
    return cases


# ---------------------------------------------------------------------------
# Suite conversion, fingerprinting, export
# ---------------------------------------------------------------------------


def campaign_to_suite(
    config: CampaignConfig, cases: list, input_file_abs: str
) -> Suite:
    """Build the in-memory Suite for preflight/execution of expanded cases."""
    return Suite(
        name=config.name,
        input_file=input_file_abs,
        time_column=config.time_column,
        baseline_enabled=config.baseline_enabled,
        baseline_expect=config.baseline_expect,
        timeout_seconds=config.timeout_seconds,
        cases=[case.to_suite_case() for case in cases],
    )


def campaign_fingerprint(config: CampaignConfig, cases: list) -> str:
    """Content fingerprint of the expanded campaign: ``"sha256:..."``.

    Covers the campaign name/version/seed, the time column, the sorted
    expanded cases (target, fault, sorted params, start, duration, cast,
    expect), the baseline, and the timeout. Never depends on absolute
    paths, time, or temp dirs: it changes exactly when meaningful
    behavior changes.
    """
    doc = {
        "name": config.name,
        "version": config.version,
        "seed": config.seed,
        "time_column": config.time_column,
        "timeout_seconds": config.timeout_seconds,
        "baseline": {
            "enabled": config.baseline_enabled,
            "expect": config.baseline_expect,
        },
        "cases": [
            {
                "target": case.target,
                "fault": case.fault_type,
                "params": case.spec.params,
                "start": case.spec.start_seconds,
                "duration": case.spec.duration_seconds,
                "cast": case.spec.cast,
                "expect": case.expect,
            }
            for case in sorted(cases, key=lambda c: c.name)
        ],
    }
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def input_fingerprints(in_path) -> tuple:
    """Fingerprint the input data file.

    Returns ``(file_sha256, sidecar_sha256_or_None)``: the sha256 hex of
    the file bytes, plus the sha256 hex of the CSV schema sidecar
    (``<stem>.schema.json``) when the input is a CSV that owns one.
    """
    in_path = Path(in_path)

    def _sha256_of(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    sidecar_hex = None
    if in_path.suffix.lower() == ".csv":
        sidecar = in_path.with_name(in_path.stem + ".schema.json")
        if sidecar.is_file():
            sidecar_hex = _sha256_of(sidecar)
    return (_sha256_of(in_path), sidecar_hex)


def _compact_number(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def expanded_suite_yaml(
    config: CampaignConfig, cases: list, input_file_display: str
) -> str:
    """Render the expanded campaign as deterministic suite-format YAML text.

    Starts with a header comment noting the file was generated and the
    campaign fingerprint. ``input_file_display`` is caller-chosen (no
    absolute temp paths are ever embedded by this function).
    """
    fingerprint = campaign_fingerprint(config, cases)
    header = "\n".join(
        [
            "# Generated by telemetry-resilience campaign expansion; "
            "do not edit by hand.",
            f"# campaign: {config.name}",
            f"# campaign fingerprint: {fingerprint}",
            f"# cases: {len(cases)}",
        ]
    )
    doc: dict = {
        "version": 1,
        "name": config.name,
        "input": {
            "file": input_file_display,
            "time_column": config.time_column,
        },
        "timeout_seconds": _compact_number(config.timeout_seconds),
    }
    if config.baseline_enabled or config.baseline_expect is not None:
        baseline: dict = {"enabled": config.baseline_enabled}
        if config.baseline_expect is not None:
            baseline["expect"] = config.baseline_expect
        doc["baseline"] = baseline
    case_docs = []
    for case in cases:
        fault: dict = {}
        if case.target_kind == "channel":
            fault["channel"] = case.channel
        fault["type"] = case.fault_type
        fault["start"] = format_duration(case.spec.start_seconds)
        fault["duration"] = format_duration(case.spec.duration_seconds)
        for key in sorted(case.spec.params):
            value = case.spec.params[key]
            if value is not None:
                fault[key] = value
        if case.spec.cast is not None:
            fault["cast"] = case.spec.cast
        entry: dict = {
            "name": case.name,
            "seed": case.seed,
            "faults": [fault],
        }
        if case.expect is not None:
            entry["expect"] = case.expect
        case_docs.append(entry)
    doc["cases"] = case_docs
    body = yaml.safe_dump(
        doc, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    return header + "\n" + body
