"""Tests for the RUN 3.1 core campaign fixes.

Covers the focused campaign.py changes (RUN 3.1): the --channel filter
excludes timeline cases, strict --channel/--fault validation, the
time_column in the campaign fingerprint, JSON-safe 53-bit case seeds,
and the effective_max_cases helper. Does not touch campaign_run.py,
cli.py, README.md, or the demo examples.
"""
import dataclasses
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from telemetry_resilience.campaign import (
    campaign_fingerprint,
    effective_max_cases,
    expand_campaign,
    parse_campaign_config,
    validate_filters,
)
from telemetry_resilience.cli import app
from telemetry_resilience.io import write_file
from telemetry_resilience.models import ScenarioError

from .conftest import write_yaml

runner = CliRunner()

#: Largest integer exactly representable as a JSON number (2**53 - 1).
MAX_SAFE_INT = 9007199254740991


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _df(n=120):
    """10 Hz synthetic frame with the channels used by the test docs."""
    rng = np.random.default_rng(7)
    timestamp = pd.date_range(
        "2026-03-01", periods=n, freq="100ms", tz="UTC"
    ).as_unit("ns")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "pressure": 101.3 + 0.05 * rng.normal(0, 1, n),
            "temp": 20.0 + rng.normal(0, 1, n),
        }
    )


def _doc():
    """Campaign with two channels (several faults) and timeline faults."""
    return {
        "version": 1,
        "name": "run31",
        "seed": 1234,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "pressure": {
                "dropout": {"durations": ["2s", "5s"]},
                "bias": {"offsets": [1.0]},
            },
            "temp": {
                "spike": {
                    "magnitudes": [5],
                    "directions": ["positive"],
                },
            },
        },
        "timeline": {
            "timestamp_jitter": {
                "max_jitter_ms": [20, 50],
                "duration": "15s",
            },
        },
    }


def _parse_expand(tmp_path, doc, **expand_kw):
    path = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(doc))
    cfg = parse_campaign_config(path)
    return cfg, expand_campaign(cfg, _df(), **expand_kw)


def _run_campaign(tmp_path, doc, *extra_cli):
    write_file(_df(), tmp_path / "data.parquet")
    camp = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(doc))
    return runner.invoke(
        app,
        ["campaign", str(camp), *extra_cli, "--",
         sys.executable, "-c", "print('unused')", "{data}"],
    )


def _seeds_by_name(cases):
    return {case.name: case.seed for case in cases}


# ---------------------------------------------------------------------------
# Fix 1: --channel filter excludes timeline
# ---------------------------------------------------------------------------

def test_channel_filter_excludes_timeline(tmp_path):
    _, cases = _parse_expand(tmp_path, _doc(), channels=("pressure",))
    assert cases, "expected channel cases"
    assert all(c.target_kind == "channel" for c in cases)
    assert {c.target for c in cases} == {"pressure"}
    assert [c.name for c in cases] == [
        "pressure__bias__offset-1",
        "pressure__dropout__duration-2s",
        "pressure__dropout__duration-5s",
    ]


def test_channel_filter_two_channels_no_timeline(tmp_path):
    _, cases = _parse_expand(
        tmp_path, _doc(), channels=("pressure", "temp")
    )
    assert cases
    assert all(c.target_kind == "channel" for c in cases)
    assert {c.target for c in cases} == {"pressure", "temp"}
    assert len(cases) == 4  # 3 pressure + 1 temp


def test_no_channel_filter_generates_channel_and_timeline(tmp_path):
    _, cases = _parse_expand(tmp_path, _doc())
    kinds = {c.target_kind for c in cases}
    assert kinds == {"channel", "timeline"}
    timeline_names = sorted(
        c.name for c in cases if c.target_kind == "timeline"
    )
    assert timeline_names == [
        "timeline__timestamp_jitter__max_jitter_ms-20",
        "timeline__timestamp_jitter__max_jitter_ms-50",
    ]
    assert len(cases) == 6  # 4 channel + 2 timeline


def test_channel_filter_excludes_timeline_cli(tmp_path):
    result = _run_campaign(tmp_path, _doc(), "--plan", "--channel", "pressure")
    assert result.exit_code == 0, result.output
    assert "TIMELINE" not in result.output
    assert "Generated cases: 3" in result.output
    # and without the filter the timeline section is present
    result = _run_campaign(tmp_path, _doc(), "--plan")
    assert result.exit_code == 0, result.output
    assert "TIMELINE" in result.output
    assert "Generated cases: 6" in result.output


# ---------------------------------------------------------------------------
# Fix 2: strictly validate filter values
# ---------------------------------------------------------------------------

def test_unknown_channel_filter_rejected(tmp_path):
    cfg, df = _parse(tmp_path, _doc()), _df()
    with pytest.raises(ScenarioError, match="unknown channel") as excinfo:
        expand_campaign(cfg, df, channels=("no-such-channel",))
    assert "no-such-channel" in str(excinfo.value)


def test_mixed_valid_invalid_channel_rejected(tmp_path):
    cfg, df = _parse(tmp_path, _doc()), _df()
    with pytest.raises(ScenarioError) as excinfo:
        expand_campaign(cfg, df, channels=("pressure", "no-such-channel"))
    assert "no-such-channel" in str(excinfo.value)


def test_unknown_fault_filter_rejected(tmp_path):
    cfg, df = _parse(tmp_path, _doc()), _df()
    with pytest.raises(ScenarioError, match="unknown fault") as excinfo:
        expand_campaign(cfg, df, faults=("frobnicator",))
    assert "frobnicator" in str(excinfo.value)


def test_valid_fault_not_configured_anywhere_rejected(tmp_path):
    # 'freeze' is a supported fault type but configured nowhere.
    cfg, df = _parse(tmp_path, _doc()), _df()
    with pytest.raises(ScenarioError, match="not configured") as excinfo:
        expand_campaign(cfg, df, faults=("freeze",))
    assert "freeze" in str(excinfo.value)


def test_valid_fault_not_configured_in_selected_channel_rejected(tmp_path):
    # 'dropout' is configured on pressure but not on the selected temp.
    cfg, df = _parse(tmp_path, _doc()), _df()
    with pytest.raises(ScenarioError, match="not configured") as excinfo:
        expand_campaign(
            cfg, df, channels=("temp",), faults=("dropout",)
        )
    assert "dropout" in str(excinfo.value)


def test_timeline_fault_not_configured_in_selected_channel_rejected(tmp_path):
    # 'timestamp_jitter' is configured only in the timeline; selecting a
    # channel means the timeline is excluded, so the fault has no source.
    cfg, df = _parse(tmp_path, _doc()), _df()
    with pytest.raises(ScenarioError, match="not configured"):
        expand_campaign(
            cfg, df, channels=("pressure",), faults=("timestamp_jitter",)
        )


def test_timeline_fault_filter_without_channel_filter_ok(tmp_path):
    # no channel filter -> timeline is a valid fault source
    _, cases = _parse_expand(
        tmp_path, _doc(), faults=("timestamp_jitter",)
    )
    assert [c.name for c in cases] == [
        "timeline__timestamp_jitter__max_jitter_ms-20",
        "timeline__timestamp_jitter__max_jitter_ms-50",
    ]


def test_duplicate_identical_filters_accepted(tmp_path):
    cfg, df = _parse(tmp_path, _doc()), _df()
    dup = expand_campaign(
        cfg, df, channels=("pressure", "pressure"),
        faults=("dropout", "dropout"),
    )
    plain = expand_campaign(cfg, df, channels=("pressure",), faults=("dropout",))
    assert [c.name for c in dup] == [c.name for c in plain]
    assert _seeds_by_name(dup) == _seeds_by_name(plain)


def test_invalid_filters_rejected_via_cli(tmp_path):
    # plan/list/run all reject with exit 2 before any artifact is written
    result = _run_campaign(
        tmp_path, _doc(), "--plan", "--channel", "no-such-channel"
    )
    assert result.exit_code == 2, result.output
    assert "no-such-channel" in result.output
    result = _run_campaign(
        tmp_path, _doc(), "--list-cases", "--fault", "frobnicator"
    )
    assert result.exit_code == 2, result.output
    assert "frobnicator" in result.output
    result = _run_campaign(
        tmp_path, _doc(), "--fault", "freeze",
        "--artifacts", str(tmp_path / "artifacts"),
    )
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "artifacts").exists()


def _parse(tmp_path, doc):
    path = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(doc))
    return parse_campaign_config(path)


# ---------------------------------------------------------------------------
# Fix 3: fingerprint includes time_column
# ---------------------------------------------------------------------------

def test_fingerprint_differs_when_only_time_column_differs(tmp_path):
    cfg, cases = _parse_expand(tmp_path, _doc())
    fp1 = campaign_fingerprint(cfg, cases)
    cfg2 = dataclasses.replace(cfg, time_column="other_column")
    fp2 = campaign_fingerprint(cfg2, cases)
    assert fp1 != fp2


def test_fingerprint_stable_under_yaml_key_reordering(tmp_path):
    cfg, cases = _parse_expand(tmp_path, _doc())
    fp1 = campaign_fingerprint(cfg, cases)
    # same campaign, keys written in a different order throughout
    reordered = {
        "timeline": {
            "timestamp_jitter": {
                "duration": "15s",
                "max_jitter_ms": [20, 50],
            },
        },
        "channels": {
            "temp": {
                "spike": {
                    "directions": ["positive"],
                    "magnitudes": [5],
                },
            },
            "pressure": {
                "bias": {"offsets": [1.0]},
                "dropout": {"durations": ["2s", "5s"]},
            },
        },
        "input": {"time_column": "timestamp", "file": "data.parquet"},
        "seed": 1234,
        "name": "run31",
        "version": 1,
    }
    cfg2, cases2 = _parse_expand(tmp_path, reordered)
    assert campaign_fingerprint(cfg2, cases2) == fp1
    # case seeds are reorder-stable too
    assert _seeds_by_name(cases2) == _seeds_by_name(cases)


# ---------------------------------------------------------------------------
# Fix 5: JSON-safe 53-bit seeds
# ---------------------------------------------------------------------------

def test_seeds_within_json_safe_53bit_range(tmp_path):
    _, cases = _parse_expand(tmp_path, _doc())
    assert len(cases) == 6
    for case in cases:
        assert isinstance(case.seed, int)
        assert 0 <= case.seed <= MAX_SAFE_INT, case.name


def test_seeds_survive_json_float_round_trip(tmp_path):
    _, cases = _parse_expand(tmp_path, _doc())
    for case in cases:
        restored = json.loads(json.dumps(case.seed))
        assert restored == case.seed, case.name
        # explicit float hop, as JSON numbers decode
        assert int(float(case.seed)) == case.seed, case.name


def test_seeds_deterministic_and_parameter_sensitive(tmp_path):
    _, run1 = _parse_expand(tmp_path, _doc())
    _, run2 = _parse_expand(tmp_path, _doc())
    # same case/root seed -> same generated seed across runs
    assert _seeds_by_name(run1) == _seeds_by_name(run2)

    # parameter change -> seed change for the affected case only
    doc2 = _doc()
    doc2["channels"]["pressure"]["bias"]["offsets"] = [2.0]
    _, run3 = _parse_expand(tmp_path, doc2)
    seeds1 = _seeds_by_name(run1)
    seeds3 = _seeds_by_name(run3)
    assert seeds3["pressure__bias__offset-2"] != seeds1["pressure__bias__offset-1"]
    for name in seeds1:
        if name != "pressure__bias__offset-1":
            assert seeds3[name] == seeds1[name], name


def test_seeds_unaffected_by_unrelated_channel_fault(tmp_path):
    # adding a fault on another channel leaves existing cases' seeds alone
    _, base = _parse_expand(tmp_path, _doc())
    doc2 = _doc()
    doc2["channels"]["pressure"]["noise"] = {"std": [0.5]}
    _, grown = _parse_expand(tmp_path, doc2)
    base_seeds = _seeds_by_name(base)
    grown_seeds = _seeds_by_name(grown)
    for name, seed in base_seeds.items():
        assert grown_seeds[name] == seed, name
    assert len(grown_seeds) == len(base_seeds) + 1


# ---------------------------------------------------------------------------
# --channel and --fault filters together
# ---------------------------------------------------------------------------

def test_channel_and_fault_filters_combined(tmp_path):
    _, cases = _parse_expand(
        tmp_path, _doc(), channels=("pressure",), faults=("dropout",)
    )
    assert [c.name for c in cases] == [
        "pressure__dropout__duration-2s",
        "pressure__dropout__duration-5s",
    ]


# ---------------------------------------------------------------------------
# Helper: effective_max_cases
# ---------------------------------------------------------------------------

def test_effective_max_cases_helper(tmp_path):
    cfg = _parse(tmp_path, _doc())
    assert cfg.max_cases == 100  # parse default
    assert effective_max_cases(cfg) == 100
    assert effective_max_cases(cfg, None) == 100
    assert effective_max_cases(cfg, 7) == 7
    for bad in (True, False, 0, -3, "10", 2.5, (3,), object()):
        with pytest.raises(ScenarioError, match="must be an integer > 0"):
            effective_max_cases(cfg, bad)


def test_expand_max_cases_override_keeps_error_message(tmp_path):
    cfg, df = _parse(tmp_path, _doc()), _df()
    with pytest.raises(
        ScenarioError,
        match=r"'max_cases' override must be an integer > 0 \(got 0\)",
    ):
        expand_campaign(cfg, df, max_cases=0)
    # valid override still respected
    assert len(expand_campaign(cfg, df, max_cases=100)) == 6


def test_validate_filters_returns_deduped(tmp_path):
    cfg = _parse(tmp_path, _doc())
    channels, faults = validate_filters(
        cfg, ("pressure", "pressure"), ("dropout", "dropout")
    )
    assert channels == ("pressure",)
    assert faults == ("dropout",)
    assert validate_filters(cfg, (), ()) == ((), ())
