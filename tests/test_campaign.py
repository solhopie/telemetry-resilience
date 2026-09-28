"""Tests for `telemetry-resilience campaign` (RUN 3).

Covers campaign parsing/validation, expansion of all 11 fault types,
deterministic seeds, window/expectation inheritance, max_cases, the
--channel/--fault filters, --plan/--list-cases, execution semantics
(baseline blocking, fail-fast), coverage reports, and campaign artifacts.
"""
import csv
import dataclasses
import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from telemetry_resilience.campaign import (
    DEFAULT_MAX_CASES,
    CampaignCase,
    CampaignConfig,
    campaign_fingerprint,
    campaign_to_suite,
    expand_campaign,
    expanded_suite_yaml,
    parse_campaign_config,
    validate_campaign_channels,
)
from telemetry_resilience.campaign_run import (
    build_coverage,
    render_coverage_csv,
    render_coverage_markdown,
)
from telemetry_resilience.cli import app
from telemetry_resilience.io import write_file
from telemetry_resilience.models import FaultSpec, ScenarioError
from telemetry_resilience.suite import _validate_case_name, load_suite

from .conftest import write_yaml

runner = CliRunner()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tiny_df(n=100):
    """Tiny synthetic telemetry frame: 10 Hz, ``n`` rows (9.9 s span)."""
    rng = np.random.default_rng(7)
    timestamp = pd.date_range(
        "2026-03-01", periods=n, freq="100ms", tz="UTC"
    ).as_unit("ns")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "pressure": 101.3 + 0.05 * rng.normal(0, 1, n),
            "temp": 20.0 + rng.normal(0, 1, n),
            "humidity": 50.0 + 2.0 * rng.normal(0, 1, n),
            "mode": pd.array(["a", "b"] * (n // 2), dtype="string"),
        }
    )


def _write_input(tmp_path, df=None, name="data.parquet"):
    df = _tiny_df() if df is None else df
    path = tmp_path / name
    write_file(df, path)
    return path


def _base_doc():
    """Minimal valid campaign doc; tests mutate copies of this."""
    return {
        "version": 1,
        "name": "t",
        "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {"pressure": {"dropout": {"durations": ["2s"]}}},
    }


def _parse(tmp_path, doc, name="campaign.yaml"):
    path = write_yaml(tmp_path, name, yaml.safe_dump(doc))
    return parse_campaign_config(path)


def _parse_expand(tmp_path, doc, df=None, **expand_kw):
    df = _tiny_df() if df is None else df
    cfg = _parse(tmp_path, doc)
    return cfg, expand_campaign(cfg, df, **expand_kw)


def _run_campaign(tmp_path, doc, target_args, *extra_cli, input_df=None,
                  input_name="data.parquet", campaign_name="campaign.yaml"):
    _write_input(tmp_path, input_df, input_name)
    camp = write_yaml(tmp_path, campaign_name, yaml.safe_dump(doc))
    return runner.invoke(
        app, ["campaign", str(camp), *extra_cli, "--", *target_args]
    )


_TARGET_OK = "import sys\nprint('ok')\n"


def _ok_target(tmp_path, name="ok_target.py"):
    path = tmp_path / name
    path.write_text(_TARGET_OK)
    return path


def _marker_target(tmp_path, name="marker_target.py"):
    """Target that writes a marker file (argv[1]) when it executes."""
    path = tmp_path / name
    path.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text('ran', encoding='utf-8')\n"
    )
    return path


def _counter_target(tmp_path, name="counter_target.py"):
    """Target that counts its own invocations in argv[1]."""
    path = tmp_path / name
    path.write_text(
        "import pathlib, sys\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        "print('ok')\n"
    )
    return path


def _dummy_target_args():
    # --plan/--list-cases never execute the target, but typer requires it.
    return [sys.executable, "-c", "print('unused')", "{data}"]


# ---------------------------------------------------------------------------
# parsing / validation
# ---------------------------------------------------------------------------

def test_valid_campaign_parses(tmp_path):
    doc = {
        "version": 1,
        "name": "demo-campaign",
        "seed": 5000,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "timeout_seconds": 30,
        "max_cases": 50,
        "baseline": {"enabled": True},
        "window": {"start": "5s", "duration": "20s"},
        "expect": {"exit_code": 0},
        "channels": {
            "pressure": {
                "expect": {"stdout_contains": ["ok"]},
                "window": {"start": "1s"},
                "cast": "float64",
                "dropout": {"durations": ["2s", "10s"], "start": "90s"},
                "spike": {"magnitudes": [5, 10], "directions": ["positive"]},
            }
        },
        "timeline": {
            "timestamp_jitter": {"max_jitter_ms": [20, 100], "duration": "15s"}
        },
    }
    cfg = _parse(tmp_path, doc)
    assert cfg.name == "demo-campaign"
    assert cfg.seed == 5000
    assert cfg.max_cases == 50
    assert cfg.baseline_enabled is True
    assert cfg.campaign_window == {"start": 5.0, "duration": 20.0}
    assert cfg.campaign_expect == {"exit_code": 0}
    assert set(cfg.raw_channels) == {"pressure"}
    assert set(cfg.raw_timeline) == {"timestamp_jitter"}


def test_unknown_top_level_key_rejected(tmp_path):
    doc = _base_doc()
    doc["bogus"] = 1
    with pytest.raises(ScenarioError, match="unknown top-level key"):
        _parse(tmp_path, doc)


def test_unknown_input_key_rejected(tmp_path):
    doc = _base_doc()
    doc["input"] = {
        "file": "data.parquet", "time_column": "timestamp", "bogus": 1
    }
    with pytest.raises(ScenarioError, match="unknown 'input' key"):
        _parse(tmp_path, doc)


def test_unknown_baseline_key_rejected(tmp_path):
    doc = _base_doc()
    doc["baseline"] = {"enabled": True, "bogus": 1}
    with pytest.raises(ScenarioError, match="unknown 'baseline' key"):
        _parse(tmp_path, doc)


def test_unknown_window_key_rejected(tmp_path):
    doc = _base_doc()
    doc["window"] = {"start": "1s", "bogus": 1}
    with pytest.raises(ScenarioError, match="unknown 'window' key"):
        _parse(tmp_path, doc)


def test_unknown_channel_window_key_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {
            "window": {"duration": "2s", "bogus": 1},
            "dropout": {"durations": ["2s"]},
        }
    }
    with pytest.raises(ScenarioError, match="unknown 'window' key"):
        _parse(tmp_path, doc)


def test_unknown_channel_key_rejected(tmp_path):
    # reserved keys are expect/window/cast; anything else must be a fault
    doc = _base_doc()
    doc["channels"] = {"pressure": {"frobnicate": {"durations": ["2s"]}}}
    with pytest.raises(ScenarioError, match="unknown fault"):
        _parse(tmp_path, doc)


def test_channel_reserved_keys_accepted(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {
            "expect": {"exit_code": 0},
            "window": {"start": "1s"},
            "cast": "float64",
            "dropout": {"durations": ["2s"]},
        }
    }
    cfg = _parse(tmp_path, doc)  # must not raise
    assert cfg.raw_channels["pressure"]["cast"] == "float64"


def test_unknown_fault_config_key_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {"dropout": {"durations": ["2s"], "bogus": 1}}
    }
    with pytest.raises(ScenarioError, match="unknown key"):
        _parse(tmp_path, doc)


def test_unknown_fault_name_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"pressure": {"frobnicator": {"durations": ["2s"]}}}
    with pytest.raises(ScenarioError, match="unknown fault"):
        _parse(tmp_path, doc)


def test_timeline_fault_under_channel_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {"timestamp_gap": {"durations": ["2s"]}}
    }
    with pytest.raises(ScenarioError, match="timeline fault"):
        _parse(tmp_path, doc)


def test_channel_fault_under_timeline_rejected(tmp_path):
    doc = _base_doc()
    doc["timeline"] = {"dropout": {"durations": ["2s"]}}
    with pytest.raises(ScenarioError, match="not a timeline fault"):
        _parse(tmp_path, doc)


def test_empty_parameter_array_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"pressure": {"dropout": {"durations": []}}}
    with pytest.raises(ScenarioError, match="non-empty list"):
        _parse(tmp_path, doc)


def test_empty_magnitudes_array_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"pressure": {"spike": {"magnitudes": []}}}
    with pytest.raises(ScenarioError, match="non-empty"):
        _parse(tmp_path, doc)


def test_duplicate_normalized_values_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"pressure": {"dropout": {"durations": [2, "2s"]}}}
    with pytest.raises(ScenarioError, match="duplicate values"):
        _parse(tmp_path, doc)


def test_duplicate_normalized_numbers_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"pressure": {"bias": {"offsets": [1, 1.0]}}}
    with pytest.raises(ScenarioError, match="duplicate values"):
        _parse(tmp_path, doc)


@pytest.mark.parametrize("bad", [0, -3])
def test_max_cases_non_positive_rejected(tmp_path, bad):
    doc = _base_doc()
    doc["max_cases"] = bad
    with pytest.raises(ScenarioError, match="max_cases"):
        _parse(tmp_path, doc)


def test_version_not_1_rejected(tmp_path):
    doc = _base_doc()
    doc["version"] = 2
    with pytest.raises(ScenarioError, match="'version' must be 1"):
        _parse(tmp_path, doc)


@pytest.mark.parametrize("bad", ["abc", 1.5, True])
def test_non_integer_seed_rejected(tmp_path, bad):
    doc = _base_doc()
    doc["seed"] = bad
    with pytest.raises(ScenarioError, match="'seed'"):
        _parse(tmp_path, doc)


def test_invalid_cast_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {"cast": "float16", "dropout": {"durations": ["2s"]}}
    }
    with pytest.raises(ScenarioError, match="'cast'"):
        _parse(tmp_path, doc)


def test_invalid_fault_cast_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {"dropout": {"durations": ["2s"], "cast": "nope"}}
    }
    with pytest.raises(ScenarioError, match="'cast'"):
        _parse(tmp_path, doc)


def test_invalid_expectation_key_rejected(tmp_path):
    doc = _base_doc()
    doc["expect"] = {"bogus_assertion": 1}
    with pytest.raises(ScenarioError, match="unknown 'expect' key"):
        _parse(tmp_path, doc)


def test_invalid_channel_expectation_key_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {
            "expect": {"bogus_assertion": 1},
            "dropout": {"durations": ["2s"]},
        }
    }
    with pytest.raises(ScenarioError, match="unknown 'expect' key"):
        _parse(tmp_path, doc)


def test_invalid_window_duration_zero_rejected(tmp_path):
    doc = _base_doc()
    doc["window"] = {"duration": "0s"}
    with pytest.raises(ScenarioError, match="must be > 0"):
        _parse(tmp_path, doc)


def test_invalid_window_duration_negative_rejected(tmp_path):
    # negative durations are not parseable at all -> still a ScenarioError
    doc = _base_doc()
    doc["window"] = {"duration": "-5s"}
    with pytest.raises(ScenarioError):
        _parse(tmp_path, doc)


def test_invalid_fault_window_duration_rejected(tmp_path):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {"bias": {"offsets": [1.0], "duration": "0s"}}
    }
    with pytest.raises(ScenarioError, match="must be > 0"):
        _parse(tmp_path, doc)


def test_unknown_channel_rejected(tmp_path):
    cfg, df = _parse(tmp_path, _base_doc()), _tiny_df()
    cfg.raw_channels["ghost"] = cfg.raw_channels.pop("pressure")
    with pytest.raises(ScenarioError, match="channel 'ghost' not found"):
        expand_campaign(cfg, df)


def test_unknown_channel_rejected_cli(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"ghost": {"dropout": {"durations": ["2s"]}}}
    result = _run_campaign(tmp_path, doc, _dummy_target_args(), "--plan")
    assert result.exit_code == 2, result.output
    assert "ghost" in result.output


def test_missing_time_column_rejected(tmp_path):
    doc = _base_doc()
    doc["input"] = {"file": "data.parquet", "time_column": "nope"}
    cfg = _parse(tmp_path, doc)  # parses fine without data
    with pytest.raises(ScenarioError, match="time column 'nope'"):
        expand_campaign(cfg, _tiny_df())


def test_missing_time_column_rejected_cli(tmp_path):
    doc = _base_doc()
    doc["input"] = {"file": "data.parquet", "time_column": "nope"}
    result = _run_campaign(tmp_path, doc, _dummy_target_args(), "--plan")
    assert result.exit_code == 2, result.output


def test_zero_case_campaign_rejected(tmp_path):
    # unknown channel filter -> strict rejection naming the bad channel
    # (fix: filters are never silently ignored)
    cfg, cases_df = _parse(tmp_path, _base_doc()), _tiny_df()
    with pytest.raises(ScenarioError, match="unknown channel") as excinfo:
        expand_campaign(cfg, cases_df, channels=("no-such-channel",))
    assert "no-such-channel" in str(excinfo.value)


def test_zero_case_campaign_rejected_cli(tmp_path):
    result = _run_campaign(
        tmp_path, _base_doc(), _dummy_target_args(),
        "--plan", "--channel", "no-such-channel",
    )
    assert result.exit_code == 2, result.output
    assert "no-such-channel" in result.output


def test_validate_campaign_channels_ok(tmp_path):
    cfg = _parse(tmp_path, _base_doc())
    validate_campaign_channels(cfg, _tiny_df())  # must not raise


# ---------------------------------------------------------------------------
# expansion (all 11 faults)
# ---------------------------------------------------------------------------

_FAULT_MATRIX = [
    # (fault type, fault config, expected case count,
    #  expected normalized-param subset on the first case, owner)
    ("dropout", {"durations": ["2s", "5s"]}, 2, {}, "channel"),
    ("freeze", {"durations": ["2s"]}, 1, {}, "channel"),
    (
        "spike",
        {"magnitudes": [10, 50], "directions": ["positive", "negative"]},
        4,
        {"magnitude": 10, "direction": "positive"},
        "channel",
    ),
    ("bias", {"offsets": [1.0, -1.0]}, 2, {"offset": 1.0}, "channel"),
    ("drift", {"rates": [0.1]}, 1, {"rate": 0.1}, "channel"),
    ("noise", {"std": [0.5]}, 1, {"std": 0.5}, "channel"),
    (
        "clipping",
        {"bounds": [{"min": 0, "max": 100}]},
        1,
        {"min": 0, "max": 100},
        "channel",
    ),
    (
        "sample_rate",
        {"transitions": [{"from": "10hz", "to": "5hz"}]},
        1,
        {"from": 10.0, "to": 5.0},
        "channel",
    ),
    (
        "scale",
        {"transforms": [{"factor": 2.0}]},
        1,
        {"factor": 2.0, "offset": 0.0},
        "channel",
    ),
    ("timestamp_gap", {"durations": ["2s"]}, 1, {}, "timeline"),
    ("timestamp_jitter", {"max_jitter_ms": [50]}, 1,
     {"max_jitter_ms": 50}, "timeline"),
]


def _single_fault_doc(ftype, fcfg, owner):
    doc = _base_doc()
    if owner == "timeline":
        doc["channels"] = {"pressure": {"dropout": {"durations": ["2s"]}}}
        # placeholder replaced below; keep structure simple instead:
        doc.pop("channels")
        doc["timeline"] = {ftype: fcfg}
    else:
        doc["channels"] = {"pressure": {ftype: fcfg}}
    return doc


@pytest.mark.parametrize(
    "ftype,fcfg,count,expected_params,owner", _FAULT_MATRIX
)
def test_expand_fault(tmp_path, ftype, fcfg, count, expected_params, owner):
    cfg, cases = _parse_expand(tmp_path, _single_fault_doc(ftype, fcfg, owner))
    assert len(cases) == count
    for case in cases:
        assert case.fault_type == ftype
        assert case.spec.type == ftype
        assert case.target_kind == owner
        # every case carries explicit window bounds
        assert case.spec.start_seconds >= 0
        assert case.spec.duration_seconds > 0
    # normalized engine params on the first expanded case
    for key, value in expected_params.items():
        assert cases[0].spec.params[key] == value


@pytest.mark.parametrize(
    "ftype,fcfg,count,expected_params,owner", _FAULT_MATRIX
)
def test_expand_case_ids_deterministic_unique_safe(
    tmp_path, ftype, fcfg, count, expected_params, owner
):
    doc = _single_fault_doc(ftype, fcfg, owner)
    _, cases_a = _parse_expand(tmp_path, doc)
    _, cases_b = _parse_expand(tmp_path, doc)
    names_a = [c.name for c in cases_a]
    names_b = [c.name for c in cases_b]
    seeds_a = [c.seed for c in cases_a]
    seeds_b = [c.seed for c in cases_b]
    assert names_a == names_b  # deterministic across runs
    assert seeds_a == seeds_b
    assert len(set(names_a)) == count  # unique
    for name in names_a:
        _validate_case_name(name, "campaign case")  # filesystem-safe


def test_spike_cartesian_product(tmp_path):
    doc = _single_fault_doc(
        "spike",
        {"magnitudes": [10, 50], "directions": ["positive", "negative"]},
        "channel",
    )
    _, cases = _parse_expand(tmp_path, doc)
    assert {c.name for c in cases} == {
        "pressure__spike__direction-negative_magnitude-10",
        "pressure__spike__direction-negative_magnitude-50",
        "pressure__spike__direction-positive_magnitude-10",
        "pressure__spike__direction-positive_magnitude-50",
    }
    by_name = {c.name: c for c in cases}
    assert by_name[
        "pressure__spike__direction-positive_magnitude-50"
    ].spec.params["magnitude"] == 50
    assert by_name[
        "pressure__spike__direction-negative_magnitude-10"
    ].spec.params["direction"] == "negative"


def test_timeline_cases_target_timeline(tmp_path):
    doc = _base_doc()
    doc.pop("channels")
    doc["timeline"] = {"timestamp_gap": {"durations": ["2s"]}}
    _, cases = _parse_expand(tmp_path, doc)
    assert len(cases) == 1
    assert cases[0].target == "timeline"
    assert cases[0].target_kind == "timeline"
    assert cases[0].channel is None
    assert cases[0].name == "timeline__timestamp_gap__duration-2s"


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_same_config_same_seeds(tmp_path):
    _, a = _parse_expand(tmp_path, _base_doc())
    _, b = _parse_expand(tmp_path, _base_doc())
    assert [c.seed for c in a] == [c.seed for c in b]


def test_seeds_stable_under_yaml_reordering(tmp_path):
    doc_a = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "pressure": {
                "dropout": {"durations": ["2s"]},
                "bias": {"offsets": [1.0]},
            },
            "temp": {"noise": {"std": [0.5]}},
        },
    }
    doc_b = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "temp": {"noise": {"std": [0.5]}},
            "pressure": {
                "bias": {"offsets": [1.0]},
                "dropout": {"durations": ["2s"]},
            },
        },
    }
    _, a = _parse_expand(tmp_path, doc_a)
    _, b = _parse_expand(tmp_path, doc_b)
    assert {c.name: c.seed for c in a} == {c.name: c.seed for c in b}


def test_seed_changes_when_param_changes(tmp_path):
    doc_a = _base_doc()
    doc_a["channels"] = {"pressure": {"spike": {"magnitudes": [10]}}}
    doc_b = _base_doc()
    doc_b["channels"] = {"pressure": {"spike": {"magnitudes": [11]}}}
    _, a = _parse_expand(tmp_path, doc_a)
    _, b = _parse_expand(tmp_path, doc_b)
    assert a[0].seed != b[0].seed


def test_seeds_not_derived_from_list_position(tmp_path):
    # inserting a new FIRST case must not change later cases' seeds
    doc_a = _base_doc()
    doc_a["channels"] = {"pressure": {"dropout": {"durations": ["2s", "5s"]}}}
    doc_b = _base_doc()
    doc_b["channels"] = {
        "pressure": {"dropout": {"durations": ["1s", "2s", "5s"]}}
    }
    _, a = _parse_expand(tmp_path, doc_a)
    _, b = _parse_expand(tmp_path, doc_b)
    seeds_a = {c.name: c.seed for c in a}
    seeds_b = {c.name: c.seed for c in b}
    assert seeds_a["pressure__dropout__duration-2s"] == seeds_b[
        "pressure__dropout__duration-2s"
    ]
    assert seeds_a["pressure__dropout__duration-5s"] == seeds_b[
        "pressure__dropout__duration-5s"
    ]


def test_seed_changes_with_root_seed(tmp_path):
    doc_a, doc_b = _base_doc(), _base_doc()
    doc_b["seed"] = 43
    _, a = _parse_expand(tmp_path, doc_a)
    _, b = _parse_expand(tmp_path, doc_b)
    assert a[0].seed != b[0].seed


# ---------------------------------------------------------------------------
# inheritance
# ---------------------------------------------------------------------------

def _inheritance_doc():
    return {
        "version": 1, "name": "inherit", "seed": 1,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "window": {"start": "1s", "duration": "3s"},  # campaign level
        "channels": {
            # fault-level window wins over everything
            "pressure": {
                "window": {"start": "2s", "duration": "4s"},
                "bias": {
                    "offsets": [1.0], "start": "5s", "duration": "6s"
                },
            },
            # channel-level window wins over campaign level
            "temp": {
                "window": {"start": "2s", "duration": "4s"},
                "bias": {"offsets": [1.0]},
            },
            # campaign-level window is the fallback
            "humidity": {"bias": {"offsets": [1.0]}},
        },
    }


def test_window_priority_fault_over_channel_over_campaign(tmp_path):
    _, cases = _parse_expand(tmp_path, _inheritance_doc())
    by_target = {c.target: c for c in cases}
    fault = by_target["pressure"].spec
    assert (fault.start_seconds, fault.duration_seconds) == (5.0, 6.0)
    channel = by_target["temp"].spec
    assert (channel.start_seconds, channel.duration_seconds) == (2.0, 4.0)
    campaign = by_target["humidity"].spec
    assert (campaign.start_seconds, campaign.duration_seconds) == (1.0, 3.0)


def test_window_defaults_to_full_input_span(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"pressure": {"bias": {"offsets": [1.0]}}}
    df = _tiny_df()
    _, cases = _parse_expand(tmp_path, doc, df)
    span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    assert cases[0].spec.start_seconds == 0.0
    assert cases[0].spec.duration_seconds == pytest.approx(span)
    assert cases[0].spec.duration_seconds > 0


def test_durations_axis_sets_window_duration(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"pressure": {"dropout": {"durations": ["7s"]}}}
    _, cases = _parse_expand(tmp_path, doc)
    assert cases[0].spec.duration_seconds == 7.0
    assert cases[0].spec.start_seconds == 0.0


def test_every_case_has_explicit_window(tmp_path):
    doc = _inheritance_doc()
    _, cases = _parse_expand(tmp_path, doc)
    for case in cases:
        assert case.spec.start_seconds >= 0
        assert case.spec.duration_seconds > 0


def test_expectation_priority_and_replacement(tmp_path):
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "expect": {
            "exit_code": 0,
            "stdout_contains": ["CAMP"],
            "max_duration_seconds": 30,
        },
        "channels": {
            "pressure": {
                "expect": {"stdout_contains": ["CHAN"]},
                "bias": {
                    "offsets": [1.0],
                    "expect": {"exit_code": 2},
                },
            }
        },
    }
    _, cases = _parse_expand(tmp_path, doc)
    expect = cases[0].expect
    # fault-level key wins
    assert expect["exit_code"] == 2
    # more-specific key REPLACES: ["CHAN"], not ["CAMP", "CHAN"]
    assert expect["stdout_contains"] == ["CHAN"]
    # unspecified keys inherited from the campaign level
    assert expect["max_duration_seconds"] == 30


def test_expectation_channel_only_inherits_campaign(tmp_path):
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "expect": {"exit_code": 0, "stderr_contains": ["warn"]},
        "channels": {
            "pressure": {"dropout": {"durations": ["2s"]}},
        },
    }
    _, cases = _parse_expand(tmp_path, doc)
    assert cases[0].expect == {"exit_code": 0, "stderr_contains": ["warn"]}


# ---------------------------------------------------------------------------
# max_cases
# ---------------------------------------------------------------------------

def _many_case_doc(n, max_cases=None):
    doc = _base_doc()
    doc["channels"] = {
        "pressure": {
            "dropout": {"durations": [f"{i}s" for i in range(1, n + 1)]}
        }
    }
    if max_cases is not None:
        doc["max_cases"] = max_cases
    return doc


def test_max_cases_default_is_100(tmp_path):
    assert DEFAULT_MAX_CASES == 100
    cfg = _parse(tmp_path, _base_doc())
    assert cfg.max_cases == 100


def test_expand_101_cases_hits_default_max(tmp_path):
    cfg = _parse(tmp_path, _many_case_doc(101))
    with pytest.raises(ScenarioError, match="Configured maximum"):
        expand_campaign(cfg, _tiny_df())


def test_max_cases_config_override_respected(tmp_path):
    cfg, cases = _parse_expand(tmp_path, _many_case_doc(101, max_cases=101))
    assert len(cases) == 101


def test_max_cases_cli_override_respected(tmp_path):
    doc = _many_case_doc(3, max_cases=2)
    result = _run_campaign(
        tmp_path, doc, _dummy_target_args(), "--list-cases", "--max-cases", "3"
    )
    assert result.exit_code == 0, result.output
    assert len([l for l in result.output.splitlines() if l.strip()]) == 3
    # without the override the same campaign is rejected
    result = _run_campaign(tmp_path, doc, _dummy_target_args(), "--list-cases")
    assert result.exit_code == 2, result.output


def test_max_cases_never_silently_truncates(tmp_path):
    doc = _many_case_doc(3, max_cases=2)
    result = _run_campaign(tmp_path, doc, _dummy_target_args(), "--list-cases")
    assert result.exit_code == 2, result.output
    assert "Configured maximum" in result.output


def test_cli_run_over_max_is_exit_2_with_zero_execution(tmp_path):
    doc = _many_case_doc(101)  # default max is 100
    script = _marker_target(tmp_path)
    marker = tmp_path / "marker.txt"
    art = tmp_path / "artifacts"
    result = _run_campaign(
        tmp_path, doc,
        [sys.executable, str(script), str(marker), "{data}"],
        "--artifacts", str(art),
    )
    assert result.exit_code == 2, result.output
    assert "Configured maximum" in result.output
    assert not marker.exists()  # zero targets executed
    assert not art.exists()  # no artifacts dir created


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def _filter_doc():
    return {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "pressure": {
                "dropout": {"durations": ["2s"]},
                "bias": {"offsets": [1.0]},
            },
            "temp": {"dropout": {"durations": ["2s"]}},
        },
    }


def _list_names(output):
    return [line.split("|")[0].strip() for line in output.splitlines()
            if line.strip()]


def test_channel_filter_selects_only_that_channel(tmp_path):
    result = _run_campaign(
        tmp_path, _filter_doc(), _dummy_target_args(),
        "--list-cases", "--channel", "pressure",
    )
    assert result.exit_code == 0, result.output
    assert set(_list_names(result.output)) == {
        "pressure__dropout__duration-2s", "pressure__bias__offset-1"
    }


def test_channel_filter_repeatable(tmp_path):
    result = _run_campaign(
        tmp_path, _filter_doc(), _dummy_target_args(),
        "--list-cases", "--channel", "pressure", "--channel", "temp",
    )
    assert result.exit_code == 0, result.output
    assert len(_list_names(result.output)) == 3


def test_fault_filter_selects_only_that_fault(tmp_path):
    result = _run_campaign(
        tmp_path, _filter_doc(), _dummy_target_args(),
        "--list-cases", "--fault", "dropout",
    )
    assert result.exit_code == 0, result.output
    assert set(_list_names(result.output)) == {
        "pressure__dropout__duration-2s", "temp__dropout__duration-2s"
    }


def test_combined_channel_and_fault_filters(tmp_path):
    result = _run_campaign(
        tmp_path, _filter_doc(), _dummy_target_args(),
        "--list-cases", "--channel", "temp", "--fault", "dropout",
    )
    assert result.exit_code == 0, result.output
    assert _list_names(result.output) == ["temp__dropout__duration-2s"]


def test_filters_apply_before_max_cases(tmp_path):
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "pressure": {
                "dropout": {
                    "durations": [f"{i}s" for i in range(1, 11)]
                }
            },
            "temp": {
                "dropout": {
                    "durations": [f"{i}s" for i in range(1, 141)]
                }
            },
        },
    }
    # 150 cases unfiltered -> over the default max of 100
    result = _run_campaign(tmp_path, doc, _dummy_target_args(), "--plan")
    assert result.exit_code == 2, result.output
    # filtered to 10 -> fine under the max
    result = _run_campaign(
        tmp_path, doc, _dummy_target_args(), "--plan", "--channel", "pressure"
    )
    assert result.exit_code == 0, result.output
    assert "Generated cases: 10" in result.output


def test_plan_output_reports_active_filters(tmp_path):
    result = _run_campaign(
        tmp_path, _filter_doc(), _dummy_target_args(),
        "--plan", "--channel", "pressure", "--fault", "bias",
    )
    assert result.exit_code == 0, result.output
    assert "Filters: channels=[pressure] faults=[bias]" in result.output
    result = _run_campaign(tmp_path, _filter_doc(), _dummy_target_args(),
                           "--plan")
    assert "Filters: none" in result.output


def test_coverage_json_reports_active_filters(tmp_path):
    script = _ok_target(tmp_path)
    art = tmp_path / "artifacts"
    result = _run_campaign(
        tmp_path, _filter_doc(),
        [sys.executable, str(script), "{data}"],
        "--artifacts", str(art), "--fault", "dropout",
    )
    assert result.exit_code == 0, result.output
    cov = json.loads((art / "coverage.json").read_text())
    assert cov["filters"] == {"channels": [], "faults": ["dropout"]}
    assert cov["configured_cases"] == 2


# ---------------------------------------------------------------------------
# --plan / --list-cases
# ---------------------------------------------------------------------------

def test_plan_executes_zero_targets_and_writes_zero_files(tmp_path):
    doc = _base_doc()
    script = _marker_target(tmp_path)
    marker = tmp_path / "marker.txt"
    _write_input(tmp_path)
    camp = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(doc))
    before = {p.name for p in tmp_path.iterdir()}
    result = runner.invoke(
        app,
        ["campaign", str(camp), "--plan", "--",
         sys.executable, str(script), str(marker), "{data}"],
    )
    assert result.exit_code == 0, result.output
    assert "CAMPAIGN PLAN" in result.output
    assert "No targets executed" in result.output
    assert not marker.exists()
    assert {p.name for p in tmp_path.iterdir()} == before


def test_list_cases_executes_zero_targets(tmp_path):
    doc = _base_doc()
    script = _marker_target(tmp_path)
    marker = tmp_path / "marker.txt"
    _write_input(tmp_path)
    camp = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(doc))
    result = runner.invoke(
        app,
        ["campaign", str(camp), "--list-cases", "--",
         sys.executable, str(script), str(marker), "{data}"],
    )
    assert result.exit_code == 0, result.output
    assert not marker.exists()


def test_plan_performs_real_applicability_preflight(tmp_path):
    # noise on a string channel is invalid: plan must fail, exit 2
    doc = _base_doc()
    doc["channels"] = {"mode": {"noise": {"std": [0.5]}}}
    result = _run_campaign(tmp_path, doc, _dummy_target_args(), "--plan")
    assert result.exit_code == 2, result.output
    assert "invalid case configuration" in result.output


def test_list_cases_performs_real_applicability_preflight(tmp_path):
    doc = _base_doc()
    doc["channels"] = {"mode": {"noise": {"std": [0.5]}}}
    result = _run_campaign(
        tmp_path, doc, _dummy_target_args(), "--list-cases"
    )
    assert result.exit_code == 2, result.output


def test_list_cases_prints_id_target_fault_params_seed(tmp_path):
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "pressure": {"dropout": {"durations": ["2s"]}},
            "temp": {"bias": {"offsets": [1.0]}},
        },
    }
    result = _run_campaign(tmp_path, doc, _dummy_target_args(), "--list-cases")
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("pressure__dropout__duration-2s | pressure | dropout |")
    assert "duration=2s" in lines[0]
    assert "| seed " in lines[0]
    assert lines[1].startswith("temp__bias__offset-1 | temp | bias |")
    assert "offset=1" in lines[1]
    assert "| seed " in lines[1]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def test_campaign_all_pass_exit_0(tmp_path):
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "baseline": {"enabled": True},
        "channels": {
            "pressure": {"dropout": {"durations": ["2s"]}},
            "temp": {"bias": {"offsets": [1.0]}},
        },
    }
    script = _ok_target(tmp_path)
    result = _run_campaign(
        tmp_path, doc, [sys.executable, str(script), "{data}"]
    )
    assert result.exit_code == 0, result.output
    assert "SUITE RESULT: PASSED" in result.output
    assert "2 passed" in result.output


def test_baseline_failure_blocks_campaign(tmp_path):
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "baseline": {"enabled": True, "expect": {"exit_code": 1}},
        "channels": {
            "pressure": {"dropout": {"durations": ["2s"]}},
            "temp": {"bias": {"offsets": [1.0]}},
        },
    }
    script = _counter_target(tmp_path)
    counter = tmp_path / "counter.txt"
    art = tmp_path / "artifacts"
    result = _run_campaign(
        tmp_path, doc,
        [sys.executable, str(script), str(counter), "{data}"],
        "--artifacts", str(art),
    )
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    # the baseline ran once; zero fault cases executed
    assert counter.read_text() == "1"
    # JUnit: only the failing __baseline__ testcase
    root = ET.parse(art / "junit.xml").getroot()
    testcases = root.findall("testcase")
    assert [t.get("name") for t in testcases] == ["__baseline__"]
    assert testcases[0].find("failure") is not None
    # coverage: 0 executed, nothing counted as passed
    cov = json.loads((art / "coverage.json").read_text())
    assert cov["configured_cases"] == 2
    assert cov["executed_cases"] == 0
    assert cov["passed_cases"] == 0
    assert cov["not_executed_cases"] == 2
    assert cov["baseline"]["status"] == "fail"
    assert all(c["status"] == "not_executed" for c in cov["cases"])


def test_fail_fast_skips_remaining_cases(tmp_path):
    # NOTE: campaign YAML goes through yaml.safe_dump (sorted keys), so
    # within a channel faults expand alphabetically: bias runs first.
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "pressure": {
                "bias": {
                    "offsets": [1.0],
                    "expect": {"stdout_contains": ["NO_SUCH_OUTPUT"]},
                },
                "dropout": {"durations": ["2s"]},
            },
            "temp": {"dropout": {"durations": ["2s"]}},
        },
    }
    script = _counter_target(tmp_path)
    counter = tmp_path / "counter.txt"
    art = tmp_path / "artifacts"
    result = _run_campaign(
        tmp_path, doc,
        [sys.executable, str(script), str(counter), "{data}"],
        "--artifacts", str(art), "--fail-fast",
    )
    assert result.exit_code == 1, result.output
    # only the first (failing) case executed
    assert counter.read_text() == "1"
    cov = json.loads((art / "coverage.json").read_text())
    assert cov["configured_cases"] == 3
    assert cov["executed_cases"] == 1
    assert cov["failed_cases"] == 1
    assert cov["not_executed_cases"] == 2
    assert cov["configured_cases"] == (
        cov["executed_cases"] + cov["not_executed_cases"]
    )
    statuses = [c["status"] for c in cov["cases"]]
    assert statuses == ["fail", "not_executed", "not_executed"]
    assert "pass" not in statuses  # skipped cases never counted as passed
    # JUnit lists only the executed case
    root = ET.parse(art / "junit.xml").getroot()
    assert [t.get("name") for t in root.findall("testcase")] == [
        "pressure__bias__offset-1"
    ]


# ---------------------------------------------------------------------------
# coverage (build_coverage unit tests with synthetic summaries)
# ---------------------------------------------------------------------------

def _cov_config(name="cov", max_cases=100, baseline_enabled=False):
    return CampaignConfig(
        name=name,
        version=1,
        seed=7,
        input_file="data.parquet",
        time_column="timestamp",
        timeout_seconds=60.0,
        max_cases=max_cases,
        baseline_enabled=baseline_enabled,
        baseline_expect=None,
        campaign_window=None,
        campaign_expect=None,
        raw_channels={},
        raw_timeline={},
    )


def _cov_case(name, channel, fault, seed=123):
    spec = FaultSpec(
        channel=channel,
        type=fault,
        start_seconds=0.0,
        duration_seconds=2.0,
        params={},
        cast=None,
    )
    return CampaignCase(
        name=name,
        target=channel,
        target_kind="channel",
        channel=channel,
        fault_type=fault,
        spec=spec,
        seed=seed,
        expect=None,
        id_params={"duration": 2.0},
    )


def _cov_timeline_case(name, fault, seed=456):
    spec = FaultSpec(
        channel=None,
        type=fault,
        start_seconds=0.0,
        duration_seconds=2.0,
        params={},
        cast=None,
    )
    return CampaignCase(
        name=name,
        target="timeline",
        target_kind="timeline",
        channel=None,
        fault_type=fault,
        spec=spec,
        seed=seed,
        expect=None,
        id_params={"duration": 2.0},
    )


def _cell_coverage_fixture():
    cfg = _cov_config()
    cases = [
        _cov_case("a-drop-1", "a", "dropout"),
        _cov_case("a-drop-2", "a", "dropout"),
        _cov_case("b-drop-1", "b", "dropout"),
        _cov_case("b-drop-2", "b", "dropout"),
        _cov_case("c-bias-1", "c", "bias"),
        _cov_case("c-bias-2", "c", "bias"),
        _cov_case("d-bias-1", "d", "bias"),
        _cov_case("d-bias-2", "d", "bias"),
    ]
    summary = {
        "cases": [
            {"name": "a-drop-1", "status": "pass"},
            {"name": "a-drop-2", "status": "pass"},
            {"name": "b-drop-1", "status": "fail"},
            {"name": "b-drop-2", "status": "fail"},
            {"name": "c-bias-1", "status": "pass"},
            {"name": "c-bias-2", "status": "fail"},
            # d-bias-* never executed
        ],
        "baseline": {"status": "pass"},
    }
    cov = build_coverage(
        cfg, cases, summary, "sha256:abc", "inputsha", None, (), ()
    )
    return cov


def test_build_coverage_cell_statuses():
    cov = _cell_coverage_fixture()
    assert cov["channels"]["a"]["dropout"]["status"] == "pass"
    assert cov["channels"]["b"]["dropout"]["status"] == "fail"
    assert cov["channels"]["c"]["bias"]["status"] == "partial"
    assert cov["channels"]["d"]["bias"]["status"] == "not_executed"
    d = cov["channels"]["d"]["bias"]
    assert (d["configured"], d["executed"], d["passed"], d["failed"],
            d["not_executed"]) == (2, 0, 0, 0, 2)
    a = cov["channels"]["a"]["dropout"]
    assert (a["configured"], a["executed"], a["passed"], a["failed"],
            a["not_executed"]) == (2, 2, 2, 0, 0)


def test_build_coverage_totals():
    cov = _cell_coverage_fixture()
    assert cov["configured_cases"] == 8
    assert cov["executed_cases"] == 6
    assert cov["passed_cases"] == 3
    assert cov["failed_cases"] == 3
    assert cov["not_executed_cases"] == 2
    rows = {r["id"]: r["status"] for r in cov["cases"]}
    assert rows["d-bias-1"] == "not_executed"
    assert rows["c-bias-2"] == "fail"


def test_configured_pass_rate_math():
    cfg = _cov_config()
    cases = [_cov_case(f"case-{i}", "a", "dropout", seed=i) for i in range(14)]
    summary = {
        "cases": [
            {"name": f"case-{i}", "status": "pass" if i < 13 else "fail"}
            for i in range(14)
        ],
        "baseline": {"status": "pass"},
    }
    cov = build_coverage(
        cfg, cases, summary, "sha256:abc", "inputsha", None, (), ()
    )
    assert cov["configured_pass_rate"] == 0.9286
    assert cov["configured_pass_rate"] == round(13 / 14, 4)


_EXPECTED_COVERAGE_KEYS = {
    "version",
    "campaign",
    "campaign_fingerprint",
    "input_sha256",
    "input_sidecar_sha256",
    "filters",
    "max_cases",
    "configured_cases",
    "executed_cases",
    "passed_cases",
    "failed_cases",
    "not_executed_cases",
    "configured_pass_rate",
    "baseline",
    "channels",
    "timeline",
    "cases",
}


def test_coverage_json_key_set_exact_and_no_safety_score():
    cov = _cell_coverage_fixture()
    assert set(cov.keys()) == _EXPECTED_COVERAGE_KEYS
    assert "safety_score" not in json.dumps(cov)


def test_coverage_csv_parses_with_correct_header_and_rows():
    cfg = _cov_config()
    cases = [
        _cov_case("a-drop-1", "a", "dropout"),
        _cov_case("a-drop-2", "a", "dropout"),
        _cov_case("b-bias-1", "b", "bias"),
        _cov_timeline_case("tl-jitter-1", "timestamp_jitter"),
    ]
    summary = {
        "cases": [
            {"name": "a-drop-1", "status": "pass"},
            {"name": "a-drop-2", "status": "pass"},
            {"name": "b-bias-1", "status": "fail"},
            {"name": "tl-jitter-1", "status": "pass"},
        ],
        "baseline": None,
    }
    cov = build_coverage(
        cfg, cases, summary, "sha256:abc", "inputsha", None, (), ()
    )
    text = render_coverage_csv(cov)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "target_kind", "target", "fault", "configured", "executed",
        "passed", "failed", "not_executed", "status",
    ]
    # one row per (kind, target, fault) cell: 2 channel cells + 1 timeline
    assert len(rows) == 1 + 3
    by_cell = {(r[0], r[1], r[2]): r for r in rows[1:]}
    assert by_cell[("channel", "a", "dropout")][8] == "pass"
    assert by_cell[("channel", "b", "bias")][8] == "fail"
    assert by_cell[("timeline", "timeline", "timestamp_jitter")][8] == "pass"


def test_coverage_markdown_escapes_pipes_and_control_chars():
    cfg = _cov_config(name="md-campaign")
    weird = _cov_case("weird-drop", "we|ird<&>", "dropout")
    tabbed = _cov_case("tab\tcase", "other", "bias")
    plain = _cov_case("plain-drop", "we|ird<&>", "dropout")
    cases = [weird, plain, tabbed]
    summary = {
        "cases": [
            {"name": "weird-drop", "status": "pass"},
            {"name": "plain-drop", "status": "pass"},
            # "tab\tcase" never executed
        ],
        "baseline": {"status": "pass"},
    }
    cov = build_coverage(
        cfg, cases, summary, "sha256:abc", "inputsha", None, (), ()
    )
    md = render_coverage_markdown(cfg.name, cov)
    # pipe escaped, raw unescaped channel name absent
    assert "we\\|ird<&>" in md
    assert "we|ird<&>" not in md
    # control char stripped (replaced with a space), never raw
    assert "\t" not in md
    assert "- tab case (reason: fail-fast)" in md
    # 'other' has no dropout cell -> NOT CONFIGURED legend
    assert "NOT CONFIGURED" in md


def test_coverage_markdown_not_configured_cell():
    cov = _cell_coverage_fixture()
    md = render_coverage_markdown("cov", cov)
    # channel d configured bias only; the dropout column shows NOT CONFIGURED
    assert "— = NOT CONFIGURED" in md
    d_line = next(
        line for line in md.splitlines() if line.startswith("| d |")
    )
    assert "—" in d_line


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def _artifacts_doc():
    return {
        "version": 1, "name": "artifacts-demo", "seed": 99,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "baseline": {"enabled": True},
        "expect": {"exit_code": 0},
        "channels": {
            "pressure": {"dropout": {"durations": ["2s", "5s"]}},
            "temp": {"bias": {"offsets": [1.0]}},
        },
    }


def _run_artifacts_campaign(tmp_path, doc=None, input_name="data.parquet"):
    doc = _artifacts_doc() if doc is None else doc
    script = _ok_target(tmp_path)
    art = tmp_path / "artifacts"
    result = _run_campaign(
        tmp_path, doc, [sys.executable, str(script), "{data}"],
        "--artifacts", str(art), input_name=input_name,
    )
    assert result.exit_code == 0, result.output
    return art


def test_expanded_suite_yaml_round_trip(tmp_path):
    doc = _artifacts_doc()
    art = _run_artifacts_campaign(tmp_path, doc)
    text = (art / "expanded-suite.yaml").read_text()
    assert "/tmp" not in text
    assert str(tmp_path) not in text
    assert "campaign fingerprint: sha256:" in text
    suite = load_suite(art / "expanded-suite.yaml")
    expected_names = [
        "pressure__dropout__duration-2s",
        "pressure__dropout__duration-5s",
        "temp__bias__offset-1",
    ]
    assert [c.name for c in suite.cases] == expected_names
    _, expanded = _parse_expand(tmp_path, doc)
    assert [c.seed for c in suite.cases] == [c.seed for c in expanded]
    assert suite.baseline_enabled is True
    assert suite.input_file == "data.parquet"
    assert suite.time_column == "timestamp"
    # all cases/seeds/normalized params/expectations/baseline/input present
    raw = yaml.safe_load(text)
    assert raw["input"] == {"file": "data.parquet", "time_column": "timestamp"}
    assert raw["baseline"] == {"enabled": True}
    assert len(raw["cases"]) == 3
    assert raw["cases"][0]["faults"][0]["duration"] == "2s"
    assert raw["cases"][2]["faults"][0]["offset"] == 1.0
    assert raw["cases"][0]["expect"] == {"exit_code": 0}
    assert all("seed" in c for c in raw["cases"])


def test_campaign_fingerprint_stable_and_sensitive(tmp_path):
    doc = _artifacts_doc()
    cfg, cases = _parse_expand(tmp_path, doc)
    fp1 = campaign_fingerprint(cfg, cases)
    assert fp1.startswith("sha256:")
    assert len(fp1) == len("sha256:") + 64
    int(fp1.split(":", 1)[1], 16)  # valid hex digest
    # stable across runs
    _, cases2 = _parse_expand(tmp_path, doc)
    assert campaign_fingerprint(cfg, cases2) == fp1
    # changes when the root seed changes
    cfg_seed = dataclasses.replace(cfg, seed=cfg.seed + 1)
    assert campaign_fingerprint(cfg_seed, cases) != fp1
    # changes when a parameter value changes
    doc2 = _artifacts_doc()
    doc2["channels"]["temp"]["bias"]["offsets"] = [2.0]
    cfg2, cases3 = _parse_expand(tmp_path, doc2)
    assert campaign_fingerprint(cfg2, cases3) != fp1


def test_input_sha256_matches_file(tmp_path):
    art = _run_artifacts_campaign(tmp_path)
    cov = json.loads((art / "coverage.json").read_text())
    expected = hashlib.sha256(
        (tmp_path / "data.parquet").read_bytes()
    ).hexdigest()
    assert cov["input_sha256"] == expected


def test_csv_sidecar_sha256_recorded(tmp_path):
    doc = _artifacts_doc()
    doc["input"] = {"file": "data.csv", "time_column": "timestamp"}
    art = _run_artifacts_campaign(tmp_path, doc, input_name="data.csv")
    sidecar = tmp_path / "data.schema.json"
    assert sidecar.is_file()
    cov = json.loads((art / "coverage.json").read_text())
    assert cov["input_sidecar_sha256"] == hashlib.sha256(
        sidecar.read_bytes()
    ).hexdigest()
    assert cov["input_sha256"] == hashlib.sha256(
        (tmp_path / "data.csv").read_bytes()
    ).hexdigest()


def test_junit_has_one_testcase_per_case(tmp_path):
    doc = {
        "version": 1, "name": "t", "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "channels": {
            "pressure": {"dropout": {"durations": ["2s", "5s"]}},
            "temp": {"bias": {"offsets": [1.0]}},
        },
    }
    script = _ok_target(tmp_path)
    art = tmp_path / "artifacts"
    result = _run_campaign(
        tmp_path, doc, [sys.executable, str(script), "{data}"],
        "--artifacts", str(art),
    )
    assert result.exit_code == 0, result.output
    root = ET.parse(art / "junit.xml").getroot()
    assert {t.get("name") for t in root.findall("testcase")} == {
        "pressure__dropout__duration-2s",
        "pressure__dropout__duration-5s",
        "temp__bias__offset-1",
    }


def test_ownership_marker_includes_campaign_files(tmp_path):
    art = _run_artifacts_campaign(tmp_path)
    marker = json.loads(
        (art / ".telemetry-resilience-artifacts.json").read_text()
    )
    owned = marker["owned_entries"]
    for name in (
        "expanded-suite.yaml", "coverage.json", "coverage.md", "coverage.csv"
    ):
        assert name in owned, f"{name} missing from owned_entries"


def test_overwrite_refused_with_untracked_file(tmp_path):
    doc = _base_doc()
    script = _ok_target(tmp_path)
    art = tmp_path / "artifacts"
    first = _run_campaign(
        tmp_path, doc, [sys.executable, str(script), "{data}"],
        "--artifacts", str(art),
    )
    assert first.exit_code == 0, first.output
    (art / "notes.txt").write_text("user notes - do not delete")
    second = _run_campaign(
        tmp_path, doc, [sys.executable, str(script), "{data}"],
        "--artifacts", str(art), "--overwrite-artifacts",
    )
    assert second.exit_code == 2, second.output
    assert "untracked" in second.output
    assert (art / "notes.txt").read_text() == "user notes - do not delete"


def test_input_file_hash_unchanged_by_plan_and_run(tmp_path):
    doc = _base_doc()
    data = _write_input(tmp_path)
    camp = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(doc))
    script = _ok_target(tmp_path)
    digest = lambda: hashlib.sha256(data.read_bytes()).hexdigest()
    h0 = digest()
    planned = runner.invoke(
        app, ["campaign", str(camp), "--plan", "--", *_dummy_target_args()]
    )
    assert planned.exit_code == 0, planned.output
    assert digest() == h0  # plan/preflight never mutates the dataframe
    art = tmp_path / "art"
    ran = runner.invoke(
        app,
        ["campaign", str(camp), "--artifacts", str(art), "--",
         sys.executable, str(script), "{data}"],
    )
    assert ran.exit_code == 0, ran.output
    assert digest() == h0


# ---------------------------------------------------------------------------
# CLI: target optional for --plan / --list-cases, required for real runs
# ---------------------------------------------------------------------------

def _plan_no_target_doc():
    return _base_doc()


def test_cli_plan_without_target(tmp_path):
    _write_input(tmp_path)
    camp = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(_base_doc()))
    res = runner.invoke(app, ["campaign", str(camp), "--plan"])
    assert res.exit_code == 0, res.output
    assert "CAMPAIGN PLAN" in res.output


def test_cli_list_cases_without_target(tmp_path):
    _write_input(tmp_path)
    camp = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(_base_doc()))
    res = runner.invoke(app, ["campaign", str(camp), "--list-cases"])
    assert res.exit_code == 0, res.output
    assert "pressure__dropout__duration-2s" in res.output


def test_cli_run_without_target_is_exit_2(tmp_path):
    _write_input(tmp_path)
    camp = write_yaml(tmp_path, "campaign.yaml", yaml.safe_dump(_base_doc()))
    res = runner.invoke(app, ["campaign", str(camp)])
    assert res.exit_code == 2
    assert "target command is required" in res.output
