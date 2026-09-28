"""Regression tests for RUN 2.1: CI reliability, artifact safety, dry-run.

Covers: artifact-dir name collisions, reserved artifact names, unsafe case
names, dry-run fault-applicability validation, target launch errors,
Ctrl+C handling, artifact-directory reuse/ownership, symlink escape
safety, blocked-baseline JUnit, Markdown escaping, and artifact write
failures. All pre-existing tests must keep passing unchanged.
"""
import json
import sys
import xml.etree.ElementTree as ET

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import telemetry_resilience.runner as runner_module
import telemetry_resilience.cli as cli_module
import telemetry_resilience.suite_run as suite_run_module
from telemetry_resilience.cli import app
from telemetry_resilience.io import write_file
from telemetry_resilience.reporting import (
    render_junit_xml,
    render_summary_markdown,
)
from telemetry_resilience.suite import load_suite
from telemetry_resilience.suite_run import _resolve_artifact_subdir
from telemetry_resilience.models import ScenarioError

from .conftest import write_yaml

runner = CliRunner()


# ---------------------------------------------------------------------------
# helpers (mirroring tests/test_suite.py, kept local for independence)
# ---------------------------------------------------------------------------

def _setup(tmp_path, telemetry_df, name="data.parquet"):
    data = tmp_path / name
    write_file(telemetry_df, data)
    return data


def _target(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def _fault(ftype, channel, start, duration, **params):
    d = {"type": ftype, "start": start, "duration": duration}
    if channel is not None:
        d["channel"] = channel
    d.update(params)
    return d


def _case(name, seed, faults, expect=None, timeout=None):
    d = {"name": name, "seed": seed, "faults": faults}
    if expect is not None:
        d["expect"] = expect
    if timeout is not None:
        d["timeout_seconds"] = timeout
    return d


def _suite_dict(cases, baseline=True, name="demo-suite", timeout=None,
                baseline_expect=None, extra_top=None):
    d = {
        "version": 1,
        "name": name,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "cases": cases,
    }
    if timeout is not None:
        d["timeout_seconds"] = timeout
    if baseline:
        d["baseline"] = {"enabled": True}
        if baseline_expect is not None:
            d["baseline"]["expect"] = baseline_expect
    if extra_top:
        d.update(extra_top)
    return d


def _write_suite(tmp_path, suite_dict, name="suite.yaml"):
    return write_yaml(tmp_path, name, yaml.safe_dump(suite_dict))


def _run(tmp_path, suite_dict, target_args, *extra_cli):
    suite = _write_suite(tmp_path, suite_dict)
    args = ["suite", str(suite), *extra_cli, "--", *target_args]
    return runner.invoke(app, args)


def _expect_exit_2_before_execution(tmp_path, telemetry_df, suite_dict):
    """Suite must be rejected before any target executes (exit 2)."""
    _setup(tmp_path, telemetry_df)
    marker = tmp_path / "marker.txt"
    target = _target(tmp_path, "marker.py",
                     f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    result = _run(tmp_path, suite_dict,
                  [sys.executable, target, "{data}"])
    assert result.exit_code == 2, result.output
    assert not marker.exists()  # nothing executed
    assert "Traceback" not in result.output
    return result


# ---------------------------------------------------------------------------
# 1. artifact directory name collisions
# ---------------------------------------------------------------------------

def test_run21_normalized_dirname_collision_rejected(tmp_path, telemetry_df):
    suite = _suite_dict([
        _case("a b", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
        _case("a_b", 12, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    result = _expect_exit_2_before_execution(tmp_path, telemetry_df, suite)
    assert "same artifact directory" in result.output


def test_run21_dirname_collision_case_insensitive(tmp_path, telemetry_df):
    suite = _suite_dict([
        _case("Case One", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
        _case("case_one", 12, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    result = _expect_exit_2_before_execution(tmp_path, telemetry_df, suite)
    assert "same artifact directory" in result.output


def test_run21_distinct_names_still_pass(tmp_path, telemetry_df):
    target = _target(tmp_path, "t.py", "print('ok')\n")
    _setup(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("case-a", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
        _case("case-b", 12, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 2. reserved artifact names + unsafe names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reserved", ["baseline", "BASELINE", "summary.json",
                                     "Summary.JSON", "summary.md", "junit.xml",
                                     "JUNIT.XML"])
def test_run21_reserved_case_names_rejected(tmp_path, telemetry_df, reserved):
    suite = _suite_dict([
        _case(reserved, 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    result = _expect_exit_2_before_execution(tmp_path, telemetry_df, suite)
    assert "reserved" in result.output.lower()


@pytest.mark.parametrize("bad", ["a/b", "a\\b", ".", "..", "trailing.",
                                 "a|b", "a<b", "nul", "COM1"])
def test_run21_unsafe_case_names_rejected(tmp_path, telemetry_df, bad):
    suite = _suite_dict([
        _case(bad, 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    _expect_exit_2_before_execution(tmp_path, telemetry_df, suite)


def test_run21_control_char_case_name_rejected(tmp_path, telemetry_df):
    suite = _suite_dict([
        _case("evil\nname", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    result = _expect_exit_2_before_execution(tmp_path, telemetry_df, suite)
    assert "control" in result.output.lower()


# ---------------------------------------------------------------------------
# 3. dry-run really validates fault applicability
# ---------------------------------------------------------------------------

def _setup_with_string_column(tmp_path, telemetry_df):
    df = telemetry_df.copy()
    df["label"] = "nominal"
    data = tmp_path / "data.parquet"
    write_file(df, data)
    return data


def test_run21_dry_run_noise_on_string_column_fails(tmp_path, telemetry_df):
    _setup_with_string_column(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("noise", "label", "1s", "2s", std=0.5)]),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--dry-run")
    assert result.exit_code == 2, result.output
    assert "configuration valid" not in result.output
    assert "label" in result.output


def test_run21_dry_run_drift_on_strict_int_without_cast_fails(
        tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("drift", "count_int", "1s", "2s", rate=0.4)]),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--dry-run")
    assert result.exit_code == 2, result.output
    assert "configuration valid" not in result.output


def test_run21_dry_run_valid_numeric_noise_succeeds(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    marker = tmp_path / "marker.txt"
    target = _target(tmp_path, "marker.py",
                     f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("noise", "motor_temperature", "1s", "2s",
                               std=0.5)]),
    ], baseline=False)
    art = tmp_path / "artifacts"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "configuration valid" in result.output
    assert not marker.exists()  # target never executed
    assert not art.exists()  # nothing written


def test_run21_dry_run_valid_explicit_cast_succeeds(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    faults = [_fault("drift", "count_int", "1s", "2s", rate=0.4)]
    faults[0]["cast"] = "float32"
    suite = _suite_dict([_case("c", 11, faults)], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--dry-run")
    assert result.exit_code == 0, result.output
    assert "configuration valid" in result.output


# ---------------------------------------------------------------------------
# 4. target executable launch errors
# ---------------------------------------------------------------------------

def test_run21_missing_executable_baseline_clean_exit_2(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    suite_file = _write_suite(tmp_path, suite)
    result = runner.invoke(
        app, ["suite", str(suite_file), "--",
              "definitely-not-a-real-command", "{data}"])
    assert result.exit_code == 2, result.output
    assert "TARGET COULD NOT START" in result.output
    assert "executable not found" in result.output
    assert "Traceback" not in result.output


def test_run21_missing_executable_case_clean_exit_2(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    suite_nb = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    suite_nb_file = _write_suite(tmp_path, suite_nb, name="suite_nb.yaml")
    result = runner.invoke(
        app, ["suite", str(suite_nb_file), "--",
              "definitely-not-a-real-command", "{data}"])
    assert result.exit_code == 2, result.output
    assert "TARGET COULD NOT START" in result.output
    assert "Traceback" not in result.output


def test_run21_launch_permission_denied_exit_2(tmp_path, telemetry_df,
                                               monkeypatch):
    _setup(tmp_path, telemetry_df)
    import subprocess as _sp

    def _raise(cmd, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(_sp, "run", _raise)
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    suite_file = _write_suite(tmp_path, suite)
    result = runner.invoke(
        app, ["suite", str(suite_file), "--", sys.executable, "t.py", "{data}"])
    assert result.exit_code == 2, result.output
    assert "TARGET COULD NOT START" in result.output
    assert "permission denied" in result.output
    assert "Traceback" not in result.output


def test_run21_test_command_missing_executable_exit_2(tmp_path, telemetry_df):
    data = _setup(tmp_path, telemetry_df)
    scenario = {
        "version": 1,
        "seed": 5,
        "input": {"file": str(data), "time_column": "timestamp"},
        "faults": [{"type": "dropout", "channel": "gps_latitude",
                    "start": "1s", "duration": "2s"}],
    }
    sc_file = tmp_path / "scenario.yaml"
    sc_file.write_text(yaml.safe_dump(scenario))
    result = runner.invoke(
        app, ["test", str(sc_file), "--",
              "definitely-not-a-real-command", "{data}"])
    assert result.exit_code == 2, result.output
    assert "TARGET COULD NOT START" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# 5. Ctrl+C handling (baseline, case, single-scenario test command)
# ---------------------------------------------------------------------------

def _raise_keyboard_interrupt(*args, **kwargs):
    raise KeyboardInterrupt()


def test_run21_interrupt_during_baseline(tmp_path, telemetry_df, monkeypatch):
    _setup(tmp_path, telemetry_df)
    monkeypatch.setattr(runner_module, "run_target",
                        _raise_keyboard_interrupt)
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    suite_file = _write_suite(tmp_path, suite)
    target = _target(tmp_path, "t.py", "print('x')\n")
    result = runner.invoke(
        app, ["suite", str(suite_file), "--",
              sys.executable, target, "{data}"])
    assert result.exit_code == 3, result.output
    assert "Interrupted by user" in result.output
    assert "Traceback" not in result.output


def test_run21_interrupt_during_case(tmp_path, telemetry_df, monkeypatch):
    _setup(tmp_path, telemetry_df)
    monkeypatch.setattr(runner_module, "run_target",
                        _raise_keyboard_interrupt)
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    suite_file = _write_suite(tmp_path, suite)
    target = _target(tmp_path, "t.py", "print('x')\n")
    result = runner.invoke(
        app, ["suite", str(suite_file), "--",
              sys.executable, target, "{data}"])
    assert result.exit_code == 3, result.output
    assert "Interrupted by user" in result.output
    assert "temporary files cleaned up" in result.output
    assert "Traceback" not in result.output


def test_run21_interrupt_during_test_command(tmp_path, telemetry_df,
                                              monkeypatch):
    data = _setup(tmp_path, telemetry_df)
    monkeypatch.setattr(cli_module, "run_target", _raise_keyboard_interrupt)
    scenario = {
        "version": 1,
        "seed": 5,
        "input": {"file": str(data), "time_column": "timestamp"},
        "faults": [{"type": "dropout", "channel": "gps_latitude",
                    "start": "1s", "duration": "2s"}],
    }
    sc_file = tmp_path / "scenario.yaml"
    sc_file.write_text(yaml.safe_dump(scenario))
    target = _target(tmp_path, "t.py", "print('x')\n")
    result = runner.invoke(
        app, ["test", str(sc_file), "--", sys.executable, target, "{data}"])
    assert result.exit_code == 3, result.output
    assert "Interrupted by user" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# 6. artifact directory reuse / ownership
# ---------------------------------------------------------------------------

def _passing_suite_no_baseline(tmp_path, telemetry_df):
    target = _target(tmp_path, "t.py", "print('ok')\n")
    _setup(tmp_path, telemetry_df)
    return _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False), [sys.executable, target, "{data}"]


def test_run21_rerun_into_owned_dir_requires_flag(tmp_path, telemetry_df):
    suite, target_args = _passing_suite_no_baseline(tmp_path, telemetry_df)
    art = tmp_path / "art"
    r1 = _run(tmp_path, suite, target_args, "--artifacts", str(art))
    assert r1.exit_code == 0, r1.output
    assert (art / ".telemetry-resilience-artifacts.json").exists()
    # second run without --overwrite-artifacts must refuse
    r2 = _run(tmp_path, suite, target_args, "--artifacts", str(art))
    assert r2.exit_code == 2, r2.output
    assert "overwrite-artifacts" in r2.output


def test_run21_overwrite_artifacts_on_owned_dir(tmp_path, telemetry_df):
    suite, target_args = _passing_suite_no_baseline(tmp_path, telemetry_df)
    art = tmp_path / "art"
    r1 = _run(tmp_path, suite, target_args, "--artifacts", str(art))
    assert r1.exit_code == 0, r1.output
    r2 = _run(tmp_path, suite, target_args, "--artifacts", str(art),
              "--overwrite-artifacts")
    assert r2.exit_code == 0, r2.output
    assert (art / "summary.json").exists()


def test_run21_stale_keep_data_removed_on_clean_rerun(tmp_path, telemetry_df):
    suite, target_args = _passing_suite_no_baseline(tmp_path, telemetry_df)
    art = tmp_path / "art"
    r1 = _run(tmp_path, suite, target_args, "--artifacts", str(art),
              "--keep-data")
    assert r1.exit_code == 0, r1.output
    corrupted = art / "c" / "corrupted.parquet"
    assert corrupted.exists()
    # clean rerun with the documented safe replacement flow
    r2 = _run(tmp_path, suite, target_args, "--artifacts", str(art),
              "--overwrite-artifacts")
    assert r2.exit_code == 0, r2.output
    assert not corrupted.exists(), "stale corrupted telemetry survived rerun"
    assert (art / "summary.json").exists()
    assert (art / "c" / "faults.json").exists()


def test_run21_unowned_artifact_dir_never_deleted(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    art = tmp_path / "art"
    art.mkdir()
    precious = art / "precious.txt"
    precious.write_text("do not delete")
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art), "--overwrite-artifacts")
    assert result.exit_code == 2, result.output
    assert precious.read_text() == "do not delete"
    assert not (art / "summary.json").exists()


def test_run21_unowned_artifact_dir_refused_without_flag(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    art = tmp_path / "art"
    art.mkdir()
    (art / "notes.txt").write_text("unrelated")
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 2, result.output
    assert (art / "notes.txt").exists()


# ---------------------------------------------------------------------------
# 7. symlink / artifact escape safety
# ---------------------------------------------------------------------------

def test_run21_symlink_escape_rejected(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    outside = tmp_path / "outside"
    outside.mkdir()
    art = tmp_path / "art"
    art.mkdir()
    (art / "case-name").symlink_to(outside, target_is_directory=True)
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([
        _case("case-name", 11,
              [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert list(outside.iterdir()) == [], "data written outside artifact root!"


def test_run21_resolve_subdir_refuses_symlink(tmp_path):
    from pathlib import Path
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "evil").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ScenarioError):
        _resolve_artifact_subdir(root, "evil")


def test_run21_resolve_subdir_refuses_escape(tmp_path):
    from pathlib import Path
    root = (tmp_path / "root").resolve()
    root.mkdir()
    with pytest.raises(ScenarioError):
        _resolve_artifact_subdir(root, "../escape")


# ---------------------------------------------------------------------------
# 8. blocked baseline must produce failing JUnit
# ---------------------------------------------------------------------------

def test_run21_blocked_baseline_junit_fails(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "crash.py", "import sys\nsys.exit(9)\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline_expect={"exit_code": 0})
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    tree = ET.parse(str(art / "junit.xml"))
    root = tree.getroot()
    assert int(root.attrib["tests"]) >= 1
    assert int(root.attrib["failures"]) >= 1
    names = [c.attrib["name"] for c in root.iter("testcase")]
    assert "__baseline__" in names
    assert "c" not in names  # never-executed case must not appear as passing
    baseline_tc = [c for c in root.iter("testcase")
                   if c.attrib["name"] == "__baseline__"][0]
    failures = list(baseline_tc.iter("failure"))
    assert len(failures) == 1
    payload = json.loads((art / "summary.json").read_text())
    assert payload["status"] == "blocked"
    assert payload["baseline"]["status"] == "fail"


def test_run21_junit_combines_multiple_failures_into_one(tmp_path):
    cases = [{
        "name": "multi",
        "status": "fail",
        "duration_seconds": 1.0,
        "failures": ["stdout did not contain 'A'", "exit_code 9 != 0"],
    }]
    xml = render_junit_xml("s", cases)
    root = ET.fromstring(xml.split("\n", 1)[1])
    tc = list(root.iter("testcase"))[0]
    failures = list(tc.iter("failure"))
    assert len(failures) == 1, "one logical failing testcase per case"
    assert "A" in failures[0].attrib["message"]
    assert "A" in (failures[0].text or "")


# ---------------------------------------------------------------------------
# 10. Markdown report safety
# ---------------------------------------------------------------------------

def test_run21_markdown_escapes_table_breaking_text():
    cases = [{
        "name": "weird|name",
        "status": "fail",
        "fault_count": 1,
        "duration_seconds": 1.2,
        "failures": ["stdout did not contain 'x|y\nz'"],
    }]
    md = render_summary_markdown("my\nsuite", None, cases, 0, 1, 1, "fail")
    assert "Suite: my suite" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("| weird")]
    assert len(rows) == 1  # name cannot inject extra rows/columns
    assert "weird\\|name" in rows[0]  # pipe escaped, not a column delimiter
    detail_lines = [ln for ln in md.splitlines()
                    if "did not contain" in ln]
    assert len(detail_lines) == 1  # newline in needle cannot break the list
    assert "### weird name" not in md  # heading sanitized, single line
    assert "### weird|name" in md


def test_run21_markdown_end_to_end_with_pipe_needle(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('nothing here')\n")
    suite = _suite_dict([
        _case("pipe-case", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_contains": ["a|b"]}),
    ], baseline=False)
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 1, result.output
    md = (art / "summary.md").read_text()
    rows = [ln for ln in md.splitlines() if ln.startswith("| pipe-case")]
    assert len(rows) == 1
    assert rows[0].startswith("| pipe-case | FAIL |")


# ---------------------------------------------------------------------------
# 11. artifact write failures -> clean documented error, no traceback
# ---------------------------------------------------------------------------

def test_run21_artifact_write_failure_clean_exit(tmp_path, telemetry_df,
                                                 monkeypatch):
    def _boom(path, payload):
        raise PermissionError("denied")

    monkeypatch.setattr(suite_run_module, "write_summary_json", _boom)
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('ok')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ], baseline=False)
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 3, result.output
    assert "Traceback" not in result.output
    assert "cannot write" in result.output.lower()
