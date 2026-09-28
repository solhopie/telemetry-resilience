"""Tests for `telemetry-resilience suite`: baselines, isolation, artifacts,
reporting, timeouts, new assertions, validation, dry-run, and interrupts."""
import glob
import hashlib
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from telemetry_resilience import runner as runner_module
from telemetry_resilience.cli import app
from telemetry_resilience.io import read_file, write_file
from telemetry_resilience.suite import load_suite
from telemetry_resilience.suite_run import run_suite
from telemetry_resilience.models import ScenarioError

from .conftest import write_yaml

runner = CliRunner()


# ---------------------------------------------------------------------------
# helpers
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


TARGET_DEGRADED = (
    "import sys\n"
    "import pandas as pd\n"
    "df = pd.read_parquet(sys.argv[1])\n"
    "print('DEGRADED_MODE' if df['gps_latitude'].isna().any() else 'NOMINAL')\n"
)

TARGET_CRASH = "import sys\nsys.exit(9)\n"

TARGET_EXIT3 = "import sys\nsys.exit(3)\n"

TARGET_SLEEP = "import time\ntime.sleep(5)\nprint('slow done')\n"

TARGET_STDERR = (
    "import sys\n"
    "print('boom happened', file=sys.stderr)\n"
    "print('all good')\n"
)


def _run(tmp_path, suite_dict, target_args, *extra_cli):
    suite = _write_suite(tmp_path, suite_dict)
    args = ["suite", str(suite), *extra_cli, "--", *target_args]
    return runner.invoke(app, args)


def _basic_data_and_target(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    return _target(tmp_path, "target.py", TARGET_DEGRADED)


# ---------------------------------------------------------------------------
# baseline behaviour
# ---------------------------------------------------------------------------

def test_suite_baseline_pass_and_cases_pass(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("case-a", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"exit_code": 0}),
        _case("case-b", 12, [_fault("dropout", "imu_z", "1s", "2s")],
              {"exit_code": 0}),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert "Baseline" in result.output
    assert "PASS  case-a" in result.output
    assert "PASS  case-b" in result.output
    assert "2 passed" in result.output
    assert "SUITE RESULT: PASSED" in result.output


def test_suite_baseline_fail_stops_suite(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict(
        [_case("case-a", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        baseline_expect={"stdout_contains": ["READY"]},
    )
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 1, result.output
    assert "BASELINE FAILED" in result.output
    assert "BLOCKED" in result.output
    assert "case-a" not in result.output  # fault cases never ran


def test_suite_baseline_custom_expect_pass(tmp_path, telemetry_df):
    target = _target(tmp_path, "ready.py", "print('READY')\n")
    _setup(tmp_path, telemetry_df)
    suite = _suite_dict(
        [_case("case-a", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        baseline_expect={"stdout_contains": ["READY"]},
    )
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert "Baseline" in result.output
    assert "  [PASS] stdout contains 'READY'" in result.output


def test_suite_no_baseline_section_runs_cases(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    d = _suite_dict(
        [_case("case-a", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        baseline=False,
    )
    art = tmp_path / "art"
    result = _run(tmp_path, d, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 0, result.output
    summary = json.loads((art / "summary.json").read_text())
    assert summary["baseline"] is None


# ---------------------------------------------------------------------------
# pass / fail counting and output
# ---------------------------------------------------------------------------

def test_suite_one_pass_one_fail(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("passer", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_contains": ["DEGRADED_MODE"]}),
        _case("failer", 12, [_fault("dropout", "imu_z", "1s", "2s")],
              {"stdout_contains": ["NO_SUCH_OUTPUT"]}),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 1, result.output
    assert "PASS  passer" in result.output
    assert "FAIL  failer" in result.output
    assert "stdout did not contain 'NO_SUCH_OUTPUT'" in result.output
    assert "1 passed" in result.output
    assert "1 failed" in result.output
    assert "SUITE RESULT: FAILED" in result.output


def test_suite_runs_all_cases_by_default(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("first-fails", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_contains": ["NO_SUCH_OUTPUT"]}),
        _case("second-passes", 12, [_fault("dropout", "imu_z", "1s", "2s")],
              {"exit_code": 0}),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 1, result.output
    assert "FAIL  first-fails" in result.output
    assert "PASS  second-passes" in result.output  # still ran


def test_suite_fail_fast_stops_after_first_failure(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("first-fails", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_contains": ["NO_SUCH_OUTPUT"]}),
        _case("never-runs", 12, [_fault("dropout", "imu_z", "1s", "2s")],
              {"exit_code": 0}),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--fail-fast")
    assert result.exit_code == 1, result.output
    assert "FAIL  first-fails" in result.output
    assert "never-runs" not in result.output
    assert "--fail-fast" in result.output


# ---------------------------------------------------------------------------
# case isolation and determinism
# ---------------------------------------------------------------------------

def test_suite_cases_start_from_original_input(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("a-nulls-gps", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
        _case("b-nulls-imu", 12, [_fault("dropout", "imu_z", "1s", "2s")]),
    ])
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art), "--keep-data")
    assert result.exit_code == 0, result.output
    a_df = read_file(art / "a-nulls-gps" / "corrupted.parquet")
    b_df = read_file(art / "b-nulls-imu" / "corrupted.parquet")
    assert a_df["gps_latitude"].isna().any()
    assert not a_df["imu_z"].isna().any()
    assert b_df["imu_z"].isna().any()
    assert not b_df["gps_latitude"].isna().any()  # case A's fault not chained


def test_suite_deterministic_seeds(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("noisy", 77, [_fault("noise", "pressure", "1s", "5s", std=0.5)]),
    ])
    art1, art2 = tmp_path / "art1", tmp_path / "art2"
    r1 = _run(tmp_path, suite, [sys.executable, target, "{data}"],
              "--artifacts", str(art1))
    r2 = _run(tmp_path, suite, [sys.executable, target, "{data}"],
              "--artifacts", str(art2))
    assert r1.exit_code == 0 and r2.exit_code == 0
    m1 = (art1 / "noisy" / "faults.json").read_bytes()
    m2 = (art2 / "noisy" / "faults.json").read_bytes()
    assert m1 == m2


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------

def test_suite_timeout_is_a_test_failure(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    sleeper = _target(tmp_path, "sleepy.py", TARGET_SLEEP)
    suite = _suite_dict(
        [_case("slow", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        baseline=False,
        timeout=1,
    )
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, sleeper, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 1, result.output
    assert "timed out" in result.output
    payload = json.loads((art / "slow" / "result.json").read_text())
    assert payload["status"] == "fail"
    assert payload["timed_out"] is True
    assert any("timed out" in f for f in payload["failures"])


def test_suite_per_case_timeout_override(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    sleeper = _target(tmp_path, "sleepy.py", TARGET_SLEEP)
    suite = _suite_dict(
        [_case("slow", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
               timeout=1)],
        baseline=False,
        timeout=60,
    )
    result = _run(tmp_path, suite, [sys.executable, sleeper, "{data}"])
    assert result.exit_code == 1, result.output
    assert "timed out" in result.output


def test_suite_invalid_timeout_rejected(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    for bad in (-5, 0, "soon"):
        suite = _suite_dict(
            [_case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
            timeout=bad,
        )
        result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
        assert result.exit_code == 2, (bad, result.output)


# ---------------------------------------------------------------------------
# assertions (old and new)
# ---------------------------------------------------------------------------

def test_suite_stdout_not_contains(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("ok", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_not_contains": ["NOMINAL"]}),
        _case("bad", 12, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_not_contains": ["DEGRADED_MODE"]}),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 1, result.output
    assert "PASS  ok" in result.output
    assert "FAIL  bad" in result.output
    assert "stdout contained 'DEGRADED_MODE' (expected absent)" in result.output


def test_suite_stderr_assertions(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "noisy_err.py", TARGET_STDERR)
    suite = _suite_dict([
        _case("ok", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stderr_contains": ["boom happened"],
               "stderr_not_contains": ["Traceback"]}),
        _case("bad", 12, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stderr_not_contains": ["boom happened"]}),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 1, result.output
    assert "PASS  ok" in result.output
    assert "FAIL  bad" in result.output


def test_suite_max_duration_seconds(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    sleeper = _target(tmp_path, "sleep1.py", "import time\ntime.sleep(1)\n")
    suite = _suite_dict([
        _case("too-slow", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"max_duration_seconds": 0.05}),
    ])
    result = _run(tmp_path, suite, [sys.executable, sleeper, "{data}"])
    assert result.exit_code == 1, result.output
    assert "exceeded max_duration_seconds" in result.output


def test_suite_default_expected_exit_code_is_zero(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    crasher = _target(tmp_path, "crash.py", TARGET_CRASH)
    suite = _suite_dict(
        [_case("crasher", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        baseline=False,
    )  # no expect block at all
    result = _run(tmp_path, suite, [sys.executable, crasher, "{data}"])
    assert result.exit_code == 1, result.output
    assert "FAIL  crasher" in result.output
    assert "expected exit_code 0, got 9" in result.output


def test_suite_explicit_nonzero_expected_exit_code(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    exiter = _target(tmp_path, "exit3.py", TARGET_EXIT3)
    suite = _suite_dict([
        _case("exits-3", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"exit_code": 3}),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, exiter, "{data}"])
    assert result.exit_code == 0, result.output
    assert "PASS  exits-3" in result.output


def test_test_command_supports_new_assertions(tmp_path, telemetry_df):
    data = _setup(tmp_path, telemetry_df)
    scenario = write_yaml(tmp_path, "scenario.yaml", """\
version: 1
seed: 42
input:
  file: data.parquet
  time_column: timestamp
faults:
  - channel: motor_temperature
    type: dropout
    start: 1s
    duration: 2s
expect:
  exit_code: 0
  stdout_not_contains: [BOOM]
  stderr_contains: [warn]
""")
    target = _target(tmp_path, "t.py",
                     "import sys\nprint('ok')\nprint('warn: x', file=sys.stderr)\n")
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert "TEST RESULT: PASS" in result.output


# ---------------------------------------------------------------------------
# artifacts and reports
# ---------------------------------------------------------------------------

def _passing_suite_artifacts(tmp_path, telemetry_df, *extra_cli):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("gps-dropout", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_contains": ["DEGRADED_MODE"]}),
        _case("imu-dropout", 12, [_fault("dropout", "imu_z", "1s", "2s")],
              {"stdout_contains": ["NO_SUCH_OUTPUT"]}),
    ])
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art), *extra_cli)
    assert result.exit_code == 1, result.output
    return art


def test_suite_artifact_directory_layout(tmp_path, telemetry_df):
    art = _passing_suite_artifacts(tmp_path, telemetry_df)
    assert (art / "summary.json").exists()
    assert (art / "summary.md").exists()
    assert (art / "junit.xml").exists()
    for name in ("baseline", "gps-dropout", "imu-dropout"):
        assert (art / name / "stdout.txt").exists()
        assert (art / "baseline" / "stderr.txt").exists()
    for name in ("gps-dropout", "imu-dropout"):
        assert (art / name / "faults.json").exists()
        assert (art / name / "result.json").exists()
        assert (art / name / "stdout.txt").exists()
        assert (art / name / "stderr.txt").exists()


def test_suite_summary_json_structure(tmp_path, telemetry_df):
    art = _passing_suite_artifacts(tmp_path, telemetry_df)
    text = (art / "summary.json").read_text()
    assert "telemetry_resilience_case_" not in text  # no temp paths
    assert "/tmp/" not in text
    payload = json.loads(text)
    assert payload["version"] == 1
    assert payload["suite"] == "demo-suite"
    assert payload["baseline"]["status"] == "pass"
    assert payload["baseline"]["exit_code"] == 0
    assert payload["passed"] == 1
    assert payload["failed"] == 1
    assert payload["total"] == 2
    cases = {c["name"]: c for c in payload["cases"]}
    assert cases["gps-dropout"]["status"] == "pass"
    assert cases["gps-dropout"]["seed"] == 11
    assert cases["gps-dropout"]["fault_count"] == 1
    assert cases["gps-dropout"]["baseline_exit_code"] == 0
    assert "baseline_duration_seconds" in cases["gps-dropout"]
    assert cases["imu-dropout"]["status"] == "fail"
    assert cases["imu-dropout"]["failures"] == [
        "stdout did not contain 'NO_SUCH_OUTPUT'"]


def test_suite_summary_markdown(tmp_path, telemetry_df):
    art = _passing_suite_artifacts(tmp_path, telemetry_df)
    md = (art / "summary.md").read_text()
    assert "# Telemetry Resilience Report" in md
    assert "Baseline: PASS" in md
    assert "| gps-dropout | PASS | 1 |" in md
    assert "| imu-dropout | FAIL | 1 |" in md
    assert "## Failures" in md
    assert "### imu-dropout" in md
    assert "stdout did not contain 'NO_SUCH_OUTPUT'" in md
    assert "SUITE RESULT: FAIL" in md


def test_suite_junit_xml_is_valid(tmp_path, telemetry_df):
    art = _passing_suite_artifacts(tmp_path, telemetry_df)
    tree = ET.parse(str(art / "junit.xml"))
    root = tree.getroot()
    assert root.tag == "testsuite"
    assert root.attrib["name"] == "demo-suite"
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "1"
    cases = {c.attrib["name"]: c for c in root.iter("testcase")}
    assert set(cases) == {"gps-dropout", "imu-dropout"}
    failures = list(cases["imu-dropout"].iter("failure"))
    assert len(failures) == 1
    assert "NO_SUCH_OUTPUT" in failures[0].attrib["message"]
    assert len(list(cases["gps-dropout"].iter("failure"))) == 0


def test_suite_junit_xml_escapes_user_text(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    needle = "<weird>&\"'tag"
    suite = _suite_dict([
        _case("esc", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"stdout_contains": [needle]}),
    ])
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 1, result.output
    raw = (art / "junit.xml").read_text()
    assert "&lt;weird&gt;&amp;" in raw  # properly escaped, not raw markup
    ET.fromstring(raw)  # still well-formed


def test_suite_keep_data_saves_corrupted_telemetry(tmp_path, telemetry_df):
    art = _passing_suite_artifacts(tmp_path, telemetry_df, "--keep-data")
    corrupted = art / "gps-dropout" / "corrupted.parquet"
    assert corrupted.exists()
    df = read_file(corrupted)
    assert df["gps_latitude"].isna().any()


def test_suite_temp_data_cleaned_by_default(tmp_path, telemetry_df):
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    before = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                        "telemetry_resilience_case_*")))
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    after = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                       "telemetry_resilience_case_*")))
    assert after - before == set()


def test_suite_parquet_case_output_round_trip(tmp_path, telemetry_df):
    art = _passing_suite_artifacts(tmp_path, telemetry_df, "--keep-data")
    df = read_file(art / "gps-dropout" / "corrupted.parquet")
    assert str(df["motor_temperature"].dtype) == "float32"
    assert "datetime64" in str(df["timestamp"].dtype)


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------

def test_suite_dry_run_executes_nothing(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    marker = tmp_path / "marker.txt"
    target = _target(tmp_path, "marker.py",
                     f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    art = tmp_path / "artifacts"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art), "--dry-run")
    assert result.exit_code == 0, result.output
    assert not marker.exists()  # target never executed
    assert not art.exists()  # nothing written
    assert "dry run" in result.output.lower()
    assert "seed 11" in result.output


def test_suite_dry_run_catches_invalid_channel(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "no_such_channel", "1s", "2s")]),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--dry-run")
    assert result.exit_code == 2, result.output
    assert "no_such_channel" in result.output


# ---------------------------------------------------------------------------
# original input safety
# ---------------------------------------------------------------------------

def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_suite_original_input_untouched(tmp_path, telemetry_df):
    data = _setup(tmp_path, telemetry_df)
    before = _sha256(data)
    target = _basic_data_and_target(tmp_path, telemetry_df)
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(tmp_path / "art"), "--keep-data")
    assert result.exit_code == 0, result.output
    assert _sha256(data) == before


def test_suite_csv_sidecar_untouched(tmp_path, telemetry_df):
    data = _setup(tmp_path, telemetry_df, name="data.csv")
    sidecar = tmp_path / "data.schema.json"
    assert sidecar.exists()
    before_csv, before_schema = _sha256(data), _sha256(sidecar)
    target = _target(tmp_path, "t.py", "print('ok')\n")
    suite_dict = _suite_dict(
        [_case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        baseline=False,
    )
    suite_dict["input"] = {"file": "data.csv", "time_column": "timestamp"}
    suite_file = _write_suite(tmp_path, suite_dict)
    result = runner.invoke(app, ["suite", str(suite_file), "--",
                                sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert _sha256(data) == before_csv
    assert _sha256(sidecar) == before_schema


# ---------------------------------------------------------------------------
# strict validation
# ---------------------------------------------------------------------------

def _expect_exit_2(tmp_path, telemetry_df, suite_dict):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    result = _run(tmp_path, suite_dict, [sys.executable, target, "{data}"])
    assert result.exit_code == 2, result.output
    return result


def test_suite_duplicate_case_names_rejected(tmp_path, telemetry_df):
    suite = _suite_dict([
        _case("dup", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
        _case("dup", 12, [_fault("dropout", "imu_z", "1s", "2s")]),
    ])
    result = _expect_exit_2(tmp_path, telemetry_df, suite)
    assert "duplicate case name" in result.output


def test_suite_path_traversal_case_names_rejected(tmp_path, telemetry_df):
    for bad in ("../evil", "a/b", "..\\evil", "/abs/path"):
        suite = _suite_dict([
            _case(bad, 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
        ])
        result = _expect_exit_2(tmp_path, telemetry_df, suite)
        assert bad not in os.listdir(tmp_path)


def test_suite_empty_case_name_rejected(tmp_path, telemetry_df):
    suite = _suite_dict([
        _case("", 11, [_fault("dropout", "gps_latitude", "1s", "2s")]),
    ])
    _expect_exit_2(tmp_path, telemetry_df, suite)


def test_suite_missing_seed_rejected(tmp_path, telemetry_df):
    case = {"name": "noseed",
            "faults": [{"type": "dropout", "channel": "gps_latitude",
                        "start": "1s", "duration": "2s"}]}
    suite = _suite_dict([case])
    result = _expect_exit_2(tmp_path, telemetry_df, suite)
    assert "seed" in result.output


def test_suite_empty_faults_rejected(tmp_path, telemetry_df):
    suite = _suite_dict([_case("empty", 11, [])])
    _expect_exit_2(tmp_path, telemetry_df, suite)


def test_suite_unknown_top_level_key_rejected(tmp_path, telemetry_df):
    suite = _suite_dict(
        [_case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        extra_top={"command": ["rm", "-rf", "/"]},  # must never be executed
    )
    result = _expect_exit_2(tmp_path, telemetry_df, suite)
    assert "unknown top-level key" in result.output


def test_suite_unknown_case_key_rejected(tmp_path, telemetry_df):
    case = _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])
    case["bogus"] = 1
    suite = _suite_dict([case])
    _expect_exit_2(tmp_path, telemetry_df, suite)


def test_suite_typo_in_expect_key_rejected(tmp_path, telemetry_df):
    suite = _suite_dict([
        _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")],
              {"expct": {"exit_code": 0}}),
    ])
    # 'expct' is not a valid expect key -> the whole expect mapping is bad
    result = _expect_exit_2(tmp_path, telemetry_df, suite)
    assert "unknown 'expect' key" in result.output


def test_suite_timestamp_jitter_rejects_channel(tmp_path, telemetry_df):
    suite = _suite_dict([
        _case("j", 11, [_fault("timestamp_jitter", "pressure", "1s", "2s",
                               max_jitter_ms=50)]),
    ])
    _expect_exit_2(tmp_path, telemetry_df, suite)


def test_suite_missing_data_placeholder_exits_2(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _write_suite(tmp_path, _suite_dict(
        [_case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])]))
    result = runner.invoke(app, ["suite", str(suite), "--",
                                 sys.executable, target])
    assert result.exit_code == 2


def test_suite_name_defaults_to_file_stem(tmp_path):
    suite = _write_suite(tmp_path, {
        "version": 1,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "cases": [{"name": "c", "seed": 1, "faults": [
            {"type": "dropout", "channel": "x", "start": "1s",
             "duration": "2s"}]}],
    }, name="my_suite.yaml")
    loaded = load_suite(str(suite))
    assert loaded.name == "my_suite"


def test_load_suite_rejects_non_mapping():
    with pytest.raises(ScenarioError):
        load_suite("/nonexistent/suite.yaml")


# ---------------------------------------------------------------------------
# interrupts
# ---------------------------------------------------------------------------

def test_suite_keyboard_interrupt_cleans_up(tmp_path, telemetry_df, monkeypatch):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _write_suite(tmp_path, _suite_dict(
        [_case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])],
        baseline=False,
    ))

    def _boom(cmd, timeout_seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner_module, "run_target", _boom)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                        "telemetry_resilience_case_*")))
    code = run_suite(str(suite), [sys.executable, target, "{data}"])
    after = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                       "telemetry_resilience_case_*")))
    assert code == 3
    assert after - before == set()
