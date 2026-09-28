"""RUN 3.1: effective max_cases reporting (Fix 4).

Verifies that the effective max-cases limit -- priority CLI --max-cases >
campaign YAML max_cases > default 100 -- is what campaign reports show:

* ``campaign --plan`` prints the effective limit;
* ``run_campaign`` prints it in the console header;
* ``coverage.json`` ``"max_cases"`` and ``summary.json``
  ``summary["campaign"]["max_cases"]`` both carry the effective value.
"""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from telemetry_resilience.campaign import DEFAULT_MAX_CASES
from telemetry_resilience.cli import app
from telemetry_resilience.io import write_file

from .conftest import write_yaml

runner = CliRunner()

_TARGET_OK = "import sys\nprint('ok')\n"


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
        }
    )


def _write_input(tmp_path, name="data.parquet"):
    path = tmp_path / name
    write_file(_tiny_df(), path)
    return path


def _doc(max_cases_yaml=None):
    """Two-case campaign doc (pressure/dropout x two durations)."""
    doc = {
        "version": 1,
        "name": "maxcases-demo",
        "seed": 42,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "expect": {"exit_code": 0},
        "channels": {"pressure": {"dropout": {"durations": ["2s", "3s"]}}},
    }
    if max_cases_yaml is not None:
        doc["max_cases"] = max_cases_yaml
    return doc


def _write_campaign(tmp_path, doc, name="campaign.yaml"):
    _write_input(tmp_path)
    return write_yaml(tmp_path, name, yaml.safe_dump(doc))


def _ok_target(tmp_path, name="ok_target.py"):
    path = tmp_path / name
    path.write_text(_TARGET_OK)
    return path


def _run(tmp_path, doc, *extra_cli):
    camp = _write_campaign(tmp_path, doc)
    script = _ok_target(tmp_path)
    return runner.invoke(
        app,
        ["campaign", str(camp), *extra_cli, "--",
         sys.executable, str(script), "{data}"],
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# effective max_cases in artifacts
# ---------------------------------------------------------------------------

def test_default_max_cases_reported_in_artifacts(tmp_path):
    """No max_cases anywhere -> reports show the default (100)."""
    art = tmp_path / "artifacts"
    result = _run(tmp_path, _doc(), "--artifacts", str(art))
    assert result.exit_code == 0, result.output
    assert f"Max cases (effective): {DEFAULT_MAX_CASES}" in result.output

    summary = _read_json(art / "summary.json")
    assert summary["campaign"]["max_cases"] == DEFAULT_MAX_CASES
    coverage = _read_json(art / "coverage.json")
    assert coverage["max_cases"] == DEFAULT_MAX_CASES


def test_yaml_max_cases_reported_in_artifacts(tmp_path):
    """Campaign YAML max_cases: 7 -> reports show 7."""
    art = tmp_path / "artifacts"
    result = _run(tmp_path, _doc(max_cases_yaml=7),
                  "--artifacts", str(art))
    assert result.exit_code == 0, result.output
    assert "Max cases (effective): 7" in result.output

    summary = _read_json(art / "summary.json")
    assert summary["campaign"]["max_cases"] == 7
    coverage = _read_json(art / "coverage.json")
    assert coverage["max_cases"] == 7


def test_cli_max_cases_overrides_yaml_in_artifacts(tmp_path):
    """CLI --max-cases 5 beats YAML max_cases: 1; 2-case campaign runs."""
    art = tmp_path / "artifacts"
    result = _run(tmp_path, _doc(max_cases_yaml=1), "--artifacts", str(art),
                  "--max-cases", "5")
    assert result.exit_code == 0, result.output
    assert "Max cases (effective): 5" in result.output

    summary = _read_json(art / "summary.json")
    assert summary["campaign"]["max_cases"] == 5
    coverage = _read_json(art / "coverage.json")
    assert coverage["max_cases"] == 5
    assert coverage["configured_cases"] == 2


# ---------------------------------------------------------------------------
# --plan output
# ---------------------------------------------------------------------------

def _plan(tmp_path, doc, *extra_cli):
    camp = _write_campaign(tmp_path, doc)
    return runner.invoke(
        app, ["campaign", str(camp), "--plan", *extra_cli, "--",
              sys.executable, "-c", "print('unused')", "{data}"]
    )


def test_plan_shows_effective_max_default(tmp_path):
    result = _plan(tmp_path, _doc())
    assert result.exit_code == 0, result.output
    assert f"Effective max cases: {DEFAULT_MAX_CASES}" in result.output


def test_plan_shows_effective_max_yaml_and_override(tmp_path):
    result = _plan(tmp_path, _doc(max_cases_yaml=7))
    assert result.exit_code == 0, result.output
    assert "Effective max cases: 7" in result.output

    result = _plan(tmp_path, _doc(max_cases_yaml=7),
                   "--max-cases", "5")
    assert result.exit_code == 0, result.output
    assert "Effective max cases: 5" in result.output


# ---------------------------------------------------------------------------
# over-limit behavior unchanged: error before execution
# ---------------------------------------------------------------------------

def test_over_limit_still_errors_before_execution(tmp_path):
    """Two-case campaign with max_cases: 1 errors with zero execution."""
    marker = tmp_path / "marker.txt"
    script = tmp_path / "marker_target.py"
    script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text('ran', encoding='utf-8')\n"
    )
    art = tmp_path / "artifacts"
    camp = _write_campaign(tmp_path, _doc(max_cases_yaml=1))
    result = runner.invoke(
        app,
        ["campaign", str(camp), "--artifacts", str(art), "--",
         sys.executable, str(script), str(marker), "{data}"],
    )
    assert result.exit_code == 2, result.output
    assert "Configured maximum" in result.output
    assert not marker.exists()  # zero targets executed
    assert not art.exists()  # no artifacts written
