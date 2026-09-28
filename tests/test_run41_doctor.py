"""Run 4.1: doctor degrades gracefully when optional dependencies are missing.

pyarrow (and friends) are simulated as missing/broken via sys.modules
manipulation and import hooks -- pyarrow is NEVER uninstalled from the
real environment.

Covers:
  * ``--version`` / ``--help`` work with pyarrow unimportable
  * ``doctor`` reports pyarrow MISSING (no traceback), STATUS: NOT READY,
    exit code 2
  * broken (not merely missing) dependencies are reported as BROKEN
  * Parquet read/write without PyArrow raises the clean
    ``Parquet support requires PyArrow.`` ValueError, mapped to CLI exit 2
    with no traceback, on every CLI path that can reach parquet IO
  * CSV-only and JSONL-only workflows are unaffected by a missing pyarrow
"""
import importlib.abc
import importlib.machinery
import sys

import pytest
from typer.testing import CliRunner

from telemetry_resilience import __version__
from telemetry_resilience.cli import app

runner = CliRunner()

SCENARIO_CSV = """\
version: 1
seed: 42
input:
  file: data.csv
  time_column: timestamp
faults:
  - channel: motor_temperature
    type: dropout
    start: 1s
    duration: 2s
"""


def _block_pyarrow(monkeypatch):
    """Make ``import pyarrow`` raise ImportError for the rest of the test."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)


class _FailingLoader(importlib.abc.Loader):
    def __init__(self, exc):
        self._exc = exc

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise self._exc


class _FailingFinder(importlib.abc.MetaPathFinder):
    """Import hook that makes importing ``name`` raise ``exc``."""

    def __init__(self, name, exc):
        self._name = name
        self._exc = exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._name or fullname.startswith(self._name + "."):
            return importlib.machinery.ModuleSpec(fullname, _FailingLoader(self._exc))
        return None


@pytest.fixture
def broken_numpy(monkeypatch):
    """Simulate a broken numpy install: importing it raises RuntimeError."""
    for mod in [m for m in sys.modules if m == "numpy" or m.startswith("numpy.")]:
        monkeypatch.delitem(sys.modules, mod)
    finder = _FailingFinder("numpy", RuntimeError("simulated broken install"))
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    importlib.invalidate_caches()
    return finder


def _assert_no_traceback(result):
    assert "Traceback" not in result.output
    assert "Traceback" not in (result.stderr or "")


# ---------------------------------------------------------------------------
# --version / --help with pyarrow unimportable
# ---------------------------------------------------------------------------


def test_version_works_with_pyarrow_blocked(monkeypatch):
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"Telemetry Resilience CLI {__version__}\n"


def test_help_works_with_pyarrow_blocked(monkeypatch):
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_doctor_help_works_with_pyarrow_blocked(monkeypatch):
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# doctor with pyarrow missing
# ---------------------------------------------------------------------------


def test_doctor_reports_pyarrow_missing_not_a_traceback(monkeypatch):
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 2
    out = result.stdout
    assert "pyarrow: MISSING" in out
    # The other dependencies are still reported with their versions.
    for component in ("pandas:", "numpy:", "pyyaml:", "typer:"):
        assert component in out
        assert f"{component} MISSING" not in out
    assert "check: csv round-trip: PASS" in out
    assert "check: jsonl round-trip: PASS" in out
    assert "check: parquet round-trip: FAIL" in out
    assert "pyarrow is not installed" in out
    assert "STATUS: NOT READY" in out
    assert "failed: dependency pyarrow" in out
    assert "failed: parquet round-trip" in out
    _assert_no_traceback(result)


def test_doctor_ready_when_everything_present():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "pyarrow:" in result.stdout
    assert " OK" in result.stdout
    assert "STATUS: READY" in result.stdout


# ---------------------------------------------------------------------------
# broken (not merely missing) dependencies
# ---------------------------------------------------------------------------


def test_lazy_import_classifies_missing():
    from telemetry_resilience import doctor

    module, state, detail = doctor._lazy_import("definitely_not_a_real_module_xyz")
    assert module is None
    assert state == "missing"
    assert "not installed" in detail


def test_lazy_import_classifies_broken(monkeypatch):
    from telemetry_resilience import doctor

    finder = _FailingFinder(
        "definitely_not_a_real_module_xyz", RuntimeError("boom")
    )
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    importlib.invalidate_caches()
    module, state, detail = doctor._lazy_import("definitely_not_a_real_module_xyz")
    assert module is None
    assert state == "broken"
    assert "boom" in detail


def test_doctor_reports_broken_dependency(broken_numpy):
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 2
    out = result.stdout
    assert "numpy: BROKEN" in out
    assert "STATUS: NOT READY" in out
    assert "failed: dependency numpy" in out
    _assert_no_traceback(result)


# ---------------------------------------------------------------------------
# Parquet without PyArrow: clean ValueError -> exit 2, no traceback
# ---------------------------------------------------------------------------


def test_parquet_adapter_raises_clean_valueerror(monkeypatch, tmp_path):
    _block_pyarrow(monkeypatch)
    from telemetry_resilience.io import parquet_adapter
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match=r"^Parquet support requires PyArrow\.$"):
        parquet_adapter.write(df, tmp_path / "out.parquet")
    with pytest.raises(ValueError, match=r"^Parquet support requires PyArrow\.$"):
        parquet_adapter.read(tmp_path / "out.parquet")


def test_demo_generate_without_pyarrow_is_clean_exit_2(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["demo", "generate"])
    assert result.exit_code == 2
    assert "Parquet support requires PyArrow." in result.stderr
    _assert_no_traceback(result)


def test_demo_init_without_pyarrow_is_clean_exit_2(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["demo", "init"])
    assert result.exit_code == 2
    assert "Parquet support requires PyArrow." in result.stderr
    _assert_no_traceback(result)


def test_inspect_parquet_without_pyarrow_is_clean_exit_2(monkeypatch, tmp_path):
    fake = tmp_path / "data.parquet"
    fake.write_bytes(b"not a real parquet file")
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["inspect", str(fake)])
    assert result.exit_code == 2
    assert "Parquet support requires PyArrow." in result.stderr
    _assert_no_traceback(result)


def test_inject_parquet_without_pyarrow_is_clean_exit_2(monkeypatch, tmp_path):
    fake = tmp_path / "data.parquet"
    fake.write_bytes(b"not a real parquet file")
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(SCENARIO_CSV.replace("data.csv", "data.parquet"))
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["inject", str(fake), "--scenario", str(scenario)])
    assert result.exit_code == 2
    assert "Parquet support requires PyArrow." in result.stderr
    _assert_no_traceback(result)


# ---------------------------------------------------------------------------
# CSV-only and JSONL-only workflows are unaffected
# ---------------------------------------------------------------------------


def test_csv_inspect_and_inject_work_without_pyarrow(
    monkeypatch, tmp_path, telemetry_df
):
    from telemetry_resilience.io import write_file

    monkeypatch.chdir(tmp_path)
    _block_pyarrow(monkeypatch)

    data = tmp_path / "data.csv"
    write_file(telemetry_df, data)
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(SCENARIO_CSV)

    inspected = runner.invoke(app, ["inspect", str(data)])
    assert inspected.exit_code == 0, inspected.output
    assert "rows: 600" in inspected.stdout

    injected = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert injected.exit_code == 0, injected.output
    assert (tmp_path / "data.corrupted.csv").is_file()
    assert (tmp_path / "data.faults.json").is_file()
    assert (tmp_path / "data.report.json").is_file()


def test_jsonl_roundtrip_works_without_pyarrow(monkeypatch, tmp_path, telemetry_df):
    from telemetry_resilience.io import read_file, write_file

    _block_pyarrow(monkeypatch)
    path = tmp_path / "data.jsonl"
    write_file(telemetry_df, path)
    back = read_file(path)
    assert len(back) == len(telemetry_df)
    assert list(back.columns) == list(telemetry_df.columns)


def test_doctor_csv_jsonl_checks_pass_without_pyarrow(monkeypatch):
    _block_pyarrow(monkeypatch)
    result = runner.invoke(app, ["doctor"])
    assert "check: csv round-trip: PASS" in result.stdout
    assert "check: jsonl round-trip: PASS" in result.stdout
