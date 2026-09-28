"""CLI tests via typer.testing.CliRunner (no subprocess of the CLI itself)."""
import json
import sys

from typer.testing import CliRunner

from telemetry_resilience.cli import app
from telemetry_resilience.io import write_file

from .conftest import write_yaml

runner = CliRunner()

SCENARIO = """\
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
  stdout_contains: [DEGRADED_MODE]
  stderr_not_contains: [Traceback]
"""

BAD_SCENARIO = """\
version: 1
seed: 42
input:
  file: data.parquet
  time_column: timestamp
faults:
  - channel: motor_temperature
    type: teleport
    start: 1s
    duration: 2s
"""


def _data_and_scenario(tmp_path, telemetry_df, scenario_text=SCENARIO):
    data = tmp_path / "data.parquet"
    write_file(telemetry_df, data)
    scenario = write_yaml(tmp_path, "scenario.yaml", scenario_text)
    return data, scenario


def _target(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_inspect_reports_rows_and_columns(tmp_path, telemetry_df):
    data, _ = _data_and_scenario(tmp_path, telemetry_df)
    result = runner.invoke(app, ["inspect", str(data)])
    assert result.exit_code == 0
    assert "rows:" in result.output
    assert "columns:" in result.output


def test_inject_creates_outputs(tmp_path, telemetry_df):
    data, scenario = _data_and_scenario(tmp_path, telemetry_df)
    result = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert result.exit_code == 0
    assert (tmp_path / "data.corrupted.parquet").exists()
    assert (tmp_path / "data.faults.json").exists()
    assert (tmp_path / "data.report.json").exists()
    manifest = json.loads((tmp_path / "data.faults.json").read_text())
    assert manifest["seed"] == 42
    assert len(manifest["faults"]) == 1


def test_inject_bad_scenario_exits_2(tmp_path, telemetry_df):
    data, scenario = _data_and_scenario(tmp_path, telemetry_df, BAD_SCENARIO)
    result = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert result.exit_code == 2


def test_inject_nonexistent_input_exits_2(tmp_path, telemetry_df):
    _, scenario = _data_and_scenario(tmp_path, telemetry_df)
    result = runner.invoke(
        app, ["inject", str(tmp_path / "nope.parquet"), "--scenario", str(scenario)]
    )
    assert result.exit_code == 2


def test_inject_refuses_overwrite_without_flag(tmp_path, telemetry_df):
    data, scenario = _data_and_scenario(tmp_path, telemetry_df)
    first = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert first.exit_code == 0
    second = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert second.exit_code == 2
    # --overwrite allows the second run
    third = runner.invoke(
        app, ["inject", str(data), "--scenario", str(scenario), "--overwrite"]
    )
    assert third.exit_code == 0


def test_test_command_pass(tmp_path, telemetry_df):
    _, scenario = _data_and_scenario(tmp_path, telemetry_df)
    target = _target(
        tmp_path,
        "target.py",
        "import sys\n"
        "print('DEGRADED_MODE: motor_temperature dropout detected')\n",
    )
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target), "{data}"]
    )
    assert result.exit_code == 0
    assert "TEST RESULT: PASS" in result.output


def test_test_command_fail_on_exit_code(tmp_path, telemetry_df):
    _, scenario = _data_and_scenario(tmp_path, telemetry_df)
    target = _target(tmp_path, "target.py", "import sys\nsys.exit(3)\n")
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target), "{data}"]
    )
    assert result.exit_code == 1
    assert "TEST RESULT: FAIL" in result.output


def test_test_command_missing_placeholder_exits_2(tmp_path, telemetry_df):
    _, scenario = _data_and_scenario(tmp_path, telemetry_df)
    target = _target(tmp_path, "target.py", "print('hi')\n")
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target)]
    )
    assert result.exit_code == 2


def test_test_command_stderr_violation_fails(tmp_path, telemetry_df):
    _, scenario = _data_and_scenario(tmp_path, telemetry_df)
    target = _target(
        tmp_path,
        "target.py",
        "import sys\n"
        "print('Traceback: simulated boom', file=sys.stderr)\n"
        "print('DEGRADED_MODE ok')\n",
    )
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target), "{data}"]
    )
    assert result.exit_code == 1
    assert "TEST RESULT: FAIL" in result.output
