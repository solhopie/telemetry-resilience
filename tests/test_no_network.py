"""Offline guarantee: no CLI command may touch the network.

Every test in this module blocks all socket use (``socket.socket`` and
``socket.create_connection`` raise on any use) and then exercises the CLI
end-to-end on small local fixtures. The CLI is a pure local tool: it reads
telemetry files, injects deterministic faults, and runs *local* target
programs. If any of these paths ever opens a socket, the test fails loudly
rather than hanging on a sandbox firewall.

Commands that do not exist yet (e.g. added by a concurrent change) are
skipped rather than failed.
"""

import socket
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from telemetry_resilience.cli import app

runner = CliRunner()

TINY_CSV = """timestamp,motor_temperature,pressure,rpm
2026-01-01T00:00:00Z,65.0,101.3,1500
2026-01-01T00:00:01Z,65.1,101.2,1501
2026-01-01T00:00:02Z,65.2,101.4,1499
2026-01-01T00:00:03Z,65.1,101.3,1500
2026-01-01T00:00:04Z,65.0,101.2,1502
"""

SCENARIO_YAML = """\
version: 1
seed: 7
input:
  file: tiny.csv
  time_column: timestamp
faults:
  - channel: pressure
    type: noise
    start: 1s
    duration: 2s
    std: 0.5
"""

SUITE_YAML = """\
version: 1
name: offline-suite
input:
  file: tiny.csv
  time_column: timestamp
timeout_seconds: 30
baseline:
  enabled: false
cases:
  - name: pressure-noise
    seed: 11
    faults:
      - channel: pressure
        type: noise
        start: 1s
        duration: 2s
        std: 0.5
    expect:
      exit_code: 0
"""

CAMPAIGN_YAML = """\
version: 1
name: offline-campaign
seed: 5000
input:
  file: tiny.csv
  time_column: timestamp
timeout_seconds: 30
window:
  start: 1s
  duration: 2s
baseline:
  enabled: false
expect:
  exit_code: 0
channels:
  pressure:
    noise:
      std: [0.5]
"""


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch):
    """Fail the test on any socket creation or connection attempt."""
    used = []

    class BlockedSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            used.append(("socket.socket", args, kwargs))
            raise OSError("network access is blocked in offline tests")

    def blocked_create_connection(*args, **kwargs):
        used.append(("socket.create_connection", args, kwargs))
        raise OSError("network access is blocked in offline tests")

    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)
    yield used
    assert not used, f"socket usage detected during offline test: {used}"


@pytest.fixture
def offline_project(tmp_path, monkeypatch):
    """A tiny self-contained project directory; cwd is switched into it."""
    (tmp_path / "tiny.csv").write_text(TINY_CSV)
    (tmp_path / "scenario.yaml").write_text(SCENARIO_YAML)
    (tmp_path / "suite.yaml").write_text(SUITE_YAML)
    (tmp_path / "campaign.yaml").write_text(CAMPAIGN_YAML)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _target_args():
    # Trivial local target; {data} is replaced with the corrupted dataset path
    # but the command itself only prints a fixed string.
    return [sys.executable, "-c", "print('target ran')", "{data}"]


def _require_command(*argv):
    """Skip the test if the CLI command under test does not exist yet."""
    result = runner.invoke(app, list(argv) + ["--help"])
    if result.exit_code != 0 or "No such command" in result.output:
        pytest.skip(f"command not available yet: {' '.join(argv)}")


def test_offline_inspect(offline_project):
    result = runner.invoke(app, ["inspect", "tiny.csv"])
    assert result.exit_code == 0, result.output
    assert "rows: 5" in result.output


def test_offline_inject(offline_project):
    result = runner.invoke(
        app, ["inject", "tiny.csv", "--scenario", "scenario.yaml"]
    )
    assert result.exit_code == 0, result.output
    assert (offline_project / "tiny.corrupted.csv").exists()
    assert (offline_project / "tiny.faults.json").exists()


def test_offline_test_command(offline_project):
    result = runner.invoke(app, ["test", "scenario.yaml", "--"] + _target_args())
    assert result.exit_code == 0, result.output
    assert "TEST RESULT: PASS" in result.output


def test_offline_suite_dry_run(offline_project):
    result = runner.invoke(
        app, ["suite", "suite.yaml", "--dry-run", "--"] + _target_args()
    )
    assert result.exit_code == 0, result.output


def test_offline_suite_full_run(offline_project):
    result = runner.invoke(
        app,
        ["suite", "suite.yaml", "--artifacts", "suite-artifacts", "--"]
        + _target_args(),
    )
    assert result.exit_code == 0, result.output
    assert (offline_project / "suite-artifacts" / "summary.json").exists()


def test_offline_campaign_plan(offline_project):
    result = runner.invoke(app, ["campaign", "campaign.yaml", "--plan"])
    assert result.exit_code == 0, result.output
    assert "Generated cases: 1" in result.output


def test_offline_campaign_list_cases(offline_project):
    result = runner.invoke(app, ["campaign", "campaign.yaml", "--list-cases"])
    assert result.exit_code == 0, result.output
    assert "pressure__noise__std-0.5" in result.output


def test_offline_campaign_full_run(offline_project):
    result = runner.invoke(
        app,
        ["campaign", "campaign.yaml", "--artifacts", "campaign-artifacts", "--"]
        + _target_args(),
    )
    assert result.exit_code == 0, result.output
    assert (offline_project / "campaign-artifacts" / "coverage.json").exists()
    assert (offline_project / "campaign-artifacts" / "summary.json").exists()
    assert (offline_project / "campaign-artifacts" / "junit.xml").exists()


def test_offline_doctor(offline_project):
    _require_command("doctor")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "READY" in result.output


def test_offline_demo_generate(tmp_path, monkeypatch):
    _require_command("demo", "generate")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["demo", "generate", "tiny.parquet", "--seconds", "2", "--hz", "2"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tiny.parquet").exists()
    # The generated file must itself be inspectable offline.
    result = runner.invoke(app, ["inspect", "tiny.parquet"])
    assert result.exit_code == 0, result.output
