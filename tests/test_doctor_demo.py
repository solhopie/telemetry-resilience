"""Tests for the `doctor` command, `--version`, and the `demo` command group.

All file-system side effects stay inside pytest tmp dirs; `doctor` and
`demo` never execute user programs and make no network requests.
"""
import importlib.resources as importlib_resources

import pytest
from typer.testing import CliRunner

from telemetry_resilience import __version__
from telemetry_resilience.cli import app
from telemetry_resilience.demo import TEMPLATE_FILES

runner = CliRunner()


def test_version_flag_prints_exact_version_string():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"Telemetry Resilience CLI {__version__}\n"


def test_doctor_exits_zero_with_status_ready():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    out = result.stdout
    assert f"Telemetry Resilience CLI {__version__}" in out
    assert "python:" in out
    assert "os:" in out
    for component in ("pandas:", "numpy:", "pyarrow:", "pyyaml:", "typer:"):
        assert component in out
    for check in (
        "csv round-trip",
        "jsonl round-trip",
        "parquet round-trip",
        "temp directory writable",
        "subprocess execution",
    ):
        assert f"check: {check}: PASS" in out
    assert "STATUS: READY" in out


def test_demo_generate_creates_parquet_inspect_can_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["demo", "generate"])
    assert result.exit_code == 0
    out = tmp_path / "demo_drive.parquet"
    assert out.is_file()
    lowered = result.stdout.lower()
    assert "synthetic" in lowered
    assert "non-authoritative" in lowered
    assert "physically accurate" not in lowered

    inspected = runner.invoke(app, ["inspect", str(out)])
    assert inspected.exit_code == 0
    assert "rows: 3000" in inspected.stdout
    assert "gps_latitude" in inspected.stdout


def test_demo_generate_custom_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["demo", "generate", "nested/out.parquet"])
    assert result.exit_code == 0
    assert (tmp_path / "nested" / "out.parquet").is_file()


def test_demo_generate_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["demo", "generate", "a.parquet"]).exit_code == 0
    assert runner.invoke(app, ["demo", "generate", "b.parquet"]).exit_code == 0
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_demo_init_creates_all_four_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["demo", "init"])
    assert result.exit_code == 0
    project = tmp_path / "telemetry-demo"
    assert project.is_dir()
    for name in ("demo_drive.parquet", *TEMPLATE_FILES):
        assert (project / name).is_file(), f"missing {name}"
    # The exact run command is printed at the end of `demo init`.
    assert (
        'telemetry-resilience campaign navigation_campaign.yaml '
        '--artifacts resilience-results -- python demo_app.py "{data}"'
    ) in result.stdout


def test_demo_init_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["demo", "init"]).exit_code == 0
    second = runner.invoke(app, ["demo", "init"])
    assert second.exit_code == 2
    assert "refusing to overwrite" in second.stderr


def test_demo_init_campaign_passes_plan(tmp_path, monkeypatch):
    """The generated demo's campaign validates with --plan (no execution)."""
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["demo", "init"]).exit_code == 0
    campaign = tmp_path / "telemetry-demo" / "navigation_campaign.yaml"
    result = runner.invoke(app, ["campaign", str(campaign), "--plan"])
    assert result.exit_code == 0, result.output
    assert "No targets executed" in result.stdout


def test_templates_resolvable_via_importlib_resources():
    templates = importlib_resources.files("telemetry_resilience") / "templates"
    for name in TEMPLATE_FILES:
        resource = templates / name
        assert resource.is_file(), f"template missing: {name}"
        assert resource.read_bytes(), f"template empty: {name}"
    readme = (templates / "README.md").read_text(encoding="utf-8")
    assert (
        'telemetry-resilience campaign navigation_campaign.yaml '
        '--artifacts resilience-results -- python demo_app.py "{data}"'
    ) in readme


def test_demo_help_mentions_no_user_program_execution():
    def flat(output: str) -> str:
        return " ".join(output.split())

    result = runner.invoke(app, ["demo", "--help"])
    assert result.exit_code == 0
    assert "Never executes user programs" in flat(result.stdout)
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "no user programs are executed" in flat(result.stdout)
