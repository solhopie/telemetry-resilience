"""Run 4.1: bundled degraded_navigation.yaml template + demo generate overwrite safety.

Covers:
- ``degraded_navigation.yaml`` bundled under ``templates/``, listed in
  ``TEMPLATE_FILES``/``list_templates()``, and copied by ``demo init``;
- the copied scenario parses as a valid scenario and injects cleanly
  against the init-generated demo data;
- ``demo generate`` refuses to overwrite an existing file (exit 2) unless
  ``--overwrite`` is given;
- ``demo init`` keeps refusing to overwrite planned files (now including
  the new template).
"""
import importlib.resources as importlib_resources
from pathlib import Path

import yaml
from typer.testing import CliRunner

from telemetry_resilience.cli import app
from telemetry_resilience.demo import TEMPLATE_FILES, list_templates
from telemetry_resilience.scenario import load_scenario

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_NAME = "degraded_navigation.yaml"


def test_degraded_navigation_yaml_in_template_files():
    assert SCENARIO_NAME in TEMPLATE_FILES
    assert SCENARIO_NAME in list_templates()


def test_bundled_scenario_matches_example_source():
    """The template is a faithful copy of examples/degraded_navigation.yaml."""
    bundled = (
        importlib_resources.files("telemetry_resilience")
        / "templates"
        / SCENARIO_NAME
    ).read_bytes()
    example = (REPO_ROOT / "examples" / SCENARIO_NAME).read_bytes()
    assert bundled == example


def test_bundled_scenario_keeps_relative_input_file():
    raw = yaml.safe_load(
        (REPO_ROOT / "src" / "telemetry_resilience" / "templates" / SCENARIO_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert raw["input"]["file"] == "demo_drive.parquet"


def test_demo_init_copies_scenario_and_prints_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["demo", "init"])
    assert result.exit_code == 0, result.output
    project = tmp_path / "telemetry-demo"
    copied = project / SCENARIO_NAME
    assert copied.is_file()
    assert SCENARIO_NAME in result.stdout


def test_copied_scenario_parses_and_injects_cleanly(tmp_path, monkeypatch):
    """The init-copied scenario is valid and applies against init's data."""
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["demo", "init"]).exit_code == 0
    project = tmp_path / "telemetry-demo"
    copied = project / SCENARIO_NAME

    scenario = load_scenario(copied)
    assert scenario.version == 1
    assert scenario.input_file == "demo_drive.parquet"
    assert len(scenario.faults) == 5

    monkeypatch.chdir(project)
    injected = runner.invoke(
        app, ["inject", "demo_drive.parquet", "--scenario", SCENARIO_NAME]
    )
    assert injected.exit_code == 0, injected.output
    assert (project / "demo_drive.corrupted.parquet").is_file()
    assert (project / "demo_drive.faults.json").is_file()
    assert (project / "demo_drive.report.json").is_file()


def test_demo_generate_refuses_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["demo", "generate", "demo_drive.parquet"]).exit_code == 0
    before = (tmp_path / "demo_drive.parquet").read_bytes()
    refused = runner.invoke(app, ["demo", "generate", "demo_drive.parquet"])
    assert refused.exit_code == 2
    assert "refusing to overwrite" in refused.stderr
    # The existing file is untouched.
    assert (tmp_path / "demo_drive.parquet").read_bytes() == before


def test_demo_generate_overwrite_flag_replaces_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        runner.invoke(app, ["demo", "generate", "demo_drive.parquet", "--seconds", "10"]).exit_code
        == 0
    )
    small = (tmp_path / "demo_drive.parquet").read_bytes()
    result = runner.invoke(
        app, ["demo", "generate", "demo_drive.parquet", "--overwrite"]
    )
    assert result.exit_code == 0, result.output
    after = (tmp_path / "demo_drive.parquet").read_bytes()
    assert after != small  # replaced with the default 300 s frame
    # And a second plain generate still refuses on the replaced file.
    refused = runner.invoke(app, ["demo", "generate", "demo_drive.parquet"])
    assert refused.exit_code == 2


def test_demo_init_still_refuses_existing_planned_files(tmp_path, monkeypatch):
    """Init's no-overwrite behavior covers the new template file too."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "telemetry-demo"
    target.mkdir()
    (target / SCENARIO_NAME).write_text("placeholder", encoding="utf-8")
    result = runner.invoke(app, ["demo", "init"])
    assert result.exit_code == 2
    assert "refusing to overwrite" in result.stderr
    # Only the new template blocked init; nothing else was written.
    assert (target / SCENARIO_NAME).read_text(encoding="utf-8") == "placeholder"
    assert not (target / "demo_drive.parquet").exists()


def test_package_data_glob_covers_templates():
    """Wheel mechanism: pyproject package-data must glob the templates dir."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "templates/*" in text


def test_manifest_in_covers_templates():
    """Sdist mechanism: MANIFEST.in must recursive-include the templates dir."""
    text = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include src/telemetry_resilience/templates" in text
