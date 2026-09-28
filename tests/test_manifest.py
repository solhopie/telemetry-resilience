"""Manifest/report tests: affected counts reflect reality, params normalized."""
from telemetry_resilience.engine import apply_scenario
from telemetry_resilience.manifest import build_manifest, build_report

from .conftest import make_scenario


def test_manifest_records_actual_window_rows(telemetry_df):
    df = telemetry_df
    # window 55s..85s extends past the 60 s of data: only rows 550..599 exist
    scenario = make_scenario(
        [
            {
                "channel": "motor_temperature",
                "type": "dropout",
                "start": 55,
                "duration": 30,
            },
            {
                "channel": "pressure",
                "type": "spike",
                "start": 55,
                "duration": 30,
                "params": {"magnitude": 5.0},  # direction omitted -> default
            },
        ],
        seed=99,
    )
    _, entries = apply_scenario(df, scenario)
    manifest = build_manifest(scenario, "data.parquet", entries)

    assert manifest["version"] == 1
    assert manifest["seed"] == 99
    assert manifest["source_file"] == "data.parquet"
    assert "software_version" in manifest
    # actual rows in the window (50), not the 300 the window requested
    assert entries[0]["affected_observations"] == 50
    assert manifest["faults"][0]["affected_observations"] == 50
    # parameters recorded normalized (spike direction default -> None)
    assert manifest["faults"][1]["parameters"] == {
        "magnitude": 5.0,
        "direction": None,
        "count": 1,
    }


def test_report_summarizes_run(telemetry_df):
    df = telemetry_df
    scenario = make_scenario(
        [
            {
                "channel": "motor_temperature",
                "type": "dropout",
                "start": 55,
                "duration": 30,
            }
        ],
        seed=99,
    )
    _, entries = apply_scenario(df, scenario)
    report = build_report(scenario, "data.parquet", "data.corrupted.parquet", entries)

    assert report["faults_applied"] == 1
    assert report["total_affected_observations"] == 50
    assert report["seed"] == 99
    assert report["output_file"] == "data.corrupted.parquet"
