"""Scenario YAML parsing/validation tests (load_scenario -> ScenarioError)."""
import pandas as pd
import pytest

from telemetry_resilience.engine import apply_scenario
from telemetry_resilience.models import FaultError, ScenarioError
from telemetry_resilience.scenario import load_scenario

from .conftest import make_scenario, write_yaml

BASE = """\
version: 1
seed: 42
input:
  file: data.parquet
  time_column: timestamp
faults:
  - channel: motor_temperature
    type: dropout
    start: 10s
    duration: 5s
"""


def _scenario_with_fault(tmp_path, fault_yaml):
    return write_yaml(
        tmp_path,
        "scenario.yaml",
        "version: 1\n"
        "seed: 42\n"
        "input:\n"
        "  file: data.parquet\n"
        "  time_column: timestamp\n"
        "faults:\n"
        f"{fault_yaml}\n",
    )


def test_valid_scenario_loads(tmp_path):
    sc = load_scenario(write_yaml(tmp_path, "scenario.yaml", BASE))
    assert sc.version == 1
    assert sc.seed == 42
    assert sc.input_file == "data.parquet"
    assert sc.time_column == "timestamp"
    assert len(sc.faults) == 1
    assert sc.faults[0].type == "dropout"
    assert sc.faults[0].start_seconds == 10.0
    assert sc.faults[0].duration_seconds == 5.0


def test_unknown_fault_type(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: motor_temperature\n"
        "    type: teleport\n"
        "    start: 10s\n"
        "    duration: 5s\n",
    )
    with pytest.raises(ScenarioError, match="unknown fault type"):
        load_scenario(path)


def test_negative_duration(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: motor_temperature\n"
        "    type: dropout\n"
        "    start: 10s\n"
        "    duration: -5s\n",
    )
    with pytest.raises(ScenarioError, match="duration"):
        load_scenario(path)


def test_malformed_start(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: motor_temperature\n"
        "    type: dropout\n"
        "    start: soon\n"
        "    duration: 5s\n",
    )
    with pytest.raises(ScenarioError, match="start"):
        load_scenario(path)


def test_sample_rate_impossible_rate(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: imu_z\n"
        "    type: sample_rate\n"
        "    start: 10s\n"
        "    duration: 5s\n"
        "    from: 2Hz\n"
        "    to: 10Hz\n",
    )
    with pytest.raises(ScenarioError, match="must be less than"):
        load_scenario(path)


def test_spike_magnitude_not_numeric(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: pressure\n"
        "    type: spike\n"
        "    start: 10s\n"
        "    duration: 5s\n"
        "    magnitude: huge\n",
    )
    with pytest.raises(ScenarioError, match="magnitude must be numeric"):
        load_scenario(path)


def test_clipping_min_greater_than_max(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: pressure\n"
        "    type: clipping\n"
        "    start: 10s\n"
        "    duration: 5s\n"
        "    min: 5\n"
        "    max: 1\n",
    )
    with pytest.raises(ScenarioError, match="min must be <="):
        load_scenario(path)


def test_timestamp_gap_with_channel_rejected(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: gps_latitude\n"
        "    type: timestamp_gap\n"
        "    start: 10s\n"
        "    duration: 5s\n",
    )
    with pytest.raises(ScenarioError, match="'channel' must be absent"):
        load_scenario(path)


def test_bad_cast_value(tmp_path):
    path = _scenario_with_fault(
        tmp_path,
        "  - channel: motor_temperature\n"
        "    type: dropout\n"
        "    start: 10s\n"
        "    duration: 5s\n"
        "    cast: float16\n",
    )
    with pytest.raises(ScenarioError, match="'cast' must be one of"):
        load_scenario(path)


def test_version_2_rejected(tmp_path):
    with pytest.raises(ScenarioError, match="'version' must be 1"):
        load_scenario(write_yaml(tmp_path, "scenario.yaml", BASE.replace("version: 1", "version: 2")))


def test_missing_seed_rejected(tmp_path):
    with pytest.raises(ScenarioError, match="'seed' must be an integer"):
        load_scenario(
            write_yaml(tmp_path, "scenario.yaml", BASE.replace("seed: 42\n", ""))
        )


def test_empty_faults_rejected(tmp_path):
    text = (
        "version: 1\n"
        "seed: 42\n"
        "input:\n"
        "  file: data.parquet\n"
        "  time_column: timestamp\n"
        "faults: []\n"
    )
    with pytest.raises(ScenarioError, match="'faults' must be a non-empty list"):
        load_scenario(write_yaml(tmp_path, "scenario.yaml", text))


def test_unknown_channel_raises_before_mutation(telemetry_df):
    df = telemetry_df
    snapshot = df.copy(deep=True)
    scenario = make_scenario(
        [
            {"channel": "nonexistent", "type": "dropout"},
            # a second, valid fault: must NOT be applied either
            {"channel": "pressure", "type": "bias", "params": {"offset": 1.0}},
        ]
    )
    with pytest.raises(FaultError, match="not found"):
        apply_scenario(df, scenario)
    # pre-validation happens before any mutation: the caller's frame is intact
    pd.testing.assert_frame_equal(df, snapshot)
