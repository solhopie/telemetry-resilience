"""Regression tests for the RUN 1.1 correctness & safety patch.

Each test below pins one of the 11 fixes so the bug cannot silently
reappear. Conventions mirror the existing suite (CliRunner for CLI,
make_scenario/window_mask helpers from conftest).
"""
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from telemetry_resilience.cli import app
from telemetry_resilience.engine import apply_scenario
from telemetry_resilience.io import read_file, write_file
from telemetry_resilience.models import ScenarioError
from telemetry_resilience.scenario import load_scenario

from .conftest import make_scenario, window_mask, write_yaml

runner = CliRunner()

BASE = """\
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
"""


def _scenario_yaml(tmp_path, name, fault_yaml, extra=""):
    return write_yaml(
        tmp_path,
        name,
        "version: 1\n"
        "seed: 42\n"
        "input:\n"
        "  file: data.parquet\n"
        "  time_column: timestamp\n"
        "faults:\n"
        f"{fault_yaml}\n"
        f"{extra}",
    )


def _rate_df(n, freq="100ms"):
    ts = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "ch": np.arange(n, dtype="float64")})


# ---------------------------------------------------------------------------
# Fix 1: CSV sidecar overwrite bug -- corrupted output owns its own sidecar
# ---------------------------------------------------------------------------


def test_csv_injection_never_touches_input_sidecar(tmp_path):
    n = 200
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    df = pd.DataFrame(
        {"timestamp": ts, "rpm": pd.Series(np.arange(n), dtype="Int64")}
    )
    data = tmp_path / "data.csv"
    write_file(df, data)
    schema_path = tmp_path / "data.schema.json"
    assert schema_path.exists()
    assert json.loads(schema_path.read_text())["columns"]["rpm"] == "Int64"
    csv_bytes_before = data.read_bytes()
    schema_bytes_before = schema_path.read_bytes()

    scenario = _scenario_yaml(
        tmp_path,
        "spike.yaml",
        "  - channel: rpm\n"
        "    type: spike\n"
        "    start: 1s\n"
        "    duration: 2s\n"
        "    magnitude: 50\n"
        "    cast: float32\n",
    )
    result = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert result.exit_code == 0

    corrupted = tmp_path / "data.corrupted.csv"
    corrupted_schema = tmp_path / "data.corrupted.schema.json"
    assert corrupted.exists()
    assert corrupted_schema.exists()

    # originals are byte-for-byte untouched
    assert data.read_bytes() == csv_bytes_before
    assert schema_path.read_bytes() == schema_bytes_before

    # the corrupted output carries its own schema; dtypes differ as intended
    assert json.loads(corrupted_schema.read_text())["columns"]["rpm"] == "float32"
    assert json.loads(schema_path.read_text())["columns"]["rpm"] == "Int64"

    # the corrupted CSV reads back through its OWN sidecar
    back = read_file(corrupted)
    assert str(back["rpm"].dtype) == "float32"


def test_csv_sidecar_is_preflighted(tmp_path):
    # The schema sidecar counts as an output: re-injecting without
    # --overwrite must refuse even when only the sidecar still exists.
    n = 60
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    df = pd.DataFrame(
        {"timestamp": ts, "rpm": pd.Series(np.arange(n), dtype="Int64")}
    )
    data = tmp_path / "data.csv"
    write_file(df, data)
    scenario = _scenario_yaml(
        tmp_path,
        "spike.yaml",
        "  - channel: rpm\n"
        "    type: spike\n"
        "    start: 1s\n"
        "    duration: 2s\n"
        "    magnitude: 50\n"
        "    cast: float32\n",
    )
    first = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert first.exit_code == 0
    # remove everything except the corrupted schema sidecar
    (tmp_path / "data.corrupted.csv").unlink()
    (tmp_path / "data.faults.json").unlink()
    (tmp_path / "data.report.json").unlink()
    assert (tmp_path / "data.corrupted.schema.json").exists()

    second = runner.invoke(app, ["inject", str(data), "--scenario", str(scenario)])
    assert second.exit_code == 2
    assert "data.corrupted.schema.json" in second.output


def test_csv_legacy_sidecar_fallback_still_reads(tmp_path):
    # Pre-1.1 layout: data.corrupted.csv with only data.schema.json present
    # must still read (fallback), preferring the specific sidecar when both exist.
    n = 60
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    df = pd.DataFrame(
        {"timestamp": ts, "rpm": pd.Series(np.arange(n), dtype="Int64")}
    )
    legacy = tmp_path / "data.corrupted.csv"
    df.to_csv(legacy, index=False)
    (tmp_path / "data.schema.json").write_text(
        json.dumps(
            {
                "columns": {"timestamp": "datetime64[ns, UTC]", "rpm": "Int64"},
                "time_column": "timestamp",
            }
        )
    )
    back = read_file(legacy)
    assert str(back["rpm"].dtype) == "Int64"
    assert str(back["timestamp"].dtype) == "datetime64[ns, UTC]"


# ---------------------------------------------------------------------------
# Fix 2: sample-rate degradation -- deterministic thinning, any ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_hz,to_hz,n,expected",
    [
        (10, 2, 100, 20),
        (10, 3, 100, 30),   # non-integer ratio: must NOT behave like ~3.4 Hz
        (100, 30, 200, 60),
        (60, 7.5, 80, 10),
    ],
)
def test_sample_rate_arbitrary_ratios(from_hz, to_hz, n, expected):
    df = _rate_df(n)
    scenario = make_scenario(
        [
            {
                "channel": "ch",
                "type": "sample_rate",
                "params": {"from": from_hz, "to": to_hz},
                "start": 0,
                "duration": 3600,
            }
        ],
        seed=5,
    )
    out, entries = apply_scenario(df, scenario)
    e = entries[0]

    assert e["requested_from_hz"] == from_hz
    assert e["requested_to_hz"] == to_hz
    assert e["window_samples"] == n
    assert e["retained_samples"] == expected
    assert e["observations_made_unavailable"] == n - expected
    assert e["effective_ratio"] == pytest.approx(expected / n)

    # global timeline intact; only the channel nulled
    assert len(out) == n
    assert out["timestamp"].equals(df["timestamp"])
    kept = out["ch"].notna()
    assert int(kept.sum()) == expected
    assert (out.loc[kept, "ch"] == df.loc[kept, "ch"]).all()

    # retained observations are spread evenly (no clumping)
    positions = np.flatnonzero(kept.to_numpy())
    assert int(np.diff(positions).max()) <= math.ceil(from_hz / to_hz) + 1


def test_sample_rate_10_to_3_not_integer_stepped():
    # The old round(10/3)=3 stepping kept ~34 of 100 (~3.4 Hz). The new
    # algorithm must retain ~30 (10 Hz -> 3 Hz), i.e. within one sample of
    # the ideal proportion.
    df = _rate_df(100)
    _, entries = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "ch",
                    "type": "sample_rate",
                    "params": {"from": "10Hz", "to": "3Hz"},
                    "start": 0,
                    "duration": 3600,
                }
            ],
            seed=5,
        ),
    )
    e = entries[0]
    assert e["retained_samples"] == 30
    assert abs(e["effective_ratio"] - 0.3) <= 1 / 100 + 1e-9


# ---------------------------------------------------------------------------
# Fix 3: test command must never false-pass a crash
# ---------------------------------------------------------------------------


def _data_and_no_expect_scenario(tmp_path, telemetry_df, expect_yaml=""):
    data = tmp_path / "data.parquet"
    write_file(telemetry_df, data)
    scenario = write_yaml(
        tmp_path,
        "scenario.yaml",
        "version: 1\n"
        "seed: 42\n"
        "input:\n"
        "  file: data.parquet\n"
        "  time_column: timestamp\n"
        "faults:\n"
        "  - channel: motor_temperature\n"
        "    type: dropout\n"
        "    start: 1s\n"
        "    duration: 2s\n" + expect_yaml,
    )
    return scenario


def _target(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_test_command_no_expect_exit_0_passes(tmp_path, telemetry_df):
    scenario = _data_and_no_expect_scenario(tmp_path, telemetry_df)
    target = _target(tmp_path, "ok.py", "print('fine')\n")
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target), "{data}"]
    )
    assert result.exit_code == 0
    assert "TEST RESULT: PASS" in result.output


def test_test_command_no_expect_crash_fails(tmp_path, telemetry_df):
    scenario = _data_and_no_expect_scenario(tmp_path, telemetry_df)
    target = _target(tmp_path, "boom.py", "import sys\nsys.exit(9)\n")
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target), "{data}"]
    )
    assert result.exit_code == 1
    assert "TEST RESULT: FAIL" in result.output
    assert "exit_code == 0" in result.output


def test_test_command_stdout_ok_but_crash_still_fails(tmp_path, telemetry_df):
    scenario = _data_and_no_expect_scenario(
        tmp_path,
        telemetry_df,
        "expect:\n  stdout_contains: [DEGRADED_MODE]\n",
    )
    target = _target(
        tmp_path,
        "boom.py",
        "import sys\nprint('DEGRADED_MODE')\nsys.exit(9)\n",
    )
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target), "{data}"]
    )
    assert result.exit_code == 1
    assert "TEST RESULT: FAIL" in result.output


def test_test_command_explicit_exit_9_passes(tmp_path, telemetry_df):
    scenario = _data_and_no_expect_scenario(
        tmp_path,
        telemetry_df,
        "expect:\n  exit_code: 9\n",
    )
    target = _target(tmp_path, "boom.py", "import sys\nsys.exit(9)\n")
    result = runner.invoke(
        app, ["test", str(scenario), "--", sys.executable, str(target), "{data}"]
    )
    assert result.exit_code == 0
    assert "TEST RESULT: PASS" in result.output


# ---------------------------------------------------------------------------
# Fix 4: timestamp_jitter is a global timeline fault (no channel)
# ---------------------------------------------------------------------------


def test_timestamp_jitter_with_channel_rejected(tmp_path):
    path = _scenario_yaml(
        tmp_path,
        "jitter.yaml",
        "  - channel: pressure\n"
        "    type: timestamp_jitter\n"
        "    start: 10s\n"
        "    duration: 5s\n"
        "    max_jitter_ms: 80\n",
    )
    with pytest.raises(ScenarioError, match="'channel' must be absent"):
        load_scenario(path)


def test_timestamp_jitter_without_channel_loads(tmp_path):
    path = _scenario_yaml(
        tmp_path,
        "jitter.yaml",
        "  - type: timestamp_jitter\n"
        "    start: 10s\n"
        "    duration: 5s\n"
        "    max_jitter_ms: 80\n",
    )
    sc = load_scenario(path)
    assert sc.faults[0].type == "timestamp_jitter"
    assert sc.faults[0].channel is None


def test_timestamp_jitter_manifest_channel_null_and_data_untouched(telemetry_df):
    df = telemetry_df
    out, entries = apply_scenario(
        df,
        make_scenario(
            [{"type": "timestamp_jitter", "params": {"max_jitter_ms": 80}}],
            seed=3,
        ),
    )
    assert entries[0]["channel"] is None
    assert entries[0]["type"] == "timestamp_jitter"
    # only the time column changes; ordinary channels are untouched
    for col in df.columns:
        if col != "timestamp":
            pd.testing.assert_series_equal(out[col], df[col])
    # rows are NOT automatically resorted
    pd.testing.assert_index_equal(out.index, df.index)


# ---------------------------------------------------------------------------
# Fix 6: strict scenario key validation
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ScenarioError, match="unknown top-level key"):
        load_scenario(write_yaml(tmp_path, "s.yaml", BASE + "expct:\n  exit_code: 0\n"))


def test_unknown_input_key_rejected(tmp_path):
    text = BASE.replace(
        "  time_column: timestamp\n",
        "  time_column: timestamp\n  tz: UTC\n",
    )
    with pytest.raises(ScenarioError, match="unknown 'input' key"):
        load_scenario(write_yaml(tmp_path, "s.yaml", text))


# ---------------------------------------------------------------------------
# Fix 9: manifest accuracy -- no-effect operations are not counted
# ---------------------------------------------------------------------------


def test_dropout_does_not_count_already_null(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)
    non_null = int(df.loc[mask, "rpm"].notna().sum())
    assert 0 < non_null < int(mask.sum())  # fixture really has NAs in the window

    _, entries = apply_scenario(
        df, make_scenario([{"channel": "rpm", "type": "dropout"}])
    )
    e = entries[0]
    assert e["targeted_observations"] == int(mask.sum())
    assert e["changed_observations"] == non_null
    assert e["affected_observations"] == non_null


def test_spike_selects_only_non_null_observations():
    n = 100
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    vals = np.arange(n, dtype="float64")
    vals[::10] = np.nan  # 10 pre-existing NaNs
    df = pd.DataFrame({"timestamp": ts, "ch": vals})

    out, entries = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "ch",
                    "type": "spike",
                    "params": {"magnitude": 5.0, "count": 20},
                    "start": 0,
                    "duration": 3600,
                }
            ],
            seed=9,
        ),
    )
    e = entries[0]
    assert e["targeted_observations"] == n
    assert e["changed_observations"] == 20
    assert e["affected_observations"] == 20
    # NaNs untouched; exactly 20 non-null observations moved by |5.0|
    assert int(out["ch"].isna().sum()) == int(df["ch"].isna().sum())
    assert int((out["ch"] - df["ch"]).abs().eq(5.0).sum()) == 20


def test_spike_all_null_window_reports_zero_changed():
    n = 50
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "ch": np.full(n, np.nan)})

    _, entries = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "ch",
                    "type": "spike",
                    "params": {"magnitude": 5.0, "count": 3},
                    "start": 0,
                    "duration": 3600,
                }
            ],
            seed=9,
        ),
    )
    e = entries[0]
    assert e["targeted_observations"] == n
    assert e["changed_observations"] == 0
    assert e["affected_observations"] == 0


def test_sample_rate_does_not_recount_already_null():
    n = 50
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    vals = pd.Series(np.arange(n, dtype="float64"))
    vals.iloc[1] = np.nan
    vals.iloc[3] = np.nan  # pre-existing nulls inside the window
    df = pd.DataFrame({"timestamp": ts, "ch": vals})

    out, entries = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "ch",
                    "type": "sample_rate",
                    "params": {"from": "10Hz", "to": "5Hz"},
                    "start": 0,
                    "duration": 3600,
                }
            ],
            seed=5,
        ),
    )
    e = entries[0]
    # 10 -> 5 Hz keeps even positions 0,2,...,48; odd positions (incl. the
    # two NaNs) are targeted for nulling
    assert e["retained_samples"] == 25
    assert e["targeted_observations"] == 25
    assert e["changed_observations"] == 23
    assert e["observations_made_unavailable"] == 23
    assert e["affected_observations"] == 23
    # total nulls grew only by the newly-nulled observations
    assert int(out["ch"].isna().sum()) == 2 + 23


# ---------------------------------------------------------------------------
# Fix 10: Parquet acceptance tests, actually executed with PyArrow
# ---------------------------------------------------------------------------


def test_parquet_acceptance_pyarrow(telemetry_df, tmp_path):
    import pyarrow  # noqa: F401 -- fails loudly if PyArrow is not installed

    df = telemetry_df
    path = tmp_path / "accept.parquet"
    write_file(df, path)
    back = read_file(path)

    assert list(back.columns) == list(df.columns)  # column order survives
    assert len(back) == len(df)  # row count survives
    assert str(back["motor_temperature"].dtype) == "float32"  # float32 survives
    assert str(back["imu_z"].dtype) == "float32"
    assert str(back["pressure"].dtype) == "float64"  # float64 survives
    assert str(back["rpm"].dtype) == "Int64"  # nullable Int64 survives
    assert int(back["rpm"].isna().sum()) == int(df["rpm"].isna().sum())  # nulls
    assert str(back["timestamp"].dtype) == "datetime64[ns, UTC]"  # UTC tz survives
    pd.testing.assert_frame_equal(back, df)
