"""Per-operator fault tests: one test per fault type (11 total)."""
import numpy as np
import pandas as pd

from telemetry_resilience.engine import apply_scenario

from .conftest import make_scenario, window_mask

MASK_ROWS = 50  # window_mask() default (10s..15s at 10 Hz) selects 50 rows


def test_dropout_nulls_only_in_window(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)
    assert int(mask.sum()) == MASK_ROWS

    out, entries = apply_scenario(
        df, make_scenario([{"channel": "motor_temperature", "type": "dropout"}])
    )

    col = out["motor_temperature"]
    assert col[mask].isna().all()
    assert int(col.isna().sum()) == MASK_ROWS
    assert (col[~mask] == df["motor_temperature"][~mask]).all()
    # other channels and rows untouched, timestamps preserved
    for c in df.columns:
        if c != "motor_temperature":
            pd.testing.assert_series_equal(out[c], df[c])
    assert out["timestamp"].equals(df["timestamp"])
    assert entries[0]["affected_observations"] == MASK_ROWS


def test_freeze_holds_last_pre_window_value(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)

    out, _ = apply_scenario(
        df, make_scenario([{"channel": "motor_temperature", "type": "freeze"}])
    )

    hold = df["motor_temperature"].iloc[99]  # last value before the window
    assert (out.loc[mask, "motor_temperature"] == hold).all()
    assert out["timestamp"].equals(df["timestamp"])  # timestamps untouched
    assert (out.loc[~mask, "motor_temperature"] == df.loc[~mask, "motor_temperature"]).all()


def test_spike_deterministic_positions(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)
    seed = 7
    fault = {
        "channel": "pressure",
        "type": "spike",
        "params": {"magnitude": 5.0, "direction": "positive", "count": 3},
    }

    out, entries = apply_scenario(df, make_scenario([fault], seed=seed))

    # replicate the engine's per-fault RNG: default_rng([seed, fault_index])
    rng = np.random.default_rng([seed, 0])
    idx = df.index[mask]
    chosen = idx[rng.choice(len(idx), size=3, replace=False)]
    deltas = out.loc[chosen, "pressure"] - df.loc[chosen, "pressure"]
    assert (deltas == 5.0).all()
    untouched = idx.difference(chosen)
    assert (out.loc[untouched, "pressure"] == df.loc[untouched, "pressure"]).all()
    assert entries[0]["affected_observations"] == 3

    # reproducible across runs
    out2, _ = apply_scenario(df, make_scenario([fault], seed=seed))
    pd.testing.assert_frame_equal(out, out2)


def test_bias_exact_offset(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)

    out, _ = apply_scenario(
        df,
        make_scenario(
            [{"channel": "pressure", "type": "bias", "params": {"offset": 2.5}}]
        ),
    )

    # identical expression on both sides -> bitwise-equal comparison
    pd.testing.assert_series_equal(
        out.loc[mask, "pressure"], df.loc[mask, "pressure"] + 2.5
    )
    assert (out.loc[~mask, "pressure"] == df.loc[~mask, "pressure"]).all()


def test_drift_ramp_from_window_start(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)

    out, _ = apply_scenario(
        df,
        make_scenario(
            [{"channel": "pressure", "type": "drift", "params": {"rate": 0.5}}]
        ),
    )

    elapsed = (
        df.loc[mask, "timestamp"] - df.loc[mask, "timestamp"].min()
    ).dt.total_seconds()
    np.testing.assert_allclose(
        (out.loc[mask, "pressure"] - df.loc[mask, "pressure"]).to_numpy(),
        (0.5 * elapsed).to_numpy(),
        rtol=1e-12,
        atol=0,
    )
    # 0 added at the window start: first affected row is unchanged
    assert out.loc[mask, "pressure"].iloc[0] == df.loc[mask, "pressure"].iloc[0]
    # ramp grows monotonically through the window
    added = out.loc[mask, "pressure"] - df.loc[mask, "pressure"]
    assert (added.diff().dropna() > 0).all()
    assert (out.loc[~mask, "pressure"] == df.loc[~mask, "pressure"]).all()


def test_noise_seed_determinism(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)
    fault = {"channel": "pressure", "type": "noise", "params": {"std": 0.8}}

    out1, _ = apply_scenario(df, make_scenario([fault], seed=1))
    out2, _ = apply_scenario(df, make_scenario([fault], seed=1))
    pd.testing.assert_frame_equal(out1, out2)  # same seed -> identical

    out3, _ = apply_scenario(df, make_scenario([fault], seed=2))
    assert not out1["pressure"].equals(out3["pressure"])  # different seed -> differs
    # noise is confined to the window
    assert (out1.loc[~mask, "pressure"] == df.loc[~mask, "pressure"]).all()


def test_clipping_enforces_bounds(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)

    # both bounds
    out, _ = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "pressure",
                    "type": "clipping",
                    "params": {"min": 101.0, "max": 101.6},
                }
            ]
        ),
    )
    w = out.loc[mask, "pressure"]
    assert (w >= 101.0).all() and (w <= 101.6).all()
    inside = (df.loc[mask, "pressure"] >= 101.0) & (
        df.loc[mask, "pressure"] <= 101.6
    )
    assert (w[inside] == df.loc[mask, "pressure"][inside]).all()
    assert (out.loc[~mask, "pressure"] == df.loc[~mask, "pressure"]).all()

    # min-only
    out_min, _ = apply_scenario(
        df,
        make_scenario(
            [{"channel": "pressure", "type": "clipping", "params": {"min": 101.0}}]
        ),
    )
    assert (out_min.loc[mask, "pressure"] >= 101.0).all()

    # max-only
    out_max, _ = apply_scenario(
        df,
        make_scenario(
            [{"channel": "pressure", "type": "clipping", "params": {"max": 101.6}}]
        ),
    )
    assert (out_max.loc[mask, "pressure"] <= 101.6).all()


def test_timestamp_gap_removes_exact_rows(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)

    out, entries = apply_scenario(df, make_scenario([{"type": "timestamp_gap"}]))

    assert len(out) == len(df) - MASK_ROWS == 550
    assert entries[0]["rows_removed"] == MASK_ROWS
    assert entries[0]["affected_observations"] == MASK_ROWS
    # remaining rows are untouched (all columns)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df[~mask].reset_index(drop=True)
    )


def test_timestamp_jitter_keeps_row_order(telemetry_df):
    # NOTE: this test relies on the fixture's datetime64[ns, UTC] timestamps.
    # timestamp_jitter.apply adds a timedelta64[ns] delta; under pandas 3 that
    # result can only be assigned back into an ns-unit column (on a us-unit
    # column the engine raises TypeError -- a genuine src/pandas-3 bug, see
    # the final test report).
    df = telemetry_df
    mask = window_mask(df)
    seed = 11
    max_jitter_ms = 500
    fault = {
        "type": "timestamp_jitter",
        "params": {"max_jitter_ms": max_jitter_ms},
    }

    out, entries = apply_scenario(df, make_scenario([fault], seed=seed))

    assert entries[0]["affected_observations"] == MASK_ROWS
    # timestamps changed inside the window...
    assert (out.loc[mask, "timestamp"] != df.loc[mask, "timestamp"]).any()
    # ...but untouched outside it, and data columns are untouched
    pd.testing.assert_series_equal(
        out.loc[~mask, "timestamp"], df.loc[~mask, "timestamp"]
    )
    pd.testing.assert_series_equal(out["pressure"], df["pressure"])
    # row order is NOT re-sorted: output order equals input order
    pd.testing.assert_index_equal(out.index, df.index)
    # self-consistent check: replicate the RNG to confirm this seed inverts
    # the order, then assert the engine output really is out of order
    rng = np.random.default_rng([seed, 0])
    expected = df.loc[mask, "timestamp"] + pd.to_timedelta(
        rng.uniform(-max_jitter_ms, max_jitter_ms, MASK_ROWS), unit="ms"
    )
    assert bool((expected.diff() < pd.Timedelta(0)).any()), (
        "test setup: chosen seed must cause a timestamp inversion"
    )
    assert bool((out["timestamp"].diff() < pd.Timedelta(0)).any())


def test_sample_rate_thins_window(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)

    out, entries = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "imu_z",
                    "type": "sample_rate",
                    "params": {"from": "10Hz", "to": "2Hz"},
                }
            ]
        ),
    )

    w = out.loc[mask, "imu_z"]
    kept = df.index[mask][::5]  # every 5th sample kept (10 Hz -> 2 Hz)
    assert int(w.isna().sum()) == 40
    assert w.loc[kept].notna().all()
    assert (w.loc[kept] == df.loc[kept, "imu_z"]).all()
    nulled = df.index[mask].difference(kept)
    assert w.loc[nulled].isna().all()
    # global timeline intact: same row count, same timestamps, outside untouched
    assert len(out) == len(df)
    assert out["timestamp"].equals(df["timestamp"])
    assert (out.loc[~mask, "imu_z"] == df.loc[~mask, "imu_z"]).all()
    assert entries[0]["retained_samples"] == 10
    assert entries[0]["observations_made_unavailable"] == 40


def test_scale_affine_exact(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)

    out, _ = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "pressure",
                    "type": "scale",
                    "params": {"factor": 2.0, "offset": 1.0},
                }
            ]
        ),
    )

    # identical expression on both sides -> bitwise-equal comparison
    pd.testing.assert_series_equal(
        out.loc[mask, "pressure"], df.loc[mask, "pressure"] * 2.0 + 1.0
    )
    assert (out.loc[~mask, "pressure"] == df.loc[~mask, "pressure"]).all()
