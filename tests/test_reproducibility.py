"""Determinism / reproducibility tests for apply_scenario."""
import pandas as pd

from telemetry_resilience.engine import apply_scenario
from telemetry_resilience.manifest import build_manifest

from .conftest import make_scenario


def _noisy_scenario(seed):
    return make_scenario(
        [
            {"channel": "pressure", "type": "noise", "params": {"std": 0.8}},
            {
                "type": "timestamp_jitter",
                "params": {"max_jitter_ms": 200},
            },
            {
                "channel": "gps_latitude",  # float64: spike widens float32 cols,
                "type": "spike",            # which pandas 3 refuses (see report)
                "params": {"magnitude": 3.0, "count": 5},
            },
        ],
        seed=seed,
    )


def test_same_seed_gives_identical_output_and_manifest(telemetry_df):
    df = telemetry_df
    scenario = _noisy_scenario(seed=2026)

    out1, entries1 = apply_scenario(df, scenario)
    out2, entries2 = apply_scenario(df, scenario)

    pd.testing.assert_frame_equal(out1, out2)
    manifest1 = build_manifest(scenario, "data.parquet", entries1)
    manifest2 = build_manifest(scenario, "data.parquet", entries2)
    assert manifest1 == manifest2


def test_different_seed_changes_randomized_faults(telemetry_df):
    df = telemetry_df

    noise = [{"channel": "pressure", "type": "noise", "params": {"std": 0.8}}]
    out1, _ = apply_scenario(df, make_scenario(noise, seed=1))
    out2, _ = apply_scenario(df, make_scenario(noise, seed=2))
    assert not out1["pressure"].equals(out2["pressure"])

    jitter = [
        {
            "type": "timestamp_jitter",
            "params": {"max_jitter_ms": 200},
        }
    ]
    j1, _ = apply_scenario(df, make_scenario(jitter, seed=1))
    j2, _ = apply_scenario(df, make_scenario(jitter, seed=2))
    assert not j1["timestamp"].equals(j2["timestamp"])


def test_apply_does_not_mutate_input(telemetry_df):
    df = telemetry_df
    snapshot = df.copy(deep=True)
    apply_scenario(
        df,
        make_scenario(
            [
                {"channel": "pressure", "type": "noise", "params": {"std": 0.8}},
                {"type": "timestamp_gap"},
            ],
            seed=5,
        ),
    )
    pd.testing.assert_frame_equal(df, snapshot)
