"""Type-safety tests: the engine refuses silent dtype corruption (FaultError)
unless an explicit cast is requested."""
import numpy as np
import pandas as pd
import pytest

from telemetry_resilience.engine import apply_scenario
from telemetry_resilience.models import FaultError

from .conftest import make_scenario, window_mask


def test_drift_on_numpy_int64_without_cast_raises(telemetry_df):
    df = telemetry_df
    assert str(df["count_int"].dtype) == "int64"
    with pytest.raises(FaultError, match="cast"):
        apply_scenario(
            df,
            make_scenario(
                [{"channel": "count_int", "type": "drift", "params": {"rate": 0.37}}]
            ),
        )


# Previously xfailed under pandas 3.0.6 (LossySetitemError on float64 ->
# float32 assignment); fixed in engine.py by widening float32 to float64
# for the op and restoring float32 afterwards.
def test_drift_on_numpy_int64_with_cast_float32(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)
    out, _ = apply_scenario(
        df,
        make_scenario(
            [
                {
                    "channel": "count_int",
                    "type": "drift",
                    "params": {"rate": 0.37},
                    "cast": "float32",
                }
            ]
        ),
    )
    assert str(out["count_int"].dtype) == "float32"
    elapsed = (
        df.loc[mask, "timestamp"] - df.loc[mask, "timestamp"].min()
    ).dt.total_seconds()
    expected = df.loc[mask, "count_int"].astype("float32") + 0.37 * elapsed
    # float32 rounding on assignment: compare with tolerance, not bitwise
    np.testing.assert_allclose(
        out.loc[mask, "count_int"].to_numpy(), expected.to_numpy(), rtol=1e-5, atol=0
    )
    # the caller's frame keeps its original dtype
    assert str(df["count_int"].dtype) == "int64"


def test_dropout_on_numpy_int64_without_cast_raises(telemetry_df):
    df = telemetry_df
    with pytest.raises(FaultError, match="cast"):
        apply_scenario(
            df, make_scenario([{"channel": "count_int", "type": "dropout"}])
        )


def test_dropout_on_nullable_int64_uses_pd_na(telemetry_df):
    df = telemetry_df
    mask = window_mask(df)
    assert int(df["rpm"].isna().sum()) > 0  # fixture really has missing values
    out, _ = apply_scenario(
        df, make_scenario([{"channel": "rpm", "type": "dropout"}])
    )

    assert str(out["rpm"].dtype) == "Int64"
    assert out.loc[mask, "rpm"].isna().all()
    # window rows that already held NA stay NA: total = outside nulls + window
    assert int(out["rpm"].isna().sum()) == int(
        df.loc[~mask, "rpm"].isna().sum() + mask.sum()
    )
    # values outside the window (incl. pre-existing NAs) are untouched
    pd.testing.assert_series_equal(out.loc[~mask, "rpm"], df.loc[~mask, "rpm"])


# Previously xfailed under pandas 3.0.6 (LossySetitemError on float64 ->
# float32 assignment); fixed in engine.py by widening float32 to float64
# for the op and restoring float32 afterwards.
def test_noise_on_float32_preserves_dtype(telemetry_df):
    df = telemetry_df
    out, _ = apply_scenario(
        df,
        make_scenario(
            [{"channel": "motor_temperature", "type": "noise", "params": {"std": 0.8}}]
        ),
    )
    # the engine restores float32 after float-widening numeric ops
    assert str(out["motor_temperature"].dtype) == "float32"
    assert str(df["motor_temperature"].dtype) == "float32"
    mask = window_mask(df)
    assert not out.loc[mask, "motor_temperature"].equals(df.loc[mask, "motor_temperature"])
