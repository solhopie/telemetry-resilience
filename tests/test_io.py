"""IO adapter round-trip tests."""
import numpy as np
import pandas as pd

from telemetry_resilience.io import read_file, write_file


def test_parquet_round_trip_exact(telemetry_df, tmp_path):
    df = telemetry_df
    path = tmp_path / "data.parquet"
    write_file(df, path)

    back = read_file(path)

    assert list(back.columns) == list(df.columns)
    assert len(back) == len(df) == 600
    # pyarrow preserves every dtype exactly, incl. datetime64[ns, UTC],
    # float32, and Int64 with pd.NA
    pd.testing.assert_frame_equal(back, df)


def test_csv_sidecar_written_and_dtypes_restored(telemetry_df, tmp_path):
    df = telemetry_df
    path = tmp_path / "run.corrupted.csv"
    write_file(df, path)

    # RUN 1.1: every CSV owns its own sidecar -- a corrupted output writes
    # "<stem>.corrupted.schema.json" and must never touch the input's
    # "<stem>.schema.json".
    own_sidecar = tmp_path / "run.corrupted.schema.json"
    assert own_sidecar.exists(), "own schema sidecar not written next to corrupted CSV"
    assert not (tmp_path / "run.schema.json").exists()

    back = read_file(path)

    assert list(back.columns) == list(df.columns)
    assert len(back) == len(df)
    assert str(back["motor_temperature"].dtype) == "float32"
    assert str(back["imu_z"].dtype) == "float32"
    assert str(back["pressure"].dtype) == "float64"
    assert str(back["count_int"].dtype) == "int64"
    assert str(back["rpm"].dtype) == "Int64"
    assert int(back["rpm"].isna().sum()) == int(df["rpm"].isna().sum())
    # The sidecar records "datetime64[ns, UTC]"; the reader restores the
    # exact unit (pandas 3 parses CSV datetimes at us, the adapter casts
    # back to the sidecar dtype).
    assert str(back["timestamp"].dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_series_equal(back["timestamp"], df["timestamp"])
    pd.testing.assert_frame_equal(
        back.drop(columns=["timestamp"]), df.drop(columns=["timestamp"])
    )


def test_jsonl_round_trip_values(telemetry_df, tmp_path):
    # NOTE on dtype limits: JSONL carries no schema sidecar, so dtypes do NOT
    # round-trip: float32 widens to float64, nullable Int64 becomes float64
    # with NaN, and tz-aware timestamps are ISO strings parsed back (by column
    # name). Values round-trip; dtypes do not.
    #
    # RUN 1.1: the project declares pandas >= 2.0, and the two majors use
    # different default datetime resolutions (ns vs us). This test therefore
    # verifies timestamp VALUES and UTC tz-awareness, not one specific
    # datetime resolution. (pandas to_json writes ISO strings at ms
    # precision, so values are compared with a 1 ms tolerance.)
    df = telemetry_df
    path = tmp_path / "data.jsonl"
    write_file(df, path)

    back = read_file(path)

    assert list(back.columns) == list(df.columns)
    assert len(back) == len(df)
    ts = back["timestamp"]
    assert pd.api.types.is_datetime64_any_dtype(ts)
    assert str(ts.dt.tz) == "UTC"
    delta_s = (ts - df["timestamp"]).abs().dt.total_seconds()
    assert bool((delta_s < 0.001).all())
    for col in ("motor_temperature", "pressure", "gps_latitude", "imu_z"):
        np.testing.assert_allclose(
            back[col].to_numpy(), df[col].to_numpy(), rtol=1e-6, atol=0, equal_nan=True
        )
    assert int(back["rpm"].isna().sum()) == int(df["rpm"].isna().sum())
    notna = df["rpm"].notna()
    np.testing.assert_allclose(
        back.loc[notna, "rpm"].to_numpy(),
        df.loc[notna, "rpm"].astype("float64").to_numpy(),
        rtol=0,
        atol=0,
    )
    pd.testing.assert_series_equal(back["count_int"], df["count_int"])
