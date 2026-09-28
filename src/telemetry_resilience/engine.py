"""Deterministic fault-injection engine.

apply_scenario(df, scenario) copies the input frame, applies each fault in
order against a stable t0, and returns (corrupted_df, fault_entries).

Type safety: the engine refuses to silently corrupt data. Numeric faults on
non-numeric channels, nulls into non-nullable integer dtypes, and
float-widening ops on integer dtypes all raise FaultError unless the user
explicitly requested a `cast` in the scenario. After each op, an untouched
float32 column is restored to float32 so dtypes don't silently widen.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .faults import FAULT_TYPES
from .models import FaultError, Scenario

_CASTABLE = ("float32", "float64", "Int64", "Int32")
_FLOAT_ONLY_OPS = ("spike", "drift", "noise")
_NUMERIC_OPS = ("spike", "bias", "drift", "noise", "scale", "clipping")


def _prep_column(df, time_col, fspec):
    """Type-safety preparation for one fault.

    Returns (orig_dtype, cast_applied, params, widened) where params is a
    copy of fspec.params with integer-safe coercions applied (e.g. an
    integral bias offset becomes int so an integer column is not silently
    upcast). The manifest keeps the original normalized params from the
    scenario. ``widened`` is True when a float32 column was temporarily
    widened to float64 for the operation (pandas >= 3 refuses to store
    float64 op results into float32 columns); the caller restores float32
    afterwards.
    """
    ftype = fspec.type
    channel = fspec.channel
    params = dict(fspec.params)
    if ftype in ("timestamp_gap", "timestamp_jitter"):
        # Timeline faults never touch a data channel; skip column prep
        # entirely (channel may be None).
        return None, False, params, False

    orig_dtype = df[channel].dtype
    cast_applied = False

    if ftype in _NUMERIC_OPS and (
        channel == time_col or not pd.api.types.is_numeric_dtype(orig_dtype)
    ):
        raise FaultError(
            f"{ftype} on '{channel}': requires a numeric data channel "
            f"(got dtype {orig_dtype}); refusing to silently corrupt data"
        )

    def _apply_cast():
        nonlocal cast_applied
        if fspec.cast not in _CASTABLE:
            raise FaultError(
                f"{ftype} on '{channel}': dtype {orig_dtype} cannot hold the "
                f"result; provide cast in {list(_CASTABLE)} to proceed "
                "(silent corruption refused)"
            )
        df[channel] = df[channel].astype(fspec.cast)
        cast_applied = True

    is_float = pd.api.types.is_float_dtype(orig_dtype)
    is_int = pd.api.types.is_integer_dtype(orig_dtype)

    if ftype in _FLOAT_ONLY_OPS:
        if is_float:
            pass
        elif fspec.cast in ("float32", "float64"):
            _apply_cast()
        else:
            raise FaultError(
                f"{ftype} on '{channel}': requires float32/float64 dtype "
                f"(got {orig_dtype}); provide cast: float32 or cast: float64"
            )
    elif ftype == "bias":
        offset = params["offset"]
        if is_float:
            pass
        elif is_int and float(offset).is_integer():
            params["offset"] = int(offset)
        else:
            _apply_cast()
    elif ftype == "scale":
        factor, offset = params["factor"], params["offset"]
        if is_float:
            pass
        elif is_int and float(factor).is_integer() and offset == 0:
            params["factor"] = int(factor)
            params["offset"] = int(offset)
        else:
            _apply_cast()
    elif ftype == "clipping":
        bounds = [b for b in (params.get("min"), params.get("max")) if b is not None]
        if is_float:
            pass
        elif is_int and all(float(b).is_integer() for b in bounds):
            if params.get("min") is not None:
                params["min"] = int(params["min"])
            if params.get("max") is not None:
                params["max"] = int(params["max"])
        else:
            _apply_cast()
    elif ftype in ("dropout", "sample_rate"):
        if is_float or isinstance(
            orig_dtype, (pd.Int64Dtype, pd.Int32Dtype, pd.BooleanDtype)
        ):
            pass
        elif is_int or pd.api.types.is_bool_dtype(orig_dtype):
            # numpy int64/int32 (or bool): NaN/NA cannot be stored; the user
            # must explicitly opt into a nullable dtype.
            _apply_cast()
        else:
            raise FaultError(
                f"{ftype} on '{channel}': cannot represent nulls in dtype "
                f"{orig_dtype}; provide cast in {list(_CASTABLE)}"
            )
    # freeze: allowed on any dtype, no prep needed.

    widened = False
    # Widen float32 -> float64 for the op itself, whether the float32 came
    # from the original data or an explicit cast: pandas >= 3 raises
    # LossySetitemError when float64 op results are stored into a float32
    # column. The caller restores float32 afterwards.
    if ftype in _NUMERIC_OPS and df[channel].dtype == np.dtype("float32"):
        df[channel] = df[channel].astype("float64")
        widened = True

    return orig_dtype, cast_applied, params, widened


def apply_scenario(df: pd.DataFrame, scenario: Scenario) -> tuple:
    """Apply every fault in the scenario; return (df, fault_entries)."""
    df = df.copy()
    time_col = scenario.time_column
    if time_col not in df.columns:
        raise FaultError(f"time column '{time_col}' not found in input data")
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise FaultError(
            f"time column '{time_col}' must be datetime64 "
            f"(got {df[time_col].dtype})"
        )

    # Pre-validate every fault's channel before touching any data.
    for i, fspec in enumerate(scenario.faults, start=1):
        if fspec.type not in FAULT_TYPES:
            raise FaultError(f"fault #{i}: unknown fault type '{fspec.type}'")
        if fspec.type in ("timestamp_gap", "timestamp_jitter"):
            if fspec.channel is not None:
                raise FaultError(
                    f"fault #{i} ({fspec.type}): 'channel' must be absent; "
                    f"{fspec.type} is a global timeline fault"
                )
        elif fspec.channel not in df.columns:
            raise FaultError(
                f"fault #{i} ({fspec.type} on '{fspec.channel}'): "
                "channel not found in input data"
            )

    t0 = df[time_col].min()  # computed once, kept stable across faults
    entries = []
    for i, fspec in enumerate(scenario.faults):
        start = t0 + pd.to_timedelta(fspec.start_seconds, "s")
        end = t0 + pd.to_timedelta(fspec.start_seconds + fspec.duration_seconds, "s")
        mask = (df[time_col] >= start) & (df[time_col] < end)
        rng = np.random.default_rng([scenario.seed, i])

        _orig_dtype, _cast_applied, params, widened = _prep_column(
            df, time_col, fspec
        )
        runtime_spec = replace(fspec, params=params)
        extras = FAULT_TYPES[fspec.type].apply(df, mask, time_col, runtime_spec, rng)

        if widened:
            # Restore float32 after the widened op (covers both original
            # float32 columns and explicit cast: float32).
            df[fspec.channel] = df[fspec.channel].astype("float32")

        entry = {
            # Timeline faults operate on the global time axis: record a
            # null channel in the manifest, never a sensor channel.
            "channel": (
                None
                if fspec.type in ("timestamp_gap", "timestamp_jitter")
                else fspec.channel
            ),
            "type": fspec.type,
            "start_seconds": fspec.start_seconds,
            "duration_seconds": fspec.duration_seconds,
            "parameters": fspec.params,
            "affected_observations": extras.pop("affected_observations", 0),
        }
        entry.update(extras)
        entries.append(entry)

    return df, entries
