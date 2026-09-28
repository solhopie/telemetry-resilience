"""Timestamp jitter fault: perturb timestamps in the window.

Each affected timestamp is shifted by a uniform random delta in
[-max_jitter_ms, +max_jitter_ms]. The row order is deliberately NOT
re-sorted afterwards, so out-of-order timestamps stay visible.
"""
import numpy as np
import pandas as pd

from ..models import FaultError
from ._common import is_number, reject_unknown


def validate(params):
    p = dict(params)
    if "max_jitter_ms" not in p:
        raise FaultError("max_jitter_ms is required")
    max_jitter_ms = p.pop("max_jitter_ms")
    if not is_number(max_jitter_ms) or max_jitter_ms <= 0:
        raise FaultError("max_jitter_ms must be a number > 0")
    reject_unknown(p)
    return {"max_jitter_ms": max_jitter_ms}


def apply(df, mask, time_col, spec, rng):
    n = int(mask.sum())
    if n:
        delta_ms = rng.uniform(
            -spec.params["max_jitter_ms"], spec.params["max_jitter_ms"], n
        )
        # Add the jitter in the column's own datetime unit (pandas >= 3
        # defaults to microsecond unit; adding a nanosecond timedelta and
        # assigning back would fail the assignment). Rounding to whole
        # units keeps the operation lossless for the column's resolution
        # and deterministic under the seed.
        unit = df[time_col].dtype.unit
        per_ms = {"s": 1e-3, "ms": 1.0, "us": 1e3, "ns": 1e6}.get(unit, 1e6)
        jitter = np.round(delta_ms * per_ms).astype("int64")
        df.loc[mask, time_col] = df.loc[mask, time_col] + pd.to_timedelta(
            jitter, unit=unit
        )
    return {"affected_observations": n}
