"""Drift fault: add a time-proportional ramp to the channel in the window.

The ``rate`` parameter is in units of the channel per second. Within the
fault window each value is adjusted as::

    value += rate * elapsed_seconds

where ``elapsed_seconds`` is the time since the start of the window
(0 at the first affected observation), measured on the time column.
"""
from ..models import FaultError
from ._common import is_number, reject_unknown


def validate(params):
    p = dict(params)
    if "rate" not in p:
        raise FaultError("rate is required")
    rate = p.pop("rate")
    if not is_number(rate):
        raise FaultError("rate must be numeric")
    reject_unknown(p)
    return {"rate": rate}


def apply(df, mask, time_col, spec, rng):
    channel = spec.channel
    n = int(mask.sum())
    if n:
        window_start = df.loc[mask, time_col].min()
        elapsed = (df.loc[mask, time_col] - window_start).dt.total_seconds()
        df.loc[mask, channel] = (
            df.loc[mask, channel] + spec.params["rate"] * elapsed.values
        )
    return {"affected_observations": n}
