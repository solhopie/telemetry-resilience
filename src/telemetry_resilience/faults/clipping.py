"""Clipping fault: clamp channel values to [min, max] in the window."""
from ..models import FaultError
from ._common import is_number, reject_unknown


def validate(params):
    p = dict(params)
    lo = p.pop("min", None)
    hi = p.pop("max", None)
    if lo is None and hi is None:
        raise FaultError("at least one of 'min' or 'max' is required")
    if lo is not None and not is_number(lo):
        raise FaultError("min must be numeric")
    if hi is not None and not is_number(hi):
        raise FaultError("max must be numeric")
    if lo is not None and hi is not None and lo > hi:
        raise FaultError("min must be <= max")
    reject_unknown(p)
    return {"min": lo, "max": hi}


def apply(df, mask, time_col, spec, rng):
    channel = spec.channel
    df.loc[mask, channel] = df.loc[mask, channel].clip(
        lower=spec.params["min"], upper=spec.params["max"]
    )
    return {"affected_observations": int(mask.sum())}
