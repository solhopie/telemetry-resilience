"""Scale fault: affine transform (value * factor + offset) in the window.

This is a pure numeric transform of the recorded values; it never claims
any unit conversion.
"""
from ..models import FaultError
from ._common import is_number, reject_unknown


def validate(params):
    p = dict(params)
    if "factor" not in p:
        raise FaultError("factor is required")
    factor = p.pop("factor")
    if not is_number(factor):
        raise FaultError("factor must be numeric")
    offset = p.pop("offset", 0.0)
    if not is_number(offset):
        raise FaultError("offset must be numeric")
    reject_unknown(p)
    return {"factor": factor, "offset": offset}


def apply(df, mask, time_col, spec, rng):
    channel = spec.channel
    df.loc[mask, channel] = (
        df.loc[mask, channel] * spec.params["factor"] + spec.params["offset"]
    )
    return {"affected_observations": int(mask.sum())}
