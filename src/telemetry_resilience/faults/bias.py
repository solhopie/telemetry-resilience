"""Bias fault: add a constant offset to the channel in the window."""
from ..models import FaultError
from ._common import is_number, reject_unknown


def validate(params):
    p = dict(params)
    has_offset = "offset" in p
    has_magnitude = "magnitude" in p
    if has_offset and has_magnitude:
        raise FaultError("specify 'offset' or 'magnitude', not both")
    if has_offset:
        offset = p.pop("offset")
    elif has_magnitude:
        offset = p.pop("magnitude")
    else:
        raise FaultError("offset is required (alias: magnitude)")
    if not is_number(offset):
        raise FaultError("offset must be numeric")
    reject_unknown(p)
    return {"offset": offset}


def apply(df, mask, time_col, spec, rng):
    channel = spec.channel
    df.loc[mask, channel] = df.loc[mask, channel] + spec.params["offset"]
    return {"affected_observations": int(mask.sum())}
