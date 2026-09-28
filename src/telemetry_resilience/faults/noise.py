"""Noise fault: add zero-mean Gaussian noise in the window."""
from ..models import FaultError
from ._common import is_number, reject_unknown


def validate(params):
    p = dict(params)
    has_std = "std" in p
    has_magnitude = "magnitude" in p
    if has_std and has_magnitude:
        raise FaultError("specify 'std' or 'magnitude', not both")
    if has_std:
        std = p.pop("std")
    elif has_magnitude:
        std = p.pop("magnitude")
    else:
        raise FaultError("std is required (alias: magnitude)")
    if not is_number(std) or std <= 0:
        raise FaultError("std must be a number > 0")
    reject_unknown(p)
    return {"std": std}


def apply(df, mask, time_col, spec, rng):
    channel = spec.channel
    n = int(mask.sum())
    df.loc[mask, channel] = (
        df.loc[mask, channel] + rng.normal(0.0, spec.params["std"], n)
    )
    return {"affected_observations": n}
