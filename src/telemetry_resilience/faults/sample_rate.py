"""Sample-rate fault: thin samples in the window down to a lower rate.

Deterministic rate thinning that supports arbitrary (including
non-integer) from/to ratios. For window positions i = 0..n-1, position i
is retained when ``floor(i * to_hz / from_hz) > floor((i-1) * to_hz /
from_hz)``; position 0 is always retained. In other words, the first
sample of each target-period bin is kept, distributing the retained
observations as evenly as practical across the window.

Rounding behavior: the retained count is exactly
``floor((n-1) * to_hz / from_hz) + 1``, i.e. approximately
``n * to_hz / from_hz`` (it may differ by at most one sample from the
ideal proportion at window edges). The rule is deterministic and uses no
randomness.

Only the selected channel's observations are made unavailable (nulled);
the global timeline -- rows and timestamps -- is preserved.
"""
import math

import numpy as np
import pandas as pd

from ..models import FaultError
from ._common import parse_hz, reject_unknown


def validate(params):
    p = dict(params)
    if "from" not in p:
        raise FaultError("'from' frequency is required")
    if "to" not in p:
        raise FaultError("'to' frequency is required")
    from_hz = parse_hz(p.pop("from"), "'from'")
    to_hz = parse_hz(p.pop("to"), "'to'")
    if not to_hz < from_hz:
        raise FaultError(
            f"'to' ({to_hz} Hz) must be less than 'from' ({from_hz} Hz): "
            "upsampling is an impossible rate"
        )
    reject_unknown(p)
    return {"from": from_hz, "to": to_hz}


def apply(df, mask, time_col, spec, rng):
    channel = spec.channel
    from_hz = spec.params["from"]
    to_hz = spec.params["to"]
    ratio = to_hz / from_hz
    n = int(mask.sum())
    window_idx = df.index[mask]

    keep = np.zeros(n, dtype=bool)
    prev_bin = -1
    for i in range(n):
        bin_no = math.floor(i * ratio)
        if bin_no > prev_bin:
            keep[i] = True
            prev_bin = bin_no

    null = np.nan if pd.api.types.is_float_dtype(df[channel].dtype) else pd.NA
    to_null = window_idx[~keep]
    needs_null = to_null[~df.loc[to_null, channel].isna().to_numpy()]
    newly_unavailable = len(needs_null)
    df.loc[needs_null, channel] = null

    retained = int(keep.sum())
    return {
        "requested_from_hz": from_hz,
        "requested_to_hz": to_hz,
        "window_samples": n,
        "expected_samples": n,
        "retained_samples": retained,
        "observations_made_unavailable": newly_unavailable,
        "targeted_observations": int((~keep).sum()),
        "changed_observations": newly_unavailable,
        "effective_ratio": (retained / n) if n else 0.0,
        "affected_observations": newly_unavailable,
    }
