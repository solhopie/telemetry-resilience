"""Spike fault: add signed impulses to a few samples in the window.

Spike positions are chosen from eligible (non-null, numeric) observations
in the window, so a requested spike never silently lands on NaN and does
nothing. If fewer eligible observations exist than ``count``, all eligible
ones are spiked and the manifest records the actual number changed.
"""
import numpy as np

from ..models import FaultError
from ._common import is_number, reject_unknown


def validate(params):
    p = dict(params)
    if "magnitude" not in p:
        raise FaultError("magnitude is required")
    magnitude = p.pop("magnitude")
    if not is_number(magnitude):
        raise FaultError("magnitude must be numeric")
    if magnitude == 0:
        raise FaultError("magnitude must be non-zero")
    direction = p.pop("direction", None)
    if direction not in (None, "positive", "negative"):
        raise FaultError("direction must be 'positive', 'negative', or omitted")
    count = p.pop("count", 1)
    if not is_number(count) or int(count) != count or count < 1:
        raise FaultError("count must be an integer >= 1")
    reject_unknown(p)
    return {"magnitude": magnitude, "direction": direction, "count": int(count)}


def apply(df, mask, time_col, spec, rng):
    channel = spec.channel
    targeted = int(mask.sum())
    eligible = df.index[mask & df[channel].notna()]
    n = min(spec.params["count"], len(eligible))
    if n == 0:
        return {
            "affected_observations": 0,
            "targeted_observations": targeted,
            "changed_observations": 0,
        }
    chosen = eligible[rng.choice(len(eligible), size=n, replace=False)]
    direction = spec.params["direction"]
    magnitude = spec.params["magnitude"]
    if direction == "positive":
        signs = np.ones(n)
    elif direction == "negative":
        signs = -np.ones(n)
    else:
        signs = rng.choice([-1, 1], size=n)
    df.loc[chosen, channel] = df.loc[chosen, channel] + signs * magnitude
    return {
        "affected_observations": int(n),
        "targeted_observations": targeted,
        "changed_observations": int(n),
    }
