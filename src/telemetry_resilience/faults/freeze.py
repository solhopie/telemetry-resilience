"""Freeze fault: hold the last valid value through the window.

If no valid value exists before the window, the first value inside the
window is held instead.
"""
from ._common import reject_unknown


def validate(params):
    reject_unknown(dict(params))
    return {}


def apply(df, mask, time_col, spec, rng):
    n = int(mask.sum())
    if n == 0:
        return {"affected_observations": 0}
    channel = spec.channel
    window_start = df.loc[mask, time_col].min()
    before = df.loc[df[time_col] < window_start, channel].dropna()
    hold = before.iloc[-1] if len(before) else df.loc[mask, channel].iloc[0]
    df.loc[mask, channel] = hold
    return {"affected_observations": n}
