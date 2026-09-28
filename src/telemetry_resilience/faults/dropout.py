"""Dropout fault: replace channel values with null inside the window.

Only observations that are actually present are counted as changed: a
dropout window covering already-null values records those in
``targeted_observations`` but not in ``changed_observations`` (the
manifest describes what actually happened, not just what was requested).
"""
import numpy as np
import pandas as pd

from ._common import reject_unknown


def validate(params):
    reject_unknown(dict(params))
    return {}


def apply(df, mask, time_col, spec, rng):
    col = df[spec.channel]
    targeted = int(mask.sum())
    eligible = df.index[mask & col.notna()]
    changed = len(eligible)
    null = np.nan if pd.api.types.is_float_dtype(col.dtype) else pd.NA
    df.loc[eligible, spec.channel] = null
    return {
        "affected_observations": changed,
        "targeted_observations": targeted,
        "changed_observations": changed,
    }
