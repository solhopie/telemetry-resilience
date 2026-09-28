"""Timestamp gap fault: drop every row whose timestamp falls in the window."""
from ._common import reject_unknown


def validate(params):
    reject_unknown(dict(params))
    return {}


def apply(df, mask, time_col, spec, rng):
    removed = int(mask.sum())
    df.drop(df.index[mask], inplace=True)
    return {"rows_removed": removed, "affected_observations": removed}
