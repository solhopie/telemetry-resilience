"""Fault type modules.

Each module exposes:
  validate(params: dict) -> dict  -- validate raw params, return normalized
                                     params, raise FaultError on bad config.
  apply(df, mask, time_col, spec, rng) -> dict -- mutate df in place; the
                                     returned dict must include
                                     "affected_observations": int.
"""
from . import (
    bias,
    clipping,
    drift,
    dropout,
    freeze,
    noise,
    sample_rate,
    scale,
    spike,
    timestamp_gap,
    timestamp_jitter,
)

FAULT_TYPES = {
    "dropout": dropout,
    "freeze": freeze,
    "spike": spike,
    "bias": bias,
    "drift": drift,
    "noise": noise,
    "clipping": clipping,
    "timestamp_gap": timestamp_gap,
    "timestamp_jitter": timestamp_jitter,
    "sample_rate": sample_rate,
    "scale": scale,
}

__all__ = [
    "FAULT_TYPES",
    "dropout",
    "freeze",
    "spike",
    "bias",
    "drift",
    "noise",
    "clipping",
    "timestamp_gap",
    "timestamp_jitter",
    "sample_rate",
    "scale",
]
