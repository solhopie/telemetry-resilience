"""Shared validation helpers for fault modules.

Kept separate from the fault modules themselves so each module stays small.
Nothing here imports engine or scenario, so there are no import cycles.
"""
import math
import re

from ..models import FaultError

_FREQUENCY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*hz\s*$", re.IGNORECASE)


def is_number(value):
    """True for finite int/float values (bools are excluded)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def reject_unknown(params):
    """Raise FaultError if any unrecognized parameter names remain."""
    if params:
        names = ", ".join(sorted(params))
        raise FaultError(f"unknown parameter(s): {names}")


def parse_hz(value, name):
    """Parse a frequency value to Hz (float); raises FaultError.

    Accepts a plain number (Hz) or a string like '100Hz' / '25 hz'.
    """
    if isinstance(value, bool):
        raise FaultError(f"{name} must be a positive frequency (got {value!r})")
    if isinstance(value, (int, float)):
        hz = float(value)
    elif isinstance(value, str):
        match = _FREQUENCY_RE.match(value)
        if not match:
            raise FaultError(
                f"{name} must be a positive frequency like '100Hz' (got {value!r})"
            )
        hz = float(match.group(1))
    else:
        raise FaultError(f"{name} must be a positive frequency (got {value!r})")
    if not math.isfinite(hz) or hz <= 0:
        raise FaultError(f"{name} must be > 0 Hz (got {value!r})")
    return hz
