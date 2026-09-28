"""Core data models for the telemetry resilience engine.

FaultError lives here (rather than in engine.py) so the fault modules can
import it without creating a circular import: engine -> faults -> models.
"""
from dataclasses import dataclass
from typing import List, Optional


class ScenarioError(ValueError):
    """Raised for invalid scenario configuration (bad YAML, bad params)."""


class FaultError(ValueError):
    """Raised when a fault cannot be applied safely to the data.

    The engine refuses to silently corrupt data (e.g. assigning NaN into a
    numpy int64 column, or upcasting dtypes); instead it raises FaultError
    explaining what cast the user must request explicitly.
    """


@dataclass
class FaultSpec:
    channel: Optional[str]
    type: str
    start_seconds: float
    duration_seconds: float
    params: dict
    cast: Optional[str] = None


@dataclass
class Scenario:
    version: int
    seed: int
    input_file: str
    time_column: str
    faults: List[FaultSpec]
    expect: Optional[dict] = None
