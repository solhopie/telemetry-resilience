"""Parquet read/write via pyarrow (exact dtype round-trip)."""

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def _require_pyarrow() -> None:
    """Fail fast with a clean, user-facing error when PyArrow is unusable.

    Raises a :class:`ValueError` (mapped to CLI exit code 2 by every caller)
    instead of letting an ImportError/traceback escape: a missing PyArrow
    is a dependency problem, not an internal error.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ValueError("Parquet support requires PyArrow.") from exc
    except Exception as exc:  # noqa: BLE001 - broken install, surfaced cleanly
        raise ValueError(
            "PyArrow is installed but broken and cannot be used for Parquet: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def read(path: PathLike) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    _require_pyarrow()
    return pd.read_parquet(Path(path), engine="pyarrow")


def write(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to Parquet."""
    _require_pyarrow()
    df.to_parquet(Path(path), engine="pyarrow", index=False)
