"""JSON Lines read/write (one JSON object per line, ISO timestamps)."""

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

_TIMESTAMP_LIKE_NAMES = {"timestamp", "time", "datetime", "date"}


def read(path: PathLike) -> pd.DataFrame:
    """Read a JSONL file, best-effort parsing timestamp-like columns."""
    df = pd.read_json(Path(path), lines=True)
    for col in df.columns:
        if str(col).lower() in _TIMESTAMP_LIKE_NAMES:
            try:
                df[col] = pd.to_datetime(df[col], utc=True)
            except Exception:
                pass
    return df


def write(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to JSONL with ISO-formatted timestamps."""
    df.to_json(Path(path), orient="records", lines=True, date_format="iso")
