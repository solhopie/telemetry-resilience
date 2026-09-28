"""IO layer: read and write telemetry files in the supported formats."""

from pathlib import Path
from typing import Union

import pandas as pd

from . import csv_adapter, jsonl_adapter, parquet_adapter

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq", ".jsonl", ".ndjson"}

PathLike = Union[str, Path]


def _suffix_of(path: Path) -> str:
    return Path(path).suffix


def read_file(path: PathLike) -> pd.DataFrame:
    """Read a telemetry file into a DataFrame, dispatching on file suffix."""
    path = Path(path)
    suffix = _suffix_of(path)
    if suffix == ".csv":
        return csv_adapter.read(path)
    if suffix in (".parquet", ".pq"):
        return parquet_adapter.read(path)
    if suffix in (".jsonl", ".ndjson"):
        return jsonl_adapter.read(path)
    raise ValueError(f"unsupported file format: '{suffix}'")


def write_file(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to disk, dispatching on file suffix."""
    path = Path(path)
    suffix = _suffix_of(path)
    if suffix == ".csv":
        csv_adapter.write(df, path)
    elif suffix in (".parquet", ".pq"):
        parquet_adapter.write(df, path)
    elif suffix in (".jsonl", ".ndjson"):
        jsonl_adapter.write(df, path)
    else:
        raise ValueError(f"unsupported file format: '{suffix}'")


def extra_output_files(path: PathLike) -> list:
    """Additional files a write to ``path`` will create (for preflight).

    Currently this is the CSV schema sidecar. Returns an empty list for
    formats with no sidecars.
    """
    path = Path(path)
    if _suffix_of(path) == ".csv":
        return [csv_adapter.sidecar_path(path)]
    return []
