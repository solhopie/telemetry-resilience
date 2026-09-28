"""CSV read/write with a JSON schema sidecar for dtype fidelity.

CSVs are plain text, so dtypes (nullable integers, timezones, float32, ...)
do not survive a bare round trip. Every CSV we write therefore gets a
sidecar file recording each column's dtype; on read the sidecar is used to
restore the original dtypes.
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

_CORRUPTED_SUFFIX = ".corrupted"
_TIMESTAMP_LIKE_NAMES = {"timestamp", "time", "datetime", "date"}


def sidecar_path(path: Path) -> Path:
    """Return the schema sidecar path OWNED by the CSV at ``path``.

    Every CSV file owns its own sidecar: ``<stem>.schema.json``. In
    particular a corrupted output ``data.corrupted.csv`` owns
    ``data.corrupted.schema.json`` -- it must never write into the input's
    ``data.schema.json`` (RUN 1.1 fix: writing the corrupted schema back to
    the input's sidecar mutated the original's metadata).
    """
    path = Path(path)
    return path.with_name(path.stem + ".schema.json")


def _legacy_sidecar_path(path: Path) -> Path:
    """Pre-1.1 sidecar location for corrupted outputs.

    Older versions wrote the corrupted CSV's schema to the input's sidecar
    (``data.schema.json`` for ``data.corrupted.csv``). Used ONLY as a
    read fallback when the output-specific sidecar is absent.
    """
    path = Path(path)
    stem = path.stem
    if stem.endswith(_CORRUPTED_SUFFIX):
        stem = stem[: -len(_CORRUPTED_SUFFIX)]
    return path.with_name(stem + ".schema.json")


def write(df: pd.DataFrame, path: PathLike) -> None:
    """Write ``df`` to CSV plus a ``.schema.json`` sidecar."""
    path = Path(path)
    df.to_csv(path, index=False)

    time_column = None
    for col in df.columns:
        if str(df[col].dtype).startswith("datetime64"):
            time_column = col
            break

    sidecar = {
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "time_column": time_column,
    }
    sidecar_path(path).write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )


def read(path: PathLike) -> pd.DataFrame:
    """Read a CSV, restoring dtypes from the schema sidecar when present."""
    path = Path(path)
    # Prefer this file's OWN sidecar; fall back to the legacy pre-1.1
    # location (the input's sidecar) only when the output-specific
    # sidecar does not exist.
    candidates = [sidecar_path(path)]
    legacy = _legacy_sidecar_path(path)
    if legacy != candidates[0]:
        candidates.append(legacy)

    df = pd.read_csv(path)

    schema = None
    for candidate in candidates:
        if candidate.exists():
            schema = json.loads(candidate.read_text(encoding="utf-8"))
            break

    if schema is not None:
        for col, dtype in schema.get("columns", {}).items():
            if col not in df.columns:
                continue
            try:
                if dtype.startswith("datetime64"):
                    # DEVIATION from the letter of the spec (see note below):
                    # the spec'd strict call is tried first; pandas >= 2 then
                    # applies the format inferred from the first element to
                    # every row, which fails on our own to_csv output whenever
                    # timestamps mix whole-second and sub-second resolutions
                    # (the normal case, e.g. 10 Hz telemetry). The fallback
                    # retries with per-element format inference. Genuine
                    # garbage still raises, preserving the mismatch error
                    # contract below.
                    utc = ", UTC" in dtype
                    try:
                        parsed = pd.to_datetime(df[col], utc=utc, errors="raise")
                    except ValueError:
                        parsed = pd.to_datetime(
                            df[col], utc=utc, errors="raise", format="mixed"
                        )
                    if str(parsed.dtype) != dtype:
                        # pandas >= 3 parses CSV datetimes at microsecond
                        # resolution; restore the exact unit recorded in the
                        # sidecar (e.g. datetime64[ns, UTC]).
                        parsed = parsed.astype(dtype)
                    df[col] = parsed
                elif dtype in ("Int64", "Int32", "boolean"):
                    df[col] = df[col].astype(dtype)
                elif dtype in ("float32", "float64", "int64", "int32", "str", "string"):
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                raise ValueError(
                    f"CSV sidecar schema mismatch for column '{col}': {e}"
                ) from e
    else:
        # Best effort: parse timestamp-like columns without a sidecar.
        for col in df.columns:
            if str(col).lower() in _TIMESTAMP_LIKE_NAMES:
                try:
                    df[col] = pd.to_datetime(df[col], utc=True)
                except Exception:
                    pass
    return df
