"""Environment health checks for the telemetry-resilience CLI.

Everything here is read-only apart from temporary files, which are always
cleaned up. No network requests, no user data is touched, and no user
programs are executed.

All third-party imports are lazy and best-effort (via importlib): a
missing or broken dependency is reported as MISSING/BROKEN, never as a
traceback. This module must stay importable even when every optional
dependency is absent, so it imports nothing but the standard library (and
the local ``__version__``) at module level.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__

# (display label, distribution name for metadata fallback, import name)
_COMPONENTS = (
    ("pandas", "pandas", "pandas"),
    ("numpy", "numpy", "numpy"),
    ("pyarrow", "pyarrow", "pyarrow"),
    ("pyyaml", "PyYAML", "yaml"),
    ("typer", "typer", "typer"),
)


def _lazy_import(name: str):
    """Import ``name`` best-effort.

    Returns ``(module_or_None, state, detail)`` where ``state`` is ``"ok"``,
    ``"missing"`` (an ImportError: not installed) or ``"broken"`` (the
    import raised something else: installed but unusable).
    """
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        return None, "missing", f"{name} is not installed ({exc})"
    except Exception as exc:  # noqa: BLE001 - broken install, reported not raised
        return None, "broken", (
            f"{name} is installed but cannot be imported: "
            f"{type(exc).__name__}: {exc}"
        )
    return module, "ok", ""


def _component_version(dist_name: str, module) -> str:
    """Best-effort version string: module ``__version__``, then dist metadata."""
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _check_dependencies():
    """Probe each reported dependency; return ``[(label, state, info)]``.

    ``state`` is ``"ok"`` / ``"missing"`` / ``"broken"``; ``info`` is the
    version string when ok, otherwise a human-readable detail.
    """
    results = []
    for label, dist_name, module_name in _COMPONENTS:
        module, state, detail = _lazy_import(module_name)
        info = _component_version(dist_name, module) if state == "ok" else detail
        results.append((label, state, info))
    return results


def print_versions():
    """Print the CLI version, interpreter, OS, and dependency versions.

    Each dependency prints ``<version> OK``, ``MISSING``, or ``BROKEN``.
    Returns the per-dependency ``[(label, state, info)]`` results.
    """
    print(f"Telemetry Resilience CLI {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"os: {platform.system()} {platform.machine()}")
    results = _check_dependencies()
    for label, state, info in results:
        if state == "ok":
            print(f"{label}: {info} OK")
        elif state == "missing":
            print(f"{label}: MISSING")
        else:
            print(f"{label}: BROKEN ({info})")
    return results


def _roundtrip(tmpdir: Path, suffix: str) -> None:
    """Write a small frame with the package IO adapters and read it back.

    Only called when pandas (and, for Parquet, pyarrow) imported cleanly.
    """
    # Local import: the io package pulls in pandas eagerly, so it must not
    # be imported at this module's top level.
    from .io import read_file, write_file

    pd = importlib.import_module("pandas")
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "value": [1.5, 2.5],
            "label": ["a", "b"],
        }
    )
    path = tmpdir / f"roundtrip{suffix}"
    write_file(df, path)
    back = read_file(path)
    if len(back) != len(df):
        raise AssertionError(
            f"row count changed on round-trip: {len(df)} -> {len(back)}"
        )
    if list(back.columns) != list(df.columns):
        raise AssertionError(
            f"columns changed on round-trip: {list(df.columns)} -> "
            f"{list(back.columns)}"
        )


def _missing_pandas() -> None:
    raise AssertionError("pandas is not installed")


def _missing_pyarrow() -> None:
    raise AssertionError("pyarrow is not installed (Parquet support requires PyArrow)")


def _temp_writable(tmpdir: Path) -> None:
    """A temporary directory can be created and written to."""
    probe = tmpdir / "writable.probe"
    probe.write_text("ok")
    if probe.read_text() != "ok":
        raise AssertionError("could not read back probe file")
    probe.unlink()


def _subprocess_ok() -> None:
    """The current interpreter can be launched as a subprocess."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "pass"],
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as the check detail
        raise AssertionError(f"could not launch subprocess: {exc}") from exc
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess exited with code {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )


def run_doctor() -> int:
    """Print versions, run capability checks, return 0 (READY) or 2 (NOT READY).

    Never raises for a missing or broken dependency: every problem is
    reported as a failed check or a failed dependency, and the summary
    names exactly what failed.
    """
    dep_results = print_versions()

    failures: list = []
    dep_ok = {}
    for label, state, info in dep_results:
        dep_ok[label] = state == "ok"
        if state == "missing":
            failures.append((f"dependency {label}", "not installed"))
        elif state == "broken":
            failures.append((f"dependency {label}", info))

    tmpdir = Path(tempfile.mkdtemp(prefix="telemetry_resilience_doctor_"))
    try:
        checks: list = []
        if dep_ok.get("pandas"):
            checks.append(("csv round-trip", lambda: _roundtrip(tmpdir, ".csv")))
            checks.append(("jsonl round-trip", lambda: _roundtrip(tmpdir, ".jsonl")))
            if dep_ok.get("pyarrow"):
                checks.append(
                    ("parquet round-trip", lambda: _roundtrip(tmpdir, ".parquet"))
                )
            else:
                checks.append(("parquet round-trip", _missing_pyarrow))
        else:
            checks.append(("csv round-trip", _missing_pandas))
            checks.append(("jsonl round-trip", _missing_pandas))
            checks.append(("parquet round-trip", _missing_pandas))
        checks.append(("temp directory writable", lambda: _temp_writable(tmpdir)))
        checks.append(("subprocess execution", _subprocess_ok))

        for name, fn in checks:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                print(f"check: {name}: FAIL")
                failures.append((name, str(exc) or type(exc).__name__))
            else:
                print(f"check: {name}: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if failures:
        print("STATUS: NOT READY")
        for name, detail in failures:
            print(f"failed: {name}: {detail}")
        return 2
    print("STATUS: READY")
    return 0
