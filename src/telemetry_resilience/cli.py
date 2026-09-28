"""Command-line interface for telemetry-resilience.

Commands:
  inspect  print a human-readable summary of a telemetry file
  inject   inject deterministic faults from a YAML scenario
  test     run a target command against fault-injected data and check expectations
  suite    run a multi-case resilience suite (with baseline) against a target
  campaign expand a telemetry mutation campaign into deterministic
           single-fault cases and run them against a target
  doctor   check this environment can run telemetry-resilience (versions and
           capability checks; never executes user programs)
  demo     generate synthetic demo telemetry and scaffold a demo project
           (never executes user programs)

Options:
  --version  print the CLI version and exit

Exit codes:
  0  ok / all test expectations passed
  1  test expectations failed (or the target command timed out)
  2  invalid user or scenario input (bad scenario, missing file, bad fault,
     unsupported format, refusing to overwrite, target could not start,
     doctor checks failed)
  3  internal error (unexpected exception) or user interrupt
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, NoReturn, Optional

import pandas as pd
import typer

from .io import extra_output_files, read_file, write_file
from .scenario import load_scenario
from .engine import apply_scenario
from .manifest import build_manifest, build_report, write_json
from .models import ScenarioError, FaultError
from .runner import evaluate_expectations, launch_error_message, run_target
from .suite_run import run_suite
from .demo import (
    TEMPLATE_FILES,
    generate_demo_frame,
    read_template,
    write_demo_parquet,
)
from . import __version__

app = typer.Typer(
    help="Test how telemetry-driven software behaves when sensor data goes bad."
)


def _version_callback(value: Optional[bool]) -> None:
    if value:
        print(f"Telemetry Resilience CLI {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the CLI version and exit.",
    ),
) -> None:
    """Test how telemetry-driven software behaves when sensor data goes bad."""


def _fail(msg: str, code: int) -> NoReturn:
    print(f"Error: {msg}", file=sys.stderr)
    raise typer.Exit(code=code)


def _load_input(path: Path) -> pd.DataFrame:
    """Read a telemetry file, mapping failures onto exit codes 2 / 3."""
    if not path.exists():
        _fail(f"input file not found: {path}", 2)
    try:
        return read_file(path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc), 2)
    except Exception as exc:  # noqa: BLE001 - mapped to the internal-error exit code
        _fail(f"internal error reading {path}: {exc}", 3)
    raise RuntimeError("unreachable")


def _apply(df: pd.DataFrame, scenario) -> tuple:
    """Apply a scenario, mapping FaultError/ValueError to exit code 2."""
    try:
        return apply_scenario(df, scenario)
    except (FaultError, ValueError) as exc:
        _fail(str(exc), 2)
    except Exception as exc:  # noqa: BLE001 - mapped to the internal-error exit code
        _fail(f"internal error applying scenario: {exc}", 3)
    raise RuntimeError("unreachable")


def _write_outputs(
    df: pd.DataFrame,
    out_path: Path,
    faults_path: Path,
    report_path: Path,
    scenario,
    in_path: Path,
    entries: list,
) -> None:
    """Write the corrupted dataset, fault manifest, and run report."""
    try:
        write_file(df, out_path)
        write_json(faults_path, build_manifest(scenario, str(in_path), entries))
        write_json(
            report_path, build_report(scenario, str(in_path), str(out_path), entries)
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc), 2)
    except Exception as exc:  # noqa: BLE001 - mapped to the internal-error exit code
        _fail(f"internal error writing outputs: {exc}", 3)


@app.command("inspect")
def inspect_cmd(
    file: str = typer.Argument(
        ..., help="Telemetry file to inspect (CSV, Parquet, or JSONL)"
    ),
) -> None:
    """Print a human-readable summary of a telemetry file."""
    in_path = Path(file)
    df = _load_input(in_path)

    print(f"format: {in_path.suffix or '(none)'}")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    likely_ts = [
        str(col)
        for col in df.columns
        if str(df[col].dtype).startswith("datetime64")
        or any(key in str(col).lower() for key in ("time", "timestamp", "date"))
    ]
    if likely_ts:
        print(f"likely timestamp columns: [{', '.join(likely_ts)}]")
    else:
        print("likely timestamp columns: none")

    if likely_ts:
        ts = pd.to_datetime(df[likely_ts[0]], utc=True, errors="coerce")
        start, end = ts.min(), ts.max()
        print(f"start: {start.isoformat() if pd.notna(start) else 'n/a'}")
        print(f"end: {end.isoformat() if pd.notna(end) else 'n/a'}")
        if len(ts) > 1:
            intervals = ts.dropna().diff().dropna()
            if len(intervals):
                median_ms = intervals.median().total_seconds() * 1000
                hz = 1000.0 / median_ms if median_ms > 0 else float("inf")
                print(f"sampling: median interval {median_ms:.2f} ms (~{hz:.1f} Hz)")

    print("missing values:")
    for col in df.columns:
        print(f"  - {col}: {int(df[col].isna().sum())}")

    if likely_ts:
        ts = pd.to_datetime(df[likely_ts[0]], utc=True, errors="coerce")
        print(f"duplicate timestamps: {int(ts.duplicated().sum())}")
        out_of_order = int((ts.diff() < pd.Timedelta(0)).sum())
        print(f"out-of-order timestamps: {out_of_order}")
    else:
        print("duplicate timestamps: n/a")
        print("out-of-order timestamps: n/a")


@app.command("inject")
def inject_cmd(
    file: str = typer.Argument(..., help="Telemetry file to inject faults into"),
    scenario: str = typer.Option(
        ..., "--scenario", "-s", help="YAML scenario file"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for the corrupted dataset"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow overwriting existing output files"
    ),
) -> None:
    """Inject deterministic sensor faults from a scenario into telemetry data.

    Writes the corrupted dataset (default: <input>.corrupted<suffix>), the fault
    manifest (<input>.faults.json), and the run report (<input>.report.json)
    next to the input file.
    """
    try:
        sc = load_scenario(scenario)
    except ScenarioError as exc:
        _fail(str(exc), 2)

    in_path = Path(file)
    df = _load_input(in_path)
    df_out, entries = _apply(df, sc)

    stem = in_path.stem
    out_path = (
        Path(output) if output else in_path.with_name(f"{stem}.corrupted{in_path.suffix}")
    )
    faults_path = in_path.with_name(f"{stem}.faults.json")
    report_path = in_path.with_name(f"{stem}.report.json")

    # Preflight EVERYTHING the inject run may create: corrupted data, any
    # format sidecars (e.g. the CSV schema sidecar), the fault manifest,
    # and the report. Never overwrite any of them without --overwrite.
    planned = [out_path, *extra_output_files(out_path), faults_path, report_path]
    for p in planned:
        if p.resolve() == in_path.resolve() and not overwrite:
            _fail(
                f"refusing to overwrite original input {in_path} (use --overwrite)",
                2,
            )
        if p.exists() and not overwrite:
            _fail(f"output already exists: {p} (use --overwrite)", 2)

    _write_outputs(df_out, out_path, faults_path, report_path, sc, in_path, entries)

    total_affected = sum(
        int(entry.get("affected_observations", 0) or 0) for entry in entries
    )
    print(f"Applied {len(entries)} faults (seed {sc.seed})")
    print(f"Wrote: {out_path}")
    print(f"Manifest: {faults_path}")
    print(f"Report: {report_path}")
    print(f"Total affected observations: {total_affected}")


@app.command("test")
def test_cmd(
    scenario: str = typer.Argument(..., help="YAML scenario file"),
    target: List[str] = typer.Argument(
        ...,
        help="Target command to run; use {data} as placeholder for the corrupted dataset path",
    ),
) -> None:
    """Inject faults into a temp copy and run a target command against it.

    The target command comes only from the arguments after `--`; the corrupted
    dataset is written to a temporary directory that is cleaned up afterwards.
    """
    try:
        sc = load_scenario(scenario)
    except ScenarioError as exc:
        _fail(str(exc), 2)

    base = Path(scenario).parent
    in_path = Path(sc.input_file)
    if not in_path.is_absolute():
        in_path = base / in_path

    df = _load_input(in_path)
    df_out, _ = _apply(df, sc)

    if "{data}" not in " ".join(target):
        _fail("target command must contain the {data} placeholder", 2)

    tmpdir = tempfile.mkdtemp(prefix="telemetry_resilience_")
    try:
        tmp_file = Path(tmpdir) / f"corrupted{in_path.suffix}"
        try:
            write_file(df_out, tmp_file)
        except (FileNotFoundError, ValueError) as exc:
            _fail(str(exc), 2)
        except Exception as exc:  # noqa: BLE001
            _fail(f"internal error writing temp data: {exc}", 3)

        cmd = [t.replace("{data}", str(tmp_file)) for t in target]
        print(f"Executing: {' '.join(cmd)}")
        try:
            result = run_target(cmd, timeout_seconds=600)
        except KeyboardInterrupt:
            print(
                "\nInterrupted by user -- test aborted; "
                "temporary files cleaned up.",
                file=sys.stderr,
            )
            raise typer.Exit(code=3)
        if result.launch_error:
            print(
                launch_error_message(cmd, result.launch_error),
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
        if result.timed_out:
            print(
                f"Error: target command timed out after "
                f"{result.timeout_seconds:g}s",
                file=sys.stderr,
            )

        print("--- stdout ---")
        print(result.stdout, end="")
        print("--- stderr ---")
        print(result.stderr, end="")
        if result.timed_out:
            print("exit code: n/a (timed out)")
        else:
            print(f"exit code: {result.exit_code}")
        print(f"duration: {result.duration_seconds:.2f}s")

        checks = evaluate_expectations(sc.expect, result)
        for check in checks:
            print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']}")

        if all(check["passed"] for check in checks):
            print("TEST RESULT: PASS")
            return
        print("TEST RESULT: FAIL")
        raise typer.Exit(code=1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.command("suite")
def suite_cmd(
    suite: str = typer.Argument(..., help="Resilience suite YAML file"),
    artifacts: Optional[str] = typer.Option(
        None,
        "--artifacts",
        help="Write suite artifacts (summary.json, summary.md, junit.xml, "
        "per-case results) to this directory",
    ),
    keep_data: bool = typer.Option(
        False,
        "--keep-data",
        help="Keep per-case corrupted telemetry inside the artifacts directory",
    ),
    fail_fast: bool = typer.Option(
        False, "--fail-fast", help="Stop after the first failing case"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the suite without executing the target or writing data",
    ),
    overwrite_artifacts: bool = typer.Option(
        False,
        "--overwrite-artifacts",
        help="Allow replacing a previous telemetry-resilience run in the "
        "artifacts directory (only works on directories carrying the "
        "telemetry-resilience ownership marker)",
    ),
    target: List[str] = typer.Argument(
        ...,
        help="Target command to run; use {data} as placeholder for the "
        "dataset path",
    ),
) -> None:
    """Run a multi-case resilience suite against a target.

    The target command comes only from the arguments after `--`; suite YAML
    holds data and expectations and is never executed.
    """
    code = run_suite(
        suite,
        list(target),
        artifacts=artifacts,
        keep_data=keep_data,
        fail_fast=fail_fast,
        dry_run=dry_run,
        overwrite_artifacts=overwrite_artifacts,
    )
    if code != 0:
        raise typer.Exit(code=code)


@app.command("campaign")
def campaign_cmd(
    campaign: str = typer.Argument(
        ..., help="Campaign YAML file describing telemetry mutations to test"
    ),
    artifacts: Optional[str] = typer.Option(
        None,
        "--artifacts",
        help="Write campaign artifacts (expanded-suite.yaml, "
        "coverage.json/.md/.csv, summary.json/.md, junit.xml, per-case "
        "results) to this directory",
    ),
    keep_data: bool = typer.Option(
        False, "--keep-data", help="Keep per-case corrupted telemetry inside the "
        "artifacts directory"
    ),
    fail_fast: bool = typer.Option(
        False, "--fail-fast", help="Stop after the first failing case"
    ),
    overwrite_artifacts: bool = typer.Option(
        False,
        "--overwrite-artifacts",
        help="Allow replacing a previous telemetry-resilience run in the "
        "artifacts directory",
    ),
    max_cases: Optional[int] = typer.Option(
        None, "--max-cases", help="Override the campaign's max_cases limit"
    ),
    channel: List[str] = typer.Option(
        [],
        "--channel",
        help="Only generate cases for this channel (repeatable)",
    ),
    fault: List[str] = typer.Option(
        [],
        "--fault",
        help="Only generate cases for this fault type (repeatable)",
    ),
    plan: bool = typer.Option(
        False,
        "--plan",
        help="Validate, expand and preflight the campaign without executing "
        "any target",
    ),
    list_cases: bool = typer.Option(
        False,
        "--list-cases",
        help="Print generated case IDs (with parameters and seeds) and exit "
        "without executing",
    ),
    target: List[str] = typer.Argument(
        [],
        help="Target command to run; use {data} as placeholder for the "
        "dataset path (required unless --plan or --list-cases is given)",
    ),
) -> None:
    """Expand a telemetry mutation campaign into deterministic single-fault cases and run them against a target. The target command comes only from the arguments after `--`; campaign YAML holds data and expectations and is never executed."""
    if plan and list_cases:
        print(
            "Error: --plan and --list-cases are mutually exclusive",
            file=sys.stderr,
        )
        raise typer.Exit(2)
    if not plan and not list_cases and not target:
        print(
            "Error: target command is required "
            "(or use --plan / --list-cases)",
            file=sys.stderr,
        )
        raise typer.Exit(2)
    if max_cases is not None and (
        isinstance(max_cases, bool)
        or not isinstance(max_cases, int)
        or max_cases <= 0
    ):
        print("Error: --max-cases must be a positive integer", file=sys.stderr)
        raise typer.Exit(2)
    # Imported here (not at module top) so the CLI keeps working even if the
    # campaign runner module is unavailable; the campaign command is the only
    # one that needs it.
    from .campaign_run import campaign_list_cases, campaign_plan, run_campaign

    if plan:
        code = campaign_plan(
            campaign,
            max_cases=max_cases,
            channels=tuple(channel),
            faults=tuple(fault),
        )
    elif list_cases:
        code = campaign_list_cases(
            campaign,
            max_cases=max_cases,
            channels=tuple(channel),
            faults=tuple(fault),
        )
    else:
        code = run_campaign(
            campaign,
            list(target),
            artifacts=artifacts,
            keep_data=keep_data,
            fail_fast=fail_fast,
            overwrite_artifacts=overwrite_artifacts,
            max_cases=max_cases,
            channels=tuple(channel),
            faults=tuple(fault),
        )
    if code != 0:
        raise typer.Exit(code=code)


@app.command("doctor")
def doctor_cmd() -> None:
    """Check this environment can run telemetry-resilience.

    Prints the CLI version, interpreter/OS, and dependency versions, then
    runs capability checks (CSV/JSONL/Parquet round-trips, temp directory,
    subprocess launch). Only temporary files are written and they are
    cleaned up; no user programs are executed and no network is used.
    Exits 0 when ready, 2 when any check fails.
    """
    # Imported lazily (not at module top) so that `--version` and `--help`
    # keep working even when an optional dependency such as pyarrow is
    # missing or broken; the doctor command is the only one that needs it.
    from .doctor import run_doctor

    raise typer.Exit(code=run_doctor())


demo_app = typer.Typer(
    help="Generate synthetic demo telemetry and scaffold a demo project. "
    "Never executes user programs; demo data is synthetic, non-authoritative."
)
app.add_typer(demo_app, name="demo")


@demo_app.command("generate")
def demo_generate_cmd(
    output: str = typer.Argument(
        "demo_drive.parquet",
        help="Output path for the generated synthetic demo telemetry "
        "(default: demo_drive.parquet in the current directory).",
    ),
    seconds: float = typer.Option(
        300.0, "--seconds", help="Duration of the synthetic telemetry, in seconds."
    ),
    hz: float = typer.Option(
        10.0, "--hz", help="Sample rate of the synthetic telemetry, in Hz."
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Allow overwriting the output file if it already exists.",
    ),
) -> None:
    """Generate synthetic, non-authoritative demo telemetry locally.

    Writes a Parquet file with a clean baseline (the demo app reports
    NOMINAL on it). Never executes user programs. Refuses to overwrite an
    existing output file unless --overwrite is given.
    """
    out_path = Path(output)
    if out_path.exists() and not overwrite:
        _fail(f"refusing to overwrite existing file: {out_path} (use --overwrite)", 2)
    df = generate_demo_frame(seconds=seconds, hz=hz)
    try:
        out = write_demo_parquet(df, output)
    except ValueError as exc:
        # Missing/broken PyArrow (or another bad output): a clean exit-2
        # error, never a traceback.
        _fail(str(exc), 2)
    print(
        "Generated synthetic, non-authoritative demo telemetry "
        "(for fault-injection demos only)."
    )
    print(f"Wrote {out} ({len(df)} rows)")
    print("Next steps:")
    print(f"  telemetry-resilience inspect {out}")
    print("  telemetry-resilience demo init   # scaffold a full demo project")


@demo_app.command("init")
def demo_init_cmd(
    dir: str = typer.Argument(
        "./telemetry-demo",
        help="Directory to create for the demo project "
        "(default: ./telemetry-demo).",
    ),
) -> None:
    """Scaffold a demo project: telemetry, campaigns, demo app and README.

    Creates DIR with demo_drive.parquet (generated locally), plus
    navigation_campaign.yaml, degraded_navigation.yaml, demo_app.py and
    README.md copied from the package's bundled templates.
    Never executes user programs.
    """
    target = Path(dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(f"could not create directory {target}: {exc}", 2)

    planned = [target / "demo_drive.parquet"] + [
        target / name for name in TEMPLATE_FILES
    ]
    for p in planned:
        if p.exists():
            _fail(f"refusing to overwrite existing file: {p}", 2)

    df = generate_demo_frame()
    try:
        write_demo_parquet(df, target / "demo_drive.parquet")
    except ValueError as exc:
        # Missing/broken PyArrow: a clean exit-2 error, never a traceback.
        _fail(str(exc), 2)
    for name in TEMPLATE_FILES:
        (target / name).write_bytes(read_template(name))

    print(
        "Created demo project in "
        f"{target} (synthetic, non-authoritative telemetry)."
    )
    print("Files:")
    print("  demo_drive.parquet        synthetic demo telemetry (clean baseline)")
    print("  navigation_campaign.yaml  17-case navigation mutation campaign")
    print("  degraded_navigation.yaml  single-scenario inject demo (5 faults, expects DEGRADED_MODE)")
    print("  demo_app.py               demo consumer app (prints NOMINAL / DEGRADED_MODE)")
    print("  README.md                 quickstart guide")
    run_command = (
        'telemetry-resilience campaign navigation_campaign.yaml '
        '--artifacts resilience-results -- python demo_app.py "{data}"'
    )
    print("Validate the campaign without executing anything:")
    print("  telemetry-resilience campaign navigation_campaign.yaml --plan")
    print("Inject the single-scenario demo (writes demo_drive.corrupted.parquet):")
    print("  telemetry-resilience inject demo_drive.parquet --scenario degraded_navigation.yaml")
    print("Run the campaign against the demo app (from the project directory):")
    print(f"  cd {dir}")
    print(f"  {run_command}")


if __name__ == "__main__":
    app()
