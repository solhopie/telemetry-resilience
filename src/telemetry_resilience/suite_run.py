"""Orchestration for `telemetry-resilience suite`.

Runs a validated Suite against a target command:

* full in-memory fault-applicability preflight for every case BEFORE
  anything executes (an invalid suite runs zero targets and writes zero
  artifacts -- same implementation as --dry-run);
* optional baseline run against a TEMPORARY byte-for-byte copy of the
  input first (the target never sees the user's original file) -- a
  baseline failure marks the suite INVALID/BLOCKED and stops everything;
* each case injects its faults from the ORIGINAL input (never chained),
  runs the target with ``{data}`` replaced by the corrupted dataset path,
  and evaluates the case's expectations;
* optional artifacts directory with summary.json / summary.md / junit.xml,
  per-case manifests and logs, and --keep-data corrupted telemetry;
* --fail-fast, per-case timeout overrides, --dry-run, clean Ctrl+C.

Only the target command from the CLI's ``-- <command>`` arguments is ever
executed, via subprocess argument arrays (never shell=True). Suite YAML
holds data and expectations only.

Artifact-directory safety: by default a non-empty --artifacts directory is
refused (it may hold a previous run or unrelated data). --overwrite-artifacts
only replaces entries recorded in the version-2 ownership marker's
``owned_entries`` list; untracked user files cause a refusal, legacy v1
markers are never destructively cleaned, and unowned directories are never
touched. Case/baseline subdirectories are resolved before writing and any
path escaping the artifact root (or a pre-existing symlink) is refused.
"""
import contextlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from . import __version__
from . import runner as _runner
from .engine import apply_scenario
from .io import read_file, write_file
from .manifest import build_manifest, write_json
from .models import FaultError, ScenarioError
from .reporting import (
    render_junit_xml,
    render_summary_markdown,
    write_summary_json,
)
from .suite import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_MARKER,
    Suite,
    load_suite,
    safe_case_dirname,
    verify_suite_channels,
)

_CASE_TMP_PREFIX = "telemetry_resilience_case_"
_BASELINE_TMP_PREFIX = "telemetry_resilience_baseline_"


@contextlib.contextmanager
def temp_case_dir():
    """Yield a temp dir that is removed afterwards, even on interrupt."""
    path = Path(tempfile.mkdtemp(prefix=_CASE_TMP_PREFIX))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def temp_baseline_input(in_path: Path):
    """Yield a byte-for-byte temporary copy of the original input file.

    The baseline target receives the COPY's path, so a misbehaving target
    can never modify or destroy the user's original telemetry. For CSV
    the exact schema sidecar is copied alongside under the corresponding
    filename. The copy is byte-for-byte (shutil.copy2) -- the data is
    never deserialized/reserialized. The temp directory is removed
    afterwards on every path: success, failure, timeout, launch error,
    Ctrl+C, and unexpected exceptions.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=_BASELINE_TMP_PREFIX))
    try:
        dest = tmpdir / in_path.name
        shutil.copy2(in_path, dest)
        if in_path.suffix == ".csv":
            from .io import csv_adapter

            src_sidecar = csv_adapter.sidecar_path(in_path)
            if src_sidecar.exists():
                shutil.copy2(src_sidecar, csv_adapter.sidecar_path(dest))
        yield dest
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_and_evaluate(cmd, expect, timeout_seconds: float) -> dict:
    """Execute the target and evaluate expectations.

    Never raises on a test failure. ``launch_error`` is set when the target
    could not even start -- callers must treat that as an invalid target
    configuration (exit 2), never as an ordinary resilience failure.
    """
    result = _runner.run_target(cmd, timeout_seconds)
    if result.launch_error:
        return {
            "launch_error": result.launch_error,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration_seconds": result.duration_seconds,
            "timed_out": False,
            "checks": [],
            "passed": False,
            "failures": [],
        }
    checks = _runner.evaluate_expectations(expect, result)
    return {
        "launch_error": None,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "timed_out": result.timed_out,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "failures": [c["detail"] for c in checks if not c["passed"]],
    }


def _print_run_result(res: dict, label: str) -> None:
    print(f"exit code: {res['exit_code']}")
    print(f"duration: {res['duration_seconds']:.2f}s")
    for check in res["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']}")


def _print_dry_run_listing(suite: Suite, in_path: Path, target: list) -> None:
    print(f"Suite: {suite.name} (dry run)")
    print(f"Input: {in_path} (time column: {suite.time_column})")
    print(f"Target: {' '.join(target)}")
    print(f"Default timeout: {suite.timeout_seconds:g}s")
    print(f"Baseline: {'enabled' if suite.baseline_enabled else 'disabled'}")
    print("")
    print("Cases:")
    for case in suite.cases:
        timeout = case.timeout_seconds or suite.timeout_seconds
        kinds = sorted({f.type for f in case.faults})
        print(
            f"  - {case.name} (seed {case.seed}): "
            f"{len(case.faults)} fault(s) [{', '.join(kinds)}], "
            f"timeout {timeout:g}s"
        )
    print("")


def preflight_cases(df, suite: Suite) -> str | None:
    """Validate every case's faults against the real input, in memory.

    Builds each case's Scenario and applies it to the original dataframe,
    catching dtype incompatibility, unsafe casts, invalid windows and any
    other FaultError/ValueError/ScenarioError the fault engine would raise
    on a real run. The generated data is discarded.

    Returns an error message when a case cannot actually be applied, else
    None. This ONE implementation is shared by --dry-run and by normal
    runs (which preflight before executing any target), so the two
    validation paths cannot drift apart. No target execution, no writes.
    """
    for case in suite.cases:
        scenario = case.to_scenario(suite.input_file, suite.time_column)
        try:
            apply_scenario(df, scenario)
        except (FaultError, ScenarioError, ValueError) as exc:
            return f"case '{case.name}': {exc}"
    return None


def _owned_entries(suite: Suite) -> list:
    """Root entries this suite run will create inside the artifacts dir."""
    entries = []
    if suite.baseline_enabled:
        entries.append("baseline")
    for case in suite.cases:
        entries.append(safe_case_dirname(case.name))
    entries.extend(["summary.json", "summary.md", "junit.xml"])
    return entries


def _read_artifact_marker(artifacts_dir: Path):
    """Return the parsed ownership marker, or None if absent/invalid.

    Accepts format versions 1 and 2. Version 2 markers must carry a valid
    ``owned_entries`` list; anything else is treated as "not ours".
    """
    try:
        data = json.loads(
            (artifacts_dir / ARTIFACT_MARKER).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("tool") != "telemetry-resilience":
        return None
    version = data.get("format_version")
    if version == 2:
        owned = data.get("owned_entries")
        if not isinstance(owned, list) or not all(
            isinstance(e, str) for e in owned
        ):
            return None
        return data
    if version == 1:
        return data
    return None


def _write_artifact_marker(artifacts_dir: Path, owned_entries: list) -> None:
    (artifacts_dir / ARTIFACT_MARKER).write_text(
        json.dumps(
            {
                "tool": "telemetry-resilience",
                "format_version": ARTIFACT_FORMAT_VERSION,
                "owned_entries": sorted(owned_entries),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _prepare_artifacts_dir(
    artifacts_dir: Path, overwrite: bool, owned_entries: list
) -> tuple | None:
    """Validate (and, if allowed, clean) the artifacts directory.

    Returns None on success, or (exit_code, message) on refusal/failure.

    Safety rules (RUN 2.2):
    * A non-empty directory without our marker is never touched.
    * A version-1 (legacy) marker cannot trigger destructive cleanup: we
      refuse and tell the user to choose a clean directory rather than
      guessing which files are ours.
    * With a version-2 marker, --overwrite-artifacts may delete/replace
      ONLY paths recorded in ``owned_entries``. If untracked user files
      exist alongside, the overwrite is REFUSED (exit 2) -- there is no
      force-delete flag.
    """
    if artifacts_dir.exists() and not artifacts_dir.is_dir():
        return (2, f"artifacts path {artifacts_dir} exists and is not a directory")
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return (3, f"cannot create artifacts directory {artifacts_dir}: {exc}")
    try:
        children = list(artifacts_dir.iterdir())
    except OSError as exc:
        return (3, f"cannot list artifacts directory {artifacts_dir}: {exc}")

    if not children:
        try:
            _write_artifact_marker(artifacts_dir, owned_entries)
        except OSError as exc:
            return (3, f"cannot write artifact ownership marker: {exc}")
        return None

    marker = _read_artifact_marker(artifacts_dir)
    if marker is None:
        return (
            2,
            f"refusing to use artifacts directory {artifacts_dir}: it contains "
            "data not created by telemetry-resilience; choose a clean "
            "directory",
        )
    if marker.get("format_version") == 1:
        return (
            2,
            f"refusing to use artifacts directory {artifacts_dir}: it was "
            "created by an older telemetry-resilience version and we cannot "
            "tell which files belong to it; choose a clean directory (your "
            "existing files are left untouched)",
        )
    if not overwrite:
        return (
            2,
            f"refusing to use artifacts directory {artifacts_dir}: it contains "
            "a previous telemetry-resilience run; choose a clean directory "
            "or pass --overwrite-artifacts",
        )
    # Owned (v2) run + explicit --overwrite-artifacts: refuse when untracked
    # user files are present, otherwise delete/replace ONLY recorded owned
    # entries (never follow symlinks -- unlink them instead).
    owned = set(marker.get("owned_entries") or [])
    untracked = sorted(
        c.name for c in children
        if c.name != ARTIFACT_MARKER and c.name not in owned
    )
    if untracked:
        return (
            2,
            f"refusing to overwrite artifacts directory {artifacts_dir}: it "
            f"contains untracked files not created by telemetry-resilience "
            f"({', '.join(untracked)}); move them elsewhere or choose "
            "another directory",
        )
    for child in children:
        if child.name == ARTIFACT_MARKER:
            continue
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            return (
                3,
                f"cannot clean previous artifacts in {artifacts_dir}: {exc}",
            )
    try:
        _write_artifact_marker(artifacts_dir, owned_entries)
    except OSError as exc:
        return (3, f"cannot write artifact ownership marker: {exc}")
    return None


def _resolve_artifact_subdir(root: Path, name: str) -> Path:
    """Resolve a case/baseline subdirectory, refusing escapes and symlinks."""
    candidate = root / name
    if candidate.is_symlink():
        raise ScenarioError(
            f"refusing to write artifacts: '{candidate}' is a symlink "
            "pointing outside the artifacts directory"
        )
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ScenarioError(
            f"refusing to write artifacts: '{candidate}' resolves outside "
            f"the artifacts directory {root}"
        )
    return candidate


def _baseline_payload(res: dict) -> dict:
    return {
        "status": "pass" if res["passed"] else "fail",
        "exit_code": res["exit_code"],
        "duration_seconds": round(res["duration_seconds"], 3),
        "failures": res["failures"],
    }


def _case_payload(case, res: dict, fault_count: int, baseline_res) -> dict:
    payload = {
        "name": case.name,
        "status": "pass" if res["passed"] else "fail",
        "seed": case.seed,
        "duration_seconds": round(res["duration_seconds"], 3),
        "fault_count": fault_count,
        "failures": res["failures"],
        "case_exit_code": res["exit_code"],
        "timed_out": res["timed_out"],
    }
    if baseline_res is not None:
        payload["baseline_exit_code"] = baseline_res["exit_code"]
        payload["baseline_duration_seconds"] = round(
            baseline_res["duration_seconds"], 3
        )
    return payload


def run_suite(
    suite_path: str,
    target: list,
    artifacts: str | None = None,
    keep_data: bool = False,
    fail_fast: bool = False,
    dry_run: bool = False,
    overwrite_artifacts: bool = False,
) -> int:
    """Run a resilience suite; return the process exit code (0/1/2/3)."""
    try:
        suite = load_suite(suite_path)
    except ScenarioError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    base = Path(suite_path).parent
    in_path = Path(suite.input_file)
    if not in_path.is_absolute():
        in_path = base / in_path

    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 2
    try:
        df = read_file(in_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - mapped to the internal-error exit code
        print(f"Error: internal error reading {in_path}: {exc}", file=sys.stderr)
        return 3

    try:
        verify_suite_channels(df, suite)
    except ScenarioError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if "{data}" not in " ".join(target):
        print(
            "Error: target command must contain the {data} placeholder",
            file=sys.stderr,
        )
        return 2

    # Full fault-applicability preflight for EVERY case, using the same
    # implementation as --dry-run. This runs BEFORE artifacts are prepared
    # and BEFORE the baseline or any case executes, so an invalid suite
    # can never execute the user's target program or leave artifacts
    # behind (RUN 2.2 fix).
    problem = preflight_cases(df, suite)
    if problem is not None:
        print(f"Error: invalid case configuration: {problem}", file=sys.stderr)
        return 2

    if dry_run:
        _print_dry_run_listing(suite, in_path, target)
        print("Dry run: configuration valid. Nothing executed, nothing written.")
        return 0

    artifacts_dir = Path(artifacts) if artifacts else None
    if artifacts_dir is not None:
        refusal = _prepare_artifacts_dir(
            artifacts_dir, overwrite_artifacts, _owned_entries(suite)
        )
        if refusal is not None:
            code, message = refusal
            print(f"Error: {message}", file=sys.stderr)
            return code
        # Resolve every case/baseline subdirectory BEFORE any target
        # executes, so a symlink or path escape can never receive output.
        try:
            baseline_dir = _resolve_artifact_subdir(artifacts_dir, "baseline")
            case_dirs = {
                case.name: _resolve_artifact_subdir(
                    artifacts_dir, safe_case_dirname(case.name)
                )
                for case in suite.cases
            }
        except ScenarioError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    print("TELEMETRY RESILIENCE")
    print("")
    print(f"Suite: {suite.name}")
    print("")

    try:
        # ---- baseline: target against a TEMPORARY UNMODIFIED COPY --------
        # The baseline target NEVER receives the user's original file: it
        # gets a byte-for-byte temp copy instead, so a misbehaving target
        # cannot modify or destroy the original telemetry (RUN 2.2 fix).
        # The temp path is never recorded in any report artifact.
        baseline_res = None
        if suite.baseline_enabled:
            print("Baseline")
            with temp_baseline_input(in_path) as baseline_data:
                cmd = [t.replace("{data}", str(baseline_data)) for t in target]
                print(f"Executing: {' '.join(cmd)}")
                baseline_res = _run_and_evaluate(
                    cmd, suite.baseline_expect, suite.timeout_seconds
                )
            if baseline_res["launch_error"]:
                print(
                    _runner.launch_error_message(
                        cmd, baseline_res["launch_error"]
                    ),
                    file=sys.stderr,
                )
                return 2
            _print_run_result(baseline_res, "baseline")
            print("PASS" if baseline_res["passed"] else "FAIL")
            if not baseline_res["passed"]:
                for detail in baseline_res["failures"]:
                    print(f"  - {detail}")
            print("")
            if artifacts_dir is not None:
                try:
                    baseline_dir.mkdir(parents=True, exist_ok=True)
                    (baseline_dir / "stdout.txt").write_text(
                        baseline_res["stdout"], encoding="utf-8"
                    )
                    (baseline_dir / "stderr.txt").write_text(
                        baseline_res["stderr"], encoding="utf-8"
                    )
                    write_json(
                        baseline_dir / "result.json",
                        _baseline_payload(baseline_res),
                    )
                except OSError as exc:
                    print(
                        f"Error: cannot write baseline artifacts: {exc}",
                        file=sys.stderr,
                    )
                    return 3
            if not baseline_res["passed"]:
                print("BASELINE FAILED")
                print(
                    "SUITE RESULT: BLOCKED "
                    "(baseline failed; fault cases not run)"
                )
                if artifacts_dir is not None:
                    _write_summary_artifacts(
                        artifacts_dir, suite, baseline_res, [], "blocked"
                    )
                return 1

        # ---- fault cases, each isolated from the ORIGINAL input ----------
        case_payloads = []
        print("Cases")
        print("")
        for case in suite.cases:
            scenario = case.to_scenario(suite.input_file, suite.time_column)
            df_out, entries = apply_scenario(df, scenario)
            timeout = case.timeout_seconds or suite.timeout_seconds
            case_dir = case_dirs[case.name] if artifacts_dir is not None else None

            with temp_case_dir() as tmpdir:
                tmp_file = tmpdir / f"corrupted{in_path.suffix}"
                try:
                    write_file(df_out, tmp_file)
                except ValueError as exc:
                    # e.g. Parquet output without a working PyArrow: a
                    # clean exit-2 error, never a traceback.
                    print(f"Error: {exc}", file=sys.stderr)
                    return 2
                cmd = [t.replace("{data}", str(tmp_file)) for t in target]
                print(f"Executing: {' '.join(cmd)}")
                res = _run_and_evaluate(cmd, case.expect, timeout)

            if res["launch_error"]:
                print(
                    _runner.launch_error_message(cmd, res["launch_error"]),
                    file=sys.stderr,
                )
                return 2

            status = "PASS" if res["passed"] else "FAIL"
            print(f"{status}  {case.name}")
            for detail in res["failures"]:
                print(f"  - {detail}")
            print("")

            if case_dir is not None:
                try:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    if keep_data:
                        write_file(
                            df_out, case_dir / f"corrupted{in_path.suffix}"
                        )
                    write_json(
                        case_dir / "faults.json",
                        build_manifest(scenario, suite.input_file, entries),
                    )
                    (case_dir / "stdout.txt").write_text(
                        res["stdout"], encoding="utf-8"
                    )
                    (case_dir / "stderr.txt").write_text(
                        res["stderr"], encoding="utf-8"
                    )
                    write_json(
                        case_dir / "result.json",
                        _case_payload(case, res, len(entries), baseline_res),
                    )
                except ValueError as exc:
                    # e.g. Parquet output without a working PyArrow: a
                    # clean exit-2 error, never a traceback.
                    print(f"Error: {exc}", file=sys.stderr)
                    return 2
                except OSError as exc:
                    print(
                        f"Error: cannot write artifacts for case "
                        f"'{case.name}': {exc}",
                        file=sys.stderr,
                    )
                    return 3

            case_payloads.append(
                _case_payload(case, res, len(entries), baseline_res)
            )
            if not res["passed"] and fail_fast:
                print("Stopping after the first failure (--fail-fast)")
                print("")
                break

        passed = sum(1 for p in case_payloads if p["status"] == "pass")
        failed = sum(1 for p in case_payloads if p["status"] != "pass")
        total = len(case_payloads)
        status = "pass" if failed == 0 else "fail"

        if artifacts_dir is not None:
            _write_summary_artifacts(
                artifacts_dir, suite, baseline_res, case_payloads, status
            )

        print("--------------------------------")
        print(f"{passed} passed")
        print(f"{failed} failed")
        print(f"{total} total")
        print("")
        print(f"SUITE RESULT: {'PASSED' if status == 'pass' else 'FAILED'}")
        return 0 if status == "pass" else 1
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user -- suite aborted; "
            "temporary files cleaned up.",
            file=sys.stderr,
        )
        return 3
    except (FaultError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            f"Error: cannot write suite artifacts: {exc}", file=sys.stderr
        )
        return 3
    except Exception as exc:  # noqa: BLE001 - mapped to the internal-error exit code
        print(f"Error: internal error running suite: {exc}", file=sys.stderr)
        return 3


def _write_summary_artifacts(
    artifacts_dir: Path,
    suite: Suite,
    baseline_res,
    case_payloads: list,
    status: str,
) -> None:
    """Write summary.json, summary.md and junit.xml.

    Filesystem errors propagate to the caller as OSError so they become a
    clean documented CLI error instead of a traceback.
    """
    passed = sum(1 for p in case_payloads if p["status"] == "pass")
    failed = sum(1 for p in case_payloads if p["status"] != "pass")
    baseline_payload = (
        _baseline_payload(baseline_res) if baseline_res is not None else None
    )
    payload = {
        "version": 1,
        "suite": suite.name,
        "software_version": __version__,
        "baseline": baseline_payload,
        "cases": case_payloads,
        "passed": passed,
        "failed": failed,
        "total": len(case_payloads),
        "status": status,
    }
    write_summary_json(artifacts_dir / "summary.json", payload)
    (artifacts_dir / "summary.md").write_text(
        render_summary_markdown(
            suite.name,
            payload["baseline"],
            case_payloads,
            passed,
            failed,
            len(case_payloads),
            status,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "junit.xml").write_text(
        render_junit_xml(suite.name, case_payloads, baseline_payload),
        encoding="utf-8",
    )
    # The run completed: refresh the ownership marker with the entries this
    # run actually created, so later --overwrite-artifacts decisions rest on
    # the true on-disk state rather than the pre-run expectation.
    actual = sorted(
        child.name
        for child in artifacts_dir.iterdir()
        if child.name != ARTIFACT_MARKER
    )
    _write_artifact_marker(artifacts_dir, actual)
