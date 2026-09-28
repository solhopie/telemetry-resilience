"""Target-process execution and expectation evaluation.

Shared by the `test` command (single scenario) and the `suite` command
(multi-case resilience suites) so both evaluate assertions identically.

Assertions:
  exit_code            expected process exit code (default 0)
  stdout_contains      strings that must appear on stdout
  stdout_not_contains  strings that must NOT appear on stdout
  stderr_contains      strings that must appear on stderr
  stderr_not_contains  strings that must NOT appear on stderr
  max_duration_seconds process must finish within this many seconds

A target that exceeds its timeout is a TEST FAILURE (not an internal
crash): the result is marked failed with reason "timeout".
"""
import subprocess
import time
from dataclasses import dataclass


@dataclass
class TargetResult:
    """Outcome of one target-process execution."""

    stdout: str
    stderr: str
    exit_code: int | None  # None when the process never produced one (timeout)
    duration_seconds: float
    timed_out: bool = False
    timeout_seconds: float = 0.0
    launch_error: str | None = None  # set when the target could not start


def launch_error_message(cmd: list, reason: str) -> str:
    """Clean, traceback-free message for a target that could not start."""
    return (
        "TARGET COULD NOT START\n"
        f"Command: {' '.join(cmd)}\n"
        f"Reason: {reason}"
    )


def run_target(cmd: list, timeout_seconds: float) -> TargetResult:
    """Run ``cmd`` (argument array, never shell=True) with a timeout.

    A target that cannot even start (missing executable, permission denied,
    other OS launch failures) is reported via ``launch_error`` instead of
    raising -- callers must treat that as an invalid target configuration
    (exit code 2), never as an ordinary resilience test failure.
    """
    t0 = time.perf_counter()
    try:
        # errors="replace": engineering targets and native binaries can
        # emit malformed UTF-8; decode it to U+FFFD instead of raising
        # UnicodeDecodeError (which would misreport a config error).
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - t0
        stdout = exc.stdout
        stderr = exc.stderr
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return TargetResult(
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=None,
            duration_seconds=duration,
            timed_out=True,
            timeout_seconds=float(timeout_seconds),
        )
    except OSError as exc:
        # NB: TimeoutExpired is an OSError subclass, so it is caught above.
        duration = time.perf_counter() - t0
        if isinstance(exc, FileNotFoundError):
            reason = "executable not found"
        elif isinstance(exc, PermissionError):
            reason = "permission denied"
        else:
            reason = str(exc) or type(exc).__name__
        return TargetResult(
            stdout="",
            stderr="",
            exit_code=None,
            duration_seconds=duration,
            timeout_seconds=float(timeout_seconds),
            launch_error=reason,
        )
    duration = time.perf_counter() - t0
    return TargetResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
        duration_seconds=duration,
        timeout_seconds=float(timeout_seconds),
    )


def evaluate_expectations(expect: dict | None, result: TargetResult) -> list:
    """Evaluate assertions against a target result.

    Returns a list of {"check", "passed", "detail"} dicts. The default
    expected exit code is 0 unless the expectations explicitly provide
    exit_code, so a crashing target can never silently pass. A timeout
    short-circuits to a single failing "timeout" check.
    """
    if result.timed_out:
        return [
            {
                "check": "timeout",
                "passed": False,
                "detail": (
                    f"target timed out after {result.timeout_seconds:g}s "
                    "(timeout_seconds exceeded)"
                ),
            }
        ]

    expect = expect or {}
    checks = []

    expected_exit = expect.get("exit_code", 0)
    checks.append(
        {
            "check": f"exit_code == {expected_exit}",
            "passed": result.exit_code == expected_exit,
            "detail": (
                f"expected exit_code {expected_exit}, got {result.exit_code}"
            ),
        }
    )

    for needle in expect.get("stdout_contains", []) or []:
        checks.append(
            {
                "check": f"stdout contains '{needle}'",
                "passed": needle in result.stdout,
                "detail": f"stdout did not contain {needle!r}",
            }
        )
    for needle in expect.get("stdout_not_contains", []) or []:
        checks.append(
            {
                "check": f"stdout not contains '{needle}'",
                "passed": needle not in result.stdout,
                "detail": f"stdout contained {needle!r} (expected absent)",
            }
        )
    for needle in expect.get("stderr_contains", []) or []:
        checks.append(
            {
                "check": f"stderr contains '{needle}'",
                "passed": needle in result.stderr,
                "detail": f"stderr did not contain {needle!r}",
            }
        )
    for needle in expect.get("stderr_not_contains", []) or []:
        checks.append(
            {
                "check": f"stderr not contains '{needle}'",
                "passed": needle not in result.stderr,
                "detail": f"stderr contained {needle!r} (expected absent)",
            }
        )

    max_duration = expect.get("max_duration_seconds")
    if max_duration is not None:
        checks.append(
            {
                "check": f"duration <= {max_duration:g}s",
                "passed": result.duration_seconds <= max_duration,
                "detail": (
                    f"duration {result.duration_seconds:.2f}s exceeded "
                    f"max_duration_seconds {max_duration:g}s"
                ),
            }
        )

    return checks
