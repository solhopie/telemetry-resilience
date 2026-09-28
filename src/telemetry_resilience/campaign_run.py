"""Runner/reporting layer for telemetry mutation campaigns.

Wave 2 of the campaign feature. This module holds the CLI-facing entry
points:

* :func:`campaign_plan` -- validate and display the expansion (no targets
  executed, no files written);
* :func:`campaign_list_cases` -- list the expanded cases (no execution);
* :func:`run_campaign` -- execute the expanded cases through the EXISTING
  suite runner (:func:`telemetry_resilience.suite_run.run_suite`) and
  report per-cell coverage.

There is no second execution engine here: campaign cases expand to plain
suite cases (see :mod:`telemetry_resilience.campaign`) and the shared
:func:`~telemetry_resilience.suite_run.preflight_cases` is used for the
zero-execution preflight of plan/list/dry paths and real runs alike.

All public functions return process exit codes (0/1/2/3), print errors to
stderr, and never raise.
"""
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

from .campaign import (
    campaign_fingerprint,
    campaign_to_suite,
    effective_max_cases,
    expand_campaign,
    expanded_suite_yaml,
    format_duration,
    input_fingerprints,
    parse_campaign_config,
    validate_campaign_channels,
    _fmt_number,
)
from .io import read_file
from .models import ScenarioError
from .reporting import _md_cell, _md_text, write_summary_json
from .suite_run import (
    ARTIFACT_MARKER,
    _write_artifact_marker,
    preflight_cases,
    run_suite,
)


class _CampaignAbort(Exception):
    """Private control-flow exception: abort with an exit code + message."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Shared load pipeline
# ---------------------------------------------------------------------------


def _load(campaign_path, max_cases, channels, faults):
    """Run the shared campaign load pipeline.

    Returns ``(config, df, in_path, input_sha, sidecar_sha, cases,
    fingerprint, effective_max)``, where ``effective_max`` is the
    resolved case cap: CLI ``--max-cases`` > campaign YAML ``max_cases``
    > default. Raises :class:`_CampaignAbort` with an exit code and
    message on any problem; the public entry points catch it, print to
    stderr, and return the code.
    """
    base = Path(campaign_path).parent
    try:
        config = parse_campaign_config(campaign_path)
    except ScenarioError as exc:
        raise _CampaignAbort(2, str(exc))

    in_path = Path(config.input_file)
    if not in_path.is_absolute():
        in_path = base / in_path

    if not in_path.exists():
        raise _CampaignAbort(2, f"input file not found: {in_path}")
    try:
        df = read_file(in_path)
    except (FileNotFoundError, ValueError) as exc:
        raise _CampaignAbort(2, str(exc))
    except Exception as exc:  # noqa: BLE001 - mapped to the internal-error exit code
        raise _CampaignAbort(3, f"internal error reading {in_path}: {exc}")

    # Fingerprint the input BEFORE anything executes.
    input_sha, sidecar_sha = input_fingerprints(in_path)

    try:
        validate_campaign_channels(config, df)
    except ScenarioError as exc:
        raise _CampaignAbort(2, str(exc))

    try:
        effective_max = effective_max_cases(config, max_cases)
    except ScenarioError as exc:
        raise _CampaignAbort(2, str(exc))

    try:
        cases = expand_campaign(
            config,
            df,
            channels=tuple(channels),
            faults=tuple(faults),
            max_cases=max_cases,
        )
    except ScenarioError as exc:
        raise _CampaignAbort(2, str(exc))

    fingerprint = campaign_fingerprint(config, cases)
    return config, df, in_path, input_sha, sidecar_sha, cases, fingerprint, effective_max


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _filters_line(channels, faults) -> str:
    """Render the filters line; 'none' when no filters are set."""
    channels = list(channels)
    faults = list(faults)
    if not channels and not faults:
        return "Filters: none"
    return (
        f"Filters: channels=[{', '.join(channels)}] "
        f"faults=[{', '.join(faults)}]"
    )


def _group_cases(cases):
    """Group cases into [(label, [(fault, [cases])])] in expansion order.

    The label is the channel name, or ``"TIMELINE"`` for timeline cases;
    channel sections come first (expansion order), then timeline.
    """
    sections = []
    section_index = {}
    for case in cases:
        label = "TIMELINE" if case.target_kind == "timeline" else case.target
        if label not in section_index:
            section_index[label] = len(sections)
            sections.append((label, [], {}))
        _, fault_groups, fault_index = sections[section_index[label]]
        if case.fault_type not in fault_index:
            fault_index[case.fault_type] = len(fault_groups)
            fault_groups.append((case.fault_type, []))
        fault_groups[fault_index[case.fault_type]][1].append(case)
    return sections


def _param_value(key, value) -> str:
    """Compact display value for one id-param (matches case-ID style)."""
    if key == "duration":
        return format_duration(float(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _params_str(id_params: dict) -> str:
    """Compact ``k=v,k=v`` rendering of a case's id-params, keys sorted."""
    return ",".join(
        f"{key}={_param_value(key, id_params[key])}" for key in sorted(id_params)
    )


# ---------------------------------------------------------------------------
# campaign_plan / campaign_list_cases
# ---------------------------------------------------------------------------


def campaign_plan(campaign_path: str, *, max_cases=None, channels=(), faults=()) -> int:
    """Validate a campaign and print its expansion plan.

    Zero target execution, zero files written. Returns 0 on success, 2 on
    any configuration problem, 3 on internal error.
    """
    try:
        try:
            config, df, in_path, input_sha, _, cases, fingerprint, effective_max = (
                _load(campaign_path, max_cases, channels, faults)
            )
        except _CampaignAbort as exc:
            print(f"Error: {exc.message}", file=sys.stderr)
            return exc.code

        suite = campaign_to_suite(config, cases, str(in_path))
        problem = preflight_cases(df, suite)
        if problem is not None:
            print(
                f"Error: invalid case configuration: {problem}",
                file=sys.stderr,
            )
            return 2

        print("CAMPAIGN PLAN")
        print(f"Campaign: {config.name} ({fingerprint})")
        print(f"Input: {in_path} (time column: {config.time_column})")
        print(f"Input SHA-256: {input_sha}")
        print(_filters_line(channels, faults))
        print(f"Generated cases: {len(cases)}")
        print(f"Effective max cases: {effective_max}")
        for label, fault_groups, _ in _group_cases(cases):
            print(f"{label}:")
            for fault, group in fault_groups:
                print(f"  {fault}: {len(group)} case(s)")
        print("Cases:")
        for case in cases:
            print(f"  {case.name}  (seed {case.seed})")
        print("Configuration valid. No targets executed, no data written.")
        return 0
    except Exception as exc:  # noqa: BLE001 - public functions never raise
        print(f"Error: internal error: {exc}", file=sys.stderr)
        return 3


def campaign_list_cases(
    campaign_path: str, *, max_cases=None, channels=(), faults=()
) -> int:
    """List the expanded cases of a campaign, one line per case.

    No target execution. Returns 0 on success, 2 on any configuration
    problem, 3 on internal error.
    """
    try:
        try:
            config, df, in_path, _, _, cases, _, effective_max = _load(
                campaign_path, max_cases, channels, faults
            )
        except _CampaignAbort as exc:
            print(f"Error: {exc.message}", file=sys.stderr)
            return exc.code

        suite = campaign_to_suite(config, cases, str(in_path))
        problem = preflight_cases(df, suite)
        if problem is not None:
            print(
                f"Error: invalid case configuration: {problem}",
                file=sys.stderr,
            )
            return 2

        for case in cases:
            print(
                f"{case.name} | {case.target} | {case.fault_type} | "
                f"{_params_str(case.id_params)} | seed {case.seed}"
            )
        return 0
    except Exception as exc:  # noqa: BLE001 - public functions never raise
        print(f"Error: internal error: {exc}", file=sys.stderr)
        return 3


# ---------------------------------------------------------------------------
# Coverage computation / rendering
# ---------------------------------------------------------------------------


def _cell_status(executed: int, passed: int, failed: int) -> str:
    """Status of one coverage cell: pass|fail|partial|not_executed."""
    if executed == 0:
        return "not_executed"
    if failed == 0:
        return "pass"
    if passed == 0:
        return "fail"
    return "partial"


def build_coverage(
    config,
    cases,
    summary,
    fingerprint,
    input_sha,
    sidecar_sha,
    channels,
    faults,
    *,
    max_cases=None,
) -> dict:
    """Build the campaign coverage payload from a suite summary.json.

    ``summary`` is the parsed summary.json written by run_suite.
    Executed/pass/fail come from summary["cases"] (matched by case
    name); configured-but-never-executed cases (fail-fast skips,
    baseline-blocked campaigns) are counted as not_executed and NEVER
    as passing.

    ``max_cases`` is the effective case cap actually used for the run
    (CLI override > campaign YAML > default); it defaults to
    ``config.max_cases`` when not passed so the function stays safe
    standalone.
    """
    summary_cases = summary.get("cases") or []
    status_by_name = {
        item["name"]: item.get("status") for item in summary_cases
    }
    executed = len(summary_cases)
    passed = sum(1 for item in summary_cases if item.get("status") == "pass")
    failed = executed - passed
    configured = len(cases)
    not_executed = configured - executed

    baseline_summary = summary.get("baseline")
    if baseline_summary is None:
        baseline_status = "not_run"
    else:
        baseline_status = baseline_summary.get("status") or "not_run"

    def cell(cases_subset):
        cell_executed = sum(
            1 for case in cases_subset if case.name in status_by_name
        )
        cell_passed = sum(
            1
            for case in cases_subset
            if status_by_name.get(case.name) == "pass"
        )
        cell_failed = cell_executed - cell_passed
        return {
            "configured": len(cases_subset),
            "executed": cell_executed,
            "passed": cell_passed,
            "failed": cell_failed,
            "not_executed": len(cases_subset) - cell_executed,
            "status": _cell_status(cell_executed, cell_passed, cell_failed),
        }

    channels_cov = {}
    timeline_cov = {}
    for case in cases:
        if case.target_kind == "timeline":
            target = timeline_cov
            key = case.fault_type
        else:
            target = channels_cov.setdefault(case.channel, {})
            key = case.fault_type
        target.setdefault(key, []).append(case)
    channels_cov = {
        ch: {fault: cell(group) for fault, group in fdict.items()}
        for ch, fdict in channels_cov.items()
    }
    timeline_cov = {
        fault: cell(group) for fault, group in timeline_cov.items()
    }

    case_rows = []
    for case in cases:
        name_status = status_by_name.get(case.name)
        case_rows.append(
            {
                "id": case.name,
                "target": case.target,
                "target_kind": case.target_kind,
                "fault": case.fault_type,
                "parameters": dict(case.id_params),
                "seed": case.seed,
                "status": (
                    "pass"
                    if name_status == "pass"
                    else "fail"
                    if name_status is not None
                    else "not_executed"
                ),
            }
        )

    return {
        "version": 1,
        "campaign": config.name,
        "campaign_fingerprint": fingerprint,
        "input_sha256": input_sha,
        "input_sidecar_sha256": sidecar_sha,
        "filters": {"channels": list(channels), "faults": list(faults)},
        "max_cases": max_cases if max_cases is not None else config.max_cases,
        "configured_cases": configured,
        "executed_cases": executed,
        "passed_cases": passed,
        "failed_cases": failed,
        "not_executed_cases": not_executed,
        "configured_pass_rate": round(passed / configured, 4) if configured else 0.0,
        "baseline": {
            "enabled": bool(config.baseline_enabled),
            "status": baseline_status,
        },
        "channels": channels_cov,
        "timeline": timeline_cov,
        "cases": case_rows,
    }


def _coverage_cell_text(cell: dict) -> str:
    """Markdown cell text: '2/2 PASS', '1/2 FAIL', 'PARTIAL', ..."""
    status = cell["status"]
    if status == "pass":
        return f"{cell['passed']}/{cell['executed']} PASS"
    if status == "fail":
        return f"{cell['passed']}/{cell['executed']} FAIL"
    if status == "partial":
        return "PARTIAL"
    return "0/0 NOT EXECUTED"


def render_coverage_markdown(campaign_name, coverage) -> str:
    """Render the coverage payload as human-readable Markdown."""
    lines = ["# Telemetry Resilience Coverage", ""]
    lines.append(f"Campaign: {_md_text(campaign_name)}")
    lines.append(f"Fingerprint: {_md_text(coverage['campaign_fingerprint'])}")
    lines.append(f"Input SHA-256: {_md_text(coverage['input_sha256'])}")
    lines.append(
        _filters_line(
            coverage["filters"]["channels"], coverage["filters"]["faults"]
        )
    )
    lines.append("")
    lines.append(
        f"Configured cases: {coverage['configured_cases']} | "
        f"Executed: {coverage['executed_cases']} | "
        f"Passed: {coverage['passed_cases']} | "
        f"Failed: {coverage['failed_cases']} | "
        f"Not executed: {coverage['not_executed_cases']}"
    )
    lines.append(
        f"Configured pass rate: {coverage['configured_pass_rate']}"
    )
    lines.append("")

    fault_columns = []
    for _ch, fault_dict in coverage["channels"].items():
        for fault in fault_dict:
            if fault not in fault_columns:
                fault_columns.append(fault)
    if coverage["channels"]:
        lines.append(
            "| Channel | "
            + " | ".join(
                _md_cell(fault[:1].upper() + fault[1:])
                for fault in fault_columns
            )
            + " |"
        )
        lines.append("|---" + "|---" * len(fault_columns) + "|")
        for channel, fault_dict in coverage["channels"].items():
            cells = []
            for fault in fault_columns:
                if fault in fault_dict:
                    cells.append(_md_cell(_coverage_cell_text(fault_dict[fault])))
                else:
                    cells.append("—")
            lines.append(f"| {_md_cell(channel)} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("— = NOT CONFIGURED")
        lines.append("")

    if coverage["timeline"]:
        lines.append("Timeline:")
        lines.append("")
        lines.append("| Fault | Result |")
        lines.append("|---|---|")
        for fault, cell in coverage["timeline"].items():
            lines.append(
                f"| {_md_cell(fault)} | {_md_cell(_coverage_cell_text(cell))} |"
            )
        lines.append("")

    not_executed = [
        case for case in coverage["cases"] if case["status"] == "not_executed"
    ]
    if not_executed:
        lines.append("## Not executed")
        lines.append("")
        reason = (
            "baseline blocked"
            if coverage["baseline"]["status"] == "fail"
            else "fail-fast"
        )
        for case in not_executed:
            lines.append(f"- {_md_text(case['id'])} (reason: {reason})")
        lines.append("")

    lines.append(
        "*Coverage of the configured telemetry resilience campaign only. "
        "It does not certify safety or reliability.*"
    )
    lines.append("")
    return "\n".join(lines)


def render_coverage_csv(coverage) -> str:
    """Render the coverage payload as CSV, one row per (kind, fault) cell."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "target_kind",
            "target",
            "fault",
            "configured",
            "executed",
            "passed",
            "failed",
            "not_executed",
            "status",
        ]
    )
    for channel, fault_dict in coverage["channels"].items():
        for fault, cell in fault_dict.items():
            writer.writerow(
                [
                    "channel",
                    channel,
                    fault,
                    cell["configured"],
                    cell["executed"],
                    cell["passed"],
                    cell["failed"],
                    cell["not_executed"],
                    cell["status"],
                ]
            )
    for fault, cell in coverage["timeline"].items():
        writer.writerow(
            [
                "timeline",
                "timeline",
                fault,
                cell["configured"],
                cell["executed"],
                cell["passed"],
                cell["failed"],
                cell["not_executed"],
                cell["status"],
            ]
        )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# run_campaign
# ---------------------------------------------------------------------------


def _campaign_summary_header(
    campaign_name, fingerprint, input_sha, channels, faults, configured, executed
) -> str:
    """Header block prepended to summary.md (all user text sanitized)."""
    lines = [f"# Campaign: {_md_text(campaign_name)}", ""]
    lines.append(f"Fingerprint: {_md_text(fingerprint)}")
    lines.append(f"Input SHA-256: {_md_text(input_sha[:12])}")
    lines.append(_filters_line(channels, faults))
    lines.append(f"Configured cases: {configured} | Executed: {executed}")
    lines.append("")
    return "\n".join(lines)


def _postprocess_artifacts(
    config,
    cases,
    fingerprint,
    input_sha,
    sidecar_sha,
    channels,
    faults,
    artifacts_dir,
    effective_max,
) -> dict:
    """Add campaign artifacts (coverage, expanded suite, enriched reports).

    Reads summary.json written by run_suite, writes expanded-suite.yaml,
    coverage.json, coverage.md and coverage.csv, enriches summary.json
    with campaign metadata (including the effective max_cases used for
    the run), prepends a campaign header to summary.md, and refreshes
    the artifact ownership marker. OSError propagates to the caller
    (mapped to exit 3).
    """
    artifacts_dir = Path(artifacts_dir)
    summary_path = artifacts_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    coverage = build_coverage(
        config, cases, summary, fingerprint, input_sha, sidecar_sha,
        channels, faults, max_cases=effective_max,
    )

    (artifacts_dir / "expanded-suite.yaml").write_text(
        expanded_suite_yaml(config, cases, input_file_display=config.input_file),
        encoding="utf-8",
    )
    write_summary_json(artifacts_dir / "coverage.json", coverage)
    (artifacts_dir / "coverage.md").write_text(
        render_coverage_markdown(config.name, coverage), encoding="utf-8"
    )
    (artifacts_dir / "coverage.csv").write_text(
        render_coverage_csv(coverage), encoding="utf-8"
    )

    summary["campaign_fingerprint"] = fingerprint
    summary["campaign"] = {
        "name": config.name,
        "input_sha256": input_sha,
        "input_sidecar_sha256": sidecar_sha,
        "filters": {"channels": list(channels), "faults": list(faults)},
        "configured_cases": len(cases),
        "executed_cases": coverage["executed_cases"],
        "max_cases": effective_max,
    }
    write_summary_json(summary_path, summary)

    md_path = artifacts_dir / "summary.md"
    md_body = md_path.read_text(encoding="utf-8")
    md_path.write_text(
        _campaign_summary_header(
            config.name,
            fingerprint,
            input_sha,
            channels,
            faults,
            len(cases),
            coverage["executed_cases"],
        )
        + md_body,
        encoding="utf-8",
    )

    actual = sorted(
        child.name
        for child in artifacts_dir.iterdir()
        if child.name != ARTIFACT_MARKER
    )
    _write_artifact_marker(artifacts_dir, actual)
    return coverage


def run_campaign(
    campaign_path: str,
    target: list,
    *,
    artifacts=None,
    keep_data=False,
    fail_fast=False,
    overwrite_artifacts=False,
    max_cases=None,
    channels=(),
    faults=(),
) -> int:
    """Execute a campaign through the existing suite runner.

    Expands the campaign to a temporary suite YAML, runs it with
    :func:`telemetry_resilience.suite_run.run_suite` (which handles the
    baseline on a safe temp copy, per-case isolation, artifacts, JUnit
    and Ctrl+C), then post-processes the artifacts directory with
    campaign coverage reports. Returns the suite runner's exit code.
    """
    try:
        try:
            config, df, in_path, input_sha, sidecar_sha, cases, fingerprint, effective_max = _load(
                campaign_path, max_cases, channels, faults
            )
        except _CampaignAbort as exc:
            print(f"Error: {exc.message}", file=sys.stderr)
            return exc.code

        # Same {data} rule as the suite command.
        if "{data}" not in " ".join(target):
            print(
                "Error: target command must contain the {data} placeholder",
                file=sys.stderr,
            )
            return 2

        # Shared full fault-applicability preflight BEFORE anything
        # executes: an invalid campaign runs zero targets and writes zero
        # artifacts.
        suite = campaign_to_suite(config, cases, str(in_path))
        problem = preflight_cases(df, suite)
        if problem is not None:
            print(
                f"Error: invalid case configuration: {problem}",
                file=sys.stderr,
            )
            return 2

        print("TELEMETRY MUTATION CAMPAIGN")
        print(f"Campaign: {config.name} ({fingerprint})")
        print(f"Input: {in_path} (time column: {config.time_column})")
        print(f"Generated cases: {len(cases)}")
        print(f"Max cases (effective): {effective_max}")
        print(_filters_line(channels, faults))

        # The expanded suite lives in a TEMP file (never in the artifacts
        # dir); the input path is absolute so run_suite resolves it from
        # anywhere. run_suite re-parses the suite from this file.
        fd, temp_path = tempfile.mkstemp(
            prefix="telemetry_resilience_campaign_", suffix=".yaml"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    expanded_suite_yaml(
                        config,
                        cases,
                        input_file_display=str(in_path.absolute()),
                    )
                )
            code = run_suite(
                temp_path,
                target,
                artifacts=artifacts,
                keep_data=keep_data,
                fail_fast=fail_fast,
                overwrite_artifacts=overwrite_artifacts,
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        coverage = None
        if artifacts is not None and code in (0, 1):
            try:
                coverage = _postprocess_artifacts(
                    config,
                    cases,
                    fingerprint,
                    input_sha,
                    sidecar_sha,
                    channels,
                    faults,
                    artifacts,
                    effective_max,
                )
            except OSError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 3

        if coverage is not None:
            print("")
            print("CAMPAIGN COVERAGE")
            print(
                f"Configured: {coverage['configured_cases']} | "
                f"Executed: {coverage['executed_cases']} | "
                f"Passed: {coverage['passed_cases']} | "
                f"Failed: {coverage['failed_cases']} | "
                f"Not executed: {coverage['not_executed_cases']}"
            )
            print(
                f"Configured pass rate: {coverage['configured_pass_rate']}"
            )
        return code
    except Exception as exc:  # noqa: BLE001 - public functions never raise
        print(f"Error: internal error: {exc}", file=sys.stderr)
        return 3
