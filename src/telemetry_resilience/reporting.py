"""Artifact writers for `telemetry-resilience suite`.

Produces three CI-friendly reports inside the artifacts directory:

  summary.json  machine-readable suite result (deterministic fields only;
                no temporary absolute machine paths)
  summary.md    human-readable report for PR comments / bug reports
  junit.xml     JUnit-compatible XML so CI systems show cases as tests

XML is built with xml.etree.ElementTree (never string concatenation) so
user-provided text is escaped properly. No HTML dashboard is generated.
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def write_summary_json(path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")


# Valid XML 1.0 characters (https://www.w3.org/TR/xml/#charsets):
#   #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
# ElementTree escapes <, >, & and quotes for us, but it does NOT remove
# illegal control characters -- those would produce XML that cannot be
# parsed. Replace them with U+FFFD instead.
_XML_ILLEGAL_CHARS = re.compile(
    "[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]"
)


def _xml_text(value) -> str:
    """Make text safe for XML 1.0: forbidden code points become U+FFFD."""
    return _XML_ILLEGAL_CHARS.sub("\ufffd", str(value))


def _md_text(value) -> str:
    """Render a value as plain Markdown text: strip control characters."""
    return re.sub(r"[\x00-\x1f\x7f]", " ", str(value))


def _md_cell(value) -> str:
    """Render a value for a Markdown table cell (escape pipes)."""
    return _md_text(value).replace("|", "\\|")


def render_summary_markdown(
    suite_name: str,
    baseline: dict | None,
    cases: list,
    passed: int,
    failed: int,
    total: int,
    status: str,
) -> str:
    """Render the Markdown suite report.

    Case names and the suite name are sanitized so they cannot break the
    table structure or inject extra columns.
    """
    lines = ["# Telemetry Resilience Report", ""]
    lines.append(f"Suite: {_md_text(suite_name)}")
    lines.append("")
    if baseline is None:
        lines.append("Baseline: not run")
    else:
        lines.append(f"Baseline: {baseline['status'].upper()}")
    lines.append("")
    lines.append("| Case | Result | Faults | Duration |")
    lines.append("|---|---|---:|---:|")
    for case in cases:
        lines.append(
            f"| {_md_cell(case['name'])} | {case['status'].upper()} "
            f"| {case['fault_count']} | {case['duration_seconds']:.1f}s |"
        )
    lines.append("")
    failing = [c for c in cases if c["status"] != "pass"]
    if failing:
        lines.append("## Failures")
        lines.append("")
        for case in failing:
            lines.append(f"### {_md_text(case['name'])}")
            lines.append("")
            for detail in case.get("failures", []):
                lines.append(f"- {_md_text(detail)}")
            lines.append("")
    lines.append(f"{passed} passed, {failed} failed, {total} total")
    lines.append("")
    lines.append(f"SUITE RESULT: {status.upper()}")
    lines.append("")
    return "\n".join(lines)


def _combined_failure(testcase, failures: list) -> None:
    """Attach ONE <failure> element combining all failed expectations.

    One resilience case is one logical failing testcase; multiple failed
    expectations are combined into a single <failure> so CI viewers do not
    count one case as several failed tests. All text is XML-1.0 sanitized.
    """
    failures = [_xml_text(f) for f in failures]
    message = failures[0] if failures else "failed"
    failure = ET.SubElement(testcase, "failure", {"message": message})
    failure.text = "\n".join(failures)


def render_junit_xml(
    suite_name: str, cases: list, baseline: dict | None = None
) -> str:
    """Render JUnit-compatible XML; each resilience case is a test case.

    A baseline that blocked the suite is represented as a failing
    ``__baseline__`` testcase so CI report viewers never see a blocked
    suite as "0 tests, 0 failures". Cases that were never executed are
    not listed at all (never as passing). Every user/process-derived
    string (suite name, case names, failure details) is sanitized so the
    output is always valid XML 1.0, even with control characters or
    non-UTF8-decoded process output in the text.
    """
    extra = (
        1
        if baseline is not None and baseline.get("status") != "pass"
        else 0
    )
    total = len(cases) + extra
    failures = sum(1 for c in cases if c["status"] != "pass") + extra
    testsuite = ET.Element(
        "testsuite",
        {
            "name": _xml_text(suite_name),
            "tests": str(total),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
        },
    )
    if extra:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            {
                "classname": "telemetry_resilience",
                "name": "__baseline__",
            },
        )
        details = list(baseline.get("failures") or ["baseline failed"])
        _combined_failure(testcase, [f"baseline: {d}" for d in details])
    for case in cases:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            {
                "classname": "telemetry_resilience",
                "name": _xml_text(case["name"]),
                "time": f"{case['duration_seconds']:.3f}",
            },
        )
        failures_list = list(case.get("failures", []))
        if case["status"] != "pass":
            _combined_failure(testcase, failures_list)
    ET.indent(testsuite)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        testsuite, encoding="unicode"
    )
