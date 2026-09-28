"""Regression tests for RUN 2.2: preflight, original-data protection & CI
report hardening.

Covers: preflight-before-any-execution, baseline temp-copy protection of
the original file (+ CSV sidecar), XML 1.0-safe JUnit, non-UTF8 target
output, untracked-file protection in owned artifact dirs, marker write
order. All pre-existing tests must keep passing unchanged.
"""
import hashlib
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import telemetry_resilience.runner as runner_module
from telemetry_resilience.cli import app
from telemetry_resilience.io import write_file
from telemetry_resilience.reporting import render_junit_xml

from .conftest import write_yaml

runner = CliRunner()


# ---------------------------------------------------------------------------
# helpers (mirroring tests/test_run21_regressions.py, kept local)
# ---------------------------------------------------------------------------

def _setup(tmp_path, telemetry_df, name="data.parquet"):
    data = tmp_path / name
    write_file(telemetry_df, data)
    return data


def _target(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def _fault(ftype, channel, start, duration, **params):
    d = {"type": ftype, "start": start, "duration": duration}
    if channel is not None:
        d["channel"] = channel
    d.update(params)
    return d


def _case(name, seed, faults, expect=None, timeout=None):
    d = {"name": name, "seed": seed, "faults": faults}
    if expect is not None:
        d["expect"] = expect
    if timeout is not None:
        d["timeout_seconds"] = timeout
    return d


def _suite_dict(cases, baseline=True, name="demo-suite", timeout=None,
                baseline_expect=None, extra_top=None, input_file="data.parquet"):
    d = {
        "version": 1,
        "name": name,
        "input": {"file": input_file, "time_column": "timestamp"},
        "cases": cases,
    }
    if timeout is not None:
        d["timeout_seconds"] = timeout
    if baseline:
        d["baseline"] = {"enabled": True}
        if baseline_expect is not None:
            d["baseline"]["expect"] = baseline_expect
    if extra_top:
        d.update(extra_top)
    return d


def _write_suite(tmp_path, suite_dict, name="suite.yaml"):
    return write_yaml(tmp_path, name, yaml.safe_dump(suite_dict))


def _run(tmp_path, suite_dict, target_args, *extra_cli):
    suite = _write_suite(tmp_path, suite_dict)
    args = ["suite", str(suite), *extra_cli, "--", *target_args]
    return runner.invoke(app, args)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _string_df(telemetry_df):
    df = telemetry_df.copy()
    df["label"] = "ok"
    return df


def _baseline_tmpdirs():
    return [
        p for p in Path(tempfile.gettempdir()).iterdir()
        if p.name.startswith("telemetry_resilience_baseline_")
    ]


def _raise_keyboard_interrupt(*args, **kwargs):
    raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# 1. preflight every case before executing any target
# ---------------------------------------------------------------------------

def test_run22_preflight_blocks_baseline_execution(tmp_path, telemetry_df):
    """Noise on a string channel: normal suite must exit 2 with zero
    target executions and zero artifacts -- the baseline must NOT run."""
    _setup(tmp_path, _string_df(telemetry_df))
    marker = tmp_path / "marker.txt"
    target = _target(tmp_path, "marker.py",
                     f"from pathlib import Path\n"
                     f"Path({str(marker)!r}).write_text('ran')\n")
    art = tmp_path / "art"
    suite = _suite_dict([
        _case("c", 11, [_fault("noise", "label", "1s", "2s", std=0.5)]),
    ], baseline=True)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 2, result.output
    assert "numeric" in result.output
    assert not marker.exists(), "target executed despite invalid suite"
    assert not art.exists(), "artifacts written despite invalid suite"
    assert "Traceback" not in result.output


def test_run22_preflight_blocks_case_execution(tmp_path, telemetry_df):
    """Same, with the baseline disabled: still exit 2, nothing executes."""
    _setup(tmp_path, _string_df(telemetry_df))
    marker = tmp_path / "marker.txt"
    target = _target(tmp_path, "marker.py",
                     f"from pathlib import Path\n"
                     f"Path({str(marker)!r}).write_text('ran')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("noise", "label", "1s", "2s", std=0.5)]),
    ], baseline=False)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 2, result.output
    assert not marker.exists()
    assert "Traceback" not in result.output


def test_run22_preflight_drift_on_strict_int_without_cast(tmp_path, telemetry_df):
    """Drift on a strict integer channel without cast is caught preflight."""
    _setup(tmp_path, telemetry_df)
    marker = tmp_path / "marker.txt"
    target = _target(tmp_path, "marker.py",
                     f"from pathlib import Path\n"
                     f"Path({str(marker)!r}).write_text('ran')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("drift", "count_int", "1s", "2s", rate=0.4)]),
    ], baseline=True)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 2, result.output
    assert not marker.exists()


# ---------------------------------------------------------------------------
# 2. baseline must never receive the user's original file
# ---------------------------------------------------------------------------

_DESTROY_TARGET = (
    "import sys\n"
    "from pathlib import Path\n"
    "Path(sys.argv[1]).write_text('DESTROYED')\n"
)


def _passing_case():
    return _case("c", 11, [_fault("dropout", "gps_latitude", "1s", "2s")])


def test_run22_mutating_baseline_cannot_modify_original_csv(
    tmp_path, telemetry_df
):
    data = _setup(tmp_path, telemetry_df, name="data.csv")
    before = _sha256(data)
    target = _target(tmp_path, "destroy.py", _DESTROY_TARGET)
    suite = _suite_dict([_passing_case()], baseline=True,
                        input_file="data.csv")
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert _sha256(data) == before, "original CSV was modified by baseline"
    assert data.read_text() != "DESTROYED"


def test_run22_mutating_baseline_cannot_modify_csv_sidecar(
    tmp_path, telemetry_df
):
    data = _setup(tmp_path, telemetry_df, name="data.csv")
    sidecar = data.with_name("data.schema.json")
    assert sidecar.exists()
    before_data = _sha256(data)
    before_sidecar = _sha256(sidecar)
    target = _target(
        tmp_path, "tamper.py",
        "import sys\n"
        "from pathlib import Path\n"
        "p = Path(sys.argv[1])\n"
        "p.with_name(p.stem + '.schema.json').write_text('{\"tampered\": true}')\n",
    )
    suite = _suite_dict([_passing_case()], baseline=True,
                        input_file="data.csv")
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert _sha256(data) == before_data
    assert _sha256(sidecar) == before_sidecar, \
        "original CSV sidecar was modified by baseline"


def test_run22_mutating_baseline_cannot_modify_original_parquet(
    tmp_path, telemetry_df
):
    data = _setup(tmp_path, telemetry_df, name="data.parquet")
    before = _sha256(data)
    target = _target(tmp_path, "destroy.py", _DESTROY_TARGET)
    suite = _suite_dict([_passing_case()], baseline=True)
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert _sha256(data) == before, "original Parquet was modified by baseline"


def test_run22_baseline_temp_cleanup_after_pass(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('ok')\n")
    suite = _suite_dict([_passing_case()], baseline=True)
    before = set(_baseline_tmpdirs())
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert set(_baseline_tmpdirs()) <= before


def test_run22_baseline_temp_cleanup_after_fail(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "import sys\nsys.exit(1)\n")
    suite = _suite_dict([_passing_case()], baseline=True)
    before = set(_baseline_tmpdirs())
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    assert set(_baseline_tmpdirs()) <= before


def test_run22_baseline_temp_cleanup_after_timeout(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "import time\ntime.sleep(30)\n")
    suite = _suite_dict([_passing_case()], baseline=True, timeout=1)
    before = set(_baseline_tmpdirs())
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"])
    assert result.exit_code == 1, result.output
    assert set(_baseline_tmpdirs()) <= before


def test_run22_baseline_temp_cleanup_after_interrupt(
    tmp_path, telemetry_df, monkeypatch
):
    _setup(tmp_path, telemetry_df)
    monkeypatch.setattr(runner_module, "run_target",
                        _raise_keyboard_interrupt)
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([_passing_case()], baseline=True)
    suite_file = _write_suite(tmp_path, suite)
    before = set(_baseline_tmpdirs())
    result = runner.invoke(
        app, ["suite", str(suite_file), "--",
              sys.executable, target, "{data}"])
    assert result.exit_code == 3, result.output
    assert "Interrupted by user" in result.output
    assert "Traceback" not in result.output
    assert set(_baseline_tmpdirs()) <= before


# ---------------------------------------------------------------------------
# 3. XML 1.0-safe JUnit
# ---------------------------------------------------------------------------

def test_run22_junit_illegal_char_in_suite_name():
    xml_text = render_junit_xml("bad\x01suite", [], None)
    root = ET.fromstring(xml_text)  # must parse
    assert root.get("name") == "bad\ufffdsuite"


def test_run22_junit_illegal_char_in_failure_detail():
    cases = [{
        "name": "c", "status": "fail", "duration_seconds": 1.0,
        "fault_count": 1, "failures": ["boom\x02detail"],
    }]
    xml_text = render_junit_xml("s", cases, None)
    root = ET.fromstring(xml_text)
    failure = root.find("testcase/failure")
    assert failure is not None
    assert "\x02" not in (failure.text or "")
    assert "\ufffd" in (failure.text or "")


def test_run22_junit_valid_unicode_preserved():
    cases = [{
        "name": "café-✓-日本語", "status": "pass", "duration_seconds": 0.5,
        "fault_count": 1, "failures": [],
    }]
    xml_text = render_junit_xml("suite-✓", cases, None)
    root = ET.fromstring(xml_text)
    assert root.find("testcase").get("name") == "café-✓-日本語"


def test_run22_junit_escapes_markup_chars():
    cases = [{
        "name": "c", "status": "fail", "duration_seconds": 1.0,
        "fault_count": 1,
        "failures": ["expected <a> & \"b\" but got 'c'"],
    }]
    xml_text = render_junit_xml("s", cases, None)
    root = ET.fromstring(xml_text)  # must parse
    failure = root.find("testcase/failure")
    assert failure.text == "expected <a> & \"b\" but got 'c'"


def test_run22_junit_end_to_end_with_control_char_suite_name(
    tmp_path, telemetry_df
):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('ok')\n")
    suite = _suite_dict([_passing_case()], baseline=False,
                        name="bad\x01suite")
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 0, result.output
    junit = (art / "junit.xml").read_text(encoding="utf-8")
    ET.fromstring(junit)  # must parse as valid XML
    assert "\x01" not in junit


# ---------------------------------------------------------------------------
# 4. non-UTF8 target output
# ---------------------------------------------------------------------------

_BAD_BYTES_TARGET = (
    "import sys\n"
    "sys.stdout.buffer.write(b'hello\\xffworld\\n')\n"
    "sys.stdout.buffer.flush()\n"
    "sys.stderr.buffer.write(b'err\\xfe!\\n')\n"
    "sys.stderr.buffer.flush()\n"
)


def test_run22_non_utf8_stdout_and_stderr(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "bad.py", _BAD_BYTES_TARGET)
    suite = _suite_dict([_passing_case()], baseline=False)
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    stdout = (art / "c" / "stdout.txt").read_text(encoding="utf-8")
    stderr = (art / "c" / "stderr.txt").read_text(encoding="utf-8")
    assert "hello\ufffdworld" in stdout
    assert "err\ufffd!" in stderr
    ET.fromstring((art / "junit.xml").read_text(encoding="utf-8"))


def test_run22_non_utf8_timeout_partial_output(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    target = _target(
        tmp_path, "slow.py",
        "import sys, time\n"
        "sys.stdout.buffer.write(b'partial\\xff')\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(30)\n",
    )
    suite = _suite_dict([_passing_case()], baseline=False, timeout=1)
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert "timed out" in result.output
    ET.fromstring((art / "junit.xml").read_text(encoding="utf-8"))


def test_run22_non_utf8_test_command(tmp_path, telemetry_df):
    """The single-scenario `test` command tolerates bad bytes too."""
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "bad.py", _BAD_BYTES_TARGET)
    scenario = {
        "version": 1,
        "seed": 5,
        "input": {"file": "data.parquet", "time_column": "timestamp"},
        "faults": [_fault("dropout", "gps_latitude", "1s", "2s")],
    }
    scenario_file = write_yaml(tmp_path, "scenario.yaml",
                               yaml.safe_dump(scenario))
    result = runner.invoke(
        app, ["test", str(scenario_file), "--",
              sys.executable, target, "{data}"])
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert "hello\ufffdworld" in result.output
    assert "TEST RESULT: PASS" in result.output


# ---------------------------------------------------------------------------
# 4b. platform-independent process-output decoding (CI compatibility fix)
# ---------------------------------------------------------------------------
#
# Regression section for the Windows decoding bug: with text=True,
# subprocess output was decoded with locale.getpreferredencoding(), so on
# Windows b"\xff" became "ÿ" (cp1252) instead of U+FFFD. Capture is now
# binary everywhere and decode_process_output() applies UTF-8/replace
# explicitly, so the same bytes decode identically on every platform.


def test_decode_process_output_replaces_invalid_utf8():
    decode = runner_module.decode_process_output
    assert decode(b"hello\xffworld") == "hello\ufffdworld"
    assert decode(b"\xfe") == "\ufffd"
    assert decode(b"") == ""
    assert decode(b"plain ascii") == "plain ascii"
    # Multi-byte UTF-8 still decodes as UTF-8 (not latin-1/cp1252).
    assert decode("héllo".encode("utf-8")) == "héllo"


def test_decode_process_output_str_and_none_passthrough():
    decode = runner_module.decode_process_output
    assert decode("already text") == "already text"
    assert decode(None) == ""


def test_decode_process_output_is_not_locale_decoding():
    # Documents the exact Windows failure mode: cp1252 decodes 0xFF as
    # "ÿ" (valid character, no replacement). The product contract is
    # UTF-8 with replacement -- never the platform code page.
    assert runner_module.decode_process_output(b"\xff") == "\ufffd"
    assert runner_module.decode_process_output(b"\xff") != b"\xff".decode(
        "cp1252"
    )


def test_run_target_never_uses_text_mode():
    # Static guard: reintroducing text=True / universal_newlines would
    # silently reintroduce the Windows code-page decoding bug. Token-based
    # so mentions inside docstrings/comments do not trip it.
    import io
    import tokenize

    src = Path(runner_module.__file__).read_text(encoding="utf-8")
    toks = [
        (tok.type, tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
    ]
    for a, b, c in zip(toks, toks[1:], toks[2:]):
        assert not (
            a == (tokenize.NAME, "text")
            and b == (tokenize.OP, "=")
            and c == (tokenize.NAME, "True")
        ), "text=True must not be used for process output capture"
    assert not any(
        ttype == tokenize.NAME and tstr == "universal_newlines"
        for ttype, tstr in toks
    ), "universal_newlines must not be used for process output capture"


def test_run_target_decodes_bad_bytes_identically():
    # End-to-end through run_target: invalid bytes become U+FFFD even when
    # the platform default encoding would decode them differently.
    result = runner_module.run_target(
        [sys.executable, "-c",
         "import sys;"
         "sys.stdout.buffer.write(bytes([104, 105, 255]));"
         "sys.stderr.buffer.write(bytes([101, 254, 33]))"],
        timeout_seconds=30,
    )
    assert result.exit_code == 0
    assert result.stdout == "hi\ufffd"
    assert result.stderr == "e\ufffd!"


def test_junit_xml_written_as_valid_utf8_bytes(tmp_path, telemetry_df):
    # The Windows charmap crash: junit.xml containing U+FFFD (from the
    # sanitized control-character suite name) must be written as UTF-8
    # bytes -- the platform default encoding (cp1252) cannot encode U+FFFD.
    _setup(tmp_path, telemetry_df)
    target = _target(tmp_path, "t.py", "print('ok')\n")
    suite = _suite_dict([_passing_case()], baseline=False,
                        name="bad\x01suite")
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 0, result.output
    raw = (art / "junit.xml").read_bytes()
    text = raw.decode("utf-8")  # strict: must be valid UTF-8
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>')
    ET.fromstring(text)
    assert "\x01" not in text
    # stdout.txt artifacts with replacement chars are UTF-8 too.
    for name in ("summary.json", "summary.md"):
        (art / name).read_bytes().decode("utf-8")


# ---------------------------------------------------------------------------
# 5. untracked files in owned artifact directories
# ---------------------------------------------------------------------------

def _passing_suite_no_baseline(tmp_path, telemetry_df):
    _setup(tmp_path, telemetry_df)
    suite = _suite_dict([_case("c", 11,
                               [_fault("dropout", "gps_latitude",
                                       "1s", "2s")])],
                        baseline=False)
    target = _target(tmp_path, "t.py", "print('x')\n")
    return suite, [sys.executable, target, "{data}"]


def test_run22_untracked_file_refuses_overwrite(tmp_path, telemetry_df):
    suite, target_args = _passing_suite_no_baseline(tmp_path, telemetry_df)
    art = tmp_path / "art"
    r1 = _run(tmp_path, suite, target_args, "--artifacts", str(art),
              "--keep-data")
    assert r1.exit_code == 0, r1.output
    notes = art / "notes.txt"
    notes.write_text("user notes -- do not delete")
    marker = json.loads((art / ".telemetry-resilience-artifacts.json")
                        .read_text())
    assert marker["format_version"] == 2
    assert "notes.txt" not in marker["owned_entries"]
    r2 = _run(tmp_path, suite, target_args, "--artifacts", str(art),
              "--overwrite-artifacts")
    assert r2.exit_code == 2, r2.output
    assert "untracked" in r2.output
    assert notes.read_text() == "user notes -- do not delete"


def test_run22_clean_owned_overwrite_still_works(tmp_path, telemetry_df):
    suite, target_args = _passing_suite_no_baseline(tmp_path, telemetry_df)
    art = tmp_path / "art"
    r1 = _run(tmp_path, suite, target_args, "--artifacts", str(art))
    assert r1.exit_code == 0, r1.output
    r2 = _run(tmp_path, suite, target_args, "--artifacts", str(art),
              "--overwrite-artifacts")
    assert r2.exit_code == 0, r2.output
    assert (art / "summary.json").exists()
    marker = json.loads((art / ".telemetry-resilience-artifacts.json")
                        .read_text())
    assert marker["format_version"] == 2
    assert sorted(marker["owned_entries"]) == sorted(
        ["c", "summary.json", "summary.md", "junit.xml"])


def test_run22_old_marker_format_refuses_destructive_cleanup(
    tmp_path, telemetry_df
):
    _setup(tmp_path, telemetry_df)
    art = tmp_path / "art"
    art.mkdir()
    (art / ".telemetry-resilience-artifacts.json").write_text(
        json.dumps({"tool": "telemetry-resilience", "format_version": 1}))
    stale = art / "stale.txt"
    stale.write_text("previous run leftovers")
    suite, target_args = _passing_suite_no_baseline(tmp_path, telemetry_df)
    result = _run(tmp_path, suite, target_args, "--artifacts", str(art),
                  "--overwrite-artifacts")
    assert result.exit_code == 2, result.output
    assert stale.read_text() == "previous run leftovers"
    assert not (art / "summary.json").exists()


def test_run22_invalid_suite_writes_no_marker(tmp_path, telemetry_df):
    """Marker write order: an invalid suite leaves no owned-looking dir."""
    _setup(tmp_path, _string_df(telemetry_df))
    target = _target(tmp_path, "t.py", "print('x')\n")
    suite = _suite_dict([
        _case("c", 11, [_fault("noise", "label", "1s", "2s", std=0.5)]),
    ], baseline=False)
    art = tmp_path / "art"
    result = _run(tmp_path, suite, [sys.executable, target, "{data}"],
                  "--artifacts", str(art))
    assert result.exit_code == 2, result.output
    assert not art.exists()
