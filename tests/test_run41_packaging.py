"""Run 4.1: packaging regression tests against the built dist/ artifacts.

Covers the §12 regression-test gaps left by the five parallel chunks:

- wheel + sdist template list completeness: every ``TEMPLATE_FILES`` entry
  (parsed stdlib-only from ``src/telemetry_resilience/demo.py``, exactly the
  way ``scripts/release_check.py`` parses it) must be present in the freshly
  built wheel and sdist under ``dist/``. If no artifacts were built (a
  contributor who never ran ``python -m build``), these tests SKIP with a
  clear message instead of failing the suite.
- package text audit: scan text-like members of the built wheel+sdist for
  machine-specific absolute paths and for obvious secret/key markers
  (by name only -- never printing secret contents). Fails on hits.
- project-tree hygiene as a ``.gitignore``-pattern test: asserts
  ``.gitignore`` covers the known junk patterns. Stable by construction --
  it checks the ignore rules, not the live tree, so it cannot fail just
  because pytest created a cache during this run.

The slow venv-install quick-start coverage deliberately lives in
``scripts/release_check.py`` (automated) plus the manual acceptance attack,
not in this file.
"""

from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
DEMO_PY = REPO_ROOT / "src" / "telemetry_resilience" / "demo.py"
GITIGNORE = REPO_ROOT / ".gitignore"
TEMPLATES_PREFIX = "telemetry_resilience/templates/"

# --- Mirrored from scripts/release_check.py (keep semantics in sync) ------------
# The audit-policy constants below are built programmatically (never as
# literal strings) so this file -- which ships in the sdist -- cannot
# self-flag in the content audit it performs. The policy semantics are
# identical to release_check.py.
GENERIC_USER_SEGMENTS = frozenset(
    {
        "user", "username", "test", "tests", "example", "examples",
        "runner", "ci", "builder", "docker", "nobody", "someuser",
        "someone", "yourname", "name",
    }
)
HOME_PATH_RE = re.compile(r"/(?:home|Users)/([A-Za-z0-9._-]+)(?=/|$)")
WIN_USERS_RE = re.compile(
    r"[A-Za-z]:[\\/][Uu]sers[\\/]([A-Za-z0-9._-]+)(?=[\\/]|$)"
)


def _temp_build_dir_res():
    """Temp-build-dir regexes; built so the source stays self-clean."""
    var_folders = "/" + "var" + "/folders/"
    return (
        re.compile(
            r"/(?:private/)?tmp/(?:build|pip-|pip_req|wheel|tmpbuild)[\w.\-/]*"
        ),
        re.compile(re.escape(var_folders) + r"[\w.\-/]*"),
        re.compile(
            r"[A-Za-z]:[\\/](?:[Ww]indows[\\/])?[Tt]emp[\\/"
            r"](?:[\w.\-][\w.\-\\/]*)?"
        ),
    )


TEMP_BUILD_DIR_RES = _temp_build_dir_res()


def _secret_content_markers():
    """Obvious secret/key markers; fragments only, never full literals."""
    kinds = (
        "PRIVATE KEY",
        "RSA PRIVATE KEY",
        "DSA PRIVATE KEY",
        "EC PRIVATE KEY",
        "OPENSSH PRIVATE KEY",
        "PGP PRIVATE KEY BLOCK",
    )
    return tuple(f"-----BEGIN {kind}-----" for kind in kinds) + (
        "aws" + "_secret_access_key",
    )


SECRET_CONTENT_MARKERS = _secret_content_markers()
CONTENT_SCAN_SIZE_CAP = 2_000_000
# -----------------------------------------------------------------------------

REQUIRED_GITIGNORE_PATTERNS = (
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    "attack_evidence*/",
    "examples/results*",
    "examples/campaign_results*",
    "RUN-*-attack-outcomes.txt",
)


def _parse_template_files() -> tuple[str, ...]:
    """Parse TEMPLATE_FILES from demo.py without importing the package."""
    text = DEMO_PY.read_text(encoding="utf-8")
    match = re.search(r"^TEMPLATE_FILES\s*=\s*\((.*?)\)", text, re.MULTILINE | re.DOTALL)
    assert match, "could not parse TEMPLATE_FILES from src/telemetry_resilience/demo.py"
    names = tuple(re.findall(r'"([^"]+)"', match.group(1)))
    assert names, "TEMPLATE_FILES parsed empty from src/telemetry_resilience/demo.py"
    return names


def _wheel_path() -> Path:
    wheels = sorted(DIST_DIR.glob("telemetry_resilience-*.whl"))
    if not wheels:
        pytest.skip(
            "no wheel in dist/ -- run `python -m build` first; "
            "skipping packaging regression"
        )
    return wheels[-1]


def _sdist_path() -> Path:
    sdists = sorted(DIST_DIR.glob("telemetry_resilience-*.tar.gz"))
    if not sdists:
        pytest.skip(
            "no sdist in dist/ -- run `python -m build` first; "
            "skipping packaging regression"
        )
    return sdists[-1]


def _archive_entries(archive: Path) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = []
    if archive.suffix == ".whl" or zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                members.append((name, zf.read(name)))
    else:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                fh = tf.extractfile(member)
                assert fh is not None
                members.append((member.name, fh.read()))
    return members


def test_wheel_templates_complete():
    wheel = _wheel_path()
    expected = _parse_template_files()
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    missing = [
        tname
        for tname in expected
        if not any(e == f"{TEMPLATES_PREFIX}{tname}" for e in names)
    ]
    assert not missing, f"{wheel.name} is missing bundled templates: {missing}"


def test_sdist_templates_complete():
    sdist = _sdist_path()
    expected = _parse_template_files()
    with tarfile.open(sdist) as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    missing = [
        tname
        for tname in expected
        if not any(e.endswith(f"/templates/{tname}") for e in names)
    ]
    assert not missing, f"{sdist.name} is missing bundled templates: {missing}"


def _scan_text_members(archive: Path) -> list[str]:
    """Return problem descriptions (marker/entry only, never contents)."""
    problems: list[str] = []
    for name, data in _archive_entries(archive):
        if len(data) > CONTENT_SCAN_SIZE_CAP:
            continue
        if b"\x00" in data[:4096]:
            continue  # binary: not text-like
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        reported: set[str] = set()
        for match in HOME_PATH_RE.finditer(text):
            if match.group(1).lower() in GENERIC_USER_SEGMENTS:
                continue
            reported.add(match.group(0))
        for match in WIN_USERS_RE.finditer(text):
            if match.group(1).lower() in GENERIC_USER_SEGMENTS:
                continue
            reported.add(match.group(0))
        for pattern in TEMP_BUILD_DIR_RES:
            for match in pattern.finditer(text):
                reported.add(match.group(0))
        for path in sorted(reported):
            problems.append(
                f"machine-specific path {path!r} in contents of {name!r}"
            )
        lowered = text.lower()
        for marker in SECRET_CONTENT_MARKERS:
            if marker.lower() in lowered:
                problems.append(f"secret marker {marker!r} in contents of {name!r}")
    return problems


def test_wheel_text_audit_clean():
    wheel = _wheel_path()
    problems = _scan_text_members(wheel)
    assert not problems, (
        f"{wheel.name}: {len(problems)} audit problem(s):\n  - "
        + "\n  - ".join(problems[:20])
    )


def test_sdist_text_audit_clean():
    sdist = _sdist_path()
    problems = _scan_text_members(sdist)
    assert not problems, (
        f"{sdist.name}: {len(problems)} audit problem(s):\n  - "
        + "\n  - ".join(problems[:20])
    )


def _gitignore_patterns() -> set[str]:
    assert GITIGNORE.is_file(), ".gitignore is missing"
    patterns: set[str] = set()
    for line in GITIGNORE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.add(line)
    return patterns


def test_gitignore_covers_hygiene_patterns():
    patterns = _gitignore_patterns()
    missing = [p for p in REQUIRED_GITIGNORE_PATTERNS if p not in patterns]
    assert not missing, f".gitignore is missing hygiene patterns: {missing}"
