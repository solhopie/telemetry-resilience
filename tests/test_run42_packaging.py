"""Run 4.2: release tooling cleanup regression tests.

Covers:

- single authoritative version: the pyproject dynamic version, the
  installed package metadata, ``telemetry-resilience --version``, and
  ``telemetry_resilience.__version__`` all agree;
- clean build input: a fake stale module placed only under ``build/lib``
  cannot enter the release wheel (acceptance test per spec section 2),
  and the clean-source-copy helper excludes stale generated output.
  The acceptance build runs in a hermetic venv whose setuptools satisfies
  the declared ``[build-system]`` floor, so the test never depends on
  whatever setuptools the ambient runner happens to ship
  (setuptools < 77 rejects ``license = "Apache-2.0"``);
- release_check.py never copies repository examples into the wheel smoke
  test (static regression guard for spec section 1: the inject smoke test
  must use the exact degraded_navigation.yaml produced by the installed
  wheel's ``demo init``);
- handoff tree hygiene: no generated build debris (build/, *.egg-info,
  attack evidence, attack-outcome notes) in the source tree. dist/ is
  intentionally NOT asserted -- it may hold the release-candidate
  artifacts for the handoff ZIP. (__pycache__ / .pytest_cache / *.pyc are
  intentionally not asserted here either: running pytest creates them.
  They are covered by .gitignore and by the wheel/sdist entry-name
  audit in scripts/release_check.py.)
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from telemetry_resilience import __version__
from telemetry_resilience.cli import app
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_CHECK = REPO_ROOT / "scripts" / "release_check.py"


def _build_system_requires() -> list:
    """The declared [build-system] requires from pyproject.toml.

    The hermetic build venv installs exactly these, so the test always
    honors the project's own declared build floor (currently
    ``setuptools>=77`` -- older setuptools rejects the PEP 639 SPDX
    ``license = "Apache-2.0"`` expression).
    """
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return list(tomllib.load(fh)["build-system"]["requires"])


def _provision_build_venv(rc, venv_dir: Path):
    """Create a venv with ``build`` + the declared build-system requires.

    Returns the venv's python, or None when provisioning is impossible
    (e.g. no network to install the toolchain): the caller must skip,
    not fail, in that case.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    venv_py = rc._venv_python(venv_dir)
    try:
        subprocess.run(
            [str(venv_py), "-m", "pip", "install", "-q",
             "build", *_build_system_requires()],
            check=True,
            capture_output=True,
            timeout=900,
        )
        subprocess.run(
            [str(venv_py), "-c",
             "import build.__main__, setuptools.build_meta"],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return venv_py


@pytest.fixture(autouse=True)
def _restore_tree_debris_state():
    """Regression guard for test isolation.

    A packaging test must clean up the build debris it creates in the
    repo tree. Snapshot the debris-relevant state before each test;
    afterwards, remove anything NEW the test left behind (so one failure
    cannot cascade into the hygiene test) and fail with a clear message.
    """

    def _snapshot() -> set:
        found = set()
        if (REPO_ROOT / "build").exists():
            found.add("build/")
        for egg_info in (REPO_ROOT / "src").glob("*.egg-info"):
            found.add(f"src/{egg_info.name}/")
        return found

    before = _snapshot()
    yield
    new_debris = sorted(_snapshot() - before)
    for entry in new_debris:
        target = REPO_ROOT / entry.rstrip("/")
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    assert not new_debris, (
        f"test left build debris in the repo tree: {new_debris}"
    )


def _load_release_check():
    """Import scripts/release_check.py (stdlib only) without a package."""
    spec = importlib.util.spec_from_file_location("release_check_42", RELEASE_CHECK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_sources_match():
    try:
        installed = importlib_metadata.version("telemetry-resilience")
    except importlib_metadata.PackageNotFoundError:
        pytest.skip("telemetry-resilience is not installed")
    assert installed == __version__
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"Telemetry Resilience CLI {__version__}"


def test_clean_source_copy_excludes_stale_build_output(tmp_path):
    rc = _load_release_check()
    fake = (
        REPO_ROOT / "build" / "lib" / "telemetry_resilience" / "stale_only_module.py"
    )
    created = not fake.exists()
    if created:
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("STALE_BUILD_ONLY = True\n", encoding="utf-8")
    try:
        dest = rc._clean_source_copy(REPO_ROOT, tmp_path)
        try:
            assert not (dest / "build").exists(), "build/ leaked into clean copy"
            assert not (dest / "dist").exists(), "dist/ leaked into clean copy"
            assert list(dest.glob("src/*.egg-info")) == [], (
                "*.egg-info leaked into clean copy"
            )
            leaked = list(dest.rglob("stale_only_module.py"))
            assert leaked == [], f"stale module leaked into clean copy: {leaked}"
            # Intended files survive the copy.
            assert (dest / "pyproject.toml").is_file()
            assert (dest / "src" / "telemetry_resilience" / "__init__.py").is_file()
            assert (
                dest
                / "src"
                / "telemetry_resilience"
                / "templates"
                / "degraded_navigation.yaml"
            ).is_file()
        finally:
            shutil.rmtree(dest, ignore_errors=True)
    finally:
        if created:
            fake.unlink(missing_ok=True)
            for parent in (
                fake.parent,
                fake.parent.parent,
                fake.parent.parent.parent,
            ):
                try:
                    parent.rmdir()
                except OSError:
                    pass


def test_stale_build_lib_excluded_from_wheel(tmp_path):
    """Acceptance test (spec section 2):

    1. create a fake stale module only under build/lib
    2. perform the release build (clean source copy, --no-isolation)
       inside a hermetic venv whose setuptools satisfies the declared
       [build-system] floor -- never the ambient runner toolchain
    3. inspect the wheel
    4. the stale module must NOT appear; METADATA version must agree
       with telemetry_resilience.__version__.
    """
    rc = _load_release_check()
    venv_py = _provision_build_venv(rc, tmp_path / "build-venv")
    if venv_py is None:
        pytest.skip(
            "could not provision an isolated build venv "
            "(venv creation or toolchain install failed)"
        )
    stale = (
        REPO_ROOT / "build" / "lib" / "telemetry_resilience" / "stale_only_module.py"
    )
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("STALE_BUILD_ONLY = True\n", encoding="utf-8")
    # clean_build clears stale repo outputs before building; back up dist/
    # (release artifacts) and restore it afterwards.
    dist_dir = REPO_ROOT / "dist"
    backup = tmp_path / "dist-backup"
    if dist_dir.exists():
        shutil.copytree(dist_dir, backup)
    try:
        wheel, _sdist = rc.clean_build(
            Path(venv_py), REPO_ROOT, tmp_path / "out", no_isolation=True
        )
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
        leaked = [n for n in names if "stale_only_module" in n]
        assert leaked == [], f"stale build/lib module leaked into wheel: {leaked}"
        meta_name = next(
            n for n in names if n.endswith(".dist-info/METADATA")
        )
        with zipfile.ZipFile(wheel) as zf:
            meta_text = zf.read(meta_name).decode("utf-8")
        meta_version = re.search(r"^Version: (.+)$", meta_text, re.M)
        assert meta_version, "no Version field in wheel METADATA"
        assert meta_version.group(1).strip() == __version__
    finally:
        if stale.exists():
            stale.unlink()
        for parent in (stale.parent, stale.parent.parent, REPO_ROOT / "build"):
            try:
                parent.rmdir()
            except OSError:
                pass
        if backup.exists():
            shutil.rmtree(dist_dir, ignore_errors=True)
            shutil.copytree(backup, dist_dir)


def test_release_check_never_copies_repo_examples():
    """The wheel smoke test must use only wheel/demo-init files.

    Regression guard for spec section 1: the inject smoke test must run
    against the exact degraded_navigation.yaml produced by the installed
    wheel's ``demo init`` -- no file may be copied from the source
    repository into the wheel smoke-test directory.
    """
    text = RELEASE_CHECK.read_text(encoding="utf-8")
    smoke_start = text.index("def step_wheel_smoke")
    smoke_end = text.index("def step_artifacts")
    smoke = text[smoke_start:smoke_end]
    assert 'repo / "examples"' not in smoke, (
        "wheel smoke test must not copy files from repo/examples"
    )
    assert "shutil.copy(" not in smoke and "shutil.copytree(" not in smoke, (
        "wheel smoke test must not copy repository files"
    )
    # The wheel-only smoke test contract: demo init from the installed
    # wheel, then the packaged scenario verified byte-identical.
    assert "demo init" in smoke
    assert "byte-identical" in smoke


def test_handoff_tree_has_no_generated_build_debris():
    problems: list[str] = []
    if (REPO_ROOT / "build").exists():
        problems.append("build/")
    for egg_info in (REPO_ROOT / "src").glob("*.egg-info"):
        problems.append(f"src/{egg_info.name}/")
    for name in ("attack_evidence_run22",):
        if (REPO_ROOT / name).exists():
            problems.append(f"{name}/")
    for outcome in REPO_ROOT.glob("RUN-*-attack-outcomes.txt"):
        problems.append(outcome.name)
    assert not problems, f"generated build debris in handoff tree: {problems}"
