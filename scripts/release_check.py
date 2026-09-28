#!/usr/bin/env python3
"""Pre-release verification for telemetry-resilience (stdlib only).

Works from a source checkout with NO prior editable install of the package.
It builds everything it needs in a throwaway directory tree:

  <tmp_root>/
    tooling-venv/   A fresh venv created with ``python -m venv``. The project
                    is installed into it NON-editable with test extras
                    (``pip install "<repo>[test]"``), plus the ``build``
                    package. This venv's python runs the FULL pytest suite
                    and ``python -m build`` (sdist + wheel) in the repo.
    wheel-venv/     A second fresh venv. ONLY the freshly built wheel file is
                    installed into it (no project source, no editable
                    install); declared dependencies resolve from the wheel
                    metadata via pip as normal.
    work/           An empty directory standing in for the end-user machine.
                    Every installed-package smoke test runs with cwd inside
                    work/.

Release flow:
  0. Remove generated build debris (build/, src/*.egg-info/, __pycache__,
     .pytest_cache) from the repo tree and verify it is clean BEFORE
     running tests that create new caches.
  1. Create the temp root and the tooling venv; pip-install
     ``<repo>[test]`` (non-editable) and ``build`` into it.
  2. Run the full pytest suite with the tooling venv's python.
  3. Remove stale generated directories (build/, dist/, src/*.egg-info/)
     and build sdist + wheel with the tooling venv's ``python -m build``
     from a temporary CLEAN SOURCE COPY containing only intended project
     files. Stale setuptools output (e.g. build/lib/) is never copied, so
     a stale module cannot enter the release wheel after its source file
     was removed.
  4. Audit the wheel and sdist:
       - entry-name checks (no __pycache__/.pytest_cache/.venv, attack
         fixtures, /tmp paths, absolute machine paths, traversal,
         secrets-looking names; every TEMPLATE_FILES template must be in
         both the wheel and the sdist file lists);
       - scan text-like archive contents for accidental machine-specific
         absolute paths (/home/<name>/, /Users/<name>/, C:\\Users\\<name>\\,
         temp build dirs), ignoring intentionally written generic test
         strings (placeholder usernames such as "user"/"test"/"example");
       - scan text-like archive contents and entry names for obvious
         secret/private-key material by marker, never printing any secret
         contents. Any audit failure => RELEASE CHECK: FAIL.
  5. Create wheel-venv and install ONLY the built wheel.
  6. Wheel smoke test (cwd inside work/): run
     ``telemetry-resilience demo init telemetry-demo`` from the wheel-venv
     CLI to prove the packaged demo works with no source checkout, then
     from work/telemetry-demo/ run the documented quick start:
       doctor (expect READY), inspect demo_drive.parquet,
       inject demo_drive.parquet --scenario degraded_navigation.yaml,
       campaign navigation_campaign.yaml --plan,
       campaign navigation_campaign.yaml --list-cases,
       and one FULL campaign run against demo_app.py (verify the baseline
       prints NOMINAL, the campaign completes, and coverage.json /
       summary.json / junit.xml exist in the artifacts dir).
     The inject smoke test uses the EXACT degraded_navigation.yaml
     produced by the installed wheel's ``demo init`` (verified
     byte-identical to the template shipped inside the wheel archive).
     NO file is copied from the source repository into the wheel
     smoke-test directory at any point.
     Then uninstall the package from wheel-venv and confirm it no longer
     imports.
  7. Clean up the entire temporary root afterward (best effort, even on
     failure).

Publishes nothing. Prints ``RELEASE CHECK: PASS`` / ``RELEASE CHECK: FAIL``
and exits 0 / 1.

Usage:  python scripts/release_check.py [--repo PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

# Archive entry-name fragments that must never ship (case-insensitive).
BAD_NAME_FRAGMENTS = (
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "attack_evidence",
    "/tmp/",
    ".env",
    "id_rsa",
    ".pem",
    "secret",
    "credential",
    "token",
    "passwd",
    "private_key",
    "pgp",
)
# Absolute-path prefixes that must never appear in an archive entry name.
MACHINE_PATH_MARKERS = ("/home/", "/Users/", "C:\\", "D:\\")
TEMPLATES_PREFIX = "telemetry_resilience/templates/"
PACKAGE_NAME = "telemetry-resilience"
DEMO_SCENARIO_NAME = "degraded_navigation.yaml"

# --- Content-scan patterns (audit of text-like archive contents) ----------------
# Generic placeholder usernames that mark an "intentionally written generic
# test string" -- never treated as an accidental machine-specific path.
GENERIC_USER_SEGMENTS = frozenset(
    {
        "user",
        "username",
        "test",
        "tests",
        "example",
        "examples",
        "runner",
        "ci",
        "builder",
        "docker",
        "nobody",
        "someuser",
        "someone",
        "yourname",
        "name",
    }
)
# /home/<seg>/ or /Users/<seg>/
HOME_PATH_RE = re.compile(r"/(?:home|Users)/([A-Za-z0-9._-]+)(?=/|$)")
# C:\Users\<seg>\ or C:/Users/<seg>/
WIN_USERS_RE = re.compile(
    r"[A-Za-z]:[\\/][Uu]sers[\\/]([A-Za-z0-9._-]+)(?=[\\/]|$)"
)
# Temp build dirs: pip/build scratch, macOS per-user temp, Windows Temp.
TEMP_BUILD_DIR_RES = (
    re.compile(r"/(?:private/)?tmp/(?:build|pip-|pip_req|wheel|tmpbuild)[\w.\-/]*"),
    re.compile(r"/var/folders/[\w.\-/]*"),
    re.compile(r"[A-Za-z]:[\\/](?:[Ww]indows[\\/])?[Tt]emp[\\/](?:[\w.\-][\w.\-\\/]*)?"),
)
# Obvious secret/private-key material: marker strings only. We report the
# marker and the entry name, NEVER any surrounding content.
SECRET_CONTENT_MARKERS = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN DSA PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN PGP PRIVATE KEY BLOCK-----",
    "aws_secret_access_key",
)
# Only members up to this size are content-scanned.
CONTENT_SCAN_SIZE_CAP = 2_000_000


class CheckFailed(Exception):
    """Raised when a release-check step fails."""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd, *, cwd=None, env=None, timeout=600, capture=True):
    """Run a command; raise CheckFailed with the output on non-zero exit."""
    _log(f"  $ {' '.join(str(c) for c in cmd)}")
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=capture,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise CheckFailed(f"timed out after {timeout}s: {' '.join(map(str, cmd))}") from exc
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        raise CheckFailed(
            f"exit {proc.returncode}: {' '.join(map(str, cmd))}\n{out.strip()}"
        )
    return proc


def _repo_version(repo: Path) -> str:
    init = repo / "src" / "telemetry_resilience" / "__init__.py"
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init.read_text())
    if not match:
        raise CheckFailed("could not read __version__ from package __init__")
    return match.group(1)


# --- Clean build input ---------------------------------------------------------
# Names that are never copied into the clean build source tree.
_CLEAN_COPY_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".eggs",
        ".tox",
        ".nox",
    }
)
_CLEAN_COPY_IGNORE_SUFFIXES = (".egg-info",)
_CLEAN_COPY_IGNORE_SUBSTRINGS = ("attack_evidence",)


def _clean_source_copy(repo: Path, dest_parent: Path) -> Path:
    """Copy the repo to a temp dir containing only intended project files.

    Stale generated output (build/, dist/, *.egg-info/, caches, attack
    fixtures, *.pyc) is never copied, so it cannot leak into the release
    wheel or sdist.
    """

    def _ignore(dirpath: str, names: list[str]) -> list[str]:
        ignored: list[str] = []
        for name in names:
            if name in _CLEAN_COPY_IGNORE_DIRS:
                ignored.append(name)
            elif name.endswith(_CLEAN_COPY_IGNORE_SUFFIXES):
                ignored.append(name)
            elif any(sub in name for sub in _CLEAN_COPY_IGNORE_SUBSTRINGS):
                ignored.append(name)
            elif name == ".DS_Store" or name.endswith(".pyc"):
                ignored.append(name)
        return ignored

    dest = dest_parent / "clean-source"
    shutil.copytree(repo, dest, ignore=_ignore, symlinks=False)
    return dest


def clean_build(
    tool_py: Path, repo: Path, outdir: Path, *, no_isolation: bool = False
) -> tuple[Path, Path]:
    """Build sdist + wheel from a clean source copy.

    Stale generated directories (build/, dist/, src/*.egg-info/) are removed
    from the repo tree first and are never copied into the build tree, so a
    stale build/lib module cannot enter the release wheel after the
    corresponding source file was removed.

    Returns (wheel_path, sdist_path) inside ``outdir``.
    """
    for stale_dir in (repo / "build", repo / "dist"):
        shutil.rmtree(stale_dir, ignore_errors=True)
    for egg_info in (repo / "src").glob("*.egg-info"):
        shutil.rmtree(egg_info, ignore_errors=True)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix="telemetry_resilience_clean_build_"))
    try:
        clean_repo = _clean_source_copy(repo, build_root)
        cmd = [str(tool_py), "-m", "build", "--outdir", str(outdir)]
        if no_isolation:
            cmd.append("--no-isolation")
        _run(cmd, cwd=clean_repo, timeout=1200)
    finally:
        shutil.rmtree(build_root, ignore_errors=True)
    wheels = sorted(outdir.glob("*.whl"))
    sdists = sorted(outdir.glob("*.tar.gz"))
    if not wheels:
        raise CheckFailed("no wheel produced")
    if not sdists:
        raise CheckFailed("no sdist produced")
    return wheels[0], sdists[0]


def _clean_repo_build_debris(repo: Path) -> list[str]:
    """Remove generated build debris from the repo tree; return what was removed."""
    removed: list[str] = []
    for stale_dir in (repo / "build",):
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
            removed.append("build/")
    for egg_info in (repo / "src").glob("*.egg-info"):
        shutil.rmtree(egg_info, ignore_errors=True)
        removed.append(f"src/{egg_info.name}/")
    for dirpath, dirnames, filenames in os.walk(repo):
        if ".venv" in Path(dirpath).parts:
            dirnames[:] = []
            continue
        for dirname in list(dirnames):
            if dirname in ("__pycache__", ".pytest_cache"):
                shutil.rmtree(Path(dirpath) / dirname, ignore_errors=True)
                removed.append(f"{dirname}/")
                dirnames.remove(dirname)
        for filename in filenames:
            if filename.endswith(".pyc"):
                Path(dirpath, filename).unlink(missing_ok=True)
                removed.append(filename)
    return removed


def _verify_repo_hygiene(repo: Path) -> list[str]:
    """List generated-debris problems in the repo tree (empty == clean)."""
    problems: list[str] = []
    if (repo / "build").exists():
        problems.append("build/ present")
    egg_infos = list((repo / "src").glob("*.egg-info"))
    if egg_infos:
        problems.append(f"*.egg-info present: {[e.name for e in egg_infos]}")
    for cache in ("__pycache__", ".pytest_cache"):
        hits = [
            d for d in repo.rglob(cache)
            if ".venv" not in d.parts and "clean-source" not in d.parts
        ]
        if hits:
            problems.append(f"{cache}/ present ({len(hits)} dirs)")
    pyc = [
        p for p in repo.rglob("*.pyc")
        if ".venv" not in p.parts and "clean-source" not in p.parts
    ]
    if pyc:
        problems.append(f"*.pyc present ({len(pyc)} files)")
    for name in ("attack_evidence_run22",):
        if (repo / name).exists():
            problems.append(f"{name}/ present")
    return problems


def _archive_members(archive: Path):
    """Yield (entry_name, read_bytes_callable) for every file in the archive."""
    if archive.suffix == ".whl" or zipfile.is_zipfile(archive):
        zf = zipfile.ZipFile(archive)
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            yield name, (lambda n=name: zf.read(n))
        zf.close()
    else:
        tf = tarfile.open(archive)
        for member in tf.getmembers():
            if not member.isfile():
                continue
            yield member.name, (lambda m=member: tf.extractfile(m).read())
        tf.close()


def _template_file_names(repo: Path) -> tuple[str, ...]:
    """Parse ``TEMPLATE_FILES`` from src/telemetry_resilience/demo.py.

    Stdlib only (this script must not import the package under test); the
    audit then requires every bundled template in both the wheel and the
    sdist file lists.
    """
    text = (repo / "src" / "telemetry_resilience" / "demo.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^TEMPLATE_FILES\s*=\s*\((.*?)\)", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise CheckFailed(
            "could not parse TEMPLATE_FILES from src/telemetry_resilience/demo.py"
        )
    names = tuple(re.findall(r'"([^"]+)"', match.group(1)))
    if not names:
        raise CheckFailed("TEMPLATE_FILES parsed empty from src/telemetry_resilience/demo.py")
    return names


def _audit_entry_names(
    entries: list[str], *, is_wheel: bool, template_names: tuple[str, ...]
) -> list[str]:
    problems: list[str] = []
    dist_info_prefix = None
    if is_wheel:
        # e.g. telemetry_resilience-0.1.0.dist-info/
        for name in entries:
            if name.endswith(".dist-info/METADATA"):
                dist_info_prefix = name[: -len("METADATA")]
                break
    has_templates = False
    for name in entries:
        lowered = name.lower()
        if name.startswith(TEMPLATES_PREFIX):
            has_templates = True
        for frag in BAD_NAME_FRAGMENTS:
            if frag in lowered:
                problems.append(f"forbidden name fragment {frag!r} in {name!r}")
        for marker in MACHINE_PATH_MARKERS:
            if marker.lower() in lowered:
                problems.append(f"machine path marker {marker!r} in {name!r}")
        if name.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", name):
            if not (dist_info_prefix and name.startswith(dist_info_prefix)):
                problems.append(f"absolute path in {name!r}")
        if ".." in Path(name).parts:
            problems.append(f"parent-directory traversal in {name!r}")
    if is_wheel and not has_templates:
        problems.append(
            f"wheel is missing bundled templates ({TEMPLATES_PREFIX}*)"
        )
    # Every bundled template must ship in both the wheel and the sdist file
    # lists (archive entries always use '/' separators).
    kind = "wheel" if is_wheel else "sdist"
    for tname in template_names:
        if not any(e.endswith(f"/templates/{tname}") for e in entries):
            problems.append(f"{kind} is missing bundled template {tname!r}")
    return problems


def _audit_entry_contents(archive: Path) -> list[str]:
    """Scan text-like archive contents for machine paths and secret markers.

    Only UTF-8-decodable, NUL-free members are treated as text. Reports name
    the entry and the matched marker/pattern -- never any secret content.
    """
    problems: list[str] = []
    for name, read in _archive_members(archive):
        try:
            data = read()
        except Exception:
            continue
        if len(data) > CONTENT_SCAN_SIZE_CAP:
            continue
        if b"\x00" in data[:4096]:
            continue  # binary: not text-like
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        reported_paths: set[str] = set()
        for match in HOME_PATH_RE.finditer(text):
            segment = match.group(1)
            if segment.lower() in GENERIC_USER_SEGMENTS:
                continue  # intentionally written generic test string
            reported_paths.add(match.group(0))
        for match in WIN_USERS_RE.finditer(text):
            segment = match.group(1)
            if segment.lower() in GENERIC_USER_SEGMENTS:
                continue
            reported_paths.add(match.group(0))
        for pattern in TEMP_BUILD_DIR_RES:
            for match in pattern.finditer(text):
                reported_paths.add(match.group(0))
        for path in sorted(reported_paths):
            problems.append(
                f"machine-specific path {path!r} in contents of {name!r}"
            )
        lowered = text.lower()
        for marker in SECRET_CONTENT_MARKERS:
            if marker.lower() in lowered:
                problems.append(
                    f"secret marker {marker!r} in contents of {name!r}"
                )
    return problems


def _audit_archive(
    archive: Path, *, is_wheel: bool, template_names: tuple[str, ...]
) -> None:
    entries = [name for name, _ in _archive_members(archive)]
    if not entries:
        raise CheckFailed(f"{archive.name}: archive is empty")
    problems = _audit_entry_names(entries, is_wheel=is_wheel, template_names=template_names)
    problems += _audit_entry_contents(archive)
    if problems:
        raise CheckFailed(
            f"{archive.name}: {len(problems)} archive problem(s):\n  - "
            + "\n  - ".join(problems[:20])
        )
    _log(f"  {archive.name}: {len(entries)} entries, clean")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_cli(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "telemetry-resilience.exe"
    return venv_dir / "bin" / "telemetry-resilience"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-release check (stdlib only).")
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repository root (default: parent of scripts/).",
    )
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "pyproject.toml").exists():
        print(f"RELEASE CHECK: FAIL\nnot a repo root: {repo}")
        return 1

    version = _repo_version(repo)
    expected_version_line = f"Telemetry Resilience CLI {version}"
    dist_dir = repo / "dist"

    tmp_root = Path(tempfile.mkdtemp(prefix="telemetry_resilience_release_check_"))
    tooling_venv = tmp_root / "tooling-venv"
    wheel_venv = tmp_root / "wheel-venv"
    work_dir = tmp_root / "work"
    work_dir.mkdir()

    failures: list[str] = []

    def step(title: str, fn) -> None:
        _log(f"[step] {title}")
        try:
            fn()
        except CheckFailed as exc:
            failures.append(f"{title}: {exc}")
            _log(f"  FAIL: {exc}")
        else:
            _log("  ok")

    wheel_path: dict[str, Path] = {}

    def step_tooling_venv():
        # Fresh venv; install the project NON-editable with test extras,
        # plus the `build` package. No reliance on a prior `pip install -e .`.
        # setuptools+wheel are installed so `python -m build --no-isolation`
        # works inside this self-contained venv (the stale-build regression
        # test builds with --no-isolation and needs the setuptools backend).
        _run([sys.executable, "-m", "venv", str(tooling_venv)], timeout=300)
        tool_py = _venv_python(tooling_venv)
        if not tool_py.exists():
            raise CheckFailed(f"tooling venv python not created at {tool_py}")
        _run(
            [tool_py, "-m", "pip", "install", "--quiet", f"{repo}[test]"],
            timeout=1800,
        )
        _run(
            [
                tool_py,
                "-m",
                "pip",
                "install",
                "--quiet",
                "build",
                "setuptools>=68",
                "wheel",
            ],
            timeout=900,
        )

    def step_pytest():
        tool_py = _venv_python(tooling_venv)
        _run([tool_py, "-m", "pytest", "-q"], cwd=repo, timeout=1800)

    def step_repo_hygiene():
        # Verify the original source tree is clean BEFORE running tests that
        # create new caches. Clean generated debris first, then verify.
        removed = _clean_repo_build_debris(repo)
        if removed:
            _log(f"  removed generated debris: {sorted(set(removed))[:8]}")
        problems = _verify_repo_hygiene(repo)
        if problems:
            raise CheckFailed(
                "repo tree not clean before tests: " + "; ".join(problems)
            )

    def step_build():
        tool_py = _venv_python(tooling_venv)
        wheel, sdist = clean_build(tool_py, repo, dist_dir)
        wheel_path["wheel"] = wheel
        _log(f"  built: {wheel.name}, {sdist.name}")

    def step_audit():
        template_names = _template_file_names(repo)
        for archive in sorted(dist_dir.glob("*.whl")) + sorted(
            dist_dir.glob("*.tar.gz")
        ):
            _audit_archive(
                archive,
                is_wheel=archive.suffix == ".whl",
                template_names=template_names,
            )

    def step_wheel_venv_install():
        _run([sys.executable, "-m", "venv", str(wheel_venv)], timeout=300)
        wheel_py = _venv_python(wheel_venv)
        if not wheel_py.exists():
            raise CheckFailed(f"wheel venv python not created at {wheel_py}")
        # Install ONLY the built wheel (its declared dependencies come
        # from the wheel metadata via pip as normal). No project source,
        # no editable install.
        _run(
            [wheel_py, "-m", "pip", "install", "--quiet", str(wheel_path["wheel"])],
            timeout=1800,
        )
        cli = _venv_cli(wheel_venv)
        if not cli.exists():
            raise CheckFailed(f"console script not installed at {cli}")

    def step_wheel_smoke():
        cli = _venv_cli(wheel_venv)
        wheel_py = _venv_python(wheel_venv)
        # Prove the packaged demo works with no source checkout: scaffold a
        # demo project from the wheel-venv CLI, cwd inside work/.
        _run(
            [cli, "demo", "init", "telemetry-demo"],
            timeout=600,
            cwd=work_dir,
        )
        demo_dir = work_dir / "telemetry-demo"
        if not (demo_dir / "demo_drive.parquet").is_file():
            raise CheckFailed("demo init did not produce demo_drive.parquet")

        # Documented quick start, from the demo project directory.
        proc = _run([cli, "doctor"], timeout=300, cwd=demo_dir)
        if "READY" not in proc.stdout:
            raise CheckFailed(
                f"doctor output did not contain READY:\n{proc.stdout.strip()}"
            )
        _run([cli, "inspect", "demo_drive.parquet"], timeout=300, cwd=demo_dir)

        # The scenario file must come from the wheel-installed package ONLY.
        # `demo init` copies the packaged degraded_navigation.yaml into the
        # demo project. Assert it is byte-identical to the template shipped
        # inside the built wheel archive, then use it directly. NOTHING is
        # copied from the source repository into the wheel smoke-test
        # directory at any point.
        scenario_path = demo_dir / DEMO_SCENARIO_NAME
        if not scenario_path.is_file():
            raise CheckFailed(
                f"demo init did not produce {DEMO_SCENARIO_NAME} from the wheel"
            )
        packaged_scenario: bytes | None = None
        with zipfile.ZipFile(wheel_path["wheel"]) as zf:
            for name in zf.namelist():
                if name.endswith(f"/templates/{DEMO_SCENARIO_NAME}"):
                    packaged_scenario = zf.read(name)
                    break
        if packaged_scenario is None:
            raise CheckFailed(
                f"built wheel does not ship templates/{DEMO_SCENARIO_NAME}"
            )
        if scenario_path.read_bytes() != packaged_scenario:
            raise CheckFailed(
                f"{DEMO_SCENARIO_NAME} in the demo project differs from the "
                "template packaged in the wheel"
            )
        _log("  packaged degraded_navigation.yaml verified byte-identical")
        _run(
            [cli, "inject", "demo_drive.parquet", "--scenario", DEMO_SCENARIO_NAME],
            timeout=600,
            cwd=demo_dir,
        )
        if not (demo_dir / "demo_drive.corrupted.parquet").is_file():
            raise CheckFailed("inject did not produce demo_drive.corrupted.parquet")
        if not (demo_dir / "demo_drive.faults.json").is_file():
            raise CheckFailed("inject did not produce demo_drive.faults.json")

        _run(
            [cli, "campaign", "navigation_campaign.yaml", "--plan"],
            timeout=600,
            cwd=demo_dir,
        )
        proc = _run(
            [cli, "campaign", "navigation_campaign.yaml", "--list-cases"],
            timeout=600,
            cwd=demo_dir,
        )
        if not proc.stdout.strip():
            raise CheckFailed("--list-cases printed no cases")

        # Baseline sanity: the clean demo telemetry must read NOMINAL.
        proc = _run(
            [wheel_py, "demo_app.py", "demo_drive.parquet"],
            timeout=300,
            cwd=demo_dir,
        )
        if "NOMINAL" not in proc.stdout:
            raise CheckFailed(
                f"demo_app.py baseline did not print NOMINAL:\n{proc.stdout.strip()}"
            )

        # One FULL campaign run against demo_app.py.
        target = [str(wheel_py), "demo_app.py", "{data}"]
        _run(
            [
                cli,
                "campaign",
                "navigation_campaign.yaml",
                "--artifacts",
                "resilience-results",
                "--",
                *target,
            ],
            timeout=1800,
            cwd=demo_dir,
        )

    def step_artifacts():
        artifacts = work_dir / "telemetry-demo" / "resilience-results"
        for name in ("coverage.json", "summary.json", "junit.xml"):
            if not (artifacts / name).is_file():
                raise CheckFailed(f"missing campaign artifact: {name}")

    def step_uninstall():
        wheel_py = _venv_python(wheel_venv)
        _run(
            [wheel_py, "-m", "pip", "uninstall", "-y", PACKAGE_NAME],
            timeout=600,
        )
        proc = subprocess.run(
            [str(wheel_py), "-c", "import telemetry_resilience"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            raise CheckFailed("package still importable after uninstall")

    try:
        step("repo hygiene: clean and verify before tests", step_repo_hygiene)
        step("tooling venv: install <repo>[test] + build/setuptools/wheel (non-editable)", step_tooling_venv)
        if _venv_python(tooling_venv).exists():
            step("full pytest suite", step_pytest)
            step("python -m build", step_build)
        else:
            failures.append("tooling venv: skipped (venv python missing)")
        if "wheel" not in wheel_path:
            failures.append("archive audit: skipped (no wheel built)")
        else:
            step("audit wheel+sdist contents", step_audit)
            step("wheel venv: install ONLY the built wheel", step_wheel_venv_install)
            if _venv_cli(wheel_venv).exists():
                step("wheel smoke test (demo quick start)", step_wheel_smoke)
                step("campaign artifacts present", step_artifacts)
                step("uninstall package from wheel venv", step_uninstall)
            else:
                failures.append("wheel smoke test: skipped (console script missing)")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        _log(f"cleaned up {tmp_root}")

    if failures:
        print("RELEASE CHECK: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RELEASE CHECK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
