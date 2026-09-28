"""Packaging coherence tests: version agreement and bundled template data."""
import importlib.metadata as importlib_metadata
import importlib.resources as importlib_resources
import re
from pathlib import Path

import pytest

from telemetry_resilience import __version__
from telemetry_resilience.demo import TEMPLATE_FILES

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    """Resolve the declared project version from pyproject.toml.

    Supports both a static ``version = "x.y.z"`` and the single-source
    dynamic form ``dynamic = ["version"]`` with
    ``[tool.setuptools.dynamic] version = {attr = "telemetry_resilience.__version__"}``.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match:
        return match.group(1)
    dyn = re.search(r'^dynamic\s*=\s*\[([^\]]*)\]', text, re.MULTILINE)
    assert dyn and '"version"' in dyn.group(1), (
        "version is neither static nor dynamic in pyproject.toml"
    )
    attr = re.search(r'version\s*=\s*\{\s*attr\s*=\s*"([^"]+)"', text)
    assert attr, "no [tool.setuptools.dynamic] version attr found in pyproject.toml"
    assert attr.group(1) == "telemetry_resilience.__version__", (
        f"dynamic version must point at telemetry_resilience.__version__, "
        f"found {attr.group(1)!r}"
    )
    return __version__  # the single authoritative source


def test_pyproject_version_matches_package_version():
    assert _pyproject_version() == __version__


def test_installed_metadata_version_matches_package_version():
    try:
        installed = importlib_metadata.version("telemetry-resilience")
    except importlib_metadata.PackageNotFoundError:
        pytest.skip("telemetry-resilience is not installed")
    assert installed == __version__


def test_templates_present_via_importlib_resources():
    templates = importlib_resources.files("telemetry_resilience") / "templates"
    assert templates.is_dir()
    for name in TEMPLATE_FILES:
        resource = templates / name
        assert resource.is_file(), f"template missing from package data: {name}"
        assert len(resource.read_bytes()) > 0
