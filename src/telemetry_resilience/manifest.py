"""Fault manifest and run report builders."""
import json
from pathlib import Path

from . import __version__


def build_manifest(scenario, source_file: str, fault_entries: list) -> dict:
    """Exact record of what was injected (written next to the output data)."""
    return {
        "version": 1,
        "seed": scenario.seed,
        "source_file": str(source_file),
        "software_version": __version__,
        "faults": fault_entries,
    }


def build_report(
    scenario, source_file: str, output_file: str, fault_entries: list
) -> dict:
    """Small summary of a run for stdout / CI checks."""
    return {
        "version": 1,
        "source_file": str(source_file),
        "output_file": str(output_file),
        "seed": scenario.seed,
        "software_version": __version__,
        "faults_applied": len(fault_entries),
        "total_affected_observations": sum(
            entry.get("affected_observations", 0) for entry in fault_entries
        ),
    }


def write_json(path, payload):
    """Write a JSON payload with 2-space indent (non-serializables via str)."""
    Path(path).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
