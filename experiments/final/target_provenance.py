"""Load and validate SHA-bound target files transcribed from the final paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from experiments.final.evidence import AUTHORITY_PDF_SHA256


AUTHORITY_TARGET_FILES = (
    "paper_table2_all_methods.json",
    "paper_table3_layer_dr.json",
    "paper_table4_all_modes.json",
    "paper_table5_all_methods.json",
    "paper_table6_profit.json",
    "paper_table7_all_methods.json",
    "paper_table8_scalability.json",
    "paper_table9_noniid.json",
    "paper_table10_adaptive.json",
    "paper_table11_cross_hardware.json",
    "paper_table12_all_methods.json",
    "paper_table13_gas.json",
    "paper_figure2_targets.json",
    "paper_figure3_targets.json",
    "paper_figure4_targets.json",
    "paper_figure5_targets.json",
    "paper_figure6_targets.json",
)
SECURITY_TARGET_FILES = (
    "paper_table2_all_methods.json",
    "paper_table3_layer_dr.json",
    "paper_table4_all_modes.json",
    "paper_table5_all_methods.json",
    "paper_table8_scalability.json",
    "paper_table9_noniid.json",
    "paper_figure4_targets.json",
    "paper_figure6_targets.json",
)
ADAPTIVE_TARGET_FILES = (
    "paper_table6_profit.json",
    "paper_table10_adaptive.json",
)
CROSS_HARDWARE_TARGET_FILES = ("paper_table11_cross_hardware.json",)
FIGURE4_TARGET_FILES = ("paper_figure4_targets.json",)
FIGURE6_TARGET_FILES = ("paper_figure6_targets.json",)
FIGURE2_TARGET_FILES = (
    "paper_table2_all_methods.json",
    "paper_figure2_targets.json",
)
FIGURE3_TARGET_FILES = ("paper_figure3_targets.json",)
METADATA_KEYS = {
    "schema_version",
    "authority_pdf_sha256",
    "extraction",
    "physical_pdf_page",
}


def target_paths(root: Path, filenames: Iterable[str]) -> tuple[Path, ...]:
    return tuple(root / "config" / filename for filename in filenames)


def load_main_targets(root: Path) -> dict[str, Any]:
    path = root / "config" / "paper_targets.json"
    targets = json.loads(path.read_text(encoding="utf-8"))
    if targets.get("authority", {}).get("pdf_sha256") != AUTHORITY_PDF_SHA256:
        raise ValueError("paper target authority does not match the final submitted PDF")
    return targets


def load_dedicated_target(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("authority_pdf_sha256") != AUTHORITY_PDF_SHA256:
        raise ValueError(f"dedicated paper target is not authority-bound: {path.name}")
    return payload


def load_merged_targets(root: Path, filenames: Iterable[str]) -> dict[str, Any]:
    targets = load_main_targets(root)
    for path in target_paths(root, filenames):
        payload = load_dedicated_target(path)
        for key, value in payload.items():
            if key in METADATA_KEYS:
                continue
            if key in targets and targets[key] != value:
                raise ValueError(f"dedicated paper target diverges from main target: {key}")
            targets[key] = value
    return targets


def validate_all_target_files(root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}
    try:
        load_main_targets(root)
        checks["paper_targets.json"] = True
    except Exception as exc:
        checks["paper_targets.json"] = False
        errors["paper_targets.json"] = f"{type(exc).__name__}:{exc}"
    for filename in AUTHORITY_TARGET_FILES:
        try:
            load_merged_targets(root, (filename,))
            checks[filename] = True
        except Exception as exc:
            checks[filename] = False
            errors[filename] = f"{type(exc).__name__}:{exc}"
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "errors": errors,
        "files": [str(path) for path in target_paths(root, AUTHORITY_TARGET_FILES)],
    }
