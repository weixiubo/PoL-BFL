#!/usr/bin/env python3
"""Audit and seal paper-to-implementation-to-test traceability."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from experiments.final.evidence import AUTHORITY_PDF_SHA256, seal_evidence
from experiments.final.manifest import sha256_file, source_identity


REQUIRED_CATEGORIES = {
    "protocol", "zk", "committee", "aggregation", "incentives",
    "contract", "experiments", "operations", "evidence", "theory",
    "storage", "threat_model",
}
REQUIREMENT_ID = re.compile(r"^[a-z0-9]+(?:[._][a-z0-9]+)+$")


def _safe_relative_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value or any(
        marker in value for marker in ("*", "?", "[", "]")
    ):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def traceability_input_paths(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[Path, ...]:
    paths = {manifest_path.resolve()}
    documentation = _safe_relative_path(manifest.get("documentation"))
    if documentation is not None:
        paths.add((root / documentation).resolve())
    for requirement in manifest.get("requirements", ()):
        for value in (*requirement.get("owners", ()), *requirement.get("tests", ())):
            relative = _safe_relative_path(value)
            if relative is not None:
                paths.add((root / relative).resolve())
    return tuple(sorted(paths))


def audit_traceability(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    paper_sha256: str,
) -> dict[str, Any]:
    requirements = tuple(manifest.get("requirements", ()))
    documentation = _safe_relative_path(manifest.get("documentation"))
    documentation_path = None if documentation is None else root / documentation
    documentation_rows = 0
    if documentation_path is not None and documentation_path.is_file():
        in_table = False
        for line in documentation_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("| Paper component", "| Paper result")):
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                continue
            if in_table and line.startswith("|"):
                documentation_rows += 1
            elif in_table and not line.strip():
                in_table = False
    ids = [str(row.get("id", "")) for row in requirements]
    categories = {str(row.get("category", "")) for row in requirements}
    checks: dict[str, bool] = {
        "schema_version": manifest.get("schema_version") == 1,
        "authority_manifest": manifest.get("authority_pdf_sha256") == AUTHORITY_PDF_SHA256,
        "authority_pdf": paper_sha256 == AUTHORITY_PDF_SHA256,
        "requirement_count": len(requirements) == 43,
        "requirement_ids_unique": len(set(ids)) == len(ids),
        "categories_complete": categories == REQUIRED_CATEGORIES,
        "documentation_declared": documentation is not None,
        "documentation_exists": documentation_path is not None
        and documentation_path.is_file(),
        "documentation_requirement_count": documentation_rows == 32,
    }
    records = []
    for index, requirement in enumerate(requirements):
        identifier = str(requirement.get("id", ""))
        owners = tuple(requirement.get("owners", ()))
        tests = tuple(requirement.get("tests", ()))
        owner_paths = [_safe_relative_path(value) for value in owners]
        test_paths = [_safe_relative_path(value) for value in tests]
        python_test_paths = [
            path for path in test_paths if path is not None and path.suffix == ".py"
        ]
        auxiliary_test_paths = [
            path for path in test_paths if path is not None and path.suffix != ".py"
        ]
        auxiliary_tests_driven = all(
            any(
                auxiliary.name
                in (root / python_test).read_text(encoding="utf-8")
                for python_test in python_test_paths
                if (root / python_test).is_file()
            )
            for auxiliary in auxiliary_test_paths
        )
        local_checks = {
            "id": bool(REQUIREMENT_ID.fullmatch(identifier)),
            "category": requirement.get("category") in REQUIRED_CATEGORIES,
            "paper_locator": bool(str(requirement.get("paper_locator", "")).strip()),
            "requirement": bool(str(requirement.get("requirement", "")).strip()),
            "owners_nonempty": bool(owners),
            "tests_nonempty": bool(tests),
            "owner_paths_exact": all(path is not None for path in owner_paths),
            "test_paths_exact": all(path is not None for path in test_paths),
            "owners_exist": bool(owners) and all(
                path is not None and (root / path).is_file() for path in owner_paths
            ),
            "tests_exist": bool(tests) and all(
                path is not None and (root / path).is_file() for path in test_paths
            ),
            "tests_under_tests": bool(tests) and all(
                path is not None and path.parts[0] == "tests" for path in test_paths
            ),
            "auxiliary_tests_driven_by_python": auxiliary_tests_driven,
        }
        for name, passed in local_checks.items():
            checks[f"requirement:{index}:{identifier}:{name}"] = passed
        records.append(
            {
                "id": identifier,
                "category": requirement.get("category"),
                "paper_locator": requirement.get("paper_locator"),
                "owner_count": len(owners),
                "test_count": len(tests),
                "passed": all(local_checks.values()),
                "checks": local_checks,
            }
        )
    return {
        "schema_version": 1,
        "kind": "paper_code_test_traceability",
        "requirement_count": len(requirements),
        "category_count": len(categories),
        "categories": sorted(categories),
        "owner_reference_count": sum(len(tuple(row.get("owners", ()))) for row in requirements),
        "test_reference_count": sum(len(tuple(row.get("tests", ()))) for row in requirements),
        "auxiliary_test_reference_count": sum(
            1
            for row in requirements
            for value in row.get("tests", ())
            if Path(str(value)).suffix != ".py"
        ),
        "documentation_requirement_count": documentation_rows,
        "records": records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "config" / "paper_traceability.json",
    )
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit_traceability(
        manifest,
        root=root,
        paper_sha256=sha256_file(args.paper.resolve()),
    )
    inputs = traceability_input_paths(args.manifest, manifest, root=root)
    report["input_sha256"] = {
        (str(path.relative_to(root)) if path.is_relative_to(root) else str(path)):
        sha256_file(path)
        for path in (*inputs, args.paper.resolve())
    }
    source = source_identity(root)
    report = seal_evidence(
        report, source_commit=source["commit"], analysis_source=source
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
