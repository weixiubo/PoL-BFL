#!/usr/bin/env python3
"""Fail-closed audit of accepted evidence for all 17 paper result routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from experiments.final.audit_final_coverage import COVERAGE
from experiments.final.evidence import (
    AUTHORITY_PDF_SHA256,
    CANONICAL_DIGEST_SCHEME,
    canonical_digest_matches,
    seal_evidence,
    valid_source_commit,
)
from experiments.final.manifest import sha256_file, source_identity


FORMAL_CELL_ROUTES = frozenset(
    {
        "table_2_main_security",
        "table_3_layer_contribution",
        "table_4_composability",
        "table_5_incentive_effectiveness",
        "table_7_system_overhead",
        "table_8_scalability",
        "table_9_noniid",
        "figure_2_convergence",
        "figure_3_reputation_evolution",
        "figure_4_spot_check_sensitivity",
        "figure_6_sybil_scalability",
    }
)


def _verify_input_hashes(
    input_hashes: object,
    *,
    root: Path,
    seen: set[Path] | None = None,
) -> tuple[bool, bool, bool, int]:
    declared = isinstance(input_hashes, Mapping) and bool(input_hashes)
    if not declared:
        return False, False, False, 0
    seen = set() if seen is None else seen
    files_exist = True
    hashes_match = True
    count = 0
    for label, expected_value in input_hashes.items():
        expected = str(expected_value)
        input_path = Path(str(label))
        if not input_path.is_absolute():
            input_path = (root / input_path).resolve()
        else:
            input_path = input_path.resolve()
        valid_expected = (
            len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected)
        )
        exists = input_path.is_file()
        files_exist = files_exist and exists
        matched = bool(exists and valid_expected and sha256_file(input_path) == expected)
        hashes_match = hashes_match and matched
        count += 1
        if matched and input_path.suffix.lower() == ".json" and input_path not in seen:
            seen.add(input_path)
            try:
                nested_payload = json.loads(input_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                nested_payload = None
            nested_is_sealed = bool(
                isinstance(nested_payload, Mapping)
                and "evidence_digest_scheme" in nested_payload
            )
            nested_digest_matches = bool(
                nested_is_sealed and canonical_digest_matches(nested_payload)
            )
            if nested_is_sealed:
                hashes_match = hashes_match and nested_digest_matches
            if nested_digest_matches and isinstance(
                nested_payload.get("input_sha256"), Mapping
            ):
                nested_declared, nested_exists, nested_match, nested_count = (
                    _verify_input_hashes(
                        nested_payload["input_sha256"],
                        root=root,
                        seen=seen,
                    )
                )
                files_exist = files_exist and nested_declared and nested_exists
                hashes_match = hashes_match and nested_declared and nested_match
                count += nested_count
    return declared, files_exist, hashes_match, count


def _accepted(payload: Mapping[str, Any]) -> bool:
    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        return acceptance.get("passed") is True
    return (
        payload.get("passed") is True
        or payload.get("formal_accepted") is True
    )


def _source_commit(payload: Mapping[str, Any]) -> str | None:
    candidates = (
        payload.get("source_commit"),
        payload.get("source", {}).get("commit")
        if isinstance(payload.get("source"), Mapping)
        else None,
        payload.get("provenance", {}).get("source_commit")
        if isinstance(payload.get("provenance"), Mapping)
        else None,
    )
    values = {
        str(value)
        for value in candidates
        if isinstance(value, str) and valid_source_commit(value)
    }
    if len(values) > 1:
        raise ValueError("evidence contains conflicting source commits")
    return next(iter(values), None)


def audit_measurements(
    evidence_map: Mapping[str, Any],
    *,
    root: Path,
    formal_cell_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if (
        evidence_map.get("authority_pdf_sha256")
        != AUTHORITY_PDF_SHA256
    ):
        raise ValueError("final evidence map does not bind the authority PDF")
    declared_commit = str(evidence_map.get("source_commit", ""))
    if not valid_source_commit(declared_commit):
        raise ValueError("final evidence map source commit is invalid")
    evidence = evidence_map.get("evidence", {})
    if set(evidence) != set(COVERAGE):
        missing = sorted(set(COVERAGE) - set(evidence))
        extra = sorted(set(evidence) - set(COVERAGE))
        raise ValueError(
            "final evidence map route mismatch: "
            + "missing="
            + ",".join(missing)
            + ";extra="
            + ",".join(extra)
        )
    checks = {}
    records = {}
    verified_formal_cells: dict[Path, tuple[bool, str | None]] = {}
    if formal_cell_verifier is None:
        from scripts.capture_formal_evidence import verify_completed_cell

        formal_cell_verifier = verify_completed_cell
    for study in sorted(COVERAGE):
        path = Path(str(evidence[study]))
        if not path.is_absolute():
            path = (root / path).resolve()
        exists = path.is_file()
        checks[study + ".exists"] = exists
        if not exists:
            records[study] = {
                "path": str(path),
                "accepted": False,
                "reason": "missing",
            }
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        accepted = _accepted(payload)
        commit = _source_commit(payload)
        commit_matches = commit == declared_commit
        digest = str(payload.get("evidence_digest", ""))
        digest_shape = (
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        )
        digest_scheme = payload.get("evidence_digest_scheme") == CANONICAL_DIGEST_SCHEME
        authority_matches = payload.get("authority_pdf_sha256") == AUTHORITY_PDF_SHA256
        analysis_source = payload.get("analysis_source")
        analysis_source_matches = bool(
            isinstance(analysis_source, Mapping)
            and analysis_source.get("dirty") is False
            and analysis_source.get("commit") == declared_commit
        )
        digest_content = canonical_digest_matches(payload)
        input_hashes = payload.get("input_sha256")
        inputs_declared, input_files_exist, input_hashes_match, input_count = (
            _verify_input_hashes(input_hashes, root=root)
        )
        requires_formal_cells = study in FORMAL_CELL_ROUTES
        formal_paths = payload.get("formal_result_paths")
        formal_results_declared = (
            not requires_formal_cells
            or (
                isinstance(formal_paths, list)
                and bool(formal_paths)
                and len(formal_paths) == len(set(map(str, formal_paths)))
            )
        )
        formal_results_verified = formal_results_declared
        formal_results_source = formal_results_declared
        formal_error = None
        if requires_formal_cells and formal_results_declared:
            for label in formal_paths:
                formal_path = Path(str(label))
                if not formal_path.is_absolute():
                    formal_path = (root / formal_path).resolve()
                if formal_path not in verified_formal_cells:
                    try:
                        receipt = formal_cell_verifier(formal_path, root=root)
                        verified_formal_cells[formal_path] = (
                            receipt.get("source_commit") == declared_commit,
                            None,
                        )
                    except Exception as exc:  # fail closed with an auditable reason
                        verified_formal_cells[formal_path] = (
                            False,
                            f"{type(exc).__name__}:{exc}",
                        )
                verified, error = verified_formal_cells[formal_path]
                formal_results_verified = formal_results_verified and verified
                formal_results_source = formal_results_source and verified
                formal_error = formal_error or error
        checks[study + ".accepted"] = accepted
        checks[study + ".source_commit"] = commit_matches
        checks[study + ".evidence_digest"] = digest_shape
        checks[study + ".evidence_scheme"] = digest_scheme
        checks[study + ".evidence_content"] = digest_content
        checks[study + ".authority_pdf"] = authority_matches
        checks[study + ".analysis_source"] = analysis_source_matches
        checks[study + ".inputs_declared"] = inputs_declared
        checks[study + ".input_files_exist"] = input_files_exist
        checks[study + ".input_hashes"] = input_hashes_match
        checks[study + ".formal_results_declared"] = formal_results_declared
        checks[study + ".formal_results_verified"] = formal_results_verified
        checks[study + ".formal_results_source"] = formal_results_source
        records[study] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "accepted": accepted,
            "source_commit": commit,
            "evidence_digest": digest or None,
            "input_count": input_count,
            "formal_result_count": len(formal_paths) if isinstance(formal_paths, list) else 0,
            "formal_error": formal_error,
        }
    complete = bool(checks) and all(checks.values())
    return {
        "schema_version": 1,
        "authority_pdf_sha256": AUTHORITY_PDF_SHA256,
        "source_commit": declared_commit,
        "measurement_complete": complete,
        "passed": complete,
        "checks": checks,
        "records": records,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-map", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence_map = json.loads(args.evidence_map.read_text(encoding="utf-8"))
    report = audit_measurements(
        evidence_map,
        root=root,
    )
    deployed_source = source_identity(root)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    preflight_inputs = _verify_input_hashes(preflight.get("input_sha256"), root=root)
    final_checks = {
        "deployed_source_clean": deployed_source.get("dirty") is False,
        "deployed_source_commit": deployed_source.get("commit") == report["source_commit"],
        "paper_digest": args.paper.is_file()
        and sha256_file(args.paper) == AUTHORITY_PDF_SHA256,
        "preflight_passed": preflight.get("passed") is True
        and all(preflight.get("checks", {}).values()),
        "preflight_authority": preflight.get("authority_pdf_sha256")
        == AUTHORITY_PDF_SHA256,
        "preflight_source": preflight.get("source_commit") == report["source_commit"],
        "preflight_digest": canonical_digest_matches(preflight),
        "preflight_inputs": all(preflight_inputs[:3]),
    }
    report["checks"].update(
        {"final." + name: passed for name, passed in final_checks.items()}
    )
    report["measurement_complete"] = bool(report["checks"]) and all(
        report["checks"].values()
    )
    report["passed"] = report["measurement_complete"]
    report["source"] = deployed_source
    report["preflight_evidence_digest"] = preflight.get("evidence_digest")
    report["input_sha256"] = {
        str(args.evidence_map.resolve()): sha256_file(args.evidence_map),
        str(args.paper.resolve()): sha256_file(args.paper),
        str(args.preflight.resolve()): sha256_file(args.preflight),
    }
    report = seal_evidence(report, analysis_source=deployed_source)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
