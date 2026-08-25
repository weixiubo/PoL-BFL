#!/usr/bin/env python3
"""Validate measured PoL-BFL and controlled Kaizen proof costs for Table 12."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


METHODS = ("PoLBFL", "Kaizen")
METRICS = (
    "proof_generation_seconds",
    "circuit_constraints",
    "witness_seconds",
    "prover_memory_gb",
    "proof_bytes",
    "verification_ms",
    "merkle_proof_kb",
    "total_verification_ms",
)


def aggregate_table12(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 12")
    by_method = {}
    for result in results:
        required = {"method", "source_commit", "evidence_digest", "metrics"}
        if not required.issubset(result) or result.get("formal_accepted") is not True:
            raise ValueError("Table 12 result is not accepted or is incomplete")
        method = str(result["method"])
        if method not in METHODS or method in by_method:
            raise ValueError("Table 12 method is invalid or duplicated")
        if result.get("proof_system") != "Groth16" or result.get("real_benchmark") is not True:
            raise ValueError("Table 12 evidence is not a real Groth16 benchmark")
        if len(str(result["evidence_digest"])) != 64:
            raise ValueError("Table 12 evidence digest is invalid")
        if method == "PoLBFL":
            if len(str(result.get("trust_setup_record_digest", ""))) != 64:
                raise ValueError("PoL-BFL benchmark lacks production trust provenance")
        elif len(str(result.get("controlled_baseline_digest", ""))) != 64:
            raise ValueError("Kaizen controlled baseline lacks its construction digest")
        by_method[method] = result
    if set(by_method) != set(METHODS):
        raise ValueError("Table 12 aggregate requires both methods")
    table = {}
    checks = {}
    provenance = {}
    for method in METHODS:
        metrics = dict(by_method[method]["metrics"])
        target = targets["table_12_all_methods"][method]
        observed = {}
        for metric in METRICS:
            target_value = target[metric]
            observed_value = metrics.get(metric)
            if target_value is None:
                if observed_value is not None:
                    raise ValueError(f"unexpected Table 12 metric for {method}: {metric}")
                observed[metric] = None
                continue
            if observed_value is None:
                raise ValueError(f"missing Table 12 metric for {method}: {metric}")
            observed[metric] = float(observed_value)
            checks[f"{method}.{metric}"] = float(observed_value) <= float(target_value) + 1e-9
        table[method] = observed
        provenance[method] = {
            "evidence_digest": str(by_method[method]["evidence_digest"]),
        }
    return {
        "source_commit": source_commit,
        "table_12_all_methods": table,
        "provenance": {
            "source_commit": source_commit,
            "methods": provenance,
        },
        "acceptance": {
            "passed": bool(checks) and all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs=2, type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table12_all_methods.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_table12(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    result["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.results
    }
    result = seal_evidence(result, analysis_root=root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
