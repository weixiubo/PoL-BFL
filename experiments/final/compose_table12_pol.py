#!/usr/bin/env python3
"""Compose core prover and bundle/quorum evidence into the PoL-BFL Table 12 row."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def compose_pol_table12(
    core: Mapping[str, Any],
    bundle: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    if core.get("formal_accepted") is not True or bundle.get("formal_accepted") is not True:
        raise ValueError("PoL-BFL Table 12 requires two accepted formal benchmarks")
    if core.get("method") != "PoLBFL" or bundle.get("method") != "PoLBFL":
        raise ValueError("Table 12 evidence method differs from PoL-BFL")
    if core.get("source_commit") != bundle.get("source_commit"):
        raise ValueError("Table 12 benchmark sources differ")
    if core.get("trust_setup_record_digest") != bundle.get("trust_setup_record_digest"):
        raise ValueError("Table 12 benchmark trust records differ")
    core_metrics = core["metrics"]
    bundle_metrics = bundle["metrics"]
    metrics = {
        "proof_generation_seconds": float(core_metrics["proof_seconds_median"]),
        "circuit_constraints": int(core_metrics["circuit_constraints"]),
        "witness_seconds": float(core_metrics["witness_seconds_median"]),
        "prover_memory_gb": float(core_metrics["prover_memory_gb_max"]),
        "proof_bytes": int(core_metrics["proof_bytes"]),
        "verification_ms": float(bundle_metrics["verification_ms"]),
        "merkle_proof_kb": float(bundle_metrics["merkle_proof_kb"]),
        "total_verification_ms": float(bundle_metrics["total_verification_ms"]),
    }
    target = targets["table_12_all_methods"]["PoLBFL"]
    checks = {
        metric: value <= float(target[metric]) + 1e-9
        for metric, value in metrics.items()
    }
    result = {
        "schema_version": 1,
        "method": "PoLBFL",
        "proof_system": "Groth16",
        "real_benchmark": True,
        "metrics": metrics,
        "source_commit": str(core["source_commit"]),
        "trust_setup_record_digest": str(core["trust_setup_record_digest"]),
        "component_evidence_digests": {
            "core": str(core["evidence_digest"]),
            "bundle": str(bundle["evidence_digest"]),
        },
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
    }
    result["formal_accepted"] = result["acceptance"]["passed"]
    body = dict(result)
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table12_all_methods.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compose_pol_table12(
        json.loads(args.core.read_text(encoding="utf-8")),
        json.loads(args.bundle.read_text(encoding="utf-8")),
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    result["input_sha256"] = {
        "core": hashlib.sha256(args.core.read_bytes()).hexdigest(),
        "bundle": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "targets": hashlib.sha256(args.targets.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["formal_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
