#!/usr/bin/env python3
"""Aggregate attested real-trace observations for final-paper Table 11."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def aggregate_cross_hardware(
    observations: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for observation in observations:
        required = {
            "hardware_pair",
            "client_id",
            "behavior",
            "accepted",
            "proof_digest",
            "trainer_attestation",
            "verifier_attestation",
            "real_trace",
            "real_groth16",
        }
        if not required.issubset(observation):
            raise ValueError("cross-hardware observation is incomplete")
        if observation["real_trace"] is not True or observation["real_groth16"] is not True:
            raise ValueError("cross-hardware evidence must use real traces and Groth16")
        if observation["behavior"] not in {"honest", "malicious"}:
            raise ValueError("cross-hardware behavior must be honest or malicious")
        identity = (
            str(observation["hardware_pair"]),
            str(observation["client_id"]),
            str(observation["proof_digest"]),
        )
        if identity in seen:
            raise ValueError("duplicate cross-hardware proof observation")
        seen.add(identity)
        if not observation["trainer_attestation"] or not observation["verifier_attestation"]:
            raise ValueError("cross-hardware observation lacks device attestation")
        groups[str(observation["hardware_pair"])].append(observation)
    table = {}
    checks = {}
    provenance = {}
    for pair, rows in sorted(groups.items()):
        honest = [row for row in rows if row["behavior"] == "honest"]
        malicious = [row for row in rows if row["behavior"] == "malicious"]
        if not honest or not malicious:
            raise ValueError(f"hardware pair lacks both behavior classes: {pair}")
        fpr = 100.0 * sum(not bool(row["accepted"]) for row in honest) / len(honest)
        honest_pass = 100.0 * sum(bool(row["accepted"]) for row in honest) / len(honest)
        block_rate = 100.0 * sum(not bool(row["accepted"]) for row in malicious) / len(malicious)
        observed = {
            "FPR": fpr,
            "honest_pass_rate": honest_pass,
            "DR": block_rate,
            "block_rate": block_rate,
        }
        table[pair] = observed
        target = targets["table_11_cross_hardware"][pair]
        checks[f"{pair}.FPR"] = observed["FPR"] <= float(target["FPR"])
        checks[f"{pair}.honest_pass_rate"] = observed["honest_pass_rate"] >= float(
            target["honest_pass_rate"]
        )
        checks[f"{pair}.DR"] = observed["DR"] >= float(target["DR"])
        checks[f"{pair}.block_rate"] = observed["block_rate"] >= float(target["block_rate"])
        provenance[pair] = {
            "honest_samples": len(honest),
            "malicious_samples": len(malicious),
            "proof_digests": sorted(str(row["proof_digest"]) for row in rows),
        }
    return {
        "table_11_cross_hardware": table,
        "provenance": provenance,
        "acceptance": {
            "passed": bool(checks) and all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table11_cross_hardware.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.observations]
    aggregate = aggregate_cross_hardware(
        rows,
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.observations
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
