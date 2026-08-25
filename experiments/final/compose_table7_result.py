#!/usr/bin/env python3
"""Compose one measured Table 7 method from source-bound raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


METHODS = ("Vanilla", "VeriblockFL", "Kaizen", "PoLBFL")
METRICS = (
    "runtime_seconds",
    "communication_mb",
    "gas_usd",
    "storage_mb_per_client",
)


def gas_usd(
    gas: int,
    *,
    gas_price_gwei: float = 1.5,
    eth_price_usd: float = 2500.0,
) -> float:
    if gas < 0 or gas_price_gwei < 0 or eth_price_usd < 0:
        raise ValueError("gas conversion inputs must be non-negative")
    return gas * gas_price_gwei * 1e-9 * eth_price_usd


def _source_commit(evidence: Mapping[str, Any]) -> str:
    value = evidence.get("source_commit")
    if value is None:
        value = evidence.get("source", {}).get("commit")
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError("Table 7 input lacks a source commit")
    return value


def _gas_metric(evidence: Mapping[str, Any]) -> float:
    if evidence.get("passed") is not True:
        raise ValueError("Table 7 gas evidence was not accepted")
    observed = evidence.get("observed_gas", {})
    gas = int(observed.get("honest_round_total", -1))
    return gas_usd(gas)


def compose_table7_result(
    *,
    method: str,
    seed: int,
    training: Mapping[str, Any],
    targets: Mapping[str, Any],
    source_lock_digest: str,
    gas_evidence: Mapping[str, Any] | None = None,
    proof_evidence: Mapping[str, Any] | None = None,
    veriblock_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if method not in METHODS or len(source_lock_digest) != 64:
        raise ValueError("Table 7 method or source lock is invalid")
    if (
        training.get("formal_accepted") is not True
        or int(training.get("rounds", 0)) != 200
        or training.get("dataset") != "CIFAR10"
    ):
        raise ValueError("Table 7 requires accepted 200-round CIFAR10 training")
    source_commit = _source_commit(training)
    base_runtime = float(
        training.get(
            "protocol_runtime_seconds",
            training["runtime_seconds"],
        )
    )
    base_communication = float(training["communication_mb"])
    if min(base_runtime, base_communication) < 0:
        raise ValueError("Table 7 training measurements are invalid")

    provenance: dict[str, Any] = {
        "training_result_digest": training.get(
            "result_digest", training.get("evidence_digest")
        ),
    }
    if method == "Vanilla":
        metrics = {
            "runtime_seconds": base_runtime,
            "communication_mb": base_communication,
            "gas_usd": 0.0,
            "storage_mb_per_client": 0.0,
        }
    elif method == "PoLBFL":
        if gas_evidence is None:
            raise ValueError("PoL-BFL Table 7 requires contract gas evidence")
        if _source_commit(gas_evidence) != source_commit:
            raise ValueError("PoL-BFL training and gas source differ")
        trust_digest = str(training.get("trust_setup_record_digest", ""))
        if (
            len(trust_digest) != 64
            or training.get("real_contract_rounds") is not True
            or int(training.get("contract_rounds", 0)) != 200
        ):
            raise ValueError("PoL-BFL Table 7 lacks production proof/contract evidence")
        metrics = {
            "runtime_seconds": float(training["runtime_seconds"]),
            "communication_mb": base_communication,
            "gas_usd": _gas_metric(gas_evidence),
            "storage_mb_per_client": float(
                training["storage_mb_per_client"]
            ),
        }
        provenance.update(
            {
                "gas_evidence_digest": gas_evidence["evidence_digest"],
                "trust_setup_record_digest": trust_digest,
                "contract_evidence_digest": training[
                    "contract_evidence_digest"
                ],
            }
        )
    elif method == "Kaizen":
        if gas_evidence is None or proof_evidence is None:
            raise ValueError("Kaizen Table 7 requires proof and gas evidence")
        if (
            _source_commit(gas_evidence) != source_commit
            or _source_commit(proof_evidence) != source_commit
            or proof_evidence.get("method") != "Kaizen"
            or proof_evidence.get("real_benchmark") is not True
            or proof_evidence.get("formal_accepted") is not True
        ):
            raise ValueError("Kaizen Table 7 inputs are incompatible")
        proof = proof_evidence["metrics"]
        per_client_proof = (
            float(proof["witness_seconds"])
            + float(proof["proof_generation_seconds"])
        )
        clients = int(training.get("num_clients", 50))
        dual_gpu_batches = math.ceil(clients / 2)
        runtime = (
            base_runtime
            + dual_gpu_batches * per_client_proof
            + clients * float(proof["verification_ms"]) / 1000.0
        )
        persistent_bytes = int(proof["proof_bytes"]) + 18 * 32
        metrics = {
            "runtime_seconds": runtime,
            "communication_mb": (
                base_communication
                + clients * persistent_bytes / 1_000_000
            ),
            "gas_usd": _gas_metric(gas_evidence),
            "storage_mb_per_client": persistent_bytes / 1_000_000,
        }
        provenance.update(
            {
                "proof_evidence_digest": proof_evidence["evidence_digest"],
                "gas_evidence_digest": gas_evidence["evidence_digest"],
                "dual_gpu_batches": dual_gpu_batches,
            }
        )
    else:
        if veriblock_evidence is None:
            raise ValueError(
                "Veriblock-FL Table 7 requires its controlled benchmark"
            )
        if (
            _source_commit(veriblock_evidence) != source_commit
            or veriblock_evidence.get("classification")
            != "controlled_veriblockfl_full_verification"
            or veriblock_evidence.get("real_benchmark") is not True
            or veriblock_evidence.get("formal_accepted") is not True
        ):
            raise ValueError("Veriblock-FL controlled evidence is invalid")
        measured = veriblock_evidence["metrics"]
        clients = int(training.get("num_clients", 50))
        per_client_verification = (
            float(measured["witness_seconds"])
            + float(measured["proof_seconds"])
            + float(measured["verification_ms"]) / 1000.0
        )
        wire_bytes = (
            int(measured["proof_bytes"])
            + int(measured["public_input_bytes"])
        )
        metrics = {
            "runtime_seconds": (
                base_runtime + clients * per_client_verification
            ),
            "communication_mb": (
                base_communication + clients * wire_bytes / 1_000_000
            ),
            "gas_usd": float(measured["gas_usd"]),
            "storage_mb_per_client": (
                int(measured["persistent_bytes_per_client"])
                / 1_000_000
            ),
        }
        provenance.update(
            {
                "veriblock_evidence_digest": veriblock_evidence[
                    "evidence_digest"
                ],
                "per_client_verification_seconds": per_client_verification,
            }
        )

    target = targets["table_7_all_methods"][method]
    checks = {
        name: float(metrics[name]) <= float(target[name]) + 1e-9
        for name in METRICS
    }
    result = {
        "schema_version": 1,
        "method": method,
        "seed": int(seed),
        "source_commit": source_commit,
        "real_measurement": True,
        "training_rounds": 200,
        **metrics,
        "baseline_source_lock_digest": source_lock_digest,
        "provenance": provenance,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "failed": sorted(
                name for name, passed in checks.items() if not passed
            ),
        },
    }
    if method == "PoLBFL":
        result.update(
            {
                "trust_setup_record_digest": training[
                    "trust_setup_record_digest"
                ],
                "real_contract_gas": True,
            }
        )
    result["formal_accepted"] = bool(result["acceptance"]["passed"])
    body = dict(result)
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--gas-evidence", type=Path)
    parser.add_argument("--proof-evidence", type=Path)
    parser.add_argument("--veriblock-evidence", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table7_all_methods.json",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=root / "config" / "baseline_sources.lock.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path | None):
        return (
            None
            if path is None
            else json.loads(path.read_text(encoding="utf-8"))
        )

    result = compose_table7_result(
        method=args.method,
        seed=args.seed,
        training=load(args.training),
        targets=load(args.targets),
        source_lock_digest=hashlib.sha256(
            args.source_lock.read_bytes()
        ).hexdigest(),
        gas_evidence=load(args.gas_evidence),
        proof_evidence=load(args.proof_evidence),
        veriblock_evidence=load(args.veriblock_evidence),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["formal_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
