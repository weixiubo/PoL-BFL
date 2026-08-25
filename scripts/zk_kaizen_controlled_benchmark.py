#!/usr/bin/env python3
"""Benchmark the declared controlled Kaizen-style Groth16 cost baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, source_identity, write_manifest_atomic
from polbfl.zk import IcicleProverPool, encode_groth16_proof
from scripts.zk_production_benchmark import (
    _gpu_processes,
    timed,
    timed_icicle_proof,
    timed_verification,
)


def validate_controlled_setup(build: Path) -> dict:
    record_path = build / "controlled_setup.json"
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("classification") != "controlled_kaizen_style_cost_baseline":
        raise ValueError("controlled baseline classification is invalid")
    body = dict(record)
    declared = body.pop("record_digest", None)
    calculated = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if declared != calculated:
        raise ValueError("controlled baseline record digest mismatch")
    for label, expected in record.get("artifacts", {}).items():
        path = build / label
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"controlled baseline artifact hash mismatch: {path}")
    return record


def evaluate_kaizen_metrics(metrics: dict, targets: dict) -> dict:
    target = targets["table_12_all_methods"]["Kaizen"]
    checks = {}
    for metric, value in metrics.items():
        if target[metric] is None:
            checks[metric] = value is None
        else:
            checks[metric] = float(value) <= float(target[metric]) + 1e-9
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--icicle-root", type=Path, required=True)
    parser.add_argument("--verifier-binary", type=Path, required=True)
    parser.add_argument(
        "--toolchain",
        type=Path,
        default=ROOT / "config" / "toolchain.lock.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "config" / "paper_table12_all_methods.json",
    )
    parser.add_argument("--witness-repeats", type=int, default=3)
    parser.add_argument("--proof-repeats", type=int, default=3)
    parser.add_argument("--verify-repeats", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.witness_repeats, args.proof_repeats, args.verify_repeats) <= 0:
        raise ValueError("controlled benchmark repeat counts must be positive")
    build = args.build.resolve()
    icicle_root = args.icicle_root.resolve()
    verifier_binary = args.verifier_binary.resolve()
    source = source_identity(ROOT)
    if not args.allow_dirty and (source["dirty"] or not source["commit"]):
        raise RuntimeError("controlled benchmark requires a clean source commit")
    setup = validate_controlled_setup(build)
    toolchain = json.loads(args.toolchain.read_text(encoding="utf-8"))
    for label, expected in toolchain["icicle_snark"]["artifacts"].items():
        path = icicle_root / label
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"controlled benchmark ICICLE artifact is not locked: {path}")
    if sha256_file(verifier_binary) != toolchain["rapidsnark"][
        "linux_x86_64_verifier_sha256"
    ]:
        raise RuntimeError("controlled benchmark verifier is not locked")
    witness_binary = build / "kaizen_controlled_cost_cpp" / "kaizen_controlled_cost"
    input_path = build / "input.json"
    witness_path = build / "benchmark-measured.wtns"
    zkey = build / "kaizen_controlled_cost_final.zkey"
    vkey = build / "verification_key.json"
    proof_path = build / "benchmark-proof.json"
    public_path = build / "benchmark-public.json"
    for path in (witness_binary, input_path, zkey, vkey):
        if not path.is_file():
            raise FileNotFoundError(path)
    witness_runs = [
        timed([str(witness_binary), str(input_path), str(witness_path)], cwd=build)
        for _ in range(args.witness_repeats)
    ]
    if _gpu_processes():
        raise RuntimeError("controlled ICICLE benchmark requires idle GPUs")
    pool = IcicleProverPool(
        icicle_root / "bin" / "icicle-snark",
        backend_directory=icicle_root / "backend",
        library_directories=(
            icicle_root / "lib",
            icicle_root / "backend" / "cuda",
        ),
        devices=(0,),
        timeout_seconds=600,
    )
    try:
        proof_runs = [
            timed_icicle_proof(
                pool,
                witness=witness_path,
                proving_key=zkey,
                proof=proof_path,
                public=public_path,
            )
            for _ in range(args.proof_repeats)
        ]
    finally:
        pool.close()
    verify_runs = [
        timed_verification(
            [str(verifier_binary), str(vkey), str(public_path), str(proof_path)],
            cwd=build,
        )
        for _ in range(args.verify_repeats)
    ]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    verification_ms = 1000.0 * statistics.median(value[0] for value in verify_runs)
    metrics = {
        "proof_generation_seconds": statistics.median(value[0] for value in proof_runs),
        "circuit_constraints": int(setup["constraints"]),
        "witness_seconds": statistics.median(value[0] for value in witness_runs),
        "prover_memory_gb": max(
            max(value[1] for value in proof_runs) / (1024 * 1024),
            max(value[2] for value in proof_runs) / 1024,
        ),
        "proof_bytes": len(encode_groth16_proof(proof)),
        "verification_ms": verification_ms,
        "merkle_proof_kb": None,
        "total_verification_ms": verification_ms,
    }
    result = {
        "schema_version": 1,
        "method": "Kaizen",
        "proof_system": "Groth16",
        "real_benchmark": True,
        "metrics": metrics,
        "source": source,
        "source_commit": source["commit"],
        "controlled_baseline_digest": setup["record_digest"],
        "acceptance": evaluate_kaizen_metrics(
            metrics,
            json.loads(args.targets.read_text(encoding="utf-8")),
        ),
        "input_sha256": {
            "controlled_setup.json": sha256_file(build / "controlled_setup.json"),
            "icicle-snark": sha256_file(icicle_root / "bin" / "icicle-snark"),
            "rapidsnark-verifier": sha256_file(verifier_binary),
        },
    }
    result["formal_accepted"] = bool(
        not args.allow_dirty and result["acceptance"]["passed"]
    )
    body = dict(result)
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_manifest_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
