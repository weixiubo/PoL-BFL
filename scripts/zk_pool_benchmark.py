#!/usr/bin/env python3
"""Benchmark valid reference proofs through persistent Rapidsnark pools."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polbfl.zk import Groth16Artifacts, Groth16Backend


def benchmark(args, pool_size: int) -> dict[str, object]:
    build = args.build.resolve()
    started_init = time.perf_counter()
    backend = Groth16Backend(
        Groth16Artifacts(
            wasm=build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
            proving_key=build / "sampled_sgd_reference_final.zkey",
            verification_key=build / "verification_key.json",
            r1cs=build / "sampled_sgd_reference.r1cs",
        ),
        snarkjs_cli=ROOT / "node_modules" / "snarkjs" / "cli.js",
        witness_binary=build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference",
        prover_binary=args.prover,
        verifier_binary=args.verifier,
        prover_library=args.library,
        prover_pool_size=pool_size,
        timeout_seconds=300,
    )
    initialization_seconds = time.perf_counter() - started_init
    circuit_input = json.loads((build / "input.json").read_text(encoding="utf-8"))
    try:
        started_batch = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            proofs = list(
                executor.map(
                    lambda _index: backend.prove(circuit_input),
                    range(args.proofs),
                )
            )
        batch_seconds = time.perf_counter() - started_batch
        verification = [backend.verify(proof) for proof in proofs]
        return {
            "pool_size": pool_size,
            "proof_count": len(proofs),
            "initialization_seconds": initialization_seconds,
            "batch_seconds": batch_seconds,
            "proofs_per_second": len(proofs) / batch_seconds,
            "witness_median_seconds": statistics.median(
                proof.witness_seconds for proof in proofs
            ),
            "prove_median_seconds": statistics.median(
                proof.prove_seconds for proof in proofs
            ),
            "verify_median_seconds": statistics.median(
                seconds for _valid, seconds in verification
            ),
            "all_valid": all(valid for valid, _seconds in verification),
            "proof_bytes": sorted({len(proof.compact_bytes) for proof in proofs}),
            "unique_proof_digests": len({proof.proof_digest for proof in proofs}),
        }
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--prover", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--pool-sizes", default="1,2,4,8")
    parser.add_argument("--proofs", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = [int(value) for value in args.pool_sizes.split(",")]
    if args.proofs <= 0 or any(size <= 0 for size in sizes):
        parser.error("proof and pool counts must be positive")
    result = {
        "schema_version": 1,
        "results": [benchmark(args, size) for size in sizes],
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not all(row["all_valid"] for row in result["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
