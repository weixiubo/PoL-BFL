#!/usr/bin/env python3
"""Benchmark the reference circuit with native witness/prove/verify binaries."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polbfl.zk import encode_groth16_proof


def timed(command: list[str], *, cwd: Path) -> tuple[float, int]:
    with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8") as timing:
        started = time.perf_counter()
        process = subprocess.run(
            ["/usr/bin/time", "-f", "%M", "-o", timing.name, *command],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}")
        timing.seek(0)
        peak_rss_kb = int(timing.read().strip())
    return elapsed, peak_rss_kb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--witness-binary", type=Path, required=True)
    parser.add_argument("--prover-binary", type=Path, required=True)
    parser.add_argument("--verifier-binary", type=Path, required=True)
    parser.add_argument("--witness-repeats", type=int, default=3)
    parser.add_argument("--proof-repeats", type=int, default=3)
    parser.add_argument("--verify-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    witness_binary = args.witness_binary.resolve()
    prover_binary = args.prover_binary.resolve()
    verifier_binary = args.verifier_binary.resolve()
    input_path = build / "input.json"
    witness_path = build / "benchmark.wtns"
    proof_path = build / "benchmark-proof.json"
    public_path = build / "benchmark-public.json"
    zkey = build / "sampled_sgd_reference_final.zkey"
    vkey = build / "verification_key.json"
    required = [input_path, zkey, vkey, witness_binary, prover_binary, verifier_binary]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    witness_runs = [
        timed([str(witness_binary), str(input_path), str(witness_path)], cwd=build)
        for _ in range(args.witness_repeats)
    ]
    proof_runs = [
        timed(
            [str(prover_binary), str(zkey), str(witness_path), str(proof_path), str(public_path)],
            cwd=build,
        )
        for _ in range(args.proof_repeats)
    ]
    verify_runs = [
        timed([str(verifier_binary), str(vkey), str(public_path), str(proof_path)], cwd=build)
        for _ in range(args.verify_repeats)
    ]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    compact_size = len(encode_groth16_proof(proof))
    metrics = {
        "circuit_constraints": 1_090_382,
        "witness_seconds_median": statistics.median(value[0] for value in witness_runs),
        "witness_peak_rss_kb_max": max(value[1] for value in witness_runs),
        "proof_seconds_median": statistics.median(value[0] for value in proof_runs),
        "prover_peak_rss_kb_max": max(value[1] for value in proof_runs),
        "proof_bytes": compact_size,
        "verification_seconds_median": statistics.median(value[0] for value in verify_runs),
        "verification_peak_rss_kb_max": max(value[1] for value in verify_runs),
    }
    gates = {
        "circuit_size": 900_000 <= metrics["circuit_constraints"] <= 1_300_000,
        "witness_seconds": metrics["witness_seconds_median"] <= 1.8,
        "proof_seconds": metrics["proof_seconds_median"] <= 4.2,
        "prover_memory": metrics["prover_peak_rss_kb_max"] <= int(2.5 * 1024 * 1024),
        "proof_size": metrics["proof_bytes"] <= 192,
        "verification_seconds": metrics["verification_seconds_median"] <= 0.0085,
    }
    result = {"metrics": metrics, "gates": gates, "passed": all(gates.values())}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
