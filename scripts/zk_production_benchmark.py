#!/usr/bin/env python3
"""Source- and trust-bound production benchmark for final-paper Table 12."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import (
    environment_identity,
    sha256_file,
    source_identity,
    write_manifest_atomic,
)
from experiments.final.trust_setup import validate_trust_setup
from polbfl.zk import IcicleProverPool, encode_groth16_proof


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
CONSTRAINT_PATTERN = re.compile(r"# of Constraints:\s*([0-9]+)")


def parse_constraint_count(output: str) -> int:
    match = CONSTRAINT_PATTERN.search(ANSI_PATTERN.sub("", output))
    if match is None or int(match.group(1)) <= 0:
        raise ValueError("snarkjs R1CS output lacks a positive constraint count")
    return int(match.group(1))


def _gpu_processes() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    result = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid, memory = (part.strip() for part in line.split(",", 1))
        result[int(pid)] = int(memory)
    return result


def timed(
    command: list[str],
    *,
    cwd: Path,
    monitor_gpu: bool = False,
) -> tuple[float, int, int]:
    if monitor_gpu and _gpu_processes():
        raise RuntimeError("production prover benchmark requires idle GPUs")
    with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8") as timing, tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8"
    ) as process_output:
        started = time.perf_counter()
        process = subprocess.Popen(
            ["/usr/bin/time", "-f", "%M", "-o", timing.name, *command],
            cwd=cwd,
            text=True,
            stdout=process_output,
            stderr=subprocess.STDOUT,
        )
        peak_gpu_mib = 0
        while process.poll() is None:
            if monitor_gpu:
                peak_gpu_mib = max(peak_gpu_mib, sum(_gpu_processes().values()))
            time.sleep(0.02)
        process.wait(timeout=5)
        elapsed = time.perf_counter() - started
        if process.returncode != 0:
            process_output.seek(0)
            message = process_output.read().strip()
            raise RuntimeError(message or f"exit {process.returncode}")
        timing.seek(0)
        peak_rss_kb = int(timing.read().strip())
    return elapsed, peak_rss_kb, peak_gpu_mib


def _process_peak_rss_kb(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return 0


def timed_icicle_proof(
    pool: IcicleProverPool,
    *,
    witness: Path,
    proving_key: Path,
    proof: Path,
    public: Path,
) -> tuple[float, int, int]:
    worker_pids = set(pool.worker_pids)
    if not worker_pids:
        raise RuntimeError("ICICLE benchmark pool has no live worker")
    peak_gpu_mib = [0]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            processes = _gpu_processes()
            peak_gpu_mib[0] = max(
                peak_gpu_mib[0],
                sum(memory for pid, memory in processes.items() if pid in worker_pids),
            )
            stop.wait(0.02)

    thread = threading.Thread(target=monitor, name="icicle-memory-monitor", daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        pool.prove(
            witness=witness,
            proving_key=proving_key,
            proof=proof,
            public=public,
        )
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        thread.join(timeout=5)
    peak_rss_kb = max(_process_peak_rss_kb(pid) for pid in worker_pids)
    return elapsed, peak_rss_kb, peak_gpu_mib[0]



def timed_verification(command: list[str], *, cwd: Path) -> tuple[float, int, int]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"exit {completed.returncode}"
        )
    return elapsed, 0, 0

def evaluate_metrics(metrics: Mapping[str, float], targets: Mapping[str, Any]) -> dict[str, Any]:
    target = targets["table_12_zk"]
    checks = {
        "circuit_constraints": int(metrics["circuit_constraints"])
        <= int(target["circuit_constraints"]),
        "witness_seconds": float(metrics["witness_seconds_median"])
        <= float(target["witness_seconds"]),
        "proof_generation_seconds": float(metrics["proof_seconds_median"])
        <= float(target["proof_generation_seconds"]),
        "prover_memory_gb": float(metrics["prover_memory_gb_max"])
        <= float(target["prover_memory_gb"]),
        "proof_bytes": int(metrics["proof_bytes"]) <= int(target["proof_bytes"]),
        "verification_ms": float(metrics["verification_ms_median"])
        <= float(target["verification_ms"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--witness-binary", type=Path, required=True)
    parser.add_argument("--prover-binary", type=Path, required=True)
    parser.add_argument("--verifier-binary", type=Path, required=True)
    parser.add_argument(
        "--snarkjs-cli",
        type=Path,
        default=ROOT / "node_modules" / "snarkjs" / "cli.js",
    )
    parser.add_argument(
        "--toolchain",
        type=Path,
        default=ROOT / "config" / "toolchain.lock.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "config" / "paper_targets.json",
    )
    parser.add_argument("--witness-repeats", type=int, default=3)
    parser.add_argument("--proof-repeats", type=int, default=3)
    parser.add_argument("--verify-repeats", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.witness_repeats, args.proof_repeats, args.verify_repeats) <= 0:
        raise ValueError("benchmark repeat counts must be positive")
    build = args.build.resolve()
    witness_binary = args.witness_binary.resolve()
    prover_binary = args.prover_binary.resolve()
    verifier_binary = args.verifier_binary.resolve()
    if prover_binary.name == "icicle-snark":
        icicle_root = prover_binary.parent.parent
        backend_root = icicle_root / "backend"
        cuda_backend = backend_root / "cuda"
        library_root = icicle_root / "lib"
        for path in (backend_root, cuda_backend, library_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        os.environ["ICICLE_BACKEND_INSTALL_DIR"] = str(backend_root)
        existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(library_root), str(cuda_backend), existing_library_path]
        ).rstrip(os.pathsep)
    input_path = build / "input.json"
    witness_path = build / "production-benchmark.wtns"
    proof_path = build / "production-benchmark-proof.json"
    public_path = build / "production-benchmark-public.json"
    r1cs = build / "sampled_sgd_reference.r1cs"
    zkey = build / "sampled_sgd_reference_final.zkey"
    vkey = build / "verification_key.json"
    trust_path = build / "trust_setup.json"
    required = (
        input_path,
        r1cs,
        zkey,
        vkey,
        trust_path,
        witness_binary,
        prover_binary,
        verifier_binary,
        args.snarkjs_cli.resolve(),
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = source_identity(ROOT)
    if not args.allow_dirty and (source["dirty"] or not source["commit"]):
        raise RuntimeError("production ZK benchmark requires a clean source commit")
    toolchain = json.loads(args.toolchain.read_text(encoding="utf-8"))
    if prover_binary.name == "icicle-snark":
        icicle_root = prover_binary.parent.parent
        for label, expected in toolchain["icicle_snark"]["artifacts"].items():
            path = icicle_root / label
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"ICICLE benchmark artifact is not locked: {path}")
    elif sha256_file(prover_binary) != toolchain["rapidsnark"][
        "linux_x86_64_prover_sha256"
    ]:
        raise RuntimeError("Rapidsnark prover binary is not locked")
    if sha256_file(verifier_binary) != toolchain["rapidsnark"][
        "linux_x86_64_verifier_sha256"
    ]:
        raise RuntimeError("Rapidsnark verifier binary is not locked")
    trust = validate_trust_setup(build=build, toolchain=toolchain)
    if not trust["passed"]:
        raise RuntimeError(f"production trust setup validation failed: {trust}")
    r1cs_info = subprocess.run(
        ["node", str(args.snarkjs_cli.resolve()), "r1cs", "info", str(r1cs)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    constraint_count = parse_constraint_count(r1cs_info.stdout + r1cs_info.stderr)
    witness_runs = [
        timed([str(witness_binary), str(input_path), str(witness_path)], cwd=build)
        for _ in range(args.witness_repeats)
    ]
    if prover_binary.name == "icicle-snark":
        if _gpu_processes():
            raise RuntimeError("production ICICLE benchmark requires idle GPUs")
        pool = IcicleProverPool(
            prover_binary,
            backend_directory=prover_binary.parent.parent / "backend",
            library_directories=(
                prover_binary.parent.parent / "lib",
                prover_binary.parent.parent / "backend" / "cuda",
            ),
            devices=(0,),
            timeout_seconds=300,
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
    else:
        proof_runs = [
            timed(
                [
                    str(prover_binary), str(zkey), str(witness_path),
                    str(proof_path), str(public_path),
                ],
                cwd=build,
            )
            for _ in range(args.proof_repeats)
        ]
    verify_runs = [
        timed_verification([str(verifier_binary), str(vkey), str(public_path), str(proof_path)], cwd=build)
        for _ in range(args.verify_repeats)
    ]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    prover_rss_gb = max(value[1] for value in proof_runs) / (1024 * 1024)
    prover_gpu_gb = max(value[2] for value in proof_runs) / 1024
    metrics = {
        "circuit_constraints": constraint_count,
        "witness_seconds_median": statistics.median(value[0] for value in witness_runs),
        "witness_peak_rss_kb_max": max(value[1] for value in witness_runs),
        "proof_seconds_median": statistics.median(value[0] for value in proof_runs),
        "prover_peak_rss_kb_max": max(value[1] for value in proof_runs),
        "prover_peak_gpu_mib_max": max(value[2] for value in proof_runs),
        "prover_memory_gb_max": max(prover_rss_gb, prover_gpu_gb),
        "proof_bytes": len(encode_groth16_proof(proof)),
        "verification_ms_median": 1000.0 * statistics.median(value[0] for value in verify_runs),
        "verification_peak_rss_kb_max": max(value[1] for value in verify_runs),
    }
    result = {
        "schema_version": 1,
        "method": "PoLBFL",
        "proof_system": "Groth16",
        "real_benchmark": True,
        "metrics": metrics,
        "acceptance": evaluate_metrics(
            metrics,
            json.loads(args.targets.read_text(encoding="utf-8")),
        ),
        "trust_setup": trust,
        "source": source,
        "source_commit": source["commit"],
        "trust_setup_record_digest": json.loads(
            Path(trust["details"]["record"]).read_text(encoding="utf-8")
        )["record_digest"],
        "environment": environment_identity(ROOT),
        "input_sha256": {
            str(path): sha256_file(path)
            for path in required
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
