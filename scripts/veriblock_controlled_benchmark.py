#!/usr/bin/env python3
"""Benchmark the locked public Veriblock-FL ZoKrates client circuit."""

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
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.final.manifest import (
    sha256_file,
    source_identity,
    write_manifest_atomic,
)


SCALAR_FIELD = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def _command(
    command: Sequence[str],
    *,
    cwd: Path,
    stdin: str | None = None,
) -> tuple[float, int, str]:
    with tempfile.NamedTemporaryFile(
        mode="r+", encoding="utf-8"
    ) as timing:
        started = time.perf_counter()
        completed = subprocess.run(
            [
                "/usr/bin/time",
                "-f",
                "%M",
                "-o",
                timing.name,
                *map(str, command),
            ],
            cwd=cwd,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        timing.seek(0)
        peak_rss_kb = int(timing.read().strip() or 0)
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "command failed"
        )
    return elapsed, peak_rss_kb, completed.stdout


def _constant(source: str, name: str) -> int:
    match = re.search(
        r"const\s+(?:u32|field)\s+" + re.escape(name) + r"\s*=\s*(\d+)\s*;",
        source,
    )
    if not match:
        raise ValueError("Veriblock-FL circuit constant is unavailable: " + name)
    return int(match.group(1))


def _round_constants(source: str) -> tuple[int, ...]:
    match = re.search(
        r"field\[64\]\s+round_constants\s*=\s*\[(.*?)\];",
        source,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Veriblock-FL MiMC constants are unavailable")
    values = tuple(int(value) for value in re.findall(r"\d+", match.group(1)))
    if len(values) != 64:
        raise ValueError("Veriblock-FL MiMC constant count is invalid")
    return values


def _field(value: int) -> int:
    return value if value >= 0 else SCALAR_FIELD + value


def _sign(value: int) -> int:
    return 0 if value > 0 else 1


def _truncate_divide(value: int, denominator: int) -> int:
    magnitude = abs(int(value)) // int(denominator)
    return -magnitude if value < 0 else magnitude


def _mimc(value: int, key: int, constants: Sequence[int]) -> int:
    current = int(value)
    for constant in constants:
        current = pow(
            (current + key + int(constant)) % SCALAR_FIELD,
            7,
            SCALAR_FIELD,
        )
    return (current + key) % SCALAR_FIELD


def _mimc_hash(
    weights: np.ndarray,
    biases: np.ndarray,
    constants: Sequence[int],
) -> int:
    key = 0
    for row, bias in zip(weights, biases):
        for value in row:
            key = _mimc(int(value), key, constants)
        key = _mimc(int(bias), key, constants)
    return key


def _strings(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _strings(value.tolist())
    if isinstance(value, list):
        return [_strings(item) for item in value]
    return str(int(value))


def generate_input(circuit_source: str, *, seed: int = 0) -> list[Any]:
    classes = _constant(circuit_source, "ac")
    features = _constant(circuit_source, "fe")
    batch_size = _constant(circuit_source, "bs")
    if (classes, features, batch_size) != (6, 9, 10):
        raise ValueError("Veriblock-FL circuit profile differs from its lock")
    constants = _round_constants(circuit_source)
    rng = np.random.RandomState(seed)
    precision = 1000
    learning_rate = 10
    biases = np.asarray(
        (rng.randn(classes) * precision).astype(int),
        dtype=object,
    )
    weights = np.asarray(
        (rng.randn(classes, features) * precision).astype(int),
        dtype=object,
    )
    samples = np.asarray(
        (rng.randn(batch_size, features) * precision).astype(int),
        dtype=object,
    )
    labels = np.asarray(
        [int(rng.randint(1, classes)) for _ in range(batch_size)],
        dtype=object,
    )
    initial_weights = weights.copy()
    initial_biases = biases.copy()
    for sample, label in zip(samples, labels):
        prediction = np.asarray(
            [
                _truncate_divide(
                    sum(
                        int(weights[row, column]) * int(sample[column])
                        for column in range(features)
                    ),
                    precision,
                )
                + int(biases[row])
                for row in range(classes)
            ],
            dtype=object,
        )
        expected = np.zeros(classes, dtype=object)
        expected[int(label) - 1] = precision
        error = np.asarray(
            [
                _truncate_divide(
                    2 * (int(prediction[row]) - int(expected[row])),
                    classes,
                )
                for row in range(classes)
            ],
            dtype=object,
        )
        for row in range(classes):
            biases[row] = int(biases[row]) - _truncate_divide(
                int(error[row]), learning_rate
            )
        for column in range(features):
            for row in range(classes):
                gradient = _truncate_divide(
                    int(error[row]) * int(sample[column]),
                    learning_rate,
                )
                weights[row, column] = int(
                    weights[row, column]
                ) - _truncate_divide(gradient, precision)

    convert = np.vectorize(_field, otypes=[object])
    signs = np.vectorize(_sign, otypes=[object])
    initial_weights_field = convert(initial_weights)
    initial_biases_field = convert(initial_biases)
    samples_field = convert(samples)
    final_weights_field = convert(weights)
    final_biases_field = convert(biases)
    digest = _mimc_hash(
        final_weights_field,
        final_biases_field,
        constants,
    )
    values = [
        initial_weights_field,
        signs(initial_weights),
        initial_biases_field,
        signs(initial_biases),
        samples_field,
        signs(samples),
        labels,
        learning_rate,
        precision,
        final_weights_field,
        final_biases_field,
        digest,
        digest,
    ]
    return _strings(values)


def _setup_record(
    *,
    build: Path,
    circuit: Path,
    zokrates: Path,
    config: dict[str, Any],
    constraints: int,
) -> dict[str, Any]:
    artifacts = {}
    for name in (
        "out",
        "out.r1cs",
        "abi.json",
        "proving.key",
        "verification.key",
        "verifier.sol",
    ):
        artifacts[name] = sha256_file(build / name)
    record = {
        "schema_version": 1,
        "classification": config["classification"],
        "upstream_commit": config["upstream"]["commit"],
        "upstream_circuit_sha256": sha256_file(circuit),
        "zokrates_binary_sha256": sha256_file(zokrates),
        "constraints": int(constraints),
        "artifacts": artifacts,
        "input_generator_correction": config["declared_correction"],
    }
    body = dict(record)
    record["record_digest"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return record


def _validate_setup(build: Path) -> dict[str, Any]:
    record = json.loads(
        (build / "controlled_setup.json").read_text(encoding="utf-8")
    )
    body = dict(record)
    declared = body.pop("record_digest")
    calculated = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if declared != calculated:
        raise ValueError("Veriblock-FL controlled setup digest mismatch")
    for name, expected in record["artifacts"].items():
        if sha256_file(build / name) != expected:
            raise ValueError("Veriblock-FL setup artifact hash mismatch: " + name)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--zokrates", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--reuse-setup", action="store_true")
    parser.add_argument("--witness-repeats", type=int, default=3)
    parser.add_argument("--proof-repeats", type=int, default=3)
    parser.add_argument("--verify-repeats", type=int, default=20)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "veriblock_controlled.json",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=ROOT / "config" / "baseline_sources.lock.json",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.witness_repeats, args.proof_repeats, args.verify_repeats) <= 0:
        raise ValueError("Veriblock-FL benchmark repeats must be positive")
    source = source_identity(ROOT)
    if not args.allow_dirty and (source["dirty"] or not source["commit"]):
        raise RuntimeError("formal Veriblock-FL benchmark requires clean source")
    circuit = args.circuit.resolve()
    zokrates = args.zokrates.resolve()
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (
        sha256_file(circuit) != config["upstream"]["circuit_sha256"]
        or sha256_file(zokrates) != config["zokrates"]["binary_sha256"]
    ):
        raise RuntimeError("Veriblock-FL source or ZoKrates binary is not locked")
    environment = {
        **dict(),
        "ZOKRATES_STDLIB": str(zokrates.parent / "stdlib"),
    }
    base_environment = dict(os.environ)
    base_environment.update(environment)

    def run(command: Sequence[str], stdin: str | None = None):
        with tempfile.NamedTemporaryFile(
            mode="r+", encoding="utf-8"
        ) as timing:
            started = time.perf_counter()
            completed = subprocess.run(
                [
                    "/usr/bin/time",
                    "-f",
                    "%M",
                    "-o",
                    timing.name,
                    *map(str, command),
                ],
                cwd=build,
                env=base_environment,
                input=stdin,
                text=True,
                capture_output=True,
                check=False,
            )
            elapsed = time.perf_counter() - started
            timing.seek(0)
            memory = int(timing.read().strip() or 0)
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or "ZoKrates command failed"
            )
        return elapsed, memory, completed.stdout

    if not args.reuse_setup:
        run(
            [
                zokrates,
                "compile",
                "-i",
                circuit,
                "-o",
                "out",
                "--abi-spec",
                "abi.json",
            ]
        )
        run(
            [
                zokrates,
                "setup",
                "-i",
                "out",
                "-p",
                "proving.key",
                "-v",
                "verification.key",
                "-s",
                "g16",
            ]
        )
        run(
            [
                zokrates,
                "export-verifier",
                "-i",
                "verification.key",
                "-o",
                "verifier.sol",
            ]
        )
        inspection = run([zokrates, "inspect", "-i", "out"])[2]
        match = re.search(r"constraint_count:\s*(\d+)", inspection)
        if not match:
            raise RuntimeError("ZoKrates constraint count is unavailable")
        setup = _setup_record(
            build=build,
            circuit=circuit,
            zokrates=zokrates,
            config=config,
            constraints=int(match.group(1)),
        )
        write_manifest_atomic(build / "controlled_setup.json", setup)
    setup = _validate_setup(build)

    input_values = generate_input(circuit.read_text(encoding="utf-8"))
    input_text = json.dumps(input_values, separators=(",", ":"))
    (build / "input.json").write_text(input_text + "\n", encoding="utf-8")
    witness_runs = [
        run(
            [
                zokrates,
                "compute-witness",
                "--abi",
                "--stdin",
                "-i",
                "out",
                "-s",
                "abi.json",
                "-o",
                "witness",
            ],
            stdin=input_text,
        )
        for _ in range(args.witness_repeats)
    ]
    proof_runs = [
        run(
            [
                zokrates,
                "generate-proof",
                "-i",
                "out",
                "-w",
                "witness",
                "-p",
                "proving.key",
                "-j",
                "proof.json",
                "-s",
                "g16",
            ]
        )
        for _ in range(args.proof_repeats)
    ]
    verify_runs = [
        run(
            [
                zokrates,
                "verify",
                "-v",
                "verification.key",
                "-j",
                "proof.json",
            ]
        )
        for _ in range(args.verify_repeats)
    ]
    if any("PASSED" not in value[2] for value in verify_runs):
        raise RuntimeError("Veriblock-FL proof verification did not pass")
    gas_process = subprocess.run(
        [
            "node",
            str(ROOT / "scripts" / "veriblock_gas_benchmark.cjs"),
            "--verifier",
            str(build / "verifier.sol"),
            "--proof",
            str(build / "proof.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if gas_process.returncode != 0:
        raise RuntimeError(
            gas_process.stderr.strip()
            or gas_process.stdout.strip()
        )
    gas = json.loads(gas_process.stdout.strip().splitlines()[-1])
    proof_size = (build / "proof.json").stat().st_size
    public_input_bytes = 5 * 32
    metrics = {
        "constraints": int(setup["constraints"]),
        "witness_seconds": statistics.median(
            value[0] for value in witness_runs
        ),
        "proof_seconds": statistics.median(
            value[0] for value in proof_runs
        ),
        "verification_ms": 1000.0
        * statistics.median(value[0] for value in verify_runs),
        "witness_peak_rss_kb": max(value[1] for value in witness_runs),
        "proof_peak_rss_kb": max(value[1] for value in proof_runs),
        "proof_bytes": proof_size,
        "public_input_bytes": public_input_bytes,
        "persistent_bytes_per_client": proof_size + public_input_bytes,
        "verification_gas": int(gas["verification_gas"]),
        "gas_usd": int(gas["verification_gas"]) * 1.5e-9 * 2500.0,
    }
    result = {
        "schema_version": 1,
        "classification": config["classification"],
        "method": "VeriblockFL",
        "real_benchmark": True,
        "source": source,
        "source_commit": source["commit"],
        "upstream_commit": config["upstream"]["commit"],
        "controlled_baseline_digest": setup["record_digest"],
        "baseline_source_lock_digest": sha256_file(args.source_lock),
        "metrics": metrics,
        "gas_evidence": gas,
        "input_generator_correction": config["declared_correction"],
        "input_sha256": {
            "circuit": sha256_file(circuit),
            "zokrates": sha256_file(zokrates),
            "controlled_setup.json": sha256_file(
                build / "controlled_setup.json"
            ),
            "proof.json": sha256_file(build / "proof.json"),
            "verifier.sol": sha256_file(build / "verifier.sol"),
            "gas_runner": sha256_file(
                ROOT / "scripts" / "veriblock_gas_benchmark.cjs"
            ),
        },
        "formal_accepted": bool(
            not args.allow_dirty and gas.get("passed") is True
        ),
    }
    body = dict(result)
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    write_manifest_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["formal_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
