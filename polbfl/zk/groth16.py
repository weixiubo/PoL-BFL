"""Fail-closed snarkjs Groth16 backend for Circom artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from polbfl.crypto import canonical_json_bytes, domain_hash
from polbfl.zk.codec import encode_groth16_proof
from polbfl.zk.icicle_pool import IcicleProverPool
from polbfl.zk.rapidsnark_pool import RapidsnarkProverPool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Groth16Artifacts:
    wasm: Path
    proving_key: Path
    verification_key: Path
    r1cs: Path | None = None

    def validate(self) -> None:
        for name in ("wasm", "proving_key", "verification_key"):
            path = Path(getattr(self, name))
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"missing Groth16 artifact: {path}")
        if self.r1cs is not None and (not Path(self.r1cs).is_file() or Path(self.r1cs).stat().st_size <= 0):
            raise FileNotFoundError(f"missing R1CS artifact: {self.r1cs}")

    @property
    def circuit_digest(self) -> str:
        self.validate()
        parts = [
            _file_sha256(Path(self.wasm)),
            _file_sha256(Path(self.proving_key)),
            _file_sha256(Path(self.verification_key)),
        ]
        if self.r1cs is not None:
            parts.append(_file_sha256(Path(self.r1cs)))
        return domain_hash("POLBFL_GROTH16_ARTIFACTS_V1", *parts)


@dataclass(frozen=True)
class Groth16Proof:
    proof: Mapping[str, Any]
    public_signals: tuple[str, ...]
    circuit_digest: str
    proof_digest: str
    prove_seconds: float
    peak_child_rss_kb: int
    witness_seconds: float = 0.0

    @property
    def compact_bytes(self) -> bytes:
        return encode_groth16_proof(self.proof)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof": dict(self.proof),
            "public_signals": list(self.public_signals),
            "circuit_digest": self.circuit_digest,
            "proof_digest": self.proof_digest,
            "prove_seconds": self.prove_seconds,
            "peak_child_rss_kb": self.peak_child_rss_kb,
            "witness_seconds": self.witness_seconds,
        }


class Groth16Backend:
    def __init__(
        self,
        artifacts: Groth16Artifacts,
        *,
        node_binary: str = "node",
        snarkjs_cli: str | Path = "node_modules/snarkjs/cli.js",
        witness_binary: str | Path | None = None,
        prover_binary: str | Path | None = None,
        verifier_binary: str | Path | None = None,
        prover_library: str | Path | None = None,
        prover_pool_size: int = 0,
        icicle_binary: str | Path | None = None,
        icicle_backend_directory: str | Path | None = None,
        icicle_library_directories: Sequence[str | Path] = (),
        icicle_devices: Sequence[int] = (),
        timeout_seconds: int = 600,
    ):
        artifacts.validate()
        self.artifacts = artifacts
        self._circuit_digest = artifacts.circuit_digest
        self.node_binary = str(node_binary)
        self.snarkjs_cli = str(snarkjs_cli)
        self.witness_binary = None if witness_binary is None else str(Path(witness_binary).resolve())
        self.prover_binary = None if prover_binary is None else str(Path(prover_binary).resolve())
        self.verifier_binary = None if verifier_binary is None else str(Path(verifier_binary).resolve())
        self.prover_library = (
            None if prover_library is None else str(Path(prover_library).resolve())
        )
        self.timeout_seconds = int(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("Groth16 timeout must be positive")
        for binary in (self.witness_binary, self.prover_binary, self.verifier_binary):
            if binary is not None and not Path(binary).is_file():
                raise FileNotFoundError(f"missing native Groth16 executable: {binary}")
        self.icicle_pool = None
        self.prover_pool = None
        if icicle_binary is not None:
            if self.witness_binary is None:
                raise ValueError("ICICLE proving requires a native witness generator")
            if icicle_backend_directory is None:
                raise ValueError("ICICLE backend directory is required")
            self.icicle_pool = IcicleProverPool(
                icicle_binary,
                backend_directory=icicle_backend_directory,
                library_directories=icicle_library_directories,
                devices=icicle_devices,
                timeout_seconds=self.timeout_seconds,
            )
        elif self.prover_library is not None:
            if self.witness_binary is None:
                raise ValueError("persistent Rapidsnark proving requires a native witness generator")
            self.prover_pool = RapidsnarkProverPool(
                self.prover_library,
                self.artifacts.proving_key,
                size=int(prover_pool_size),
            )

    def _run(self, args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.setdefault("NODE_OPTIONS", "--max-old-space-size=8192")
        return subprocess.run(
            [self.node_binary, self.snarkjs_cli, *map(str, args)],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )

    @staticmethod
    def _proof_digest(proof: Mapping[str, Any], public_signals: Sequence[str], circuit_digest: str) -> str:
        return domain_hash(
            "POLBFL_GROTH16_PROOF_V1",
            canonical_json_bytes(proof),
            canonical_json_bytes([str(value) for value in public_signals]),
            bytes.fromhex(circuit_digest),
        )

    def prove(self, circuit_input: Mapping[str, Any]) -> Groth16Proof:
        before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        with tempfile.TemporaryDirectory(prefix="polbfl-groth16-") as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            proof_path = root / "proof.json"
            public_path = root / "public.json"
            witness_path = root / "witness.wtns"
            input_path.write_text(json.dumps(circuit_input, sort_keys=True), encoding="utf-8")
            witness_elapsed = 0.0
            if self.witness_binary is not None:
                witness_started = time.perf_counter()
                witness_process = subprocess.run(
                    [self.witness_binary, str(input_path), str(witness_path)],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                witness_elapsed = time.perf_counter() - witness_started
                if witness_process.returncode != 0:
                    raise RuntimeError(
                        "Groth16 witness generation failed: "
                        + (
                            witness_process.stderr.strip()
                            or witness_process.stdout.strip()
                            or f"exit {witness_process.returncode}"
                        )
                    )
            if self.prover_binary is not None and self.witness_binary is None:
                witness_started = time.perf_counter()
                witness_process = self._run(
                    (
                        "wtns",
                        "calculate",
                        str(Path(self.artifacts.wasm).resolve()),
                        str(input_path),
                        str(witness_path),
                    ),
                    cwd=root,
                )
                witness_elapsed = time.perf_counter() - witness_started
                if witness_process.returncode != 0:
                    raise RuntimeError(
                        "Groth16 witness generation failed: "
                        + (witness_process.stderr.strip() or witness_process.stdout.strip())
                    )

            started = time.perf_counter()
            pooled_output = None
            if self.icicle_pool is not None:
                pooled_output = self.icicle_pool.prove(
                    witness=witness_path,
                    proving_key=self.artifacts.proving_key,
                    proof=proof_path,
                    public=public_path,
                )
                process = None
            elif self.prover_pool is not None:
                pooled_output = self.prover_pool.prove(witness_path)
                process = None
            elif self.prover_binary is not None:
                process = subprocess.run(
                    [
                        self.prover_binary,
                        str(Path(self.artifacts.proving_key).resolve()),
                        str(witness_path),
                        str(proof_path),
                        str(public_path),
                    ],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            elif self.witness_binary is not None:
                process = self._run(
                    (
                        "groth16",
                        "prove",
                        str(Path(self.artifacts.proving_key).resolve()),
                        str(witness_path),
                        str(proof_path),
                        str(public_path),
                    ),
                    cwd=root,
                )
            else:
                process = self._run(
                    (
                        "groth16",
                        "fullprove",
                        str(input_path),
                        str(Path(self.artifacts.wasm).resolve()),
                        str(Path(self.artifacts.proving_key).resolve()),
                        str(proof_path),
                        str(public_path),
                    ),
                    cwd=root,
                )
            elapsed = time.perf_counter() - started
            if process is not None and process.returncode != 0:
                raise RuntimeError(
                    "Groth16 proof generation failed: "
                    + (process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}")
                )
            if pooled_output is not None:
                proof, public = pooled_output
            else:
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
                public = tuple(
                    str(value)
                    for value in json.loads(public_path.read_text(encoding="utf-8"))
                )
        circuit_digest = self._circuit_digest
        peak_rss = max(0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss - before_rss)
        return Groth16Proof(
            proof=proof,
            public_signals=public,
            circuit_digest=circuit_digest,
            proof_digest=self._proof_digest(proof, public, circuit_digest),
            prove_seconds=elapsed,
            peak_child_rss_kb=int(peak_rss),
            witness_seconds=witness_elapsed,
        )

    def verify(self, proof: Groth16Proof) -> tuple[bool, float]:
        if proof.circuit_digest != self._circuit_digest:
            return False, 0.0
        expected_digest = self._proof_digest(
            proof.proof,
            proof.public_signals,
            proof.circuit_digest,
        )
        if expected_digest != proof.proof_digest:
            return False, 0.0
        with tempfile.TemporaryDirectory(prefix="polbfl-groth16-verify-") as temporary:
            root = Path(temporary)
            proof_path = root / "proof.json"
            public_path = root / "public.json"
            proof_path.write_text(json.dumps(proof.proof, sort_keys=True), encoding="utf-8")
            public_path.write_text(json.dumps(list(proof.public_signals)), encoding="utf-8")
            started = time.perf_counter()
            if self.verifier_binary is not None:
                process = subprocess.run(
                    [
                        self.verifier_binary,
                        str(Path(self.artifacts.verification_key).resolve()),
                        str(public_path),
                        str(proof_path),
                    ],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            else:
                process = self._run(
                    (
                        "groth16",
                        "verify",
                        str(Path(self.artifacts.verification_key).resolve()),
                        str(public_path),
                        str(proof_path),
                    ),
                    cwd=root,
                )
            elapsed = time.perf_counter() - started
        valid_output = self.verifier_binary is not None or "OK" in process.stdout
        return process.returncode == 0 and valid_output, elapsed

    def close(self) -> None:
        if self.icicle_pool is not None:
            self.icicle_pool.close()
            self.icicle_pool = None
        if self.prover_pool is not None:
            self.prover_pool.close()
            self.prover_pool = None

    def __del__(self):  # pragma: no cover - interpreter shutdown fallback
        try:
            self.close()
        except Exception:
            pass
