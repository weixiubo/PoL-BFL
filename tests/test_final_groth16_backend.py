import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from polbfl.zk import Groth16Artifacts, Groth16Backend, decode_groth16_proof


def _backend_and_input():
    root_raw = os.getenv("POL_ZK_SMOKE_BUILD")
    if not root_raw:
        pytest.skip("POL_ZK_SMOKE_BUILD is not configured")
    root = Path(root_raw)
    artifacts = Groth16Artifacts(
        wasm=root / "sampled_sgd_smoke_js" / "sampled_sgd_smoke.wasm",
        proving_key=root / "sampled_sgd_smoke_final.zkey",
        verification_key=root / "verification_key.json",
        r1cs=root / "sampled_sgd_smoke.r1cs",
    )
    backend = Groth16Backend(
        artifacts,
        snarkjs_cli=os.getenv("SNARKJS_CLI", "node_modules/snarkjs/cli.js"),
    )
    return backend, json.loads((root / "input.json").read_text(encoding="utf-8"))


def test_real_groth16_backend_proves_verifies_and_rejects_tampering():
    backend, circuit_input = _backend_and_input()
    proof = backend.prove(circuit_input)
    valid, verify_seconds = backend.verify(proof)
    assert valid
    assert proof.prove_seconds > 0
    assert verify_seconds > 0
    assert len(proof.proof_digest) == 64
    assert len(proof.compact_bytes) == 192
    decoded = decode_groth16_proof(proof.compact_bytes)
    assert decoded["pi_a"] == proof.proof["pi_a"]
    assert decoded["pi_b"] == proof.proof["pi_b"]
    assert decoded["pi_c"] == proof.proof["pi_c"]

    tampered_public = list(proof.public_signals)
    tampered_public[0] = str(int(tampered_public[0]) + 1)
    tampered = replace(proof, public_signals=tuple(tampered_public))
    valid, _ = backend.verify(tampered)
    assert not valid
