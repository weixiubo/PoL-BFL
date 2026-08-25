import json
import os
from pathlib import Path

import pytest

from polbfl.zk import Groth16Artifacts, Groth16Backend


def test_persistent_rapidsnark_pool_reuses_key_and_produces_valid_proofs():
    raw_build = os.getenv("POL_ZK_REFERENCE_BUILD")
    prover = os.getenv("RAPIDSNARK_PROVER")
    verifier = os.getenv("RAPIDSNARK_VERIFIER")
    library = os.getenv("RAPIDSNARK_LIBRARY")
    if not all((raw_build, prover, verifier, library)):
        pytest.skip("reference Groth16 artifacts and Rapidsnark library are required")
    build = Path(raw_build)
    backend = Groth16Backend(
        Groth16Artifacts(
            wasm=build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
            proving_key=build / "sampled_sgd_reference_final.zkey",
            verification_key=build / "verification_key.json",
            r1cs=build / "sampled_sgd_reference.r1cs",
        ),
        snarkjs_cli="node_modules/snarkjs/cli.js",
        witness_binary=build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference",
        prover_binary=prover,
        verifier_binary=verifier,
        prover_library=library,
        prover_pool_size=1,
        timeout_seconds=300,
    )
    try:
        circuit_input = json.loads((build / "input.json").read_text(encoding="utf-8"))
        first = backend.prove(circuit_input)
        second = backend.prove(circuit_input)
        assert len(first.compact_bytes) == 192
        assert len(second.compact_bytes) == 192
        assert first.proof_digest != second.proof_digest
        assert backend.verify(first)[0]
        assert backend.verify(second)[0]
    finally:
        backend.close()
