import os
import pytest
import torch

from client.zkp.ZKPProver import ZKPProver
from server.zkp.ZKPVerifier import ZKPVerifier


def _mk_prover_verifier(simulation: bool = False):
    prover = ZKPProver(
        circuit_js_dir='circuits/build/parameter_update_js',
        proving_key_path='circuits/build/parameter_update_0001.zkey',
        use_simulation=simulation,
    )
    verifier = ZKPVerifier(
        verification_key_path='circuits/build/parameter_update.vkey.json',
        use_simulation=simulation,
        use_onchain=False,
    )
    return prover, verifier


def _rand_weights(n=120, seed=123):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


def test_zkp_real_verify_and_bind_ok():
    # Skip if artifacts missing
    assert os.path.exists('circuits/build/parameter_update_js/parameter_update.wasm')
    assert os.path.exists('circuits/build/parameter_update_0001.zkey')
    assert os.path.exists('circuits/build/parameter_update.vkey.json')

    prover, verifier = _mk_prover_verifier(simulation=False)

    torch.manual_seed(42)
    W_t = _rand_weights()
    # small update (distance is small; prover internally sets max_distance = dist2 + 1)
    W_t1 = W_t + torch.randn(120) * 0.001
    indices = list(range(32))

    proof, public = prover.generate_proof(W_t, W_t1, indices, max_distance=None)

    ckpt0 = {'data': {'model_state': {'w': W_t}}}
    ckpt1 = {'data': {'model_state': {'w': W_t1}}}

    ok = verifier.verify_proof_with_binding(ckpt0, ckpt1, indices, proof, public)
    assert ok is True


def test_zkp_real_bind_fails_on_mismatch():
    # Skip if artifacts missing
    assert os.path.exists('circuits/build/parameter_update_js/parameter_update.wasm')
    assert os.path.exists('circuits/build/parameter_update_0001.zkey')
    assert os.path.exists('circuits/build/parameter_update.vkey.json')

    prover, verifier = _mk_prover_verifier(simulation=False)

    torch.manual_seed(31415)
    W_t = _rand_weights(seed=2025)
    W_t1 = W_t + torch.randn(120) * 0.001
    indices = list(range(32))

    # Generate proof on original tensors
    proof, public = prover.generate_proof(W_t, W_t1, indices, max_distance=None)

    # Tamper: alter checkpoint slightly -> binding should fail even if Groth16 verifies
    W_t_tampered = W_t.clone()
    W_t_tampered[0] += 1.0  # change one entry enough to alter Poseidon fold

    ckpt0_bad = {'data': {'model_state': {'w': W_t_tampered}}}
    ckpt1_ok = {'data': {'model_state': {'w': W_t1}}}

    ok = verifier.verify_proof_with_binding(ckpt0_bad, ckpt1_ok, indices, proof, public)
    assert ok is False

