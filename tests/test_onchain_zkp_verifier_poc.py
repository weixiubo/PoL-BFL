import os
import pytest

import chainfl.interact as ci
from client.zkp.ZKPProver import ZKPProver
from server.zkp.ZKPVerifier import ZKPVerifier


@pytest.mark.skipif(
    not hasattr(ci, 'Verifier') and not hasattr(ci, 'Groth16Verifier'),
    reason='Groth16 Verifier contract not compiled; run snarkjs export to chainEnv/contracts/Groth16Verifier.sol'
)
@pytest.mark.timeout(120)
def test_onchain_groth16_verifier_roundtrip():
    # Ensure offline fallback is disabled
    os.environ.pop('POL_OFFLINE_FALLBACK', None)

    # Prepare prover/verifier (real mode)
    prover = ZKPProver(
        circuit_js_dir='circuits/build/parameter_update_js',
        proving_key_path='circuits/build/parameter_update_0001.zkey',
        use_simulation=False,
    )

    # Synthetic small vectors (shape must match circuit expectations)
    param_size = prover.param_size if hasattr(prover, 'param_size') else 120
    import torch
    g = torch.Generator().manual_seed(42)
    W_t = torch.randn(param_size, dtype=torch.float64, generator=g)
    W_t1 = W_t + torch.randn(param_size, dtype=torch.float64, generator=g) * 1e-6

    # Distance margin (as in circuit public input)
    max_distance = 1000000  # rely on circuit-side scaling

    proof, public = prover.generate_proof(W_t, W_t1, data_indices=[0, 1, 2, 3], max_distance=max_distance)

    # On-chain verify (view)
    ok = ci.chain_proxy.verify_zkp_onchain(proof, public)
    assert ok is True

    # Negative case: tamper with a public signal
    bad_public = dict(public)
    # Convert to int then +1; keep as str to match downstream parsing
    bad_public['W_t_hash'] = str(int(public['W_t_hash']) + 1)
    ok2 = ci.chain_proxy.verify_zkp_onchain(proof, bad_public)
    assert ok2 is False

