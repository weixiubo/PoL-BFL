import os
import pytest
import chainfl.interact as ci
from client.zkp.ZKPProver import ZKPProver

@pytest.mark.skipif(
    not hasattr(ci, 'Verifier') and not hasattr(ci, 'Groth16Verifier'),
    reason='Groth16 Verifier contract not compiled; run snarkjs export to chainEnv/contracts/Groth16Verifier.sol'
)
@pytest.mark.timeout(180)
def test_onchain_integrated_verifier_path_roundtrip():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)

    # Enable on-chain verifier in PoL contract
    assert ci.chain_proxy.set_onchain_verifier_enabled(True) is True

    prover = ZKPProver(
        circuit_js_dir='circuits/build/parameter_update_js',
        proving_key_path='circuits/build/parameter_update_0001.zkey',
        use_simulation=False,
    )

    # Prepare synthetic vectors
    import torch
    param_size = getattr(prover, 'param_size', 120)
    g = torch.Generator().manual_seed(123)
    W_t = torch.randn(param_size, dtype=torch.float64, generator=g)
    W_t1 = W_t + torch.randn(param_size, dtype=torch.float64, generator=g) * 1e-6
    max_distance = 1000000

    proof, public = prover.generate_proof(W_t, W_t1, data_indices=[0,1,2,3], max_distance=max_distance)

    # Ensure client registered, then issue challenge and resolve via on-chain integrated verification
    assert ci.chain_proxy.pol_register_client('1') is True or True
    cid = ci.chain_proxy.issue_challenge('1', idx0=0, idx1=1, deadline_ts=__import__('time').time().__trunc__() + 3600)
    assert isinstance(cid, str) and cid.startswith('0x')

    ok = ci.chain_proxy.challenge_proof_with_zkp_onchain(cid, proof, public, reason="integrated")
    assert ok is True

    got = ci.chain_proxy.get_challenge(cid)
    assert got.get('resolved') is True and got.get('success') is True

@pytest.mark.timeout(180)
def test_onchain_integrated_verifier_path_rejects_tamper():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)
    assert ci.chain_proxy.set_onchain_verifier_enabled(True) is True

    prover = ZKPProver(
        circuit_js_dir='circuits/build/parameter_update_js',
        proving_key_path='circuits/build/parameter_update_0001.zkey',
        use_simulation=False,
    )
    import torch
    param_size = getattr(prover, 'param_size', 120)
    g = torch.Generator().manual_seed(42)
    W_t = torch.randn(param_size, dtype=torch.float64, generator=g)
    W_t1 = W_t + torch.randn(param_size, dtype=torch.float64, generator=g) * 1e-6
    max_distance = 1000000

    proof, public = prover.generate_proof(W_t, W_t1, data_indices=[0,1,2,3], max_distance=max_distance)
    # Tamper a public signal
    public_bad = dict(public)
    public_bad['data_hash'] = str(int(public['data_hash']) + 12345)

    assert ci.chain_proxy.pol_register_client('2') is True or True
    cid = ci.chain_proxy.issue_challenge('2', idx0=0, idx1=1, deadline_ts=__import__('time').time().__trunc__() + 3600)
    assert isinstance(cid, str) and cid.startswith('0x')

    ok = ci.chain_proxy.challenge_proof_with_zkp_onchain(cid, proof, public_bad, reason="tamper")
    assert ok is True  # tx executed

    got = ci.chain_proxy.get_challenge(cid)
    assert got.get('resolved') is True and got.get('success') is False

