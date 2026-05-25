import json
import os
import sys
import time

# Ensure project root on sys.path
CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CUR, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import chainfl.interact as ci
from client.zkp.ZKPProver import ZKPProver


def main():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)

    # Enable contract-inline verifier path
    ci.chain_proxy.set_onchain_verifier_enabled(True)

    # Prepare prover (real mode)
    prover = ZKPProver(
        circuit_js_dir='circuits/build/parameter_update_js',
        proving_key_path='circuits/build/parameter_update_0001.zkey',
        use_simulation=False,
    )

    import torch
    param_size = getattr(prover, 'param_size', 120)
    g = torch.Generator().manual_seed(7)
    W_t = torch.randn(param_size, dtype=torch.float64, generator=g)
    W_t1 = W_t + torch.randn(param_size, dtype=torch.float64, generator=g) * 1e-6
    indices = [0, 1, 2, 3]

    # Prove
    t0 = time.time()
    proof, public = prover.generate_proof(W_t, W_t1, indices, max_distance=1000000)
    t_prove = time.time() - t0

    # Proof size (bytes) based on JSON serialization
    try:
        import json as _json
        proof_size_bytes = len(_json.dumps(proof).encode('utf-8'))
    except Exception:
        proof_size_bytes = None

    # Path A: On-chain Verifier PoC (view)
    t1 = time.time()
    ok_view = ci.chain_proxy.verify_zkp_onchain(proof, public)
    t_view = time.time() - t1

    # Path B: Contract-inline verification (tx)
    # Ensure client is registered and issue a challenge
    ci.chain_proxy.pol_register_client('4')
    cid = ci.chain_proxy.issue_challenge('4', idx0=0, idx1=1, deadline_ts=int(time.time()) + 3600)

    t2 = time.time()
    tx = ci.chain_proxy.challenge_proof_with_zkp_onchain_receipt(cid, proof, public, reason='bench')
    t_tx = time.time() - t2
    gas_used = getattr(tx, 'gas_used', None) if tx is not None else None

    result = {
        'prove_time_s': round(t_prove, 6),
        'proof_size_bytes': int(proof_size_bytes) if proof_size_bytes is not None else None,
        'verify_view': {'ok': bool(ok_view), 'time_s': round(t_view, 6)},
        'verify_onchain': {
            'tx_ok': bool(tx is not None),
            'time_s': round(t_tx, 6),
            'gas_used': int(gas_used) if gas_used is not None else None,
            'txid': getattr(tx, 'txid', None) if tx is not None else None,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

