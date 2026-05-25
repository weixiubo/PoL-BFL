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

import torch

def run_one(size: int):
    # Select artifacts for this size
    circuit_js_dir = f'circuits/build_p{size}/param_update_{size}_js'
    proving_key_path = f'circuits/build_p{size}/param_update_{size}_0000.zkey'

    # Enable on-chain path
    ci.chain_proxy.set_onchain_verifier_enabled(True)

    # Prepare prover and inputs
    prover = ZKPProver(
        circuit_js_dir=circuit_js_dir,
        proving_key_path=proving_key_path,
        param_size=size,
        use_simulation=False,
    )

    g = torch.Generator().manual_seed(7)
    W_t = torch.randn(size, dtype=torch.float64, generator=g)
    W_t1 = W_t + torch.randn(size, dtype=torch.float64, generator=g) * 1e-6
    indices = [0, 1, 2, 3]

    # Prove
    t0 = time.time()
    proof, public = prover.generate_proof(W_t, W_t1, indices, max_distance=1000000)
    t_prove = time.time() - t0

    # Proof size
    try:
        proof_size_bytes = len(json.dumps(proof).encode('utf-8'))
    except Exception:
        proof_size_bytes = None

    # View verify
    t1 = time.time()
    ok_view = ci.chain_proxy.verify_zkp_onchain(proof, public)
    t_view = time.time() - t1

    # Tx verify
    client_id = '4'
    try:
        ci.chain_proxy.pol_register_client(client_id)
    except Exception:
        pass  # tolerate already-registered or other benign errors
    cid = ci.chain_proxy.issue_challenge(client_id, idx0=0, idx1=1, deadline_ts=int(time.time()) + 3600)
    t2 = time.time()
    tx = ci.chain_proxy.challenge_proof_with_zkp_onchain_receipt(cid, proof, public, reason=f'bench-{size}')
    t_tx = time.time() - t2
    gas_used = getattr(tx, 'gas_used', None) if tx is not None else None

    return {
        'param_size': size,
        'prove_time_s': round(t_prove, 6),
        'proof_size_bytes': int(proof_size_bytes) if proof_size_bytes is not None else None,
        'verify_view_ms': int(round(t_view * 1000)),
        'tx_time_s': round(t_tx, 6),
        'gas_used': int(gas_used) if gas_used is not None else None,
        'txid': getattr(tx, 'txid', None) if tx is not None else None,
    }


def main():
    sizes = [50, 100, 150, 200]
    results = []
    for s in sizes:
        r = run_one(s)
        print(json.dumps(r, ensure_ascii=False))
        results.append(r)
    print("=== summary ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

