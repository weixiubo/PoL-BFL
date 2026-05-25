import json
import os
import sys
import time

# Ensure project root on sys.path
CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CUR, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from client.zkp.ZKPProver import ZKPProver
from zkp.hash import quantize_to_field, poseidon_fold


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--param-size', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # Offline simulation: no brownie, no snarkjs
    prover = ZKPProver(use_simulation=True, param_size=args.param_size, batch_size=args.batch_size)

    g = torch.Generator().manual_seed(7)
    param_size = args.param_size
    W_t = torch.randn(param_size, dtype=torch.float64, generator=g)
    W_t1 = W_t + torch.randn(param_size, dtype=torch.float64, generator=g) * 1e-6
    indices = [0, 1, 2, 3]

    t0 = time.time()
    proof, public = prover.generate_proof(W_t, W_t1, indices, max_distance=None)
    t_prove = time.time() - t0

    # Lightweight "view" verification: recompute hashes and check equality
    q_t = quantize_to_field(W_t.reshape(-1)[:param_size], prover.scale)
    q_t1 = quantize_to_field(W_t1.reshape(-1)[:param_size], prover.scale)
    di = (indices or [])[:prover.batch_size]
    di = di + [0] * (prover.batch_size - len(di))

    t1 = time.time()
    ok_view = (
        int(public['W_t_hash']) == int(poseidon_fold(q_t)) and
        int(public['W_t1_hash']) == int(poseidon_fold(q_t1)) and
        int(public['data_hash']) == int(poseidon_fold(di))
    )
    t_view = time.time() - t1

    # No on-chain path in offline mode
    result = {
        'prove_time_s': round(t_prove, 6),
        'verify_view': {'ok': bool(ok_view), 'time_s': round(t_view, 6)},
        'verify_onchain': {
            'tx_ok': False,
            'time_s': None,
            'gas_used': None,
            'txid': None,
            'note': 'offline simulation; on-chain path requires brownie/ganache and snarkjs assets'
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

