#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test RemoteVerifierAdapter against a running VerifierNode (distance-only).
"""
import argparse
import torch
from server.pol.PoLVerifier import PoLVerifier
from client.pol.MerkleTree import MerkleTree
from server.pol.verifier_adapter import RemoteVerifierAdapter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', default='http://127.0.0.1:8091')
    ap.add_argument('--delta', type=float, default=0.01)
    args = ap.parse_args()

    # Build synthetic response
    params0 = {"w": torch.zeros(4)}
    params1 = {"w": torch.ones(4) * (args.delta / 10.0)}
    ck0 = {"data": {"model_state": params0, "optimizer_state": {}, "step": 0}}
    ck1 = {"data": {"model_state": params1, "optimizer_state": {}, "step": 1}}

    pv = PoLVerifier({"delta": args.delta, "distance_metric": "l2", "min_pair_success_rate": 0.99, "device": "cpu"})
    leaf0 = pv._compute_checkpoint_hash(ck0["data"])  # noqa
    leaf1 = pv._compute_checkpoint_hash(ck1["data"])  # noqa

    mt = MerkleTree([leaf0, leaf1])
    root = mt.get_root()
    ck0["merkle_proof"] = mt.get_proof(0)
    ck1["merkle_proof"] = mt.get_proof(1)
    response = {"checkpoints": [ck0, ck1]}

    rv = RemoteVerifierAdapter([args.endpoint], verifier_params={"delta": args.delta, "distance_metric": "l2", "min_pair_success_rate": 0.99})
    ok = rv.verify_response({}, response, root, None, None, None, torch.optim.SGD, 0.1)
    print({"adapter_valid": ok})

if __name__ == '__main__':
    main()

