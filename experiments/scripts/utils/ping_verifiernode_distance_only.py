#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ping a running VerifierNode (distance-only mode) with a synthetic PoL response.
This creates two checkpoints with a very small parameter change so that
parameter distance <= delta, builds a Merkle tree and proofs, serializes the
response via torch.save+base64, and posts to the given URL.

Usage:
  python -m experiments.scripts.utils.ping_verifiernode_distance_only --url http://127.0.0.1:8091/verify_full
"""
import argparse
import base64
import io
import json
import urllib.request

import torch
from server.pol.PoLVerifier import PoLVerifier
from client.pol.MerkleTree import MerkleTree


def build_payload(delta: float = 0.01):
    params0 = {"w": torch.zeros(4)}
    params1 = {"w": torch.ones(4) * (delta / 10.0)}  # ensure distance <= delta

    ck0 = {"data": {"model_state": params0, "optimizer_state": {}, "step": 0}}
    ck1 = {"data": {"model_state": params1, "optimizer_state": {}, "step": 1}}

    pv = PoLVerifier({
        "delta": delta,
        "distance_metric": "l2",
        "min_pair_success_rate": 0.99,
        "device": "cpu",
    })
    leaf0 = pv._compute_checkpoint_hash(ck0["data"])  # noqa: SLF001 (internal use for ping)
    leaf1 = pv._compute_checkpoint_hash(ck1["data"])  # noqa: SLF001

    mt = MerkleTree([leaf0, leaf1])
    root = mt.get_root()
    ck0["merkle_proof"] = mt.get_proof(0)
    ck1["merkle_proof"] = mt.get_proof(1)

    response = {"checkpoints": [ck0, ck1]}

    buf = io.BytesIO()
    torch.save(response, buf)
    ser = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "ser_response": ser,
        "commitment": root,
        "verifier_params": {"delta": float(delta), "distance_metric": "l2", "min_pair_success_rate": 0.99},
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091/verify_full")
    ap.add_argument("--delta", type=float, default=0.01)
    args = ap.parse_args()

    payload = build_payload(delta=args.delta)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(args.url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        print(body)


if __name__ == "__main__":
    main()

