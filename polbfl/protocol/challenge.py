"""Commitment-bound recent-plus-random checkpoint challenge sampling."""

from __future__ import annotations

import hashlib
import random

from polbfl.crypto.canonical import domain_hash
from polbfl.protocol.models import Challenge, TraceCommitment


class HybridChallengeSampler:
    def __init__(self, *, recent_pairs: int, random_pairs: int):
        if recent_pairs < 0 or random_pairs < 0 or recent_pairs + random_pairs <= 0:
            raise ValueError("hybrid challenge must select at least one pair")
        self.recent_pairs = recent_pairs
        self.random_pairs = random_pairs

    def sample(
        self,
        commitment: TraceCommitment,
        *,
        vrf_output: bytes,
        issued_at_ns: int,
        deadline_ns: int,
        proof_mode: str = "zk",
    ) -> Challenge:
        if len(vrf_output) < 32:
            raise ValueError("VRF output must contain at least 256 bits")
        pair_count = commitment.checkpoint_count - 1
        recent_count = min(self.recent_pairs, pair_count)
        recent = set(range(pair_count - recent_count, pair_count))
        candidates = [index for index in range(pair_count) if index not in recent]
        random_count = min(self.random_pairs, len(candidates))
        seed_material = bytes.fromhex(
            domain_hash(
                "POLBFL_CHALLENGE_SEED_V1",
                vrf_output,
                bytes.fromhex(commitment.trace_digest),
            )
        )
        rng = random.Random(int.from_bytes(seed_material, "big"))
        selected = recent | set(rng.sample(candidates, random_count))
        pair_indices = tuple(sorted(selected))
        randomness_digest = hashlib.sha256(vrf_output).hexdigest()
        challenge_id = domain_hash(
            "POLBFL_CHALLENGE_ID_V1",
            commitment.round_id,
            commitment.client_id,
            bytes.fromhex(commitment.merkle_root),
            bytes.fromhex(randomness_digest),
            issued_at_ns,
            deadline_ns,
            ",".join(map(str, pair_indices)),
        )
        return Challenge(
            challenge_id=challenge_id,
            protocol_version=commitment.protocol_version,
            round_id=commitment.round_id,
            client_id=commitment.client_id,
            commitment_root=commitment.merkle_root,
            pair_indices=pair_indices,
            randomness_digest=randomness_digest,
            issued_at_ns=issued_at_ns,
            deadline_ns=deadline_ns,
            proof_mode=proof_mode,
        )
