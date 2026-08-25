"""BN254 field adapters for binding SHA-256 protocol records to Circom."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from polbfl.crypto import domain_hash


BN254_SCALAR_FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def digest_to_field(digest: str) -> int:
    """Map an authenticated SHA-256 digest into the Circom scalar field."""

    if len(digest) != 64:
        raise ValueError("field digest must be a 32-byte hexadecimal value")
    try:
        value = int(digest, 16)
    except ValueError as exc:
        raise ValueError("field digest must be hexadecimal") from exc
    return value % BN254_SCALAR_FIELD


def interval_batch_commitment(step_evidence: Iterable[Mapping[str, Any]]) -> str:
    """Commit, in optimizer-step order, to every private batch SHA-256 value."""

    ordered = sorted(step_evidence, key=lambda item: int(item["step"]))
    if not ordered:
        raise ValueError("a ZK interval requires committed step evidence")
    steps = [int(item["step"]) for item in ordered]
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("ZK interval step evidence must be contiguous")
    digests: list[bytes] = []
    for item in ordered:
        digest = str(item.get("batch_digest", ""))
        if len(digest) != 64:
            raise ValueError("step evidence has an invalid batch digest")
        try:
            digests.append(bytes.fromhex(digest))
        except ValueError as exc:
            raise ValueError("step evidence batch digest must be hexadecimal") from exc
    return domain_hash("POLBFL_INTERVAL_BATCH_COMMITMENT_V1", *digests)


def protocol_binding_field(
    *,
    context_digest: str,
    commitment_root: str,
    challenge_id: str,
    pair_index: int,
    batch_commitment: str,
) -> int:
    """Reproduce the public linear field binding enforced by the circuit."""

    if pair_index < 0 or pair_index >= 2**32:
        raise ValueError("pair index must fit in 32 bits")
    return (
        digest_to_field(context_digest)
        + 2 * digest_to_field(commitment_root)
        + 4 * digest_to_field(challenge_id)
        + 8 * int(pair_index)
        + 16 * digest_to_field(batch_commitment)
    ) % BN254_SCALAR_FIELD
