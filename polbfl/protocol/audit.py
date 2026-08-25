"""Post-commitment, contract-reproducible probabilistic client auditing."""

from __future__ import annotations
import hashlib

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from polbfl.crypto import canonical_json_bytes, domain_hash
from polbfl.protocol.models import TraceCommitment


BPS = 10_000
ZERO = Decimal("0")
ONE = Decimal("1")
AUDIT_TICKET_DOMAIN = hashlib.sha256(
    b"POLBFL_CLIENT_AUDIT_TICKET_V2"
).digest()


@dataclass(frozen=True)
class AuditSelection:
    round_id: str
    population_size: int
    probability: Decimal
    selected_clients: tuple[str, ...]
    randomness_digest: str
    transcript_digest: str


def audit_round_id_bytes(round_id: str) -> bytes:
    """Map the protocol's textual round identity to Solidity bytes32."""

    if not round_id:
        raise ValueError("audit round identity is required")
    return hashlib.sha256(round_id.encode("utf-8")).digest()


def audit_ticket_from_material(
    *,
    round_id: str,
    commitment_root: str,
    vrf_output: bytes,
) -> int:
    """Return the Python/Solidity audit ticket for committed material."""

    if len(vrf_output) != 32:
        raise ValueError("audit VRF output must contain exactly 256 bits")
    try:
        root_bytes = bytes.fromhex(commitment_root)
    except ValueError as exc:
        raise ValueError("audit commitment root must be hexadecimal") from exc
    if len(root_bytes) != 32:
        raise ValueError("audit commitment root must contain exactly 256 bits")
    material = (
        AUDIT_TICKET_DOMAIN
        + vrf_output
        + audit_round_id_bytes(round_id)
        + root_bytes
    )
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def client_audit_ticket(
    commitment: TraceCommitment,
    *,
    vrf_output: bytes,
) -> int:
    """Return the exact ticket integer evaluated by PoLBFLProtocol.

    The commitment root is already identity-, context-, data-, and
    trajectory-bound. Using it avoids a second client-identity encoding that
    could differ between Python identifiers and Ethereum addresses.
    """

    return audit_ticket_from_material(
        round_id=commitment.round_id,
        commitment_root=commitment.merkle_root,
        vrf_output=vrf_output,
    )


def select_audit_clients(
    commitments: Iterable[TraceCommitment],
    *,
    vrf_output: bytes,
    probability: Decimal = Decimal("0.2"),
) -> AuditSelection:
    population = tuple(commitments)
    if not population or not ZERO <= probability <= ONE:
        raise ValueError("audit population and probability are invalid")
    if len(vrf_output) != 32:
        raise ValueError("audit VRF output must contain exactly 256 bits")
    clients = [item.client_id for item in population]
    rounds = {item.round_id for item in population}
    if len(set(clients)) != len(clients) or len(rounds) != 1:
        raise ValueError("audit commitments must be client-unique within one round")
    threshold_bps = int(probability * BPS)
    selected = tuple(
        sorted(
            item.client_id
            for item in population
            if client_audit_ticket(item, vrf_output=vrf_output) % BPS
            < threshold_bps
        )
    )
    randomness_digest = domain_hash("POLBFL_CLIENT_AUDIT_RANDOMNESS_V1", vrf_output)
    body = {
        "round_id": population[0].round_id,
        "population": sorted(clients),
        "probability": str(probability),
        "selected_clients": list(selected),
        "randomness_digest": randomness_digest,
    }
    return AuditSelection(
        round_id=population[0].round_id,
        population_size=len(population),
        probability=probability,
        selected_clients=selected,
        randomness_digest=randomness_digest,
        transcript_digest=domain_hash(
            "POLBFL_CLIENT_AUDIT_SELECTION_V1", canonical_json_bytes(body)
        ),
    )
