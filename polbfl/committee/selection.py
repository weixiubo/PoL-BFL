"""Auditable stake-and-reputation weighted verifier selection."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from polbfl.crypto.canonical import domain_hash


@dataclass(frozen=True)
class CommitteeCandidate:
    verifier_id: str
    stake: Decimal
    reputation: Decimal
    active: bool = True

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("verifier ID must be non-empty")
        if self.stake < 0 or not (Decimal("0") <= self.reputation <= Decimal("1")):
            raise ValueError("invalid verifier stake or reputation")


def select_committee(
    candidates: Iterable[CommitteeCandidate],
    *,
    vrf_seed: bytes,
    round_id: str,
    committee_size: int,
    minimum_stake: Decimal = Decimal("0"),
    minimum_reputation: Decimal = Decimal("0"),
) -> tuple[CommitteeCandidate, ...]:
    if len(vrf_seed) < 32:
        raise ValueError("VRF seed must contain at least 256 bits")
    if committee_size <= 0:
        raise ValueError("committee size must be positive")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.active
        and candidate.stake >= minimum_stake
        and candidate.reputation >= minimum_reputation
    ]
    if len(eligible) < committee_size:
        raise ValueError("not enough eligible verifiers")
    total_stake = sum((candidate.stake for candidate in eligible), Decimal("0"))
    if total_stake <= 0:
        raise ValueError("eligible verifier stake must be positive")
    max_ticket = Decimal(2**256 - 1)

    def rank(candidate: CommitteeCandidate) -> tuple[Decimal, str]:
        ticket_hex = domain_hash(
            "POLBFL_VERIFIER_VRF_TICKET_V1", vrf_seed, round_id, candidate.verifier_id
        )
        ticket = Decimal(int(ticket_hex, 16)) / max_ticket
        score = ticket * (candidate.stake / total_stake) * candidate.reputation
        return score, candidate.verifier_id

    return tuple(sorted(eligible, key=rank, reverse=True)[:committee_size])
