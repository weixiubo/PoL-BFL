"""Strictly bound signed verification receipts and M-of-N decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from polbfl.crypto.canonical import canonical_json_bytes, domain_hash
from polbfl.protocol.models import Challenge


SignatureVerifier = Callable[[str, bytes, str], bool]


class QuorumDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    NO_QUORUM = "no_quorum"


@dataclass(frozen=True)
class VerificationReceipt:
    protocol_version: str
    challenge_id: str
    round_id: str
    client_id: str
    commitment_root: str
    proof_digest: str
    verifier_id: str
    valid: bool
    verified_at_ns: int
    signature: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.protocol_version,
                self.challenge_id,
                self.round_id,
                self.client_id,
                self.verifier_id,
            )
        ):
            raise ValueError("receipt identifiers must be non-empty")
        for name in ("commitment_root", "proof_digest"):
            value = str(getattr(self, name))
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be hexadecimal") from exc
        if self.verified_at_ns < 0:
            raise ValueError("receipt timestamp must be non-negative")

    def unsigned_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "challenge_id": self.challenge_id,
            "round_id": self.round_id,
            "client_id": self.client_id,
            "commitment_root": self.commitment_root,
            "proof_digest": self.proof_digest,
            "verifier_id": self.verifier_id,
            "valid": self.valid,
            "verified_at_ns": self.verified_at_ns,
        }

    @property
    def signing_bytes(self) -> bytes:
        return canonical_json_bytes(self.unsigned_dict())

    @property
    def receipt_digest(self) -> str:
        return domain_hash("POLBFL_VERIFICATION_RECEIPT_V1", self.signing_bytes, self.signature)


class ReceiptQuorum:
    def __init__(self, *, committee: Iterable[str], threshold: int, verify_signature: SignatureVerifier):
        self.committee = frozenset(str(member) for member in committee)
        self.threshold = int(threshold)
        self.verify_signature = verify_signature
        if not self.committee or not (len(self.committee) // 2 < self.threshold <= len(self.committee)):
            raise ValueError("threshold must be an honest majority within the committee")

    def decide(
        self,
        challenge: Challenge,
        receipts: Iterable[VerificationReceipt],
        *,
        proof_digest: str,
        deadline_ns: int | None = None,
    ) -> QuorumDecision:
        cutoff = challenge.deadline_ns if deadline_ns is None else min(deadline_ns, challenge.deadline_ns)
        votes: dict[str, bool] = {}
        for receipt in receipts:
            if receipt.verifier_id in votes or receipt.verifier_id not in self.committee:
                continue
            if (
                receipt.protocol_version != challenge.protocol_version
                or receipt.challenge_id != challenge.challenge_id
                or receipt.round_id != challenge.round_id
                or receipt.client_id != challenge.client_id
                or receipt.commitment_root != challenge.commitment_root
                or receipt.proof_digest != proof_digest
                or receipt.verified_at_ns < challenge.issued_at_ns
                or receipt.verified_at_ns > cutoff
            ):
                continue
            if not self.verify_signature(receipt.verifier_id, receipt.signing_bytes, receipt.signature):
                continue
            votes[receipt.verifier_id] = receipt.valid
        accepts = sum(votes.values())
        rejects = len(votes) - accepts
        if accepts >= self.threshold:
            return QuorumDecision.ACCEPT
        if rejects >= self.threshold:
            return QuorumDecision.REJECT
        return QuorumDecision.NO_QUORUM
