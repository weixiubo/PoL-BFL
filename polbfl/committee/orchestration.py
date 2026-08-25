"""Authenticated committee assignment and independent proof-set voting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Iterable, Protocol, Sequence

from polbfl.committee.ecdsa import ECDSASigner
from polbfl.committee.receipts import QuorumDecision, ReceiptQuorum, VerificationReceipt
from polbfl.committee.selection import CommitteeCandidate, select_committee
from polbfl.crypto import canonical_json_bytes, domain_hash
from polbfl.protocol import Challenge, RoundContext
from polbfl.zk import ZKIntervalBundle


RandomnessProofVerifier = Callable[[str, str, bytes, bytes], bool]


@dataclass(frozen=True)
class AuthenticatedVRFSeed:
    round_id: str
    provider_id: str
    output: bytes
    proof: bytes

    def __post_init__(self) -> None:
        if not self.round_id or not self.provider_id:
            raise ValueError("VRF round and provider must be non-empty")
        if len(self.output) < 32 or not self.proof:
            raise ValueError("VRF output and proof are required")

    @property
    def digest(self) -> str:
        return domain_hash(
            "POLBFL_AUTHENTICATED_VRF_SEED_V1",
            self.round_id,
            self.provider_id,
            self.output,
            self.proof,
        )

    def verify(self, verifier: RandomnessProofVerifier) -> bool:
        return bool(verifier(self.provider_id, self.round_id, self.output, self.proof))


@dataclass(frozen=True)
class CommitteeAssignment:
    round_id: str
    aggregator_id: str
    members: tuple[CommitteeCandidate, ...]
    threshold: int
    randomness_digest: str
    transcript_digest: str

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member.verifier_id for member in self.members)

    @classmethod
    def create(
        cls,
        *,
        round_id: str,
        aggregator_id: str,
        candidates: Iterable[CommitteeCandidate],
        vrf: AuthenticatedVRFSeed,
        verify_vrf: RandomnessProofVerifier,
        committee_size: int = 5,
        threshold: int = 3,
        minimum_stake: Decimal = Decimal("0"),
        minimum_reputation: Decimal = Decimal("0"),
    ) -> "CommitteeAssignment":
        if vrf.round_id != round_id or not vrf.verify(verify_vrf):
            raise ValueError("VRF proof is invalid or bound to another round")
        if not aggregator_id:
            raise ValueError("aggregator ID must be non-empty")
        eligible = tuple(
            candidate for candidate in candidates if candidate.verifier_id != aggregator_id
        )
        members = select_committee(
            eligible,
            vrf_seed=vrf.output,
            round_id=round_id,
            committee_size=committee_size,
            minimum_stake=minimum_stake,
            minimum_reputation=minimum_reputation,
        )
        if not (committee_size // 2 < threshold <= committee_size):
            raise ValueError("committee threshold must be an honest majority")
        body = {
            "round_id": round_id,
            "aggregator_id": aggregator_id,
            "member_ids": [member.verifier_id for member in members],
            "member_stakes": [str(member.stake) for member in members],
            "member_reputations": [str(member.reputation) for member in members],
            "threshold": threshold,
            "randomness_digest": vrf.digest,
        }
        return cls(
            round_id=round_id,
            aggregator_id=aggregator_id,
            members=members,
            threshold=threshold,
            randomness_digest=vrf.digest,
            transcript_digest=domain_hash(
                "POLBFL_COMMITTEE_ASSIGNMENT_V1", canonical_json_bytes(body)
            ),
        )


def proof_set_digest(challenge: Challenge, bundles: Sequence[ZKIntervalBundle]) -> str:
    pairs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for bundle in sorted(bundles, key=lambda item: item.pair_index):
        if bundle.pair_index in seen:
            raise ValueError("proof set contains a duplicate interval")
        seen.add(bundle.pair_index)
        pairs.append(
            {
                "pair_index": bundle.pair_index,
                "proof_digest": bundle.proof.proof_digest,
                "circuit_digest": bundle.proof.circuit_digest,
            }
        )
    return domain_hash(
        "POLBFL_ZK_PROOF_SET_V1",
        challenge.challenge_id,
        canonical_json_bytes(pairs),
    )


class BundleVerifier(Protocol):
    def verify(self, context: RoundContext, bundle: ZKIntervalBundle) -> Any: ...


@dataclass(frozen=True)
class VerifierVote:
    receipt: VerificationReceipt
    reports: tuple[Any, ...]
    complete: bool


class IndependentVerifier:
    """Verify every challenged interval locally, then emit one bound receipt."""

    def __init__(self, signer: ECDSASigner, verifier: BundleVerifier):
        self.signer = signer
        self.verifier = verifier

    def verify_and_sign(
        self,
        *,
        assignment: CommitteeAssignment,
        context: RoundContext,
        challenge: Challenge,
        bundles: Sequence[ZKIntervalBundle],
        verified_at_ns: int,
    ) -> VerifierVote:
        if self.signer.verifier_id not in assignment.member_ids:
            raise PermissionError("verifier is not assigned to this round")
        if self.signer.verifier_id == assignment.aggregator_id:
            raise PermissionError("aggregator and verifier roles must be disjoint")
        if assignment.round_id != challenge.round_id or context.round_id != challenge.round_id:
            raise ValueError("committee, context, and challenge rounds differ")
        if verified_at_ns > challenge.deadline_ns:
            raise TimeoutError("verification completed after the challenge deadline")

        digest = proof_set_digest(challenge, bundles)
        expected_pairs = set(challenge.pair_indices)
        submitted_pairs = {bundle.pair_index for bundle in bundles}
        complete = submitted_pairs == expected_pairs and len(bundles) == len(expected_pairs)
        reports = tuple(self.verifier.verify(context, bundle) for bundle in bundles)
        valid = complete and all(bool(getattr(report, "valid", False)) for report in reports)
        receipt = self.signer.receipt(
            challenge,
            proof_digest=digest,
            valid=valid,
            verified_at_ns=verified_at_ns,
        )
        return VerifierVote(receipt=receipt, reports=reports, complete=complete)


def decide_proof_set(
    *,
    assignment: CommitteeAssignment,
    challenge: Challenge,
    bundles: Sequence[ZKIntervalBundle],
    receipts: Iterable[VerificationReceipt],
    verify_signature: Callable[[str, bytes, str], bool],
) -> QuorumDecision:
    quorum = ReceiptQuorum(
        committee=assignment.member_ids,
        threshold=assignment.threshold,
        verify_signature=verify_signature,
    )
    return quorum.decide(
        challenge,
        receipts,
        proof_digest=proof_set_digest(challenge, bundles),
    )
