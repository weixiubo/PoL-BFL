import hashlib
import hmac
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

import pytest

pytest.importorskip("cryptography")

from polbfl.committee import (
    AuthenticatedVRFSeed,
    CommitteeAssignment,
    CommitteeCandidate,
    ECDSAPublicKeyRegistry,
    ECDSASigner,
    IndependentVerifier,
    QuorumDecision,
    decide_proof_set,
)
from polbfl.protocol import HybridChallengeSampler, PoLTraceBuilder, RoundContext
from polbfl.zk import Groth16Proof


def _challenge():
    context = RoundContext(
        protocol_version="1",
        round_id="round-committee",
        client_id="client-a",
        model_id="toy",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD",
        learning_rate=0.1,
        local_epochs=1,
        batch_size=2,
        checkpoint_interval=1,
    )
    builder = PoLTraceBuilder(context)
    for step in range(4):
        builder.append_checkpoint(
            step=step,
            epoch=0,
            timestamp_ns=step + 1,
            model_state={"w": [step]},
            batch_data=[[step]],
            batch_labels=[0],
            batch_indices=[step],
            auxiliary={"step": step},
        )
    trace = builder.finalize()
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=1).sample(
        trace.commitment,
        vrf_output=b"c" * 32,
        issued_at_ns=10,
        deadline_ns=100,
    )
    return context, challenge


def _vrf(round_id="round-committee"):
    output = b"r" * 32
    proof = hmac.new(b"oracle-key", round_id.encode() + output, hashlib.sha256).digest()
    return AuthenticatedVRFSeed(round_id, "oracle-1", output, proof)


def _verify_vrf(provider_id, round_id, output, proof):
    if provider_id != "oracle-1":
        return False
    expected = hmac.new(b"oracle-key", round_id.encode() + output, hashlib.sha256).digest()
    return hmac.compare_digest(expected, proof)


def _assignment():
    candidates = [
        CommitteeCandidate("aggregator", Decimal("100"), Decimal("1")),
        *[
            CommitteeCandidate(f"verifier-{index}", Decimal(index + 1), Decimal("0.9"))
            for index in range(7)
        ],
    ]
    return CommitteeAssignment.create(
        round_id="round-committee",
        aggregator_id="aggregator",
        candidates=candidates,
        vrf=_vrf(),
        verify_vrf=_verify_vrf,
        committee_size=5,
        threshold=3,
    )


@dataclass(frozen=True)
class _Bundle:
    pair_index: int
    proof: Groth16Proof


class _Verifier:
    def __init__(self, invalid_pair=None):
        self.invalid_pair = invalid_pair

    def verify(self, _context, bundle):
        return SimpleNamespace(valid=bundle.pair_index != self.invalid_pair)


def _bundles(challenge):
    return [
        _Bundle(
            pair,
            Groth16Proof(
                proof={"pair": pair},
                public_signals=(),
                circuit_digest=hashlib.sha256(b"circuit").hexdigest(),
                proof_digest=hashlib.sha256(f"proof-{pair}".encode()).hexdigest(),
                prove_seconds=0.1,
                peak_child_rss_kb=1,
            ),
        )
        for pair in challenge.pair_indices
    ]


def test_authenticated_assignment_is_deterministic_and_disjoint():
    assignment = _assignment()
    assert len(assignment.members) == 5
    assert assignment.threshold == 3
    assert "aggregator" not in assignment.member_ids
    assert len(assignment.transcript_digest) == 64

    invalid = AuthenticatedVRFSeed(
        "round-committee", "oracle-1", b"r" * 32, b"invalid"
    )
    with pytest.raises(ValueError, match="VRF proof"):
        CommitteeAssignment.create(
            round_id="round-committee",
            aggregator_id="aggregator",
            candidates=assignment.members,
            vrf=invalid,
            verify_vrf=_verify_vrf,
        )


def test_independent_three_of_five_receipts_accept_complete_set_and_reject_missing_set():
    context, challenge = _challenge()
    assignment = _assignment()
    signers = {member: ECDSASigner.generate(member) for member in assignment.member_ids}
    registry = ECDSAPublicKeyRegistry(
        {member: signer.public_pem for member, signer in signers.items()}
    )
    bundles = _bundles(challenge)
    receipts = [
        IndependentVerifier(signers[member], _Verifier())
        .verify_and_sign(
            assignment=assignment,
            context=context,
            challenge=challenge,
            bundles=bundles,
            verified_at_ns=50,
        )
        .receipt
        for member in assignment.member_ids[:3]
    ]
    assert (
        decide_proof_set(
            assignment=assignment,
            challenge=challenge,
            bundles=bundles,
            receipts=receipts,
            verify_signature=registry.verify,
        )
        == QuorumDecision.ACCEPT
    )

    incomplete = bundles[:-1]
    rejects = [
        IndependentVerifier(signers[member], _Verifier())
        .verify_and_sign(
            assignment=assignment,
            context=context,
            challenge=challenge,
            bundles=incomplete,
            verified_at_ns=50,
        )
        .receipt
        for member in assignment.member_ids[:3]
    ]
    assert (
        decide_proof_set(
            assignment=assignment,
            challenge=challenge,
            bundles=incomplete,
            receipts=rejects,
            verify_signature=registry.verify,
        )
        == QuorumDecision.REJECT
    )


def test_unassigned_or_late_verifier_cannot_emit_countable_receipt():
    context, challenge = _challenge()
    assignment = _assignment()
    worker = IndependentVerifier(ECDSASigner.generate("outsider"), _Verifier())
    with pytest.raises(PermissionError):
        worker.verify_and_sign(
            assignment=assignment,
            context=context,
            challenge=challenge,
            bundles=_bundles(challenge),
            verified_at_ns=50,
        )

    member = ECDSASigner.generate(assignment.member_ids[0])
    with pytest.raises(TimeoutError):
        IndependentVerifier(member, _Verifier()).verify_and_sign(
            assignment=assignment,
            context=context,
            challenge=challenge,
            bundles=_bundles(challenge),
            verified_at_ns=101,
        )
