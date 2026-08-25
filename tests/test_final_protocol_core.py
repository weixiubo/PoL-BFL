import hashlib
import hmac
from dataclasses import replace
from decimal import Decimal

import pytest

from polbfl.committee import (
    CommitteeCandidate,
    QuorumDecision,
    ReceiptQuorum,
    VerificationReceipt,
    select_committee,
)
from polbfl.crypto import (
    MerkleTree,
    canonical_json_bytes,
    decode_merkle_proof,
    domain_hash,
    encode_merkle_proof,
)
from polbfl.incentives import EconomicParameters, IncentiveEngine
from polbfl.protocol import (
    HybridChallengeSampler,
    PoLTraceBuilder,
    RoundContext,
    TraceCommitment,
    AUDIT_TICKET_DOMAIN,
    audit_round_id_bytes,
    select_audit_clients,
)


def _context():
    return RoundContext(
        protocol_version="1",
        round_id="round-7",
        client_id="client-3",
        model_id="resnet18-cifar10",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD(momentum=0.9,weight_decay=0.0005)",
        learning_rate=0.01,
        local_epochs=5,
        batch_size=32,
        checkpoint_interval=10,
    )


def _trace():
    builder = PoLTraceBuilder(_context())
    for position, step in enumerate((0, 10, 20, 30, 40, 50)):
        builder.append_checkpoint(
            step=step,
            epoch=position // 2,
            timestamp_ns=1_000_000 + position,
            model_state={"weight": [position, position + 1], "bias": [position]},
            batch_data=[[position, position + 2]],
            batch_labels=[position % 10],
            batch_indices=[position * 32 + offset for offset in range(32)],
            auxiliary={"gradient_sample": [position, position + 0.5]},
        )
    return builder.finalize()


def test_canonical_serialization_is_order_independent_and_float_exact():
    left = canonical_json_bytes({"b": 2, "a": 0.1})
    right = canonical_json_bytes({"a": 0.1, "b": 2})
    assert left == right
    assert b"0x1.999999999999ap-4" in left


def test_merkle_proof_rejects_tampering():
    leaves = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(5)]
    tree = MerkleTree(leaves)
    proof = tree.proof(3)
    assert MerkleTree.verify(leaves[3], proof, tree.root)
    assert not MerkleTree.verify(hashlib.sha256(b"tampered").hexdigest(), proof, tree.root)
    encoded = encode_merkle_proof(proof)
    assert decode_merkle_proof(encoded) == proof


def test_reference_trace_endpoint_merkle_openings_fit_paper_size():
    leaves = [hashlib.sha256(f"checkpoint-{index}".encode()).hexdigest() for index in range(39)]
    tree = MerkleTree(leaves)
    pair_index = 20
    payload = encode_merkle_proof(tree.proof(pair_index)) + encode_merkle_proof(tree.proof(pair_index + 1))
    assert len(payload) <= int(1.2 * 1024)


def test_trace_binds_order_context_data_indices_and_auxiliary_evidence():
    trace = _trace()
    assert trace.verify_structure()
    assert trace.commitment.checkpoint_count == 6
    assert MerkleTree.verify(
        trace.checkpoints[2].chain_digest,
        trace.checkpoint_proof(2),
        trace.commitment.merkle_root,
    )
    assert trace.checkpoints[2].previous_chain_digest == trace.checkpoints[1].chain_digest


def test_prescribed_optimizer_step_count_rejects_lazy_training_trace():
    context = replace(_context(), expected_steps=50)
    builder = PoLTraceBuilder(context)
    for position, step in enumerate((0, 10, 20, 30, 40)):
        builder.append_checkpoint(
            step=step,
            epoch=position,
            timestamp_ns=position + 1,
            model_state={"weight": [position]},
            batch_data=[[position]],
            batch_labels=[0],
            batch_indices=[position],
            auxiliary={"position": position},
        )
    with pytest.raises(ValueError, match="prescribed optimizer-step count"):
        builder.finalize()


def test_hybrid_challenge_is_deterministic_and_covers_recent_pairs():
    trace = _trace()
    sampler = HybridChallengeSampler(recent_pairs=2, random_pairs=2)
    first = sampler.sample(
        trace.commitment,
        vrf_output=b"v" * 32,
        issued_at_ns=10,
        deadline_ns=20,
    )
    second = sampler.sample(
        trace.commitment,
        vrf_output=b"v" * 32,
        issued_at_ns=10,
        deadline_ns=20,
    )
    assert first == second
    assert {3, 4}.issubset(first.pair_indices)
    assert len(first.pair_indices) == 4


def test_post_commitment_audit_uses_contract_reproducible_probability_tickets():
    commitments = [
        TraceCommitment(
            protocol_version="1",
            round_id="round-audit",
            client_id=f"client-{index}",
            context_digest=hashlib.sha256(f"context-{index}".encode()).hexdigest(),
            merkle_root=hashlib.sha256(f"root-{index}".encode()).hexdigest(),
            checkpoint_count=2,
            first_step=0,
            final_step=5,
            final_model_digest=hashlib.sha256(f"model-{index}".encode()).hexdigest(),
            trace_digest=hashlib.sha256(f"trace-{index}".encode()).hexdigest(),
        )
        for index in range(50)
    ]
    first = select_audit_clients(commitments, vrf_output=b"a" * 32, probability=Decimal("0.2"))
    second = select_audit_clients(reversed(commitments), vrf_output=b"a" * 32, probability=Decimal("0.2"))
    assert first.selected_clients == second.selected_clients
    expected = tuple(
        sorted(
            item.client_id
            for item in commitments
            if int.from_bytes(
                hashlib.sha256(
                    AUDIT_TICKET_DOMAIN
                    + b"a" * 32
                    + audit_round_id_bytes(item.round_id)
                    + bytes.fromhex(item.merkle_root)
                ).digest(),
                "big",
            )
            % 10_000
            < 2_000
        )
    )
    assert first.selected_clients == expected
    assert len(first.transcript_digest) == 64


def test_committee_selection_is_publicly_reproducible():
    candidates = [
        CommitteeCandidate(f"verifier-{i}", Decimal(str(i + 1)), Decimal("0.9"))
        for i in range(8)
    ]
    first = select_committee(
        candidates,
        vrf_seed=b"s" * 32,
        round_id="round-7",
        committee_size=5,
    )
    second = select_committee(
        reversed(candidates),
        vrf_seed=b"s" * 32,
        round_id="round-7",
        committee_size=5,
    )
    assert first == second
    assert len({candidate.verifier_id for candidate in first}) == 5


def test_receipt_quorum_requires_distinct_valid_bound_signatures():
    trace = _trace()
    challenge = HybridChallengeSampler(recent_pairs=2, random_pairs=1).sample(
        trace.commitment,
        vrf_output=b"q" * 32,
        issued_at_ns=10,
        deadline_ns=100,
    )
    secrets = {f"v{i}": f"secret-{i}".encode() for i in range(5)}

    def sign(verifier, payload):
        return hmac.new(secrets[verifier], payload, hashlib.sha256).hexdigest()

    def verify(verifier, payload, signature):
        expected = sign(verifier, payload)
        return hmac.compare_digest(expected, signature)

    proof_digest = domain_hash("TEST_PROOF", b"proof")
    receipts = []
    for i in range(3):
        unsigned = VerificationReceipt(
            protocol_version="1",
            challenge_id=challenge.challenge_id,
            round_id=challenge.round_id,
            client_id=challenge.client_id,
            commitment_root=challenge.commitment_root,
            proof_digest=proof_digest,
            verifier_id=f"v{i}",
            valid=True,
            verified_at_ns=50,
            signature="",
        )
        receipts.append(
            VerificationReceipt(**unsigned.unsigned_dict(), signature=sign(f"v{i}", unsigned.signing_bytes))
        )
    quorum = ReceiptQuorum(committee=secrets, threshold=3, verify_signature=verify)
    assert quorum.decide(challenge, receipts, proof_digest=proof_digest) == QuorumDecision.ACCEPT
    assert quorum.decide(challenge, receipts[:2], proof_digest=proof_digest) == QuorumDecision.NO_QUORUM
    forged = VerificationReceipt(**receipts[0].unsigned_dict(), signature="00")
    assert quorum.decide(challenge, [forged, *receipts[1:]], proof_digest=proof_digest) == QuorumDecision.NO_QUORUM


def test_economic_rules_enforce_reward_reputation_stake_and_equilibrium():
    params = EconomicParameters(
        base_reward=Decimal("0.172"),
        beta_work=Decimal("0"),
        beta_reputation=Decimal("0"),
        reputation_decay=Decimal("0.9"),
        slashing_ratio=Decimal("1"),
        challenge_probability=Decimal("0.2"),
        detection_probability=Decimal("0.965"),
        base_minimum_stake=Decimal("0.05"),
    )
    engine = IncentiveEngine(params)
    assert engine.reward(normalized_work=Decimal("1"), reputation=Decimal("1")) == Decimal("0.172")
    assert engine.update_reputation(current=Decimal("0.5"), verification_success=True) == Decimal("0.55")
    remaining, penalty = engine.slash(Decimal("0.05"))
    assert remaining == Decimal("0") and penalty == Decimal("0.05")
    assert engine.honest_dominates(
        stake=Decimal("0.05"), expected_reward=Decimal("0.172"), saved_cost=Decimal("0.02")
    )
    assert engine.participation_is_rational(
        expected_reward=Decimal("0.172"), honest_cost=Decimal("0.022")
    )
