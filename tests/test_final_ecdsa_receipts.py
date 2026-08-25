import hashlib

import pytest

pytest.importorskip("cryptography")

from polbfl.committee import ECDSAPublicKeyRegistry, ECDSASigner, QuorumDecision, ReceiptQuorum
from polbfl.protocol import HybridChallengeSampler, PoLTraceBuilder, RoundContext


def _trace():
    context = RoundContext(
        protocol_version="1",
        round_id="round-ecdsa",
        client_id="client-ecdsa",
        model_id="toy",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD",
        learning_rate=0.1,
        local_epochs=1,
        batch_size=1,
        checkpoint_interval=1,
    )
    builder = PoLTraceBuilder(context)
    for step in range(3):
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
    return builder.finalize()


def test_ecdsa_receipts_form_quorum_and_reject_forgery():
    trace = _trace()
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=1).sample(
        trace.commitment,
        vrf_output=b"e" * 32,
        issued_at_ns=10,
        deadline_ns=100,
    )
    signers = [ECDSASigner.generate(f"verifier-{i}") for i in range(5)]
    registry = ECDSAPublicKeyRegistry({signer.verifier_id: signer.public_pem for signer in signers})
    proof_digest = hashlib.sha256(b"proof").hexdigest()
    receipts = [
        signer.receipt(challenge, proof_digest=proof_digest, valid=True, verified_at_ns=50)
        for signer in signers[:3]
    ]
    quorum = ReceiptQuorum(
        committee=[signer.verifier_id for signer in signers],
        threshold=3,
        verify_signature=registry.verify,
    )
    assert quorum.decide(challenge, receipts, proof_digest=proof_digest) == QuorumDecision.ACCEPT

    forged = type(receipts[0])(**{**receipts[0].unsigned_dict(), "signature": "00"})
    assert quorum.decide(challenge, [forged, *receipts[1:]], proof_digest=proof_digest) == QuorumDecision.NO_QUORUM

    stale = [
        signer.receipt(challenge, proof_digest=proof_digest, valid=True, verified_at_ns=9)
        for signer in signers[:3]
    ]
    assert quorum.decide(challenge, stale, proof_digest=proof_digest) == QuorumDecision.NO_QUORUM
