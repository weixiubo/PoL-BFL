from copy import deepcopy

from polbfl.protocol import HybridChallengeSampler, PoLTraceBuilder, RoundContext
from polbfl.verification import (
    ChallengeResponse,
    CheckpointOpening,
    IntervalWitness,
    StrictTraceVerifier,
)


def _build_trace_and_evidence():
    context = RoundContext(
        protocol_version="1",
        round_id="round-1",
        client_id="client-1",
        model_id="toy",
        global_model_digest="11" * 32,
        optimizer="SGD",
        learning_rate=0.1,
        local_epochs=1,
        batch_size=2,
        checkpoint_interval=1,
    )
    builder = PoLTraceBuilder(context)
    evidence = {}
    for index in range(6):
        item = {
            "model_state": {"w": [float(index), float(index + 1)], "b": [float(index)]},
            "batch_data": [[index, index + 1]],
            "batch_labels": [index % 2],
            "batch_indices": (index * 2, index * 2 + 1),
            "auxiliary": {"gradient": [1.0, 1.0, 1.0]},
        }
        builder.append_checkpoint(
            step=index,
            epoch=0,
            timestamp_ns=100 + index,
            **item,
        )
        evidence[index] = item
    return builder.finalize(), evidence


def _opening(trace, evidence, index):
    item = evidence[index]
    return CheckpointOpening(
        index=index,
        record=trace.checkpoints[index],
        merkle_proof=trace.checkpoint_proof(index),
        model_state=item["model_state"],
        batch_data=item["batch_data"],
        batch_labels=item["batch_labels"],
        batch_indices=item["batch_indices"],
        auxiliary=item["auxiliary"],
    )


def _response(trace, evidence, challenge, *, delta=1.0, uploaded=None):
    required = {trace.commitment.checkpoint_count - 1}
    for pair in challenge.pair_indices:
        required.update((pair, pair + 1))
    openings = {index: _opening(trace, evidence, index) for index in required}
    witnesses = {
        pair: IntervalWitness(
            pair_index=pair,
            private_batches=(evidence[pair + 1]["batch_data"],),
            optimizer_state={},
            replay_metadata={"delta": float(delta)},
        )
        for pair in challenge.pair_indices
    }
    return ChallengeResponse(
        challenge_id=challenge.challenge_id,
        commitment=trace.commitment,
        openings=openings,
        interval_witnesses=witnesses,
        uploaded_model_state=uploaded or evidence[5]["model_state"],
    )


def _replay(_context, opening, _end, witness):
    delta = witness.replay_metadata["delta"]
    return {key: [value + delta for value in values] for key, values in opening.model_state.items()}


def test_strict_verifier_accepts_bound_replay_and_final_model():
    trace, evidence = _build_trace_and_evidence()
    challenge = HybridChallengeSampler(recent_pairs=2, random_pairs=2).sample(
        trace.commitment,
        vrf_output=b"r" * 32,
        issued_at_ns=10,
        deadline_ns=20,
    )
    report = StrictTraceVerifier(pair_tolerance=1e-12, final_tolerance=1e-12).verify(
        context=trace.context,
        challenge=challenge,
        response=_response(trace, evidence, challenge),
        replay_interval=_replay,
    )
    assert report.valid, report.reasons
    assert set(report.pair_results) == set(challenge.pair_indices)


def test_strict_verifier_rejects_bad_replay():
    trace, evidence = _build_trace_and_evidence()
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=1).sample(
        trace.commitment,
        vrf_output=b"b" * 32,
        issued_at_ns=10,
        deadline_ns=20,
    )
    report = StrictTraceVerifier(pair_tolerance=1e-12, final_tolerance=1e-12).verify(
        context=trace.context,
        challenge=challenge,
        response=_response(trace, evidence, challenge, delta=2.0),
        replay_interval=_replay,
    )
    assert not report.valid
    assert any(reason.startswith("pair_tolerance_exceeded") for reason in report.reasons)


def test_strict_verifier_rejects_uploaded_model_not_bound_to_final_checkpoint():
    trace, evidence = _build_trace_and_evidence()
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=1).sample(
        trace.commitment,
        vrf_output=b"c" * 32,
        issued_at_ns=10,
        deadline_ns=20,
    )
    uploaded = deepcopy(evidence[5]["model_state"])
    uploaded["w"][0] += 0.1
    report = StrictTraceVerifier(pair_tolerance=1e-12, final_tolerance=1e-5).verify(
        context=trace.context,
        challenge=challenge,
        response=_response(trace, evidence, challenge, uploaded=uploaded),
        replay_interval=_replay,
    )
    assert not report.valid
    assert "final_tolerance_exceeded" in report.reasons


def test_strict_verifier_rejects_tampered_checkpoint_evidence():
    trace, evidence = _build_trace_and_evidence()
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=1).sample(
        trace.commitment,
        vrf_output=b"d" * 32,
        issued_at_ns=10,
        deadline_ns=20,
    )
    response = _response(trace, evidence, challenge)
    index = next(iter(response.openings))
    original = response.openings[index]
    tampered = CheckpointOpening(
        index=original.index,
        record=original.record,
        merkle_proof=original.merkle_proof,
        model_state=original.model_state,
        batch_data=[[999, 999]],
        batch_labels=original.batch_labels,
        batch_indices=original.batch_indices,
        auxiliary=original.auxiliary,
    )
    openings = dict(response.openings)
    openings[index] = tampered
    response = ChallengeResponse(
        challenge_id=response.challenge_id,
        commitment=response.commitment,
        openings=openings,
        interval_witnesses=response.interval_witnesses,
        uploaded_model_state=response.uploaded_model_state,
    )
    report = StrictTraceVerifier(pair_tolerance=1e-12, final_tolerance=1e-12).verify(
        context=trace.context,
        challenge=challenge,
        response=response,
        replay_interval=_replay,
    )
    assert not report.valid
    assert any(reason.startswith("invalid_checkpoint_opening") for reason in report.reasons)
