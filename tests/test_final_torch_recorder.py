import hashlib

import pytest

torch = pytest.importorskip("torch")

from polbfl.protocol import HybridChallengeSampler, RoundContext
from polbfl.storage import ContentAddressedStore
from polbfl.training import TorchPoLRecorder
from polbfl.verification import (
    ChallengeResponse,
    CheckpointOpening,
    IntervalWitness,
    StrictTraceVerifier,
)
from polbfl.verification.torch_replay import TorchSGDReplay, TorchSGDReplayConfig


def _model():
    return torch.nn.Linear(2, 2, bias=False)


def test_recorder_builds_retrievable_trace_and_strict_replay(tmp_path):
    torch.manual_seed(11)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    context = RoundContext(
        protocol_version="1",
        round_id="round-11",
        client_id="client-4",
        model_id="linear",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD",
        learning_rate=0.1,
        local_epochs=1,
        batch_size=2,
        checkpoint_interval=2,
    )
    store = ContentAddressedStore(tmp_path / "evidence")
    recorder = TorchPoLRecorder(
        context,
        store,
        sampling_seed=b"gradient-seed".ljust(32, b"!"),
        gradient_sample_rate=0.01,
    )
    recorder.start(model=model, optimizer=optimizer, timestamp_ns=100)
    batches = (
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), torch.tensor([1, 0])),
    )
    for step, (data, labels) in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(data), labels)
        loss.backward()
        optimizer.step()
        recorder.record_optimizer_step(
            step=step,
            epoch=0,
            model=model,
            optimizer=optimizer,
            batch_data=data,
            batch_labels=labels,
            batch_indices=(2 * step - 2, 2 * step - 1),
            timestamp_ns=100 + step,
        )
    recorded = recorder.finalize()
    assert recorded.trace.verify_structure()
    assert recorded.trace.commitment.checkpoint_count == 2
    assert all(store.has(step.blob) for step in recorded.steps.values())
    assert all(store.has(checkpoint.blob) for checkpoint in recorded.checkpoints.values())

    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=0).sample(
        recorded.trace.commitment,
        vrf_output=b"z" * 32,
        issued_at_ns=200,
        deadline_ns=300,
    )
    openings = {}
    for index, material in recorded.checkpoints.items():
        openings[index] = CheckpointOpening(
            index=index,
            record=recorded.trace.checkpoints[index],
            merkle_proof=recorded.trace.checkpoint_proof(index),
            model_state=material.model_state,
            batch_data=material.batch_data,
            batch_labels=material.batch_labels,
            batch_indices=material.batch_indices,
            auxiliary=material.auxiliary,
        )
    witness = IntervalWitness(
        pair_index=0,
        private_batches=batches,
        optimizer_state=recorded.checkpoints[0].optimizer_state,
        replay_metadata={"step_evidence": [recorded.steps[1].blob.digest, recorded.steps[2].blob.digest]},
    )
    response = ChallengeResponse(
        challenge_id=challenge.challenge_id,
        commitment=recorded.trace.commitment,
        openings=openings,
        interval_witnesses={0: witness},
        uploaded_model_state=recorded.checkpoints[1].model_state,
    )
    replay = TorchSGDReplay(
        TorchSGDReplayConfig(
            model_factory=_model,
            criterion_factory=torch.nn.CrossEntropyLoss,
            momentum=0.9,
            weight_decay=0.001,
        )
    )
    report = StrictTraceVerifier(pair_tolerance=1e-12, final_tolerance=1e-12).verify(
        context=context,
        challenge=challenge,
        response=response,
        replay_interval=replay,
    )
    assert report.valid, report.reasons
