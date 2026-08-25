import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import pytest

torch = pytest.importorskip("torch")

from polbfl.protocol import HybridChallengeSampler, RoundContext
from polbfl.crypto import hash_state_dict
from polbfl.storage import ContentAddressedStore
from polbfl.training import TorchPoLRecorder
from polbfl.zk import PoseidonBridge, ZKCircuitConfig, ZKPoLProver


ROOT = Path(__file__).parents[1]


def _load(store, reference):
    payload = store.get(reference)
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(io.BytesIO(payload), map_location="cpu")


def test_recorder_commits_circuit_ready_five_step_witness(tmp_path):
    if not (ROOT / "node_modules" / "circomlibjs").exists():
        pytest.skip("circomlibjs is not installed")
    torch.manual_seed(23)
    model = torch.nn.Linear(200, 10, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    context = RoundContext(
        protocol_version="1",
        round_id="round-zk-recorder",
        client_id="client-zk-recorder",
        model_id="linear-200x10",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD(momentum=0x0.0p+0,weight_decay=0x0.0p+0)",
        learning_rate=0.01,
        local_epochs=1,
        batch_size=32,
        checkpoint_interval=5,
    )
    store = ContentAddressedStore(tmp_path / "evidence")
    recorder = TorchPoLRecorder(
        context,
        store,
        sampling_seed=b"z" * 32,
        gradient_sample_rate=0.01,
        zk_config=ZKCircuitConfig(),
        poseidon_bridge=PoseidonBridge(),
    )
    recorder.start(model=model, optimizer=optimizer, timestamp_ns=100)
    for step in range(1, 6):
        data = torch.randn(2, 200) * 0.1
        labels = torch.tensor([step % 10, (step + 1) % 10])
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, labels)
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
            activations={"logits": logits.detach()},
            timestamp_ns=100 + step,
        )
    recorded = recorder.finalize(timestamp_ns=200)
    assert recorded.trace.verify_structure()
    assert recorded.trace.commitment.checkpoint_count == 2
    assert recorded.checkpoints[0].auxiliary["model_commitment"] == "zk_sampled_parameters_v1"
    assert recorded.checkpoints[1].auxiliary["model_commitment"] == "full_model_state_v1"
    assert recorded.trace.commitment.final_model_digest == hash_state_dict(model.state_dict())
    start_zk = recorded.checkpoints[0].auxiliary["zk"]
    end_zk = recorded.checkpoints[1].auxiliary["zk"]
    assert start_zk["gradient_sample_rate"] == 0.01
    assert start_zk["sample_plan_hash"] == end_zk["sample_plan_hash"]
    assert end_zk["interval"]["active_step_count"] == 5
    assert all(
        str(end_zk["interval"][name]).isdigit()
        for name in (
            "gradients_hash",
            "data_indices_hash",
            "auxiliary_hash",
            "batch_commitment_hash",
        )
    )
    private = [_load(store, recorded.steps[step].blob)["zk_witness"] for step in range(1, 6)]
    assert all(len(step["sample_indices"]) == 14 for step in private)
    assert all(len(step["data_indices"]) == 32 for step in private)
    assert all(private[index]["weights_after"] == private[index + 1]["weights_before"] for index in range(4))
    projected_cifar_storage = (
        max(item.blob.size for item in recorded.steps.values()) * 190
        + max(item.blob.size for item in recorded.checkpoints.values()) * 39
    )
    assert projected_cifar_storage <= int(2.5 * 1024 * 1024)

    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=0).sample(
        recorded.trace.commitment,
        vrf_output=b"q" * 32,
        issued_at_ns=300,
        deadline_ns=400,
        proof_mode="zk",
    )
    circuit_input = ZKPoLProver(None, ZKCircuitConfig(), store=store).build_circuit_input(
        recorded=recorded,
        challenge=challenge,
        pair_index=0,
    )
    assert circuit_input["activeStepCount"] == "5"
    assert circuit_input["stepActive"] == ["1"] * 5
    assert len(circuit_input["weightMagnitude"]) == 6
    assert len(circuit_input["activationMagnitude"]) == 5
    assert len(circuit_input["activationMagnitude"][0]) == 14
    assert len(circuit_input["activationMagnitude"][0][0]) == 32

    reference_build = os.getenv("POL_ZK_REFERENCE_BUILD")
    if reference_build:
        build = Path(reference_build)
        input_path = tmp_path / "circuit-input.json"
        witness_path = tmp_path / "witness.wtns"
        input_path.write_text(json.dumps(circuit_input), encoding="utf-8")
        subprocess.run(
            [
                os.getenv("NODE_BINARY", "node"),
                str(build / "sampled_sgd_reference_js" / "generate_witness.js"),
                str(build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm"),
                str(input_path),
                str(witness_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert witness_path.stat().st_size > 0
