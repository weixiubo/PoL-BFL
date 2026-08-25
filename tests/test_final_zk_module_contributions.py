import hashlib
import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from polbfl.protocol import HybridChallengeSampler, RoundContext
from polbfl.storage import ContentAddressedStore
from polbfl.training import TorchPoLRecorder
from polbfl.zk import PoseidonBridge, ZKCircuitConfig, ZKPoLProver


ROOT = Path(__file__).parents[1]


def test_subresolution_zero_sum_is_consistent_only_for_zero_circuit_gradient(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 1e-12)
    context = RoundContext(
        protocol_version="1",
        round_id="subresolution",
        client_id="client-subresolution",
        model_id="linear",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD(momentum=0x0.0p+0,weight_decay=0x0.0p+0)",
        learning_rate=0.01,
        local_epochs=1,
        batch_size=2,
        checkpoint_interval=5,
    )
    recorder = TorchPoLRecorder(
        context,
        ContentAddressedStore(tmp_path / "evidence-tiny"),
        sampling_seed=b"t" * 32,
        gradient_sample_rate=1.0,
        zk_config=ZKCircuitConfig(),
        poseidon_bridge=object(),
    )
    recorder._auxiliary_modules = {"": model}
    recorder._auxiliary_parameters = {"weight": model.weight}
    recorder._module_inputs = {"": torch.zeros(2, 1)}
    recorder._module_grad_outputs = {"": torch.zeros(2, 1)}
    contribution, invalid = recorder._per_example_gradient_contributions(
        model,
        parameter_name="weight",
        local_index=0,
        expected_batch=2,
        expected_quantized_gradient=0,
    )
    assert torch.equal(contribution, torch.zeros(2))
    assert not bool(invalid)
    _contribution, invalid = recorder._per_example_gradient_contributions(
        model,
        parameter_name="weight",
        local_index=0,
        expected_batch=2,
        expected_quantized_gradient=1,
    )
    assert bool(invalid)


def test_float32_reduction_cancellation_is_reconciled_in_float64(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    context = RoundContext(
        protocol_version="1",
        round_id="float32-cancellation",
        client_id="client-float32-cancellation",
        model_id="linear",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD(momentum=0x0.0p+0,weight_decay=0x0.0p+0)",
        learning_rate=0.01,
        local_epochs=1,
        batch_size=3,
        checkpoint_interval=5,
    )
    recorder = TorchPoLRecorder(
        context,
        ContentAddressedStore(tmp_path / "evidence-cancellation"),
        sampling_seed=b"c" * 32,
        gradient_sample_rate=1.0,
        zk_config=ZKCircuitConfig(),
        poseidon_bridge=object(),
    )
    recorder._auxiliary_modules = {"": model}
    recorder._auxiliary_parameters = {"weight": model.weight}
    recorder._module_inputs = {"": torch.ones(3, 1)}
    recorder._module_grad_outputs = {
        "": torch.tensor([[1e8], [1.0], [-1e8]], dtype=torch.float32)
    }

    contribution, invalid = recorder._per_example_gradient_contributions(
        model,
        parameter_name="weight",
        local_index=0,
        expected_batch=3,
        expected_quantized_gradient=1,
    )

    assert contribution.dtype == torch.float64
    assert not bool(invalid)
    assert contribution.sum().item() == pytest.approx(1.0)


class MiniConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc = torch.nn.Linear(32 * 4 * 4, 8)

    def forward(self, value):
        value = torch.relu(self.bn(self.conv(value)))
        return self.fc(torch.flatten(self.pool(value), 1))


def _load(store, reference):
    payload = store.get(reference)
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(io.BytesIO(payload), map_location="cpu")


def test_conv_batchnorm_linear_contributions_satisfy_padded_reference_witness(tmp_path):
    if not (ROOT / "node_modules" / "circomlibjs").exists():
        pytest.skip("circomlibjs is not installed")
    torch.manual_seed(31)
    model = MiniConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    context = RoundContext(
        protocol_version="1",
        round_id="round-module-contributions",
        client_id="client-module-contributions",
        model_id="mini-conv-bn-linear",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD(momentum=0x0.0p+0,weight_decay=0x0.0p+0)",
        learning_rate=0.01,
        local_epochs=1,
        batch_size=32,
        checkpoint_interval=5,
        expected_steps=1,
    )
    store = ContentAddressedStore(tmp_path / "evidence")
    config = ZKCircuitConfig()
    recorder = TorchPoLRecorder(
        context,
        store,
        sampling_seed=b"m" * 32,
        gradient_sample_rate=1.0,
        zk_config=config,
        poseidon_bridge=PoseidonBridge(),
    )
    recorder.start(model=model, optimizer=optimizer, timestamp_ns=1)
    sampled_modules = {name.rsplit(".", 1)[0] for _, name, _ in recorder._circuit_plan}
    assert {"conv", "bn", "fc"}.issubset(sampled_modules)

    data = torch.randn(2, 1, 8, 8)
    labels = torch.tensor([1, 6])
    optimizer.zero_grad(set_to_none=True)
    logits = model(data)
    torch.nn.functional.cross_entropy(logits, labels).backward()
    optimizer.step()
    recorder.record_optimizer_step(
        step=1,
        epoch=0,
        model=model,
        optimizer=optimizer,
        batch_data=data,
        batch_labels=labels,
        batch_indices=(11, 12),
        activations={"logits": logits.detach()},
        timestamp_ns=2,
    )
    recorded = recorder.finalize(timestamp_ns=3)
    private = _load(store, recorded.steps[1].blob)["zk_witness"]
    assert all(sum(row) == gradient for row, gradient in zip(private["error_factors"], private["gradients"]))

    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=0).sample(
        recorded.trace.commitment,
        vrf_output=b"n" * 32,
        issued_at_ns=4,
        deadline_ns=5,
    )
    circuit_input = ZKPoLProver(None, config, store=store).build_circuit_input(
        recorded=recorded,
        challenge=challenge,
        pair_index=0,
    )
    assert circuit_input["activeStepCount"] == "1"
    assert circuit_input["stepActive"] == ["1", "0", "0", "0", "0"]
