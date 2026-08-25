import hashlib

import pytest

torch = pytest.importorskip("torch")

from polbfl.protocol import RoundContext
from polbfl.verification import CheckpointOpening, IntervalWitness
from polbfl.verification.torch_replay import TorchSGDReplay, TorchSGDReplayConfig


def _model():
    return torch.nn.Linear(2, 2, bias=False)


def test_torch_replay_matches_original_sgd_transition():
    torch.manual_seed(7)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer_initial = optimizer.state_dict()
    batches = (
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), torch.tensor([1, 0])),
    )
    for data, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(data), labels)
        loss.backward()
        optimizer.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}

    context = RoundContext(
        protocol_version="1",
        round_id="round",
        client_id="client",
        model_id="linear",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD",
        learning_rate=0.1,
        local_epochs=1,
        batch_size=2,
        checkpoint_interval=2,
    )
    dummy = type("Record", (), {"step": 0})()
    end_dummy = type("Record", (), {"step": 2})()
    start = CheckpointOpening(0, dummy, (), initial, [], [], (), {})
    end = CheckpointOpening(1, end_dummy, (), expected, [], [], (), {})
    witness = IntervalWitness(0, batches, optimizer_initial, {})

    replay = TorchSGDReplay(
        TorchSGDReplayConfig(
            model_factory=_model,
            criterion_factory=torch.nn.CrossEntropyLoss,
            momentum=0.9,
            weight_decay=0.001,
        )
    )
    actual = replay(context, start, end, witness)
    for key in expected:
        assert torch.equal(actual[key], expected[key])
