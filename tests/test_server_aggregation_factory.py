from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset

from server.aggregation_alg.aggFac import available_aggregators, create_aggregator
from server.serverSimulator import serverSimulator


def _state(value):
    tensor = (
        value.detach().clone().to(dtype=torch.float32)
        if torch.is_tensor(value)
        else torch.tensor(value, dtype=torch.float32)
    )
    return OrderedDict(weight=tensor)


def test_aggregation_factory_resolves_public_method_names():
    assert {"fedavg", "foolsgold", "krum", "median", "shapleyfl"} <= set(
        available_aggregators()
    )
    assert create_aggregator("FedAvg").__class__.__name__ == "fedavgAggregator"
    assert create_aggregator("Shapley-FL").__class__.__name__ == "ShapleyFLAggregator"


def test_fedavg_and_coordinate_median_preserve_tensor_shapes():
    updates = [
        _state([[1.0, 4.0], [3.0, 8.0]]),
        _state([[3.0, 2.0], [5.0, 6.0]]),
        _state([[5.0, 0.0], [7.0, 4.0]]),
    ]

    average = create_aggregator("fedavg").aggregate(updates)
    median = create_aggregator("median").aggregate(updates)

    assert average["weight"].shape == torch.Size([2, 2])
    assert median["weight"].shape == torch.Size([2, 2])
    assert torch.equal(median["weight"], updates[1]["weight"])


def test_krum_factory_rejects_the_distant_update():
    updates = [
        _state([0.0, 0.0]),
        _state([0.1, 0.0]),
        _state([0.0, 0.1]),
        _state([0.1, 0.1]),
        _state([100.0, 100.0]),
    ]

    selected = create_aggregator("krum", num_byzantine=1).aggregate(updates)

    assert float(torch.linalg.vector_norm(selected["weight"]).item()) < 1.0


def test_server_simulator_aggregates_persists_and_evaluates(tmp_path):
    simulator = serverSimulator(
        create_aggregator("fedavg"),
        client_num=2,
        args={"checkpoint_folder": str(tmp_path)},
    )
    identity = torch.eye(2)
    simulator.upload_model({"state_dict": _state(identity)})
    simulator.upload_model({"state_dict": _state(identity)})

    downloaded = simulator.download_model()
    saved = simulator.save_model("global.pt")
    model = torch.nn.Linear(2, 2, bias=False)
    loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])),
        batch_size=2,
    )
    metrics = simulator.test(model=model, test_dataset=loader)

    assert torch.equal(downloaded["weight"], identity)
    assert saved.endswith("global.pt")
    assert metrics == {"accuracy": 1.0, "loss": None, "samples": 2}
