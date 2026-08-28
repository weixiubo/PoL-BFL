from collections import OrderedDict

import torch

from server.aggregation_alg.fools_gold import FoolsGoldAggregator
from server.aggregation_alg.shapley_fl import ShapleyFLAggregator


def _update(*values):
    return OrderedDict(weight=torch.tensor(values, dtype=torch.float32))


def test_shapley_adapter_uses_measured_coalition_utility():
    aggregator = ShapleyFLAggregator(
        num_mc_samples=200,
        evaluation_fn=lambda state: float(state["weight"].mean()),
        seed=1337,
    )

    aggregated = aggregator.aggregate([_update(0.0), _update(10.0)])

    assert float(aggregated["weight"].item()) > 9.0
    assert len(aggregator.get_shapley_history()) == 1


def test_shapley_adapter_has_a_defined_fedavg_path_without_validation():
    aggregator = ShapleyFLAggregator(seed=7)

    aggregated = aggregator.aggregate([_update(2.0), _update(6.0)])

    assert float(aggregated["weight"].item()) == 4.0


def test_foolsgold_adapter_emits_auditable_weights_and_resizes_memory():
    aggregator = FoolsGoldAggregator(use_memory=True)
    first = [
        _update(1.0, 1.0),
        _update(1.0, 1.0),
        _update(1.0, -1.0),
    ]

    aggregated = aggregator.aggregate(first)
    diagnostics = aggregator.test()

    assert aggregated["weight"].shape == torch.Size([2])
    assert diagnostics["rounds"] == 1
    assert diagnostics["memory_shape"] == [3, 2]
    assert diagnostics["latest_weights"][2] >= diagnostics["latest_weights"][0]

    aggregator.aggregate([_update(1.0, 0.0), _update(0.0, 1.0)])
    assert aggregator.test()["memory_shape"] == [2, 2]
