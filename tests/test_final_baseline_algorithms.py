import math

import pytest

torch = pytest.importorskip("torch")

from experiments.final.baseline_algorithms import (
    fedcoin_posap_weights,
    foolsgold_decision,
    krum_decision,
    monte_carlo_shapley,
    optimize_sdea_weights,
    update_shapley_history,
    weighted_average_updates,
)


def _updates(values):
    return [{"w": torch.tensor([float(value), 0.0])} for value in values]


def test_weighted_average_and_krum_use_real_update_coordinates():
    updates = _updates((0.0, 0.1, -0.1, 0.05, -0.05, 0.02, 10.0))
    average = weighted_average_updates(updates[:2], (0.25, 0.75))
    assert float(average["w"][0]) == pytest.approx(0.075)
    decision = krum_decision(updates, byzantine_bound=1)
    assert decision.included_indices[0] != 6
    assert 6 in decision.flagged_indices
    assert abs(float(decision.update["w"][0])) <= 0.1


def test_foolsgold_downweights_colluding_histories():
    updates = [
        {"w": torch.tensor([1.0, 0.0, 0.0, 0.0])},
        {"w": torch.tensor([1.0, 0.0, 0.0, 0.0])},
        {"w": torch.tensor([0.0, 1.0, 0.0, 0.0])},
        {"w": torch.tensor([0.0, 0.0, 1.0, 0.0])},
        {"w": torch.tensor([0.0, 0.0, 0.0, 1.0])},
    ]
    decision, history = foolsgold_decision(
        updates,
        cumulative_history=None,
        byzantine_bound=2,
    )
    assert history.shape[0] == len(updates)
    assert decision.weights[0] < decision.weights[2]
    assert decision.weights[1] < decision.weights[3]
    assert {0, 1}.issubset(decision.flagged_indices)


def test_shapley_estimator_and_history_are_seeded_and_normalized():
    clients = ("a", "b", "c")

    def utility(coalition):
        return sum({"a": 1.0, "b": 2.0, "c": 3.0}[client] for client in coalition)

    values = monte_carlo_shapley(clients, utility, permutations=10, seed=7)
    assert values == {"a": 1.0, "b": 2.0, "c": 3.0}
    history = update_shapley_history((1.0, 2.0, 3.0), None, gamma=0.3)
    assert history[0] < history[1] < history[2]
    assert all(0 <= value <= 1 for value in history)


def test_fedcoin_posap_pays_only_nonnegative_measured_contributions():
    weights = fedcoin_posap_weights((-1.0, 0.0, 2.0, 3.0))
    assert weights == (0.0, 0.0, 2.0, 3.0)
    assert fedcoin_posap_weights((-2.0, -1.0)) == (1.0, 1.0)


def test_sdea_optimizes_instance_sharpness_and_batch_diversity_on_two_views():
    client_logits = torch.tensor(
        [
            [[5.0, 0.0], [0.0, 5.0]],
            [[0.2, 0.0], [0.0, 0.2]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float64,
    )

    def evaluate(weights):
        logits = torch.einsum("c,csd->sd", weights, client_logits)
        return logits, logits * 0.95

    weights = optimize_sdea_weights(3, evaluate, iterations=30, learning_rate=0.1)
    assert sum(weights) == pytest.approx(1.0)
    assert all(math.isfinite(value) and value >= 0 for value in weights)
    assert weights[0] > weights[1]
