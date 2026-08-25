import numpy as np

from experiments.final.baselines import (
    foolsgold_weights,
    monte_carlo_shapley,
    sdea_entropy_weights,
)


def test_foolsgold_downweights_colluding_updates():
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    weights = foolsgold_weights(matrix)
    assert weights[0] < weights[2]
    assert weights[1] < weights[3]
    np.testing.assert_allclose(np.sum(weights), 1.0)


def test_shapley_estimator_and_sdea_entropy_are_deterministic():
    clients = ("a", "b", "c")

    def utility(coalition):
        return sum({"a": 1.0, "b": 2.0, "c": 3.0}[client] for client in coalition)

    values = monte_carlo_shapley(clients, utility, permutations=20, seed=7)
    assert values == {"a": 1.0, "b": 2.0, "c": 3.0}
    probabilities = np.asarray(
        [
            [[0.99, 0.01], [0.98, 0.02]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )
    weights = sdea_entropy_weights(probabilities)
    assert weights[0] > weights[1]
    np.testing.assert_allclose(np.sum(weights), 1.0)
