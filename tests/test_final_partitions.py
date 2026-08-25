import numpy as np
import pytest

from experiments.final.partitions import dirichlet_partition_indices


def test_dirichlet_partition_is_deterministic_total_disjoint_and_nonempty():
    labels = np.repeat(np.arange(10), 100)
    first = dirichlet_partition_indices(
        labels,
        num_clients=50,
        alpha=0.1,
        seed=1337,
    )
    second = dirichlet_partition_indices(
        labels,
        num_clients=50,
        alpha=0.1,
        seed=1337,
    )
    assert first == second
    assert all(first)
    flattened = [index for client in first for index in client]
    assert sorted(flattened) == list(range(len(labels)))
    assert len(flattened) == len(set(flattened))
    changed = dirichlet_partition_indices(
        labels,
        num_clients=50,
        alpha=0.1,
        seed=2026,
    )
    assert changed != first


def test_dirichlet_partition_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        dirichlet_partition_indices([], num_clients=2, alpha=0.5, seed=1)
    with pytest.raises(ValueError):
        dirichlet_partition_indices([0, 1], num_clients=3, alpha=0.5, seed=1)
    with pytest.raises(ValueError):
        dirichlet_partition_indices([0, 1], num_clients=2, alpha=0.0, seed=1)
