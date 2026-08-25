"""Deterministic paper-study data partitions with exact index provenance."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def dirichlet_partition_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    num_clients: int,
    alpha: float,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_array.size == 0:
        raise ValueError("Dirichlet partition requires labels")
    if num_clients <= 0 or alpha <= 0:
        raise ValueError("Dirichlet client count and alpha must be positive")
    if num_clients > labels_array.size:
        raise ValueError("Dirichlet partition cannot give every client one sample")
    rng = np.random.default_rng(int(seed))
    client_indices: list[list[int]] = [[] for _ in range(int(num_clients))]
    for label in sorted(int(value) for value in np.unique(labels_array)):
        class_indices = np.flatnonzero(labels_array == label)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
        boundaries = np.floor(np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client, split in enumerate(np.split(class_indices, boundaries)):
            client_indices[client].extend(int(index) for index in split)

    # A rare empty client is repaired deterministically by moving one sample
    # from the largest donor. This preserves a total, disjoint partition.
    for empty in [index for index, values in enumerate(client_indices) if not values]:
        donor = max(range(num_clients), key=lambda index: (len(client_indices[index]), -index))
        if len(client_indices[donor]) <= 1:
            raise ValueError("cannot repair empty Dirichlet client")
        client_indices[empty].append(client_indices[donor].pop())
    for values in client_indices:
        values.sort()
    flattened = [index for values in client_indices for index in values]
    if len(flattened) != labels_array.size or len(set(flattened)) != labels_array.size:
        raise AssertionError("Dirichlet partition is not a total disjoint cover")
    return tuple(tuple(values) for values in client_indices)


def dataset_labels(dataset: Any) -> np.ndarray:
    for name in ("targets", "labels"):
        if hasattr(dataset, name):
            return np.asarray(getattr(dataset, name), dtype=np.int64)
    raise ValueError("dataset does not expose targets or labels")


def partition_dataset_dirichlet(
    dataset: Any,
    *,
    num_clients: int,
    alpha: float,
    seed: int,
):
    from torch.utils.data import Subset

    return [
        Subset(dataset, list(indices))
        for indices in dirichlet_partition_indices(
            dataset_labels(dataset),
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )
    ]
