"""Deterministic baseline primitives retained for analysis compatibility."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Hashable

import numpy as np


def foolsgold_weights(
    update_vectors: np.ndarray,
    *,
    cumulative_history: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.asarray(update_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("FoolsGold requires a finite client-by-coordinate matrix")
    history = matrix if cumulative_history is None else np.asarray(cumulative_history, dtype=np.float64) + matrix
    if history.shape != matrix.shape or not np.all(np.isfinite(history)):
        raise ValueError("FoolsGold cumulative history shape is invalid")
    norms = np.linalg.norm(history, axis=1, keepdims=True)
    normalized = history / np.maximum(norms, 1e-12)
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(cosine, 0.0)
    maximum = np.max(cosine, axis=1)
    pardoned = cosine.copy()
    for left in range(len(maximum)):
        for right in range(len(maximum)):
            if left != right and maximum[left] < maximum[right]:
                pardoned[left, right] *= maximum[left] / max(maximum[right], 1e-12)
    weights = np.clip(1.0 - np.max(pardoned, axis=1), 0.0, 1.0)
    maximum_weight = float(np.max(weights))
    if maximum_weight > 0:
        weights /= maximum_weight
    weights = np.minimum(weights, 0.99)
    safe = np.clip(weights, 1e-6, 1 - 1e-6)
    weights = np.clip(np.log(safe / (1 - safe)) + 0.5, 0.0, 1.0)
    if float(np.sum(weights)) <= 0:
        return np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    return weights / np.sum(weights)


def monte_carlo_shapley(
    clients: Sequence[Hashable],
    utility: Callable[[tuple[Hashable, ...]], float],
    *,
    permutations: int,
    seed: int,
) -> dict[Hashable, float]:
    if not clients or len(set(clients)) != len(clients) or permutations <= 0:
        raise ValueError("Shapley estimation requires unique clients and permutations")
    rng = np.random.default_rng(int(seed))
    values = {client: 0.0 for client in clients}
    cache: dict[tuple[Hashable, ...], float] = {(): float(utility(()))}

    def coalition_utility(coalition: tuple[Hashable, ...]) -> float:
        key = tuple(sorted(coalition, key=str))
        if key not in cache:
            value = float(utility(key))
            if not math.isfinite(value):
                raise ValueError("Shapley utility must be finite")
            cache[key] = value
        return cache[key]

    ordered = tuple(clients)
    for _ in range(int(permutations)):
        permutation = [ordered[index] for index in rng.permutation(len(ordered))]
        coalition: tuple[Hashable, ...] = ()
        previous = coalition_utility(coalition)
        for client in permutation:
            coalition = (*coalition, client)
            current = coalition_utility(coalition)
            values[client] += current - previous
            previous = current
    return {client: value / permutations for client, value in values.items()}


def sdea_entropy_weights(public_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(public_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] < 2:
        raise ValueError("SDEA probabilities must have client, sample, and class axes")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("SDEA probabilities must be finite and non-negative")
    normalizer = np.sum(probabilities, axis=2, keepdims=True)
    if np.any(normalizer <= 0):
        raise ValueError("SDEA probability row has zero mass")
    probabilities = probabilities / normalizer
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=2,
    ).mean(axis=1)
    centered = entropy - np.median(entropy)
    scale = max(float(np.median(np.abs(centered))) * 1.4826, 1e-12)
    logits = -centered / scale
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)
