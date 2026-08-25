"""Deterministic implementations of the final paper's public baselines."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np
import torch


BASELINE_METHODS = ("VanillaFL", "Krum", "SDEA", "ShapleyFL", "FoolsGold")


@dataclass(frozen=True)
class BaselineDecision:
    method: str
    update: OrderedDict[str, torch.Tensor]
    included_indices: tuple[int, ...]
    flagged_indices: frozenset[int]
    scores: tuple[float, ...]
    weights: tuple[float, ...]


def update_matrix(
    updates: Sequence[Mapping[str, torch.Tensor]],
    *,
    max_coordinates: int = 65_536,
) -> torch.Tensor:
    if len(updates) < 2 or max_coordinates <= 0:
        raise ValueError("baseline vectorization requires multiple updates and coordinates")
    keys = tuple(
        key
        for key in sorted(updates[0])
        if torch.is_tensor(updates[0][key]) and updates[0][key].is_floating_point()
    )
    if not keys:
        raise ValueError("baseline updates contain no floating coordinates")
    vectors = []
    for update in updates:
        if any(key not in update or update[key].shape != updates[0][key].shape for key in keys):
            raise ValueError("baseline update structures differ")
        vector = torch.cat(
            [update[key].detach().to(dtype=torch.float32, device="cpu").reshape(-1) for key in keys]
        )
        if not torch.isfinite(vector).all():
            raise ValueError("baseline update contains non-finite coordinates")
        if vector.numel() > max_coordinates:
            indices = torch.linspace(
                0,
                vector.numel() - 1,
                steps=max_coordinates,
                dtype=torch.float64,
            ).round().to(torch.int64)
            vector = vector.index_select(0, indices)
        vectors.append(vector)
    if len({int(vector.numel()) for vector in vectors}) != 1:
        raise ValueError("baseline update vector lengths differ")
    return torch.stack(vectors)


def weighted_average_updates(
    updates: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> OrderedDict[str, torch.Tensor]:
    if not updates or len(updates) != len(weights):
        raise ValueError("weighted baseline aggregation requires aligned updates and weights")
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64)
    if not torch.isfinite(weight_tensor).all() or torch.any(weight_tensor < 0):
        raise ValueError("baseline weights must be finite and non-negative")
    total = float(weight_tensor.sum())
    if total <= 0:
        raise ValueError("baseline weights have zero mass")
    normalized = [float(value / total) for value in weight_tensor]
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, reference in updates[0].items():
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float32, device="cpu")
            for weight, update in zip(normalized, updates):
                value = update[key].detach().to(dtype=torch.float32, device="cpu")
                if value.shape != reference.shape:
                    raise ValueError("baseline tensor shapes differ")
                accumulator.add_(value, alpha=weight)
            result[key] = accumulator.to(dtype=reference.dtype)
        else:
            result[key] = torch.zeros_like(reference, device="cpu")
    return result


def robust_high_outliers(values: Sequence[float], *, maximum: int) -> frozenset[int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)) or maximum < 0:
        raise ValueError("robust outlier input is invalid")
    median = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - median)))
    if scale <= 1e-12:
        return frozenset()
    candidates = np.flatnonzero(array > median + 3.5 * scale)
    ordered = sorted((int(index) for index in candidates), key=lambda index: (-array[index], index))
    return frozenset(ordered[:maximum])


def low_weight_cluster(values: Sequence[float], *, maximum: int) -> frozenset[int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 3 or not np.all(np.isfinite(array)) or maximum < 0:
        raise ValueError("low-weight cluster input is invalid")
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    gaps = np.diff(sorted_values)
    split = int(np.argmax(gaps)) + 1
    median_gap = float(np.median(gaps))
    mad_gap = float(np.median(np.abs(gaps - median_gap)))
    threshold = median_gap + 3.5 * 1.4826 * mad_gap
    if float(gaps[split - 1]) <= max(threshold, 1e-12) or split > maximum:
        return frozenset()
    return frozenset(int(index) for index in order[:split])


def krum_decision(
    updates: Sequence[Mapping[str, torch.Tensor]],
    *,
    byzantine_bound: int,
) -> BaselineDecision:
    matrix = update_matrix(updates)
    count = matrix.shape[0]
    if byzantine_bound < 0 or count <= 2 * byzantine_bound + 2:
        raise ValueError("Krum requires n > 2f + 2")
    distances = torch.cdist(matrix, matrix, p=2).square()
    neighbor_count = count - byzantine_bound - 2
    scores = []
    for index in range(count):
        row = torch.cat((distances[index, :index], distances[index, index + 1 :]))
        scores.append(float(torch.topk(row, k=neighbor_count, largest=False).values.sum()))
    winner = min(range(count), key=lambda index: (scores[index], index))
    flagged = robust_high_outliers(scores, maximum=byzantine_bound)
    weights = [1.0 if index == winner else 0.0 for index in range(count)]
    return BaselineDecision(
        method="Krum",
        update=weighted_average_updates(updates, weights),
        included_indices=(winner,),
        flagged_indices=flagged,
        scores=tuple(scores),
        weights=tuple(weights),
    )


def foolsgold_weights(
    vectors: torch.Tensor,
    *,
    cumulative_history: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = vectors.to(dtype=torch.float64, device="cpu")
    history = matrix if cumulative_history is None else cumulative_history + matrix
    if history.shape != matrix.shape or not torch.isfinite(history).all():
        raise ValueError("FoolsGold history is incompatible")
    normalized = history / history.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = (normalized @ normalized.T).clamp(-1.0, 1.0)
    cosine.fill_diagonal_(0.0)
    maximum = cosine.max(dim=1).values
    pardoned = cosine.clone()
    for left in range(len(maximum)):
        for right in range(len(maximum)):
            if left != right and maximum[left] < maximum[right]:
                pardoned[left, right] *= maximum[left] / maximum[right].clamp_min(1e-12)
    weights = (1.0 - pardoned.max(dim=1).values).clamp(0.0, 1.0)
    if float(weights.max()) > 0:
        weights /= weights.max()
    weights[weights == 1] = 0.99
    safe = weights.clamp(1e-12, 1 - 1e-12)
    weights = (torch.log(safe / (1 - safe)) + 0.5).clamp(0.0, 1.0)
    if float(weights.sum()) <= 0:
        weights.fill_(1.0)
    return weights, history


def foolsgold_decision(
    updates: Sequence[Mapping[str, torch.Tensor]],
    *,
    cumulative_history: torch.Tensor | None,
    byzantine_bound: int,
) -> tuple[BaselineDecision, torch.Tensor]:
    weights, history = foolsgold_weights(
        update_matrix(updates),
        cumulative_history=cumulative_history,
    )
    flagged = low_weight_cluster(weights.tolist(), maximum=byzantine_bound)
    included = tuple(index for index in range(len(updates)) if index not in flagged)
    decision = BaselineDecision(
        method="FoolsGold",
        update=weighted_average_updates(updates, weights.tolist()),
        included_indices=included,
        flagged_indices=flagged,
        scores=tuple(float(1 - value) for value in weights),
        weights=tuple(float(value) for value in weights),
    )
    return decision, history


def update_shapley_history(
    contributions: Sequence[float],
    previous: Sequence[float] | None,
    *,
    gamma: float = 0.3,
) -> tuple[float, ...]:
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or not 0 < gamma <= 1:
        raise ValueError("Shapley contribution history input is invalid")
    span = float(values.max() - values.min())
    normalized = np.full(len(values), 0.5) if span <= 1e-12 else (values - values.min()) / span
    prior = np.full(len(values), 0.5) if previous is None else np.asarray(previous, dtype=np.float64)
    if prior.shape != values.shape or not np.all(np.isfinite(prior)):
        raise ValueError("Shapley history shape is invalid")
    updated = (1.0 - gamma) * prior + gamma * normalized
    return tuple(float(value) for value in updated)


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
    cache: dict[tuple[Hashable, ...], float] = {}

    def evaluate(coalition: tuple[Hashable, ...]) -> float:
        key = tuple(sorted(coalition, key=str))
        if key not in cache:
            result = float(utility(key))
            if not math.isfinite(result):
                raise ValueError("Shapley utility must be finite")
            cache[key] = result
        return cache[key]

    ordered = tuple(clients)
    for _ in range(int(permutations)):
        permutation = [ordered[index] for index in rng.permutation(len(ordered))]
        coalition: tuple[Hashable, ...] = ()
        previous = evaluate(coalition)
        for client in permutation:
            coalition = (*coalition, client)
            current = evaluate(coalition)
            values[client] += current - previous
            previous = current
    return {client: value / permutations for client, value in values.items()}


def fedcoin_posap_weights(
    contributions: Sequence[float],
) -> tuple[float, ...]:
    """Map measured PoSap marginal accuracies to trainer payment weights."""

    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("FedCoin PoSap contributions are invalid")
    positive = np.maximum(values, 0.0)
    if float(positive.sum()) <= 0:
        positive.fill(1.0)
    return tuple(float(value) for value in positive)


def optimize_sdea_weights(
    client_count: int,
    evaluate_weighted_model: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    iterations: int = 20,
    learning_rate: float = 0.1,
) -> tuple[float, ...]:
    if client_count < 2 or iterations <= 0 or learning_rate <= 0:
        raise ValueError("SDEA optimizer configuration is invalid")
    logits = torch.zeros(client_count, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam((logits,), lr=learning_rate)
    for _ in range(iterations):
        weights = torch.softmax(logits, dim=0)
        first_logits, second_logits = evaluate_weighted_model(weights)
        if (
            first_logits.ndim != 2
            or first_logits.shape != second_logits.shape
            or not torch.isfinite(first_logits).all()
            or not torch.isfinite(second_logits).all()
        ):
            raise ValueError("SDEA weighted model returned invalid logits")
        losses = []
        for model_logits in (first_logits, second_logits):
            probabilities = torch.softmax(model_logits, dim=-1).clamp_min(1e-12)
            instance_entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
            batch_distribution = probabilities.mean(dim=0)
            batch_entropy = -(batch_distribution * batch_distribution.log()).sum()
            losses.append(instance_entropy - batch_entropy)
        loss = 0.5 * (losses[0] + losses[1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    output = torch.softmax(logits.detach(), dim=0)
    return tuple(float(value) for value in output)
