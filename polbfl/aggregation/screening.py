"""Client-level robust screening used before coordinate aggregation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class UpdateScreeningReport:
    flagged_clients: frozenset[str]
    distances: Mapping[str, float]
    threshold: float


def _floating_vector(update: Mapping[str, Any]) -> np.ndarray:
    chunks = []
    for name in sorted(update):
        value = update[name]
        obj = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if np.issubdtype(obj.dtype, np.floating):
            chunks.append(np.asarray(obj, dtype=np.float32).reshape(-1))
    if not chunks:
        raise ValueError("screening update has no floating coordinates")
    vector = np.concatenate(chunks)
    if not np.all(np.isfinite(vector)):
        raise ValueError("screening update contains non-finite coordinates")
    return vector


def screen_update_outliers(
    client_updates: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    randomness: bytes,
    coordinate_sample: int = 8192,
    mad_multiplier: float = 3.5,
) -> UpdateScreeningReport:
    if len(client_updates) < 3 or len(randomness) < 32:
        raise ValueError("robust screening requires three clients and 256-bit randomness")
    if len({client for client, _ in client_updates}) != len(client_updates):
        raise ValueError("duplicate screening client")
    vectors = [_floating_vector(update) for _, update in client_updates]
    width = vectors[0].size
    if any(vector.size != width for vector in vectors):
        raise ValueError("screening update layouts differ")
    count = min(int(coordinate_sample), width)
    seed = int.from_bytes(
        hashlib.sha256(b"POLBFL_UPDATE_SCREEN_V1" + randomness).digest(), "big"
    )
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(width, size=count, replace=False))
    matrix = np.stack([vector[indices] for vector in vectors])
    center = np.median(matrix, axis=0)
    distances_array = np.linalg.norm(matrix - center, axis=1) / np.sqrt(count)
    median = float(np.median(distances_array))
    mad = float(np.median(np.abs(distances_array - median)))
    robust_scale = 1.4826 * mad
    threshold = median + float(mad_multiplier) * max(robust_scale, 1e-12)
    distances = {
        client: float(distance)
        for (client, _), distance in zip(client_updates, distances_array)
    }
    flagged = frozenset(client for client, distance in distances.items() if distance > threshold)
    return UpdateScreeningReport(flagged, distances, threshold)
