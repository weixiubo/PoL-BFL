"""Reusable server-side filters for federated model updates."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Mapping

import numpy as np


class serverDefender:
    """Filter suspicious client updates before aggregation.

    ``alg1`` and ``alg2`` remain as compatibility aliases for the Krum and
    robust-norm filters, respectively.
    """

    def __init__(
        self,
        num_byzantine: int = 0,
        norm_multiplier: float = 2.5,
    ) -> None:
        if num_byzantine < 0:
            raise ValueError("num_byzantine cannot be negative")
        if norm_multiplier <= 0:
            raise ValueError("norm_multiplier must be positive")
        self.num_byzantine = int(num_byzantine)
        self.norm_multiplier = float(norm_multiplier)

    @staticmethod
    def _flatten(update) -> np.ndarray:
        if isinstance(update, Mapping):
            pieces = []
            for key in sorted(update):
                value = update[key]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                pieces.append(np.asarray(value, dtype=np.float64).reshape(-1))
            if not pieces:
                raise ValueError("model update cannot be empty")
            return np.concatenate(pieces)
        vector = np.asarray(update, dtype=np.float64).reshape(-1)
        if vector.size == 0:
            raise ValueError("model update cannot be empty")
        return vector

    @classmethod
    def _matrix(cls, updates) -> np.ndarray:
        if not updates:
            raise ValueError("at least one model update is required")
        vectors = [cls._flatten(update) for update in updates]
        width = vectors[0].size
        if any(vector.size != width for vector in vectors):
            raise ValueError("all model updates must share one parameter shape")
        matrix = np.stack(vectors)
        if not np.isfinite(matrix).all():
            raise ValueError("model updates must be finite")
        return matrix

    def krum_filter(self, raw_client_model_or_grad_list: List[OrderedDict]):
        """Return the update with the minimum standard Krum score."""
        matrix = self._matrix(raw_client_model_or_grad_list)
        count = len(matrix)
        neighbor_count = count - self.num_byzantine - 2
        if neighbor_count <= 0 or count < 2 * self.num_byzantine + 3:
            raise ValueError("Krum requires n >= 2f + 3 clients")
        pairwise = np.sum(
            (matrix[:, None, :] - matrix[None, :, :]) ** 2,
            axis=2,
        )
        scores = []
        for index in range(count):
            distances = np.delete(pairwise[index], index)
            scores.append(float(np.sort(distances)[:neighbor_count].sum()))
        return [raw_client_model_or_grad_list[int(np.argmin(scores))]]

    def robust_norm_filter(
        self,
        raw_client_model_or_grad_list: List[OrderedDict],
    ):
        """Keep updates within a robust median/MAD norm envelope."""
        matrix = self._matrix(raw_client_model_or_grad_list)
        center = np.median(matrix, axis=0)
        distances = np.linalg.norm(matrix - center, axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        scale = max(1.4826 * mad, np.finfo(np.float64).eps)
        threshold = median + self.norm_multiplier * scale
        selected = [
            update
            for update, distance in zip(raw_client_model_or_grad_list, distances)
            if float(distance) <= threshold
        ]
        if not selected:
            selected = [raw_client_model_or_grad_list[int(np.argmin(distances))]]
        return selected

    def alg1(self, raw_client_model_or_grad_list: List[OrderedDict]):
        return self.krum_filter(raw_client_model_or_grad_list)

    def alg2(self, raw_client_model_or_grad_list: List[OrderedDict]):
        return self.robust_norm_filter(raw_client_model_or_grad_list)
