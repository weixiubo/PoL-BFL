"""Fail-closed Trimmed Mean, Krum, and coordinate Median implementations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class AggregationMethod(str, Enum):
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MEDIAN = "median"


@dataclass(frozen=True)
class VerifiedUpdate:
    client_id: str
    update: Mapping[str, Any]
    reputation: float
    proof_eligible: bool = True
    sybil_flagged: bool = False

    def __post_init__(self) -> None:
        if not self.client_id or not self.update:
            raise ValueError("client ID and model update are required")
        if not 0 <= float(self.reputation) <= 1:
            raise ValueError("reputation must be normalized to [0, 1]")


@dataclass(frozen=True)
class _Entry:
    name: str
    shape: tuple[int, ...]
    size: int
    template: Any


class _VectorCodec:
    def __init__(self, update: Mapping[str, Any]):
        entries: list[_Entry] = []
        for name in sorted(update):
            value = update[name]
            array = self._array(value)
            if not np.issubdtype(array.dtype, np.floating):
                continue
            entries.append(_Entry(str(name), tuple(array.shape), int(array.size), value))
        if not entries:
            raise ValueError("model update cannot be empty")
        self.entries = tuple(entries)

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        obj = value
        detach = getattr(obj, "detach", None)
        if callable(detach):
            obj = detach()
        cpu = getattr(obj, "cpu", None)
        if callable(cpu):
            obj = cpu()
        numpy = getattr(obj, "numpy", None)
        if callable(numpy):
            obj = numpy()
        array = np.asarray(obj)
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError("model updates must contain only finite numeric arrays")
        return array

    def encode(self, update: Mapping[str, Any]) -> np.ndarray:
        floating_keys = tuple(
            str(name)
            for name in sorted(update)
            if np.issubdtype(self._array(update[name]).dtype, np.floating)
        )
        if floating_keys != tuple(entry.name for entry in self.entries):
            raise ValueError("all model updates must have identical keys")
        chunks: list[np.ndarray] = []
        for entry in self.entries:
            array = self._array(update[entry.name])
            if tuple(array.shape) != entry.shape:
                raise ValueError(f"model update shape mismatch for {entry.name}")
            chunks.append(array.astype(np.float32, copy=False).reshape(-1))
        return np.concatenate(chunks)

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {}
        cursor = 0
        for entry in self.entries:
            chunk = vector[cursor : cursor + entry.size].reshape(entry.shape)
            cursor += entry.size
            template = entry.template
            if hasattr(template, "detach"):
                import torch

                if not getattr(template.dtype, "is_floating_point", False):
                    chunk = (
                        int(round(float(chunk)))
                        if entry.shape == ()
                        else np.asarray(np.rint(chunk), dtype=np.int64)
                    )
                result[entry.name] = torch.as_tensor(
                    chunk,
                    dtype=template.dtype,
                    device=template.device,
                )
            else:
                array = np.asarray(template)
                if not np.issubdtype(array.dtype, np.floating):
                    chunk = np.asarray(np.rint(chunk), dtype=array.dtype)
                result[entry.name] = chunk.astype(array.dtype, copy=False)
        if cursor != vector.size:
            raise AssertionError("aggregation vector was not consumed exactly")
        return result

    def encode_torch(self, update: Mapping[str, Any], *, device: str):
        import torch

        floating_keys = tuple(
            str(name)
            for name in sorted(update)
            if np.issubdtype(self._array(update[name]).dtype, np.floating)
        )
        if floating_keys != tuple(entry.name for entry in self.entries):
            raise ValueError("all model updates must have identical keys")
        chunks = []
        for entry in self.entries:
            value = update[entry.name]
            shape = tuple(int(dimension) for dimension in getattr(value, "shape", np.asarray(value).shape))
            if shape != entry.shape:
                raise ValueError(f"model update shape mismatch for {entry.name}")
            tensor = torch.as_tensor(value).detach().to(device=device, dtype=torch.float32)
            chunks.append(tensor.reshape(-1))
        vector = torch.cat(chunks)
        if not bool(torch.isfinite(vector).all().item()):
            raise ValueError("model updates must contain only finite numeric arrays")
        return vector

    def decode_torch(self, vector) -> dict[str, Any]:
        import torch

        result: dict[str, Any] = {}
        cursor = 0
        for entry in self.entries:
            chunk = vector[cursor : cursor + entry.size].reshape(entry.shape)
            cursor += entry.size
            template = entry.template
            if hasattr(template, "detach"):
                result[entry.name] = chunk.to(
                    dtype=template.dtype,
                    device=template.device,
                ).clone()
            else:
                result[entry.name] = chunk.detach().cpu().numpy().astype(
                    np.asarray(template).dtype,
                    copy=False,
                )
        if cursor != int(vector.numel()):
            raise AssertionError("aggregation vector was not consumed exactly")
        return result


@dataclass(frozen=True)
class AggregationResult:
    method: AggregationMethod
    update: Mapping[str, Any]
    included_clients: tuple[str, ...]
    excluded_clients: Mapping[str, str]
    krum_winner: str | None = None


def _trimmed_mean(matrix: np.ndarray, byzantine_bound: int) -> np.ndarray:
    count = matrix.shape[0]
    if byzantine_bound < 0 or 2 * byzantine_bound >= count:
        raise ValueError("Trimmed Mean requires 0 <= f < n/2")
    ordered = np.sort(matrix, axis=0)
    kept = ordered[byzantine_bound : count - byzantine_bound or None]
    return np.mean(kept, axis=0)


def _krum(matrix: np.ndarray, byzantine_bound: int) -> int:
    count = matrix.shape[0]
    if byzantine_bound < 0 or count < 2 * byzantine_bound + 3:
        raise ValueError("Krum requires n >= 2f + 3")
    norms = np.einsum("ij,ij->i", matrix, matrix)
    distances = norms[:, None] + norms[None, :] - 2.0 * (matrix @ matrix.T)
    np.maximum(distances, 0.0, out=distances)
    neighbor_count = count - byzantine_bound - 2
    scores = []
    for index in range(count):
        neighbors = np.delete(distances[index], index)
        scores.append(float(np.sum(np.partition(neighbors, neighbor_count - 1)[:neighbor_count])))
    return min(range(count), key=lambda index: (scores[index], index))


def _aggregate_torch(
    codec: _VectorCodec,
    eligible: Sequence[VerifiedUpdate],
    *,
    method: AggregationMethod,
    byzantine_bound: int,
    device: str,
):
    import torch

    with torch.no_grad():
        weighted = torch.stack(
            [
                codec.encode_torch(item.update, device=device) * float(item.reputation)
                for item in eligible
            ],
            dim=0,
        )
        count = int(weighted.shape[0])
        winner_index = None
        if method == AggregationMethod.TRIMMED_MEAN:
            if byzantine_bound < 0 or 2 * byzantine_bound >= count:
                raise ValueError("Trimmed Mean requires 0 <= f < n/2")
            ordered = torch.sort(weighted, dim=0).values
            aggregated = ordered[byzantine_bound : count - byzantine_bound].mean(dim=0)
        elif method == AggregationMethod.MEDIAN:
            ordered = torch.sort(weighted, dim=0).values
            midpoint = count // 2
            aggregated = (
                ordered[midpoint]
                if count % 2
                else (ordered[midpoint - 1] + ordered[midpoint]) * 0.5
            )
        else:
            if byzantine_bound < 0 or count < 2 * byzantine_bound + 3:
                raise ValueError("Krum requires n >= 2f + 3")
            norms = torch.einsum("ij,ij->i", weighted, weighted)
            distances = norms[:, None] + norms[None, :] - 2.0 * (weighted @ weighted.T)
            distances.clamp_min_(0.0)
            distances.fill_diagonal_(float("inf"))
            neighbor_count = count - byzantine_bound - 2
            nearest = torch.topk(
                distances,
                k=neighbor_count,
                dim=1,
                largest=False,
                sorted=False,
            ).values
            winner_index = int(torch.argmin(nearest.sum(dim=1)).item())
            aggregated = weighted[winner_index]
        return codec.decode_torch(aggregated), winner_index


def aggregate_verified_updates(
    updates: Sequence[VerifiedUpdate],
    *,
    method: AggregationMethod | str = AggregationMethod.TRIMMED_MEAN,
    byzantine_bound: int = 0,
    device: str | None = None,
) -> AggregationResult:
    if not updates:
        raise ValueError("at least one submitted update is required")
    method = AggregationMethod(method)
    seen: set[str] = set()
    eligible: list[VerifiedUpdate] = []
    excluded: dict[str, str] = {}
    for update in updates:
        if update.client_id in seen:
            raise ValueError("duplicate client update")
        seen.add(update.client_id)
        if not update.proof_eligible:
            excluded[update.client_id] = "proof_ineligible"
        elif update.sybil_flagged:
            excluded[update.client_id] = "sybil_screened"
        else:
            eligible.append(update)
    if not eligible:
        raise ValueError("no verified non-Sybil updates remain")

    codec = _VectorCodec(eligible[0].update)
    winner = None
    if device is not None:
        decoded, winner_index = _aggregate_torch(
            codec,
            eligible,
            method=method,
            byzantine_bound=int(byzantine_bound),
            device=str(device),
        )
        if winner_index is not None:
            winner = eligible[winner_index].client_id
    else:
        weighted = np.stack(
            [codec.encode(item.update) * float(item.reputation) for item in eligible],
            axis=0,
        )
        if not np.all(np.isfinite(weighted)):
            raise ValueError("reputation weighting produced a non-finite update")
        if method == AggregationMethod.TRIMMED_MEAN:
            aggregated = _trimmed_mean(weighted, int(byzantine_bound))
        elif method == AggregationMethod.MEDIAN:
            aggregated = np.median(weighted, axis=0)
        else:
            winner_index = _krum(weighted, int(byzantine_bound))
            aggregated = weighted[winner_index]
            winner = eligible[winner_index].client_id
        decoded = codec.decode(aggregated)

    if winner is not None:
        for index, item in enumerate(eligible):
            if item.client_id != winner:
                excluded[item.client_id] = "krum_not_selected"

    included = (
        (winner,)
        if winner is not None
        else tuple(item.client_id for item in eligible)
    )
    return AggregationResult(
        method=method,
        update=decoded,
        included_clients=included,
        excluded_clients=excluded,
        krum_winner=winner,
    )
