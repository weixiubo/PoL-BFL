"""Paper attack constructions over floating model-update coordinates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class _LayoutEntry:
    key: str
    shape: torch.Size
    count: int
    dtype: torch.dtype
    device: torch.device


class UpdateCodec:
    def __init__(self, template: Mapping[str, Any]):
        entries = []
        for key in sorted(template):
            value = template[key]
            if torch.is_tensor(value) and value.is_floating_point():
                entries.append(_LayoutEntry(key, value.shape, value.numel(), value.dtype, value.device))
        if not entries:
            raise ValueError("attack update has no floating coordinates")
        self.entries = tuple(entries)
        self.template = template

    def encode(self, update: Mapping[str, Any]) -> torch.Tensor:
        chunks = []
        for entry in self.entries:
            value = update[entry.key]
            if not torch.is_tensor(value) or value.shape != entry.shape or not value.is_floating_point():
                raise ValueError(f"attack update layout mismatch: {entry.key}")
            chunks.append(value.detach().to(dtype=torch.float32, device="cpu").reshape(-1))
        vector = torch.cat(chunks)
        if not torch.isfinite(vector).all():
            raise ValueError("attack update contains non-finite coordinates")
        return vector

    def decode(self, vector: torch.Tensor) -> OrderedDict:
        result = OrderedDict()
        cursor = 0
        entries = {entry.key: entry for entry in self.entries}
        for key, original in self.template.items():
            entry = entries.get(key)
            if entry is None:
                result[key] = original.detach().clone() if torch.is_tensor(original) else original
                continue
            values = vector[cursor : cursor + entry.count].reshape(entry.shape)
            cursor += entry.count
            result[key] = values.to(dtype=entry.dtype, device=entry.device)
        if cursor != vector.numel():
            raise AssertionError("attack vector was not decoded exactly")
        return result


def alie_update(benign_updates: Sequence[Mapping[str, Any]], *, z_max: float = 2.5) -> OrderedDict:
    if len(benign_updates) < 2:
        raise ValueError("ALIE requires at least two benign reference updates")
    codec = UpdateCodec(benign_updates[0])
    matrix = torch.stack([codec.encode(update) for update in benign_updates])
    malicious = matrix.mean(dim=0) + float(z_max) * matrix.std(dim=0, unbiased=False)
    return codec.decode(malicious)


def minmax_update(
    benign_updates: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 24,
) -> OrderedDict:
    """Maximize displacement while staying within the benign diameter."""

    if len(benign_updates) < 3 or iterations <= 0:
        raise ValueError("MinMax requires at least three references and positive iterations")
    codec = UpdateCodec(benign_updates[0])
    matrix = torch.stack([codec.encode(update) for update in benign_updates])
    mean = matrix.mean(dim=0)
    direction = -torch.sign(mean)
    if float(torch.linalg.vector_norm(direction)) == 0:
        direction = torch.ones_like(direction)
    direction /= torch.linalg.vector_norm(direction)
    benign_diameter = torch.cdist(matrix, matrix).max()

    def admissible(scale: float) -> bool:
        candidate = mean + scale * direction
        return bool(torch.linalg.vector_norm(matrix - candidate, dim=1).max() <= benign_diameter)

    low = 0.0
    high = max(1.0, float(benign_diameter))
    while admissible(high) and high < 2**20:
        low = high
        high *= 2
    for _ in range(iterations):
        middle = (low + high) / 2
        if admissible(middle):
            low = middle
        else:
            high = middle
    return codec.decode(mean + low * direction)


def random_noise_update(
    template: Mapping[str, Any],
    *,
    generator: torch.Generator,
    scale: float = 1.0,
) -> OrderedDict:
    result = OrderedDict()
    for key, value in template.items():
        if torch.is_tensor(value) and value.is_floating_point():
            reference = value.detach().float().cpu()
            magnitude = max(float(reference.std(unbiased=False)), float(reference.abs().mean()), 1e-4)
            noise = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
            result[key] = (noise * (float(scale) * magnitude)).to(dtype=value.dtype, device=value.device)
        else:
            result[key] = torch.zeros_like(value) if torch.is_tensor(value) else value
    return result


def model_replacement_update(
    malicious_model: Mapping[str, Any],
    global_model: Mapping[str, Any],
    *,
    amplification: float,
) -> OrderedDict:
    if amplification <= 0:
        raise ValueError("model replacement amplification must be positive")
    result = OrderedDict()
    for key in global_model:
        global_value = global_model[key]
        malicious_value = malicious_model[key]
        if torch.is_tensor(global_value) and global_value.is_floating_point():
            result[key] = (malicious_value - global_value) * float(amplification)
        else:
            result[key] = torch.zeros_like(global_value) if torch.is_tensor(global_value) else global_value
    return result
