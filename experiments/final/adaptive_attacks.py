"""Executable adaptive trace/update attacks with measured work, not paper constants."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class TimedAttack(Generic[T]):
    output: T
    elapsed_seconds: float


def measure_attack(operation: Callable[[], T]) -> TimedAttack[T]:
    started = time.perf_counter()
    output = operation()
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("adaptive attack timer did not advance")
    return TimedAttack(output, elapsed)


def checkpoint_interpolation(
    checkpoints: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if len(checkpoints) < 2:
        raise ValueError("checkpoint interpolation requires two endpoints")
    first = checkpoints[0].detach()
    final = checkpoints[-1].detach()
    if first.shape != final.shape or not first.is_floating_point() or not final.is_floating_point():
        raise ValueError("checkpoint interpolation endpoints are incompatible")
    count = len(checkpoints)
    return tuple(
        first + (final - first) * (index / (count - 1))
        for index in range(count)
    )


def gradient_mimicry(
    benign_gradients: Sequence[torch.Tensor],
    *,
    generator: torch.Generator,
    standard_deviations: float = 0.25,
) -> torch.Tensor:
    if len(benign_gradients) < 2 or standard_deviations < 0:
        raise ValueError("gradient mimicry requires references and non-negative scale")
    shape = benign_gradients[0].shape
    if any(gradient.shape != shape for gradient in benign_gradients):
        raise ValueError("gradient mimicry reference shapes differ")
    matrix = torch.stack(
        [gradient.detach().to(dtype=torch.float32, device="cpu") for gradient in benign_gradients]
    )
    if not torch.isfinite(matrix).all():
        raise ValueError("gradient mimicry references must be finite")
    mean = matrix.mean(dim=0)
    standard = matrix.std(dim=0, unbiased=False)
    noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
    return mean + float(standard_deviations) * standard * noise.clamp(-2.0, 2.0)


def partial_replay(
    updates: Sequence[torch.Tensor],
    *,
    honest_fraction: float = 0.3,
) -> tuple[torch.Tensor, ...]:
    if not updates or not 0 < honest_fraction <= 1:
        raise ValueError("partial replay requires updates and a valid honest fraction")
    honest_count = max(1, min(len(updates), math.ceil(len(updates) * honest_fraction)))
    prefix = tuple(update.detach().clone() for update in updates[:honest_count])
    return tuple(
        prefix[index] if index < honest_count else prefix[(index - honest_count) % honest_count]
        for index in range(len(updates))
    )


def combined_adaptive_trajectory(
    checkpoints: Sequence[torch.Tensor],
    benign_gradients: Sequence[torch.Tensor],
    *,
    generator: torch.Generator,
    honest_fraction: float = 0.3,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    interpolated = checkpoint_interpolation(checkpoints)
    replayed = partial_replay(interpolated, honest_fraction=honest_fraction)
    mimicked_gradient = gradient_mimicry(
        benign_gradients,
        generator=generator,
    )
    return replayed, mimicked_gradient


__all__ = [
    "TimedAttack",
    "measure_attack",
    "checkpoint_interpolation",
    "gradient_mimicry",
    "partial_replay",
    "combined_adaptive_trajectory",
]
