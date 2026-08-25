"""Fixed-point witness encoding shared by training and proof generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


PADDED_DATA_INDEX = 2**32 - 1


@dataclass(frozen=True)
class ZKCircuitConfig:
    sample_count: int = 14
    steps: int = 5
    batch_terms: int = 32
    value_bits: int = 48
    scale: int = 1_000_000
    pair_tolerance: float = 1e-5
    final_tolerance: float = 1e-3
    max_update_l2: float = 10.0
    auxiliary_pairs_per_chunk: int = 4

    def __post_init__(self) -> None:
        if min(self.sample_count, self.steps, self.batch_terms, self.value_bits, self.scale) <= 0:
            raise ValueError("ZK circuit dimensions and scale must be positive")
        if self.batch_terms % self.auxiliary_pairs_per_chunk:
            raise ValueError("batch terms must divide into auxiliary Poseidon chunks")
        if self.value_bits > 248:
            raise ValueError("ZK signed values must fit safely below the BN254 modulus")
        if min(self.pair_tolerance, self.final_tolerance, self.max_update_l2) < 0:
            raise ValueError("ZK tolerances and update bound must be non-negative")

    @property
    def learning_rate_scale(self) -> int:
        return self.scale

    @property
    def max_rounding_error(self) -> int:
        return int(math.ceil(self.pair_tolerance * self.scale * self.scale))

    @property
    def max_cumulative_rounding_error_squared(self) -> int:
        return int(math.ceil((self.final_tolerance * self.scale * self.scale) ** 2))

    @property
    def max_distance_squared(self) -> int:
        return int(math.ceil((self.max_update_l2 * self.scale) ** 2))


def quantize(value: float, *, scale: int, bits: int) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite values cannot enter a ZK witness")
    encoded = int(round(float(value) * int(scale)))
    if abs(encoded) >= 2**bits:
        raise OverflowError(f"fixed-point value exceeds {bits}-bit magnitude")
    return encoded


def require_signed_range(values: Iterable[int], *, bits: int) -> None:
    limit = 2**bits
    if any(abs(int(value)) >= limit for value in values):
        raise OverflowError(f"ZK witness value exceeds {bits}-bit magnitude")


def signed_components(values: Sequence[int]) -> tuple[list[str], list[str]]:
    magnitudes: list[str] = []
    signs: list[str] = []
    for value in values:
        integer = int(value)
        magnitudes.append(str(abs(integer)))
        signs.append("1" if integer < 0 else "0")
    return magnitudes, signs


def pad_batch_indices(indices: Sequence[int], *, batch_terms: int) -> tuple[int, ...]:
    ordered = tuple(int(index) for index in indices)
    if len(ordered) > batch_terms:
        raise ValueError("private batch exceeds the proof circuit batch size")
    if any(index < 0 or index >= PADDED_DATA_INDEX for index in ordered):
        raise ValueError("private data index must fit in the circuit range")
    return ordered + (PADDED_DATA_INDEX,) * (batch_terms - len(ordered))
