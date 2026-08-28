"""Checkpoint-trajectory and exact batch-index Sybil detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from polbfl.training import RecordedTrace


@dataclass(frozen=True)
class TraceFingerprint:
    client_id: str
    commitment_root: str
    checkpoint_vectors: tuple[tuple[float, ...], ...]
    batch_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.client_id or len(self.commitment_root) != 64:
            raise ValueError("trace identity and commitment root are required")
        if len(self.checkpoint_vectors) < 2:
            raise ValueError("Sybil screening requires at least two checkpoint vectors")
        width = len(self.checkpoint_vectors[0])
        if width <= 0 or any(len(vector) != width for vector in self.checkpoint_vectors):
            raise ValueError("checkpoint fingerprint vectors must have one fixed width")
        matrix = np.asarray(self.checkpoint_vectors, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("checkpoint fingerprints must be finite")
        if any(index < 0 for index in self.batch_indices):
            raise ValueError("batch indices must be non-negative")

    @classmethod
    def from_recorded(cls, recorded: "RecordedTrace") -> "TraceFingerprint":
        checkpoints: list[tuple[float, ...]] = []
        indices: list[int] = []
        for position in sorted(recorded.checkpoints):
            material = recorded.checkpoints[position]
            if not material.zk_private or "sampled_weights" not in material.zk_private:
                raise ValueError("recorded trace lacks committed sampled checkpoint vectors")
            checkpoints.append(
                tuple(float(value) for value in material.zk_private["sampled_weights"])
            )
        for step in sorted(recorded.steps):
            indices.extend(int(index) for index in recorded.steps[step].batch_indices)
        return cls(
            client_id=recorded.trace.commitment.client_id,
            commitment_root=recorded.trace.commitment.merkle_root,
            checkpoint_vectors=tuple(checkpoints),
            batch_indices=tuple(indices),
        )

    @property
    def trajectory(self) -> np.ndarray:
        matrix = np.asarray(self.checkpoint_vectors, dtype=np.float64)
        return np.diff(matrix, axis=0).reshape(-1)


@dataclass(frozen=True)
class PairwiseSybilEvidence:
    left_client: str
    right_client: str
    trajectory_cosine: float
    identical_batch_indices: bool
    flagged: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SybilScreeningReport:
    flagged_clients: frozenset[str]
    pairs: tuple[PairwiseSybilEvidence, ...]

    def is_flagged(self, client_id: str) -> bool:
        return client_id in self.flagged_clients


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("trace trajectories must have identical dimensions")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))


def _resampled_trajectory(
    fingerprint: TraceFingerprint,
    *,
    checkpoint_count: int,
) -> np.ndarray:
    matrix = np.asarray(fingerprint.checkpoint_vectors, dtype=np.float64)
    if checkpoint_count < 2:
        raise ValueError("trajectory comparison requires at least two checkpoints")
    source_positions = np.linspace(0.0, 1.0, num=matrix.shape[0])
    target_positions = np.linspace(0.0, 1.0, num=checkpoint_count)
    resampled = np.stack(
        [
            np.interp(target_positions, source_positions, matrix[:, coordinate])
            for coordinate in range(matrix.shape[1])
        ],
        axis=1,
    )
    return np.diff(resampled, axis=0).reshape(-1)


def _trajectory_cosine(left: TraceFingerprint, right: TraceFingerprint) -> float:
    left_width = len(left.checkpoint_vectors[0])
    right_width = len(right.checkpoint_vectors[0])
    if left_width != right_width:
        return 0.0
    checkpoint_count = max(len(left.checkpoint_vectors), len(right.checkpoint_vectors))
    return _cosine(
        _resampled_trajectory(left, checkpoint_count=checkpoint_count),
        _resampled_trajectory(right, checkpoint_count=checkpoint_count),
    )


def screen_trace_fingerprints(
    fingerprints: tuple[TraceFingerprint, ...] | list[TraceFingerprint],
    *,
    trajectory_cosine_threshold: float = 0.995,
) -> SybilScreeningReport:
    if not -1 <= trajectory_cosine_threshold <= 1:
        raise ValueError("trajectory cosine threshold must be in [-1, 1]")
    if len({item.client_id for item in fingerprints}) != len(fingerprints):
        raise ValueError("duplicate client fingerprint")
    pairs: list[PairwiseSybilEvidence] = []
    flagged: set[str] = set()
    ordered = sorted(fingerprints, key=lambda item: item.client_id)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            cosine = _trajectory_cosine(left, right)
            identical_indices = bool(left.batch_indices) and left.batch_indices == right.batch_indices
            reasons: list[str] = []
            if cosine >= trajectory_cosine_threshold:
                reasons.append("checkpoint_trajectory_similarity")
            if identical_indices:
                reasons.append("identical_batch_index_sequence")
            is_flagged = bool(reasons)
            if is_flagged:
                flagged.update((left.client_id, right.client_id))
            pairs.append(
                PairwiseSybilEvidence(
                    left_client=left.client_id,
                    right_client=right.client_id,
                    trajectory_cosine=cosine,
                    identical_batch_indices=identical_indices,
                    flagged=is_flagged,
                    reasons=tuple(reasons),
                )
            )
    return SybilScreeningReport(frozenset(flagged), tuple(pairs))
