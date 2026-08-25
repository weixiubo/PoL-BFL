"""Validated immutable protocol records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polbfl.crypto.canonical import canonical_json_bytes, domain_hash


def _require_digest(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc


@dataclass(frozen=True)
class RoundContext:
    protocol_version: str
    round_id: str
    client_id: str
    model_id: str
    global_model_digest: str
    optimizer: str
    learning_rate: float
    local_epochs: int
    batch_size: int
    checkpoint_interval: int
    expected_steps: int | None = None

    def __post_init__(self) -> None:
        if not all((self.protocol_version, self.round_id, self.client_id, self.model_id, self.optimizer)):
            raise ValueError("round context identifiers must be non-empty")
        _require_digest("global_model_digest", self.global_model_digest)
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if min(self.local_epochs, self.batch_size, self.checkpoint_interval) <= 0:
            raise ValueError("epoch, batch, and checkpoint values must be positive")
        if self.expected_steps is not None and self.expected_steps <= 0:
            raise ValueError("expected optimizer steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "round_id": self.round_id,
            "client_id": self.client_id,
            "model_id": self.model_id,
            "global_model_digest": self.global_model_digest,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "checkpoint_interval": self.checkpoint_interval,
            "expected_steps": self.expected_steps,
        }

    @property
    def digest(self) -> str:
        return domain_hash("POLBFL_ROUND_CONTEXT_V1", canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class CheckpointRecord:
    step: int
    epoch: int
    timestamp_ns: int
    model_digest: str
    data_digest: str
    indices_digest: str
    auxiliary_digest: str
    checkpoint_digest: str
    previous_chain_digest: str
    chain_digest: str

    def __post_init__(self) -> None:
        if min(self.step, self.epoch, self.timestamp_ns) < 0:
            raise ValueError("checkpoint counters and timestamp must be non-negative")
        for name in (
            "model_digest",
            "data_digest",
            "indices_digest",
            "auxiliary_digest",
            "checkpoint_digest",
            "previous_chain_digest",
            "chain_digest",
        ):
            _require_digest(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "timestamp_ns": self.timestamp_ns,
            "model_digest": self.model_digest,
            "data_digest": self.data_digest,
            "indices_digest": self.indices_digest,
            "auxiliary_digest": self.auxiliary_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "previous_chain_digest": self.previous_chain_digest,
            "chain_digest": self.chain_digest,
        }


@dataclass(frozen=True)
class TraceCommitment:
    protocol_version: str
    round_id: str
    client_id: str
    context_digest: str
    merkle_root: str
    checkpoint_count: int
    first_step: int
    final_step: int
    final_model_digest: str
    trace_digest: str

    def __post_init__(self) -> None:
        if self.checkpoint_count < 2:
            raise ValueError("a verifiable trace requires at least two checkpoints")
        if self.first_step < 0 or self.final_step <= self.first_step:
            raise ValueError("invalid trace step bounds")
        for name in ("context_digest", "merkle_root", "final_model_digest", "trace_digest"):
            _require_digest(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "round_id": self.round_id,
            "client_id": self.client_id,
            "context_digest": self.context_digest,
            "merkle_root": self.merkle_root,
            "checkpoint_count": self.checkpoint_count,
            "first_step": self.first_step,
            "final_step": self.final_step,
            "final_model_digest": self.final_model_digest,
            "trace_digest": self.trace_digest,
        }


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    protocol_version: str
    round_id: str
    client_id: str
    commitment_root: str
    pair_indices: tuple[int, ...]
    randomness_digest: str
    issued_at_ns: int
    deadline_ns: int
    proof_mode: str = "zk"

    def __post_init__(self) -> None:
        if not self.challenge_id:
            raise ValueError("challenge ID must be non-empty")
        _require_digest("commitment_root", self.commitment_root)
        _require_digest("randomness_digest", self.randomness_digest)
        if not self.pair_indices or tuple(sorted(set(self.pair_indices))) != self.pair_indices:
            raise ValueError("challenge pair indices must be sorted and unique")
        if min(self.pair_indices) < 0:
            raise ValueError("challenge pair indices must be non-negative")
        if self.deadline_ns <= self.issued_at_ns:
            raise ValueError("challenge deadline must follow issuance")
        if self.proof_mode not in {"zk", "strict_replay"}:
            raise ValueError("challenge proof mode must be zk or strict_replay")

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "protocol_version": self.protocol_version,
            "round_id": self.round_id,
            "client_id": self.client_id,
            "commitment_root": self.commitment_root,
            "pair_indices": list(self.pair_indices),
            "randomness_digest": self.randomness_digest,
            "issued_at_ns": self.issued_at_ns,
            "deadline_ns": self.deadline_ns,
            "proof_mode": self.proof_mode,
        }
