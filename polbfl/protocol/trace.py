"""PoL trace construction, hash chaining, and Merkle commitment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from polbfl.crypto.canonical import (
    canonical_json_bytes,
    domain_hash,
    hash_batch,
    hash_batch_indices,
    hash_state_dict,
)
from polbfl.crypto.merkle import MerkleProofStep, MerkleTree
from polbfl.protocol.models import CheckpointRecord, RoundContext, TraceCommitment


ZERO_DIGEST = "0" * 64


def create_checkpoint_record(
    context: RoundContext,
    *,
    step: int,
    epoch: int,
    timestamp_ns: int,
    model_state: Mapping[str, Any],
    batch_data: Any,
    batch_labels: Any,
    batch_indices: Sequence[int],
    auxiliary: Any,
    previous_chain_digest: str,
    precomputed_model_digest: str | None = None,
) -> CheckpointRecord:
    """Create the exact record committed by both training and verification."""

    model_digest = (
        hash_state_dict(model_state)
        if precomputed_model_digest is None
        else str(precomputed_model_digest)
    )
    if len(model_digest) != 64:
        raise ValueError("precomputed model digest must be SHA-256 sized")
    try:
        bytes.fromhex(model_digest)
    except ValueError as exc:
        raise ValueError("precomputed model digest must be hexadecimal") from exc
    data_digest = hash_batch(batch_data, batch_labels)
    indices_digest = hash_batch_indices(batch_indices)
    auxiliary_digest = domain_hash("POLBFL_AUXILIARY_V1", canonical_json_bytes(auxiliary))
    checkpoint_digest = domain_hash(
        "POLBFL_CHECKPOINT_V1",
        bytes.fromhex(context.digest),
        bytes.fromhex(model_digest),
        bytes.fromhex(data_digest),
        bytes.fromhex(indices_digest),
        timestamp_ns,
        step,
        epoch,
        bytes.fromhex(auxiliary_digest),
    )
    chain_digest = domain_hash(
        "POLBFL_TRACE_CHAIN_V1",
        bytes.fromhex(previous_chain_digest),
        bytes.fromhex(checkpoint_digest),
    )
    return CheckpointRecord(
        step=step,
        epoch=epoch,
        timestamp_ns=timestamp_ns,
        model_digest=model_digest,
        data_digest=data_digest,
        indices_digest=indices_digest,
        auxiliary_digest=auxiliary_digest,
        checkpoint_digest=checkpoint_digest,
        previous_chain_digest=previous_chain_digest,
        chain_digest=chain_digest,
    )


@dataclass(frozen=True)
class PoLTrace:
    context: RoundContext
    checkpoints: tuple[CheckpointRecord, ...]
    commitment: TraceCommitment

    def checkpoint_proof(self, index: int) -> tuple[MerkleProofStep, ...]:
        return MerkleTree(record.chain_digest for record in self.checkpoints).proof(index)

    def verify_structure(self) -> bool:
        if len(self.checkpoints) != self.commitment.checkpoint_count:
            return False
        previous = ZERO_DIGEST
        previous_step = -1
        for record in self.checkpoints:
            if record.step <= previous_step or record.previous_chain_digest != previous:
                return False
            expected_chain = domain_hash(
                "POLBFL_TRACE_CHAIN_V1", bytes.fromhex(previous), bytes.fromhex(record.checkpoint_digest)
            )
            if record.chain_digest != expected_chain:
                return False
            previous = record.chain_digest
            previous_step = record.step
        tree = MerkleTree(record.chain_digest for record in self.checkpoints)
        return (
            tree.root == self.commitment.merkle_root
            and self.context.digest == self.commitment.context_digest
            and self.checkpoints[-1].model_digest == self.commitment.final_model_digest
            and (
                self.context.expected_steps is None
                or self.commitment.final_step == self.context.expected_steps
            )
        )


class PoLTraceBuilder:
    def __init__(self, context: RoundContext):
        self.context = context
        self._records: list[CheckpointRecord] = []

    def append_checkpoint(
        self,
        *,
        step: int,
        epoch: int,
        timestamp_ns: int,
        model_state: Mapping[str, Any],
        batch_data: Any,
        batch_labels: Any,
        batch_indices: Sequence[int],
        auxiliary: Any,
        precomputed_model_digest: str | None = None,
    ) -> CheckpointRecord:
        if self._records and step <= self._records[-1].step:
            raise ValueError("checkpoint steps must increase strictly")
        previous = self._records[-1].chain_digest if self._records else ZERO_DIGEST
        record = create_checkpoint_record(
            self.context,
            step=step,
            epoch=epoch,
            timestamp_ns=timestamp_ns,
            model_state=model_state,
            batch_data=batch_data,
            batch_labels=batch_labels,
            batch_indices=batch_indices,
            auxiliary=auxiliary,
            previous_chain_digest=previous,
            precomputed_model_digest=precomputed_model_digest,
        )
        self._records.append(record)
        return record

    def finalize(self) -> PoLTrace:
        if len(self._records) < 2:
            raise ValueError("a verifiable trace requires at least two checkpoints")
        if self.context.expected_steps is not None and self._records[-1].step != self.context.expected_steps:
            raise ValueError("trace does not contain the prescribed optimizer-step count")
        tree = MerkleTree(record.chain_digest for record in self._records)
        body = {
            "protocol_version": self.context.protocol_version,
            "round_id": self.context.round_id,
            "client_id": self.context.client_id,
            "context_digest": self.context.digest,
            "merkle_root": tree.root,
            "checkpoint_count": len(self._records),
            "first_step": self._records[0].step,
            "final_step": self._records[-1].step,
            "final_model_digest": self._records[-1].model_digest,
        }
        commitment = TraceCommitment(
            **body,
            trace_digest=domain_hash("POLBFL_TRACE_COMMITMENT_V1", canonical_json_bytes(body)),
        )
        trace = PoLTrace(self.context, tuple(self._records), commitment)
        if not trace.verify_structure():  # pragma: no cover - defensive invariant
            raise AssertionError("newly created trace failed structural verification")
        return trace
