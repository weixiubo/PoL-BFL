"""Reference strict replay verifier with dual numerical tolerances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from polbfl.crypto import MerkleProofStep, MerkleTree, hash_state_dict
from polbfl.protocol import Challenge, CheckpointRecord, RoundContext, TraceCommitment
from polbfl.protocol.trace import create_checkpoint_record


def _flat_numeric(value: Any) -> list[float]:
    obj = value
    for method_name in ("detach", "cpu", "contiguous"):
        method = getattr(obj, method_name, None)
        if callable(method):
            obj = method()
    reshape = getattr(obj, "reshape", None)
    if callable(reshape):
        try:
            obj = reshape(-1)
        except TypeError:
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        obj = tolist()

    flattened: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, bool):
            flattened.append(float(item))
        elif isinstance(item, (int, float)):
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("non-finite model parameter")
            flattened.append(number)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)
        else:
            raise TypeError(f"model state contains non-numeric value {type(item)!r}")

    visit(obj)
    return flattened


def parameter_l2_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        raise ValueError("model states have different parameter keys")
    squared = 0.0
    for key in sorted(left):
        left_values = _flat_numeric(left[key])
        right_values = _flat_numeric(right[key])
        if len(left_values) != len(right_values):
            raise ValueError(f"parameter shape mismatch for {key}")
        squared += sum((a - b) ** 2 for a, b in zip(left_values, right_values))
    return math.sqrt(squared)


@dataclass(frozen=True)
class CheckpointOpening:
    index: int
    record: CheckpointRecord
    merkle_proof: tuple[MerkleProofStep, ...]
    model_state: Mapping[str, Any]
    batch_data: Any
    batch_labels: Any
    batch_indices: tuple[int, ...]
    auxiliary: Any


@dataclass(frozen=True)
class IntervalWitness:
    pair_index: int
    private_batches: tuple[Any, ...]
    optimizer_state: Any
    replay_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ChallengeResponse:
    challenge_id: str
    commitment: TraceCommitment
    openings: Mapping[int, CheckpointOpening]
    interval_witnesses: Mapping[int, IntervalWitness]
    uploaded_model_state: Mapping[str, Any]


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    pair_results: Mapping[int, bool]
    pair_distances: Mapping[int, float]
    final_distance: float | None
    reasons: tuple[str, ...]


ReplayInterval = Callable[
    [RoundContext, CheckpointOpening, CheckpointOpening, IntervalWitness],
    Mapping[str, Any],
]


class StrictTraceVerifier:
    def __init__(self, *, pair_tolerance: float, final_tolerance: float):
        if pair_tolerance < 0 or final_tolerance < 0:
            raise ValueError("verification tolerances must be non-negative")
        self.pair_tolerance = float(pair_tolerance)
        self.final_tolerance = float(final_tolerance)

    @staticmethod
    def _opening_is_bound(
        context: RoundContext,
        opening: CheckpointOpening,
        commitment: TraceCommitment,
    ) -> bool:
        if opening.index < 0 or opening.index >= commitment.checkpoint_count:
            return False
        recreated = create_checkpoint_record(
            context,
            step=opening.record.step,
            epoch=opening.record.epoch,
            timestamp_ns=opening.record.timestamp_ns,
            model_state=opening.model_state,
            batch_data=opening.batch_data,
            batch_labels=opening.batch_labels,
            batch_indices=opening.batch_indices,
            auxiliary=opening.auxiliary,
            previous_chain_digest=opening.record.previous_chain_digest,
        )
        return recreated == opening.record and MerkleTree.verify(
            opening.record.chain_digest,
            opening.merkle_proof,
            commitment.merkle_root,
        )

    def verify(
        self,
        *,
        context: RoundContext,
        challenge: Challenge,
        response: ChallengeResponse,
        replay_interval: ReplayInterval,
    ) -> VerificationReport:
        reasons: list[str] = []
        pair_results: dict[int, bool] = {}
        pair_distances: dict[int, float] = {}
        final_distance: float | None = None

        if response.challenge_id != challenge.challenge_id:
            reasons.append("challenge_id_mismatch")
        commitment = response.commitment
        if (
            commitment.protocol_version != challenge.protocol_version
            or commitment.round_id != challenge.round_id
            or commitment.client_id != challenge.client_id
            or commitment.merkle_root != challenge.commitment_root
            or commitment.context_digest != context.digest
            or (
                context.expected_steps is not None
                and commitment.final_step != context.expected_steps
            )
        ):
            reasons.append("commitment_context_mismatch")

        required_indices = {commitment.checkpoint_count - 1}
        for pair_index in challenge.pair_indices:
            required_indices.update((pair_index, pair_index + 1))
        missing = sorted(required_indices - set(response.openings))
        if missing:
            reasons.append("missing_checkpoint_openings:" + ",".join(map(str, missing)))

        bound: dict[int, bool] = {}
        for index in required_indices & set(response.openings):
            bound[index] = self._opening_is_bound(context, response.openings[index], commitment)
            if not bound[index]:
                reasons.append(f"invalid_checkpoint_opening:{index}")

        for pair_index in challenge.pair_indices:
            start = response.openings.get(pair_index)
            end = response.openings.get(pair_index + 1)
            witness = response.interval_witnesses.get(pair_index)
            if start is None or end is None or witness is None:
                pair_results[pair_index] = False
                reasons.append(f"missing_interval_evidence:{pair_index}")
                continue
            if not bound.get(pair_index, False) or not bound.get(pair_index + 1, False):
                pair_results[pair_index] = False
                continue
            if end.record.previous_chain_digest != start.record.chain_digest:
                pair_results[pair_index] = False
                reasons.append(f"non_adjacent_chain:{pair_index}")
                continue
            if end.record.step <= start.record.step or witness.pair_index != pair_index:
                pair_results[pair_index] = False
                reasons.append(f"invalid_interval_bounds:{pair_index}")
                continue
            try:
                replayed = replay_interval(context, start, end, witness)
                distance = parameter_l2_distance(replayed, end.model_state)
                pair_distances[pair_index] = distance
                pair_results[pair_index] = distance <= self.pair_tolerance
                if not pair_results[pair_index]:
                    reasons.append(f"pair_tolerance_exceeded:{pair_index}")
            except Exception as exc:
                pair_results[pair_index] = False
                reasons.append(f"replay_error:{pair_index}:{type(exc).__name__}")

        final_opening = response.openings.get(commitment.checkpoint_count - 1)
        if final_opening is not None and bound.get(commitment.checkpoint_count - 1, False):
            try:
                final_distance = parameter_l2_distance(response.uploaded_model_state, final_opening.model_state)
                if final_distance > self.final_tolerance:
                    reasons.append("final_tolerance_exceeded")
                if hash_state_dict(final_opening.model_state) != commitment.final_model_digest:
                    reasons.append("final_commitment_digest_mismatch")
            except Exception as exc:
                reasons.append(f"final_model_error:{type(exc).__name__}")

        all_pairs_valid = bool(pair_results) and all(pair_results.values())
        final_valid = final_distance is not None and final_distance <= self.final_tolerance
        return VerificationReport(
            valid=not reasons and all_pairs_valid and final_valid,
            pair_results=pair_results,
            pair_distances=pair_distances,
            final_distance=final_distance,
            reasons=tuple(reasons),
        )
