"""Composition of Groth16 verification with SHA-256 trace/Merkle binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from polbfl.crypto import MerkleProofStep, MerkleTree, canonical_json_bytes, domain_hash
from polbfl.protocol import Challenge, CheckpointRecord, RoundContext, TraceCommitment
from polbfl.zk.groth16 import Groth16Backend, Groth16Proof
from polbfl.zk.field import digest_to_field, interval_batch_commitment, protocol_binding_field


PUBLIC_SIGNAL_NAMES = (
    "contextHash",
    "commitmentRootHash",
    "challengeHash",
    "pairIndex",
    "batchCommitmentHash",
    "protocolBindingHash",
    "activeStepCount",
    "startWeightsHash",
    "endWeightsHash",
    "gradientsHash",
    "dataIndicesHash",
    "samplePlanHash",
    "auxiliaryHash",
    "scale",
    "learningRate",
    "maxDistanceSquared",
    "maxRoundingError",
    "maxCumulativeRoundingErrorSquared",
)


@dataclass(frozen=True)
class ZKCheckpointOpening:
    index: int
    record: CheckpointRecord
    merkle_proof: tuple[MerkleProofStep, ...]
    auxiliary_public: Mapping[str, Any]

    def is_bound(self, commitment: TraceCommitment) -> bool:
        expected_auxiliary = domain_hash(
            "POLBFL_AUXILIARY_V1",
            canonical_json_bytes(self.auxiliary_public),
        )
        return (
            0 <= self.index < commitment.checkpoint_count
            and expected_auxiliary == self.record.auxiliary_digest
            and MerkleTree.verify(
                self.record.chain_digest,
                self.merkle_proof,
                commitment.merkle_root,
            )
        )


@dataclass(frozen=True)
class ZKIntervalBundle:
    challenge: Challenge
    commitment: TraceCommitment
    pair_index: int
    start: ZKCheckpointOpening
    end: ZKCheckpointOpening
    proof: Groth16Proof
    uploaded_final_model_digest: str | None = None


@dataclass(frozen=True)
class ZKBundleReport:
    valid: bool
    proof_valid: bool
    verify_seconds: float
    reasons: tuple[str, ...]


class ZKBundleVerifier:
    def __init__(self, backend: Groth16Backend):
        self.backend = backend

    @staticmethod
    def _signals(proof: Groth16Proof) -> dict[str, str]:
        if len(proof.public_signals) != len(PUBLIC_SIGNAL_NAMES):
            raise ValueError("unexpected Groth16 public-signal count")
        return dict(zip(PUBLIC_SIGNAL_NAMES, proof.public_signals))

    def verify(self, context: RoundContext, bundle: ZKIntervalBundle) -> ZKBundleReport:
        reasons: list[str] = []
        challenge = bundle.challenge
        commitment = bundle.commitment
        if challenge.proof_mode != "zk":
            reasons.append("wrong_proof_mode")
        if bundle.pair_index not in challenge.pair_indices:
            reasons.append("pair_not_challenged")
        if bundle.start.index != bundle.pair_index or bundle.end.index != bundle.pair_index + 1:
            reasons.append("opening_index_mismatch")
        if (
            challenge.protocol_version != commitment.protocol_version
            or challenge.round_id != commitment.round_id
            or challenge.client_id != commitment.client_id
            or challenge.commitment_root != commitment.merkle_root
            or commitment.context_digest != context.digest
            or (
                context.expected_steps is not None
                and commitment.final_step != context.expected_steps
            )
        ):
            reasons.append("challenge_commitment_context_mismatch")
        if not bundle.start.is_bound(commitment):
            reasons.append("start_checkpoint_not_bound")
        if not bundle.end.is_bound(commitment):
            reasons.append("end_checkpoint_not_bound")
        if bundle.end.record.previous_chain_digest != bundle.start.record.chain_digest:
            reasons.append("checkpoint_chain_not_adjacent")

        try:
            signals = self._signals(bundle.proof)
            start_zk = dict(bundle.start.auxiliary_public.get("zk", {}))
            end_zk = dict(bundle.end.auxiliary_public.get("zk", {}))
            interval = dict(end_zk.get("interval", {}))
            batch_commitment = interval_batch_commitment(
                bundle.end.auxiliary_public.get("step_evidence", ())
            )
            binding = protocol_binding_field(
                context_digest=context.digest,
                commitment_root=commitment.merkle_root,
                challenge_id=challenge.challenge_id,
                pair_index=bundle.pair_index,
                batch_commitment=batch_commitment,
            )
            expected = {
                "contextHash": str(digest_to_field(context.digest)),
                "commitmentRootHash": str(digest_to_field(commitment.merkle_root)),
                "challengeHash": str(digest_to_field(challenge.challenge_id)),
                "pairIndex": str(bundle.pair_index),
                "batchCommitmentHash": str(digest_to_field(batch_commitment)),
                "protocolBindingHash": str(binding),
                "activeStepCount": str(interval.get("active_step_count", "")),
                "startWeightsHash": str(start_zk.get("sampled_weights_hash", "")),
                "endWeightsHash": str(end_zk.get("sampled_weights_hash", "")),
                "gradientsHash": str(interval.get("gradients_hash", "")),
                "dataIndicesHash": str(interval.get("data_indices_hash", "")),
                "samplePlanHash": str(start_zk.get("sample_plan_hash", "")),
                "auxiliaryHash": str(interval.get("auxiliary_hash", "")),
                "scale": str(interval.get("scale", "")),
                "learningRate": str(interval.get("learning_rate", "")),
                "maxDistanceSquared": str(interval.get("max_distance_squared", "")),
                "maxRoundingError": str(interval.get("max_rounding_error", "")),
                "maxCumulativeRoundingErrorSquared": str(
                    interval.get("max_cumulative_rounding_error_squared", "")
                ),
            }
            if str(start_zk.get("context_hash")) != expected["contextHash"]:
                reasons.append("start_context_field_hash_mismatch")
            if str(end_zk.get("context_hash")) != expected["contextHash"]:
                reasons.append("context_field_hash_mismatch")
            if start_zk.get("sample_plan_hash") != end_zk.get("sample_plan_hash"):
                reasons.append("sample_plan_mismatch")
            if str(interval.get("batch_commitment_hash")) != expected["batchCommitmentHash"]:
                reasons.append("batch_commitment_mismatch")
            for name, expected_value in expected.items():
                if not expected_value or signals.get(name) != expected_value:
                    reasons.append(f"public_signal_mismatch:{name}")
        except (TypeError, ValueError):
            reasons.append("invalid_public_signal_layout")

        if bundle.uploaded_final_model_digest is not None:
            if bundle.pair_index + 1 != commitment.checkpoint_count - 1:
                reasons.append("final_digest_on_nonfinal_interval")
            elif bundle.uploaded_final_model_digest != commitment.final_model_digest:
                reasons.append("uploaded_final_model_digest_mismatch")

        proof_valid, verify_seconds = self.backend.verify(bundle.proof)
        if not proof_valid:
            reasons.append("groth16_verification_failed")
        return ZKBundleReport(
            valid=not reasons and proof_valid,
            proof_valid=proof_valid,
            verify_seconds=verify_seconds,
            reasons=tuple(reasons),
        )
