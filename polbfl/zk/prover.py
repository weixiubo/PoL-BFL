"""Challenge-bound construction of production ZK-PoL interval proofs."""

from __future__ import annotations

import io
from typing import Any, Mapping, TYPE_CHECKING

from polbfl.protocol import Challenge
from polbfl.zk.bundle import ZKCheckpointOpening, ZKIntervalBundle
from polbfl.zk.field import digest_to_field, interval_batch_commitment, protocol_binding_field
from polbfl.zk.groth16 import Groth16Backend
from polbfl.zk.witness import PADDED_DATA_INDEX, ZKCircuitConfig, signed_components

if TYPE_CHECKING:  # pragma: no cover
    from polbfl.training import RecordedTrace


class ZKPoLProver:
    def __init__(self, backend: Groth16Backend, config: ZKCircuitConfig, *, store):
        self.backend = backend
        self.config = config
        self.store = store

    def _load_step(self, recorded: "RecordedTrace", step: int) -> Mapping[str, Any]:
        import torch

        payload = self.store.get(recorded.steps[step].blob)
        try:
            loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.6
            loaded = torch.load(io.BytesIO(payload), map_location="cpu")
        if not isinstance(loaded, Mapping) or not isinstance(loaded.get("zk_witness"), Mapping):
            raise ValueError("step evidence does not contain a ZK witness")
        return loaded

    def _inactive_step(self, sample_indices: tuple[int, ...], weights: tuple[int, ...]) -> dict[str, Any]:
        zero_samples = (0,) * self.config.sample_count
        zero_terms = tuple(
            (0,) * self.config.batch_terms for _ in range(self.config.sample_count)
        )
        return {
            "sample_indices": sample_indices,
            "weights_before": weights,
            "weights_after": weights,
            "gradients": zero_samples,
            "rounding": zero_samples,
            "data_indices": (PADDED_DATA_INDEX,) * self.config.batch_terms,
            "activation_factors": zero_terms,
            "error_factors": zero_terms,
        }

    @staticmethod
    def _signed_2d(rows):
        magnitudes = []
        signs = []
        for row in rows:
            row_magnitude, row_sign = signed_components(tuple(int(value) for value in row))
            magnitudes.append(row_magnitude)
            signs.append(row_sign)
        return magnitudes, signs

    @staticmethod
    def _signed_3d(blocks):
        magnitudes = []
        signs = []
        for block in blocks:
            block_magnitude, block_sign = ZKPoLProver._signed_2d(block)
            magnitudes.append(block_magnitude)
            signs.append(block_sign)
        return magnitudes, signs

    def build_circuit_input(
        self,
        *,
        recorded: "RecordedTrace",
        challenge: Challenge,
        pair_index: int,
    ) -> dict[str, Any]:
        trace = recorded.trace
        if challenge.proof_mode != "zk" or pair_index not in challenge.pair_indices:
            raise ValueError("proof request is not part of this ZK challenge")
        if (
            challenge.protocol_version != trace.commitment.protocol_version
            or challenge.round_id != trace.commitment.round_id
            or challenge.client_id != trace.commitment.client_id
            or challenge.commitment_root != trace.commitment.merkle_root
        ):
            raise ValueError("challenge does not bind the recorded trace")
        if not 0 <= pair_index < trace.commitment.checkpoint_count - 1:
            raise IndexError("challenged checkpoint interval is out of range")

        start_record = trace.checkpoints[pair_index]
        end_record = trace.checkpoints[pair_index + 1]
        step_numbers = tuple(range(start_record.step + 1, end_record.step + 1))
        if not step_numbers or len(step_numbers) > self.config.steps:
            raise ValueError("challenged interval does not fit the reference circuit")
        payloads = [self._load_step(recorded, step)["zk_witness"] for step in step_numbers]
        sample_indices = tuple(int(value) for value in payloads[0]["sample_indices"])
        if len(sample_indices) != self.config.sample_count:
            raise ValueError("ZK sample count differs from the proving circuit")
        if any(tuple(int(value) for value in payload["sample_indices"]) != sample_indices for payload in payloads):
            raise ValueError("ZK sample plan changed inside an interval")
        for left, right in zip(payloads, payloads[1:]):
            if tuple(left["weights_after"]) != tuple(right["weights_before"]):
                raise ValueError("ZK private trajectory is not contiguous")
        final_weights = tuple(int(value) for value in payloads[-1]["weights_after"])
        padded = list(payloads)
        while len(padded) < self.config.steps:
            padded.append(self._inactive_step(sample_indices, final_weights))

        weights = [tuple(int(value) for value in payloads[0]["weights_before"])]
        weights.extend(tuple(int(value) for value in payload["weights_after"]) for payload in padded)
        gradients = [tuple(int(value) for value in payload["gradients"]) for payload in padded]
        rounding = [tuple(int(value) for value in payload["rounding"]) for payload in padded]
        data_indices = [tuple(int(value) for value in payload["data_indices"]) for payload in padded]
        activations = [payload["activation_factors"] for payload in padded]
        errors = [payload["error_factors"] for payload in padded]
        weight_magnitude, weight_sign = self._signed_2d(weights)
        gradient_magnitude, gradient_sign = self._signed_2d(gradients)
        rounding_magnitude, rounding_sign = self._signed_2d(rounding)
        activation_magnitude, activation_sign = self._signed_3d(activations)
        error_magnitude, error_sign = self._signed_3d(errors)

        start_aux = dict(recorded.checkpoints[pair_index].auxiliary.get("zk", {}))
        end_auxiliary = recorded.checkpoints[pair_index + 1].auxiliary
        end_aux = dict(end_auxiliary.get("zk", {}))
        interval = dict(end_aux.get("interval", {}))
        batch_commitment = interval_batch_commitment(end_auxiliary.get("step_evidence", ()))
        binding = protocol_binding_field(
            context_digest=trace.context.digest,
            commitment_root=trace.commitment.merkle_root,
            challenge_id=challenge.challenge_id,
            pair_index=pair_index,
            batch_commitment=batch_commitment,
        )
        expected_active = int(interval.get("active_step_count", -1))
        if expected_active != len(payloads):
            raise ValueError("active step count differs from committed interval metadata")

        return {
            "contextHash": str(digest_to_field(trace.context.digest)),
            "commitmentRootHash": str(digest_to_field(trace.commitment.merkle_root)),
            "challengeHash": str(digest_to_field(challenge.challenge_id)),
            "pairIndex": str(pair_index),
            "batchCommitmentHash": str(digest_to_field(batch_commitment)),
            "protocolBindingHash": str(binding),
            "activeStepCount": str(len(payloads)),
            "startWeightsHash": str(start_aux["sampled_weights_hash"]),
            "endWeightsHash": str(end_aux["sampled_weights_hash"]),
            "gradientsHash": str(interval["gradients_hash"]),
            "dataIndicesHash": str(interval["data_indices_hash"]),
            "samplePlanHash": str(start_aux["sample_plan_hash"]),
            "auxiliaryHash": str(interval["auxiliary_hash"]),
            "scale": str(interval["scale"]),
            "learningRate": str(interval["learning_rate"]),
            "maxDistanceSquared": str(interval["max_distance_squared"]),
            "maxRoundingError": str(interval["max_rounding_error"]),
            "maxCumulativeRoundingErrorSquared": str(
                interval["max_cumulative_rounding_error_squared"]
            ),
            "sampleIndices": [str(value) for value in sample_indices],
            "dataIndices": [[str(value) for value in row] for row in data_indices],
            "stepActive": ["1"] * len(payloads) + ["0"] * (self.config.steps - len(payloads)),
            "weightMagnitude": weight_magnitude,
            "weightSign": weight_sign,
            "gradientMagnitude": gradient_magnitude,
            "gradientSign": gradient_sign,
            "roundingMagnitude": rounding_magnitude,
            "roundingSign": rounding_sign,
            "activationMagnitude": activation_magnitude,
            "activationSign": activation_sign,
            "errorMagnitude": error_magnitude,
            "errorSign": error_sign,
        }

    def prove_interval(
        self,
        *,
        recorded: "RecordedTrace",
        challenge: Challenge,
        pair_index: int,
    ) -> ZKIntervalBundle:
        circuit_input = self.build_circuit_input(
            recorded=recorded,
            challenge=challenge,
            pair_index=pair_index,
        )
        proof = self.backend.prove(circuit_input)
        trace = recorded.trace
        return ZKIntervalBundle(
            challenge=challenge,
            commitment=trace.commitment,
            pair_index=pair_index,
            start=ZKCheckpointOpening(
                pair_index,
                trace.checkpoints[pair_index],
                trace.checkpoint_proof(pair_index),
                recorded.checkpoints[pair_index].auxiliary,
            ),
            end=ZKCheckpointOpening(
                pair_index + 1,
                trace.checkpoints[pair_index + 1],
                trace.checkpoint_proof(pair_index + 1),
                recorded.checkpoints[pair_index + 1].auxiliary,
            ),
            proof=proof,
            uploaded_final_model_digest=(
                trace.commitment.final_model_digest
                if pair_index + 1 == trace.commitment.checkpoint_count - 1
                else None
            ),
        )

    def prove_challenge(
        self,
        *,
        recorded: "RecordedTrace",
        challenge: Challenge,
    ) -> tuple[ZKIntervalBundle, ...]:
        return tuple(
            self.prove_interval(recorded=recorded, challenge=challenge, pair_index=pair_index)
            for pair_index in challenge.pair_indices
        )
