"""Paper-protocol PoL trainer with private evidence recording."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import torch

from client.trainer.PoLTrainer import PoLTrainer
from polbfl.crypto import hash_state_dict
from polbfl.protocol import Challenge, RoundContext
from polbfl.storage import ContentAddressedStore
from polbfl.training import RecordedTrace, TorchPoLRecorder
from polbfl.verification import ChallengeResponse, CheckpointOpening, IntervalWitness
from polbfl.zk import Groth16Backend, PoseidonBridge, ZKCircuitConfig, ZKPoLProver


def _seed_bytes(raw: Any, *, fallback: str) -> bytes:
    if isinstance(raw, bytes):
        seed = raw
    elif isinstance(raw, str) and raw:
        text = raw[2:] if raw.startswith("0x") else raw
        try:
            seed = bytes.fromhex(text)
        except ValueError:
            seed = raw.encode("utf-8")
    elif raw is None:
        seed = b""
    else:
        seed = str(raw).encode("utf-8")
    if len(seed) < 32:
        seed = hashlib.sha256(seed + fallback.encode("utf-8")).digest()
    return seed


class ProtocolPoLTrainer(PoLTrainer):
    """Train and expose challenge-bound PoL evidence under protocol version 1."""

    def __init__(self, model, dataloader, criterion, args=None, watermarks=None):
        normalized_args = dict(args or {})
        normalized_args.setdefault("pol_save_freq", 5)
        normalized_args.setdefault("enable_zkp", True)
        normalized_args.setdefault("clip_norm", None)
        super().__init__(model, dataloader, criterion, normalized_args, watermarks or {})
        self.protocol_version = str(self.args.get("protocol_version", "1"))
        self.zk_config = None
        if bool(self.args.get("enable_zkp", True)):
            if int(self.pol_save_freq) != 5:
                raise ValueError("the reference ZK-PoL circuit requires five-step checkpoints")
            self.zk_config = ZKCircuitConfig(
                sample_count=int(self.args.get("zk_sample_count", 14)),
                steps=int(self.pol_save_freq),
                batch_terms=int(self.args.get("zk_batch_terms", 32)),
                value_bits=int(self.args.get("zk_value_bits", 48)),
                scale=int(self.args.get("zk_scale", 1_000_000)),
                pair_tolerance=float(self.args.get("pair_tolerance", 1e-5)),
                final_tolerance=float(self.args.get("final_tolerance", 1e-3)),
                max_update_l2=float(self.args.get("max_update_l2", 10.0)),
                auxiliary_pairs_per_chunk=int(self.args.get("zk_auxiliary_pairs_per_chunk", 4)),
            )
        self.trace_recorder: TorchPoLRecorder | None = None
        self.recorded_trace: RecordedTrace | None = None
        self._commitment_payload: dict[str, Any] | None = None

    def _round_context(self, total_epoch: int) -> RoundContext:
        round_number = int(self.args.get("round_num", self.args.get("round", 0)))
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        momentum = float(self.optimizer.param_groups[0].get("momentum", 0.0))
        weight_decay = float(self.optimizer.param_groups[0].get("weight_decay", 0.0))
        return RoundContext(
            protocol_version=self.protocol_version,
            round_id=str(self.args.get("round_id") or f"round-{round_number}"),
            client_id=str(self.client_id),
            model_id=str(self.args.get("model_id") or self.model.__class__.__name__),
            global_model_digest=hash_state_dict(self.model.state_dict()),
            optimizer=(
                f"{self.args.get('optimizer', 'SGD')}"
                f"(momentum={momentum.hex()},weight_decay={weight_decay.hex()})"
            ),
            learning_rate=learning_rate,
            local_epochs=int(total_epoch),
            batch_size=int(getattr(self.dataloader, "batch_size", None) or self.args.get("batch_size", 32)),
            checkpoint_interval=int(self.pol_save_freq),
            expected_steps=(int(total_epoch) - int(self.start_epoch)) * len(self.dataloader),
        )

    def train(self, total_epoch):
        if not self.enable_pol:
            return super().train(total_epoch)
        self.construct_optimizer()
        if self.zk_config is not None:
            if str(self.args.get("optimizer", "SGD")) != "SGD":
                raise ValueError("the reference ZK-PoL circuit requires SGD")
            group = self.optimizer.param_groups[0]
            if float(group.get("momentum", 0.0)) != 0.0 or float(group.get("weight_decay", 0.0)) != 0.0:
                raise ValueError("the reference ZK-PoL circuit requires vanilla SGD")
        context = self._round_context(total_epoch)
        raw_randomness = self.args.get("round_randomness") or os.getenv("POL_ROUND_RANDOMNESS")
        if raw_randomness is None and os.getenv("POL_INTEGRITY", "0") == "1":
            raise RuntimeError("POL_ROUND_RANDOMNESS is required for paper-protocol training")
        sampling_seed = _seed_bytes(raw_randomness, fallback=context.digest)
        evidence_root = (
            Path(self.pol_save_dir)
            / f"client_{self.client_id}"
            / context.round_id
            / "private_evidence_v1"
        )
        self.trace_recorder = TorchPoLRecorder(
            context,
            ContentAddressedStore(
                evidence_root,
                packed=bool(self.args.get("packed_evidence", False)),
                durable=bool(self.args.get("durable_evidence", True)),
            ),
            sampling_seed=sampling_seed,
            gradient_sample_rate=float(self.args.get("gradient_sample_rate", 0.01)),
            zk_config=self.zk_config,
            poseidon_bridge=(
                PoseidonBridge(
                    node_binary=str(self.args.get("node_binary", "node")),
                    native_binary=self.args.get("poseidon_binary") or None,
                    persistent=True,
                )
                if self.zk_config is not None
                else None
            ),
        )
        self.trace_recorder.start(model=self.model, optimizer=self.optimizer)
        results = []
        for epoch in range(self.start_epoch, total_epoch):
            results.append(self._train_epoch(epoch))
        return results

    def _record_protocol_step(self, **kwargs):
        if self.trace_recorder is None:
            return None
        if not kwargs.get("batch_indices") and os.getenv("POL_INTEGRITY", "0") == "1":
            raise RuntimeError("paper-protocol training requires exact batch indices")
        self.trace_recorder.record_optimizer_step(**kwargs)
        return None

    def _get_batch_indices(self, batch_idx):
        if os.getenv("POL_INTEGRITY", "0") == "1":
            raise RuntimeError("paper-protocol training requires dataset-supplied global indices")
        return super()._get_batch_indices(batch_idx)

    def _save_checkpoint(self, epoch: int, batch_idx: int, loss: float):
        if self.trace_recorder is not None:
            return None
        return super()._save_checkpoint(epoch, batch_idx, loss)

    def finalize_pol(self, epoch: int, dataset=None):
        del epoch, dataset
        if self._commitment_payload is not None:
            return dict(self._commitment_payload)
        if self.trace_recorder is None:
            raise RuntimeError("paper-protocol trace recorder is not initialized")
        self.recorded_trace = self.trace_recorder.finalize()
        if self.trace_recorder.poseidon_bridge is not None:
            self.trace_recorder.poseidon_bridge.close()
        self.trace_recorder.store.finalize()
        commitment = self.recorded_trace.trace.commitment
        self._commitment_payload = {
            "protocol_version": commitment.protocol_version,
            "commitment": commitment.merkle_root,
            "trace_digest": commitment.trace_digest,
            "context_digest": commitment.context_digest,
            "data_hash": self.recorded_trace.data_root,
            "num_checkpoints": commitment.checkpoint_count,
            "total_steps": commitment.final_step,
            "save_freq": self.pol_save_freq,
            "final_model_digest": commitment.final_model_digest,
            "client_id": commitment.client_id,
            "round_id": commitment.round_id,
            "proof_system": "Groth16" if self.zk_config is not None else "strict_replay",
        }
        return dict(self._commitment_payload)

    @staticmethod
    def _load_step_blob(store: ContentAddressedStore, reference):
        payload = store.get(reference)
        try:
            return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.6
            return torch.load(io.BytesIO(payload), map_location="cpu")

    def respond_to_challenge(self, challenge_data):
        if not isinstance(challenge_data, Challenge):
            raise TypeError("paper-protocol challenges must use the Challenge record")
        if self.recorded_trace is None or self.trace_recorder is None:
            raise RuntimeError("finalize_pol() must run before challenge response")
        if challenge_data.proof_mode != "strict_replay":
            raise ValueError("ZK challenges must be answered by the ZK-PoL prover")
        if self.zk_config is not None:
            raise PermissionError("ZK-enabled traces do not disclose compacted private replay data")
        if not bool(self.args.get("allow_private_replay", False)):
            raise PermissionError("private strict replay is disabled for this client")

        trace = self.recorded_trace.trace
        required = {trace.commitment.checkpoint_count - 1}
        for pair_index in challenge_data.pair_indices:
            required.update((pair_index, pair_index + 1))
        openings = {}
        for index in sorted(required):
            material = self.recorded_trace.checkpoints[index]
            openings[index] = CheckpointOpening(
                index=index,
                record=trace.checkpoints[index],
                merkle_proof=trace.checkpoint_proof(index),
                model_state=material.model_state,
                batch_data=material.batch_data,
                batch_labels=material.batch_labels,
                batch_indices=material.batch_indices,
                auxiliary=material.auxiliary,
            )
        witnesses = {}
        for pair_index in challenge_data.pair_indices:
            start_step = trace.checkpoints[pair_index].step
            end_step = trace.checkpoints[pair_index + 1].step
            step_payloads = [
                self._load_step_blob(
                    self.trace_recorder.store,
                    self.recorded_trace.steps[step].blob,
                )
                for step in range(start_step + 1, end_step + 1)
            ]
            witnesses[pair_index] = IntervalWitness(
                pair_index=pair_index,
                private_batches=tuple(
                    (payload["batch_data"], payload["batch_labels"])
                    for payload in step_payloads
                ),
                optimizer_state=self.recorded_trace.checkpoints[pair_index].optimizer_state,
                replay_metadata={
                    "step_evidence": [
                        self.recorded_trace.steps[step].blob.digest
                        for step in range(start_step + 1, end_step + 1)
                    ]
                },
            )
        return ChallengeResponse(
            challenge_id=challenge_data.challenge_id,
            commitment=trace.commitment,
            openings=openings,
            interval_witnesses=witnesses,
            uploaded_model_state={
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
        )

    def respond_to_zk_challenge(
        self,
        challenge: Challenge,
        *,
        backend: Groth16Backend,
    ):
        if challenge.proof_mode != "zk":
            raise ValueError("strict replay challenges use respond_to_challenge()")
        if self.recorded_trace is None or self.trace_recorder is None or self.zk_config is None:
            raise RuntimeError("a finalized ZK-enabled trace is required")
        return ZKPoLProver(
            backend,
            self.zk_config,
            store=self.trace_recorder.store,
        ).prove_challenge(recorded=self.recorded_trace, challenge=challenge)
