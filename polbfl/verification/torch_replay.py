"""Deterministic PyTorch replay backend for the reference verifier."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from polbfl.protocol import RoundContext
from polbfl.verification.strict import CheckpointOpening, IntervalWitness


ModelFactory = Callable[[], torch.nn.Module]
CriterionFactory = Callable[[], torch.nn.Module]


@dataclass(frozen=True)
class TorchSGDReplayConfig:
    model_factory: ModelFactory
    criterion_factory: CriterionFactory
    device: str = "cpu"
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False


class TorchSGDReplay:
    """Replay exact ordered SGD batches from a committed checkpoint."""

    def __init__(self, config: TorchSGDReplayConfig):
        self.config = config
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    @staticmethod
    def _unpack_batch(raw_batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(raw_batch, Mapping):
            data = raw_batch.get("data")
            labels = raw_batch.get("labels")
        elif isinstance(raw_batch, (tuple, list)) and len(raw_batch) >= 2:
            data, labels = raw_batch[0], raw_batch[1]
        else:
            raise TypeError("replay batches must contain data and labels")
        if not torch.is_tensor(data) or not torch.is_tensor(labels):
            raise TypeError("replay data and labels must be tensors")
        return data, labels

    def __call__(
        self,
        context: RoundContext,
        start: CheckpointOpening,
        end: CheckpointOpening,
        witness: IntervalWitness,
    ) -> Mapping[str, torch.Tensor]:
        expected_steps = end.record.step - start.record.step
        if expected_steps <= 0 or len(witness.private_batches) != expected_steps:
            raise ValueError("private replay batch count does not match checkpoint step distance")
        if not context.optimizer.upper().startswith("SGD"):
            raise ValueError("paper reference replay requires SGD")

        device = torch.device(self.config.device)
        model = self.config.model_factory().to(device)
        model.load_state_dict(start.model_state, strict=True)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(context.learning_rate),
            momentum=float(self.config.momentum),
            dampening=float(self.config.dampening),
            weight_decay=float(self.config.weight_decay),
            nesterov=bool(self.config.nesterov),
        )
        if witness.optimizer_state:
            optimizer.load_state_dict(copy.deepcopy(witness.optimizer_state))
            self._move_optimizer_state(optimizer, device)
        criterion = self.config.criterion_factory().to(device)

        deterministic_before = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for raw_batch in witness.private_batches:
                data, labels = self._unpack_batch(raw_batch)
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data), labels)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite loss during strict replay")
                loss.backward()
                optimizer.step()
        finally:
            torch.use_deterministic_algorithms(deterministic_before)

        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
