"""PyTorch recorder for private PoL trace and interval evidence."""

from __future__ import annotations

import copy
import io
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from polbfl.crypto import (
    MerkleTree,
    canonical_json_bytes,
    domain_hash,
    hash_batch,
    hash_batch_indices,
    hash_object,
)
from polbfl.protocol import PoLTrace, PoLTraceBuilder, RoundContext
from polbfl.storage import BlobRef, ContentAddressedStore
from polbfl.zk import (
    PADDED_DATA_INDEX,
    PoseidonBridge,
    ZKCircuitConfig,
    digest_to_field,
    interval_batch_commitment,
    pad_batch_indices,
    quantize,
    require_signed_range,
)


def _cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(_cpu_clone(value), buffer)
    return buffer.getvalue()


@dataclass(frozen=True)
class StepEvidence:
    step: int
    epoch: int
    batch_indices: tuple[int, ...]
    batch_digest: str
    indices_digest: str
    gradient_digest: str
    blob: BlobRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "batch_indices": list(self.batch_indices),
            "batch_digest": self.batch_digest,
            "indices_digest": self.indices_digest,
            "gradient_digest": self.gradient_digest,
            "blob": self.blob.to_dict(),
        }


@dataclass(frozen=True)
class CheckpointMaterial:
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    batch_data: Any
    batch_labels: Any
    batch_indices: tuple[int, ...]
    auxiliary: Mapping[str, Any]
    zk_private: Mapping[str, Any] | None
    blob: BlobRef


@dataclass(frozen=True)
class RecordedTrace:
    trace: PoLTrace
    data_root: str
    steps: Mapping[int, StepEvidence]
    checkpoints: Mapping[int, CheckpointMaterial]


class TorchPoLRecorder:
    """Record every private optimizer step and commit checkpoint intervals."""

    def __init__(
        self,
        context: RoundContext,
        store: ContentAddressedStore,
        *,
        sampling_seed: bytes,
        gradient_sample_rate: float = 0.01,
        zk_config: ZKCircuitConfig | None = None,
        poseidon_bridge: PoseidonBridge | None = None,
    ):
        if len(sampling_seed) < 32:
            raise ValueError("gradient sampling seed must contain at least 256 bits")
        if not 0 < gradient_sample_rate <= 1:
            raise ValueError("gradient sample rate must be in (0, 1]")
        self.context = context
        self.store = store
        self.sampling_seed = bytes(sampling_seed)
        self.gradient_sample_rate = float(gradient_sample_rate)
        self.zk_config = zk_config
        self.poseidon_bridge = poseidon_bridge
        if self.zk_config is not None:
            if self.context.checkpoint_interval != self.zk_config.steps:
                raise ValueError("checkpoint interval must match the ZK circuit step capacity")
            self.poseidon_bridge = self.poseidon_bridge or PoseidonBridge()
        self.builder = PoLTraceBuilder(context)
        self.steps: dict[int, StepEvidence] = {}
        self.checkpoints: dict[int, CheckpointMaterial] = {}
        self._pending_steps: list[StepEvidence] = []
        self._last_step = -1
        self._last_epoch = 0
        self._last_model_state: Mapping[str, Any] | None = None
        self._last_optimizer_state: Mapping[str, Any] | None = None
        self._last_model_ref: torch.nn.Module | None = None
        self._last_optimizer_ref: torch.optim.Optimizer | None = None
        self._last_batch_data: Any = []
        self._last_batch_labels: Any = []
        self._last_batch_indices: tuple[int, ...] = ()
        self._started = False
        self._sample_plan: dict[str, tuple[int, ...]] = {}
        self._selected_global_indices: tuple[int, ...] = ()
        self._circuit_plan: tuple[tuple[int, str, int], ...] = ()
        self._circuit_sample_plan_hash: str | None = None
        self._raw_sample_plan_digest: str | None = None
        self._previous_circuit_weights: tuple[int, ...] | None = None
        self._pending_zk_steps: list[Mapping[str, Any]] = []
        self._last_zk_checkpoint_weights: tuple[int, ...] | None = None
        self._module_inputs: dict[str, torch.Tensor] = {}
        self._module_grad_outputs: dict[str, torch.Tensor] = {}
        self._auxiliary_modules: dict[str, torch.nn.Module] = {}
        self._auxiliary_parameters: dict[str, torch.nn.Parameter] = {}
        self._conv_padded_inputs: dict[str, torch.Tensor] = {}
        self._batchnorm_normalized: dict[str, torch.Tensor] = {}
        self._hook_handles: list[Any] = []
        self._sample_index_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._previous_sampled_weights: Mapping[str, Any] | None = None
        self._previous_sampled_momentum: Mapping[str, Any] | None = None
        self.timings: dict[str, float] = {}

    def _add_timing(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + float(seconds)

    @staticmethod
    def _state_dict(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        state = model.state_dict()
        grouped: dict[tuple[torch.device, torch.dtype], list[tuple[str, torch.Tensor]]] = {}
        for name, value in state.items():
            grouped.setdefault((value.device, value.dtype), []).append((name, value))
        copied: dict[str, torch.Tensor] = {}
        for (device, _dtype), entries in grouped.items():
            if device.type == "cpu":
                for name, value in entries:
                    copied[name] = value.detach().clone()
                continue
            sizes = [value.numel() for _, value in entries]
            flat = torch.cat([value.detach().reshape(-1) for _, value in entries]).cpu()
            cursor = 0
            for (name, value), size in zip(entries, sizes):
                copied[name] = flat[cursor : cursor + size].reshape(value.shape).clone()
                cursor += size
        return {name: copied[name] for name in state}

    def _build_sample_plan(self, model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
        parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        total = sum(parameter.numel() for _, parameter in parameters)
        if total <= 0:
            raise ValueError("model has no trainable parameters")
        sample_count = max(1, int(math.ceil(total * self.gradient_sample_rate)))
        shared_sampling_context = {
            "protocol_version": self.context.protocol_version,
            "round_id": self.context.round_id,
            "model_id": self.context.model_id,
            "global_model_digest": self.context.global_model_digest,
            "optimizer": self.context.optimizer,
            "learning_rate": self.context.learning_rate,
            "local_epochs": self.context.local_epochs,
            "batch_size": self.context.batch_size,
            "checkpoint_interval": self.context.checkpoint_interval,
            "expected_steps": self.context.expected_steps,
        }
        seed_hex = domain_hash(
            "POLBFL_GRADIENT_SAMPLE_PLAN_V1",
            self.sampling_seed,
            canonical_json_bytes(shared_sampling_context),
        )
        rng = random.Random(int(seed_hex, 16))
        selected_global = sorted(rng.sample(range(total), min(sample_count, total)))
        self._selected_global_indices = tuple(selected_global)
        plan: dict[str, list[int]] = {name: [] for name, _ in parameters}
        cursor = 0
        selected_cursor = 0
        for name, parameter in parameters:
            end = cursor + parameter.numel()
            while selected_cursor < len(selected_global) and selected_global[selected_cursor] < end:
                plan[name].append(selected_global[selected_cursor] - cursor)
                selected_cursor += 1
            cursor = end
        return {name: tuple(indices) for name, indices in plan.items() if indices}

    def _build_circuit_plan(self, model: torch.nn.Module) -> tuple[tuple[int, str, int], ...]:
        if self.zk_config is None:
            return ()
        if len(self._selected_global_indices) < self.zk_config.sample_count:
            raise ValueError("the committed 1% gradient sample is smaller than the ZK circuit sample")
        count = self.zk_config.sample_count
        selected = tuple(
            self._selected_global_indices[(position * len(self._selected_global_indices)) // count]
            for position in range(count)
        )
        parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        locations: list[tuple[int, str, int]] = []
        parameter_position = 0
        offset = 0
        for global_index in selected:
            while global_index >= offset + parameters[parameter_position][1].numel():
                offset += parameters[parameter_position][1].numel()
                parameter_position += 1
            name, _parameter = parameters[parameter_position]
            locations.append((global_index, name, global_index - offset))
        if tuple(sorted(global_index for global_index, _, _ in locations)) != tuple(
            global_index for global_index, _, _ in locations
        ):
            raise AssertionError("ZK circuit plan must be strictly ordered")
        return tuple(locations)

    def _circuit_values_from_model(
        self,
        model: torch.nn.Module,
        *,
        gradients: bool,
    ) -> tuple[float, ...]:
        named = dict(model.named_parameters())
        values: list[torch.Tensor] = []
        for _global_index, name, local_index in self._circuit_plan:
            tensor = named[name].grad if gradients else named[name]
            if tensor is None:
                raise ValueError(f"missing gradient for ZK circuit parameter {name}")
            values.append(tensor.detach().reshape(-1)[local_index])
        if not values:
            raise ValueError("ZK circuit sample plan is empty")
        return tuple(float(value) for value in torch.stack(values).cpu().tolist())

    def _circuit_values_from_state(self, state: Mapping[str, Any]) -> tuple[float, ...]:
        values: list[float] = []
        for _global_index, name, local_index in self._circuit_plan:
            if name not in state:
                raise ValueError(f"checkpoint is missing ZK circuit parameter {name}")
            values.append(float(state[name].detach().reshape(-1)[local_index].item()))
        return tuple(values)

    def _quantize_values(self, values: Sequence[float]) -> tuple[int, ...]:
        if self.zk_config is None:
            raise AssertionError("ZK circuit configuration is not active")
        return tuple(
            quantize(value, scale=self.zk_config.scale, bits=self.zk_config.value_bits)
            for value in values
        )

    def _install_auxiliary_hooks(self, model: torch.nn.Module) -> None:
        if self.zk_config is None:
            return
        modules = dict(model.named_modules())
        self._auxiliary_parameters = dict(model.named_parameters())
        selected_names = {
            parameter_name.rsplit(".", 1)[0] if "." in parameter_name else ""
            for _global, parameter_name, _local in self._circuit_plan
        }

        def make_hook(module_name: str):
            def capture(_module, inputs, output):
                if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                    raise TypeError(f"unsupported ZK auxiliary module output: {module_name}")
                self._module_inputs[module_name] = inputs[0].detach()

                def capture_gradient(gradient):
                    self._module_grad_outputs[module_name] = gradient.detach()

                output.register_hook(capture_gradient)

            return capture

        supported = (
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
        )
        for module_name in sorted(selected_names):
            module = modules.get(module_name)
            if module is None or not isinstance(module, supported):
                raise TypeError(f"unsupported sampled ZK parameter module: {module_name}")
            self._auxiliary_modules[module_name] = module
            self._hook_handles.append(module.register_forward_hook(make_hook(module_name)))

    def _per_example_gradient_contributions(
        self,
        model: torch.nn.Module,
        *,
        parameter_name: str,
        local_index: int,
        expected_batch: int,
        expected_quantized_gradient: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        module_name, leaf_name = (
            parameter_name.rsplit(".", 1) if "." in parameter_name else ("", parameter_name)
        )
        module = self._auxiliary_modules[module_name]
        inputs = self._module_inputs.get(module_name)
        errors = self._module_grad_outputs.get(module_name)
        if inputs is None or errors is None:
            raise ValueError(f"missing activation/error capture for {parameter_name}")
        if inputs.shape[0] != expected_batch or errors.shape[0] != expected_batch:
            raise ValueError("captured activation batch differs from committed batch")

        if isinstance(module, torch.nn.Linear):
            if leaf_name == "weight":
                output_index = local_index // module.in_features
                input_index = local_index % module.in_features
                contribution = errors[..., output_index] * inputs[..., input_index]
                contribution = contribution.reshape(expected_batch, -1).sum(dim=1)
            elif leaf_name == "bias" and module.bias is not None:
                contribution = errors[..., local_index].reshape(expected_batch, -1).sum(dim=1)
            else:
                raise TypeError(f"unsupported Linear parameter: {parameter_name}")
        elif isinstance(module, torch.nn.Conv2d):
            if leaf_name == "weight":
                per_output = module.weight[0].numel()
                output_index = local_index // per_output
                within_output = local_index % per_output
                kernel_area = module.kernel_size[0] * module.kernel_size[1]
                input_within_group = within_output // kernel_area
                kernel_offset = within_output % kernel_area
                kernel_row = kernel_offset // module.kernel_size[1]
                kernel_column = kernel_offset % module.kernel_size[1]
                outputs_per_group = module.out_channels // module.groups
                group = output_index // outputs_per_group
                input_index = group * (module.in_channels // module.groups) + input_within_group
                output_errors = errors.reshape(expected_batch, module.out_channels, -1)
                padded = self._conv_padded_inputs.get(module_name)
                if padded is None:
                    padding_height, padding_width = module.padding
                    padded = torch.nn.functional.pad(
                        inputs,
                        (padding_width, padding_width, padding_height, padding_height),
                    )
                    self._conv_padded_inputs[module_name] = padded
                output_height, output_width = errors.shape[-2:]
                start_height = kernel_row * module.dilation[0]
                start_width = kernel_column * module.dilation[1]
                end_height = start_height + module.stride[0] * (output_height - 1) + 1
                end_width = start_width + module.stride[1] * (output_width - 1) + 1
                selected_inputs = padded[
                    :,
                    input_index,
                    start_height:end_height:module.stride[0],
                    start_width:end_width:module.stride[1],
                ]
                if selected_inputs.shape[-2:] != (output_height, output_width):
                    raise ValueError("direct convolution contribution shape mismatch")
                contribution = (
                    selected_inputs.reshape(expected_batch, -1)
                    * output_errors[:, output_index, :]
                ).sum(dim=1)
            elif leaf_name == "bias" and module.bias is not None:
                contribution = errors[:, local_index].reshape(expected_batch, -1).sum(dim=1)
            else:
                raise TypeError(f"unsupported Conv2d parameter: {parameter_name}")
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            channel = local_index
            if leaf_name == "weight":
                normalized = self._batchnorm_normalized.get(module_name)
                if normalized is None:
                    reduction = (0,) + tuple(range(2, inputs.ndim))
                    mean = inputs.mean(dim=reduction, keepdim=True)
                    variance = inputs.var(dim=reduction, unbiased=False, keepdim=True)
                    normalized = (inputs - mean) / torch.sqrt(variance + module.eps)
                    self._batchnorm_normalized[module_name] = normalized
                contribution = (errors[:, channel] * normalized[:, channel]).reshape(expected_batch, -1).sum(dim=1)
            elif leaf_name == "bias":
                contribution = errors[:, channel].reshape(expected_batch, -1).sum(dim=1)
            else:
                raise TypeError(f"unsupported BatchNorm parameter: {parameter_name}")
        else:  # pragma: no cover - guarded during hook installation
            raise TypeError(f"unsupported ZK parameter module: {type(module).__name__}")

        parameter = self._auxiliary_parameters[parameter_name]
        if parameter.grad is None:
            raise ValueError(f"missing final sampled gradient for {parameter_name}")
        # CUDA convolution backward and the explicit contribution calculation
        # use different deterministic reduction trees.  Reconcile the sampled
        # terms in float64 so FP32 cancellation cannot erase a small gradient
        # that remains representable in the circuit's fixed-point domain.
        contribution = contribution.to(torch.float64)
        target = (
            parameter.grad.detach()
            .reshape(-1)[local_index]
            .to(torch.float64)
        )
        raw_sum = contribution.sum()
        nonzero_sum = raw_sum.abs() > 1e-20
        safe_sum = torch.where(nonzero_sum, raw_sum, torch.ones_like(raw_sum))
        contribution = torch.where(
            nonzero_sum,
            contribution * (target / safe_sum),
            torch.zeros_like(contribution),
        )
        invalid = torch.logical_and(
            ~nonzero_sum,
            torch.as_tensor(
                int(expected_quantized_gradient) != 0,
                dtype=torch.bool,
                device=target.device,
            ),
        )
        return contribution.detach(), invalid.detach()

    def _quantized_gradient_contributions(
        self,
        model: torch.nn.Module,
        gradient_values: tuple[int, ...],
        *,
        actual_batch_size: int,
    ) -> tuple[tuple[int, ...], ...]:
        if self.zk_config is None:
            raise AssertionError("ZK contribution encoding requires a circuit configuration")
        raw_rows: list[torch.Tensor] = []
        invalid_rows: list[torch.Tensor] = []
        for (_global, name, local_index), gradient in zip(self._circuit_plan, gradient_values):
            values, invalid = self._per_example_gradient_contributions(
                model,
                parameter_name=name,
                local_index=local_index,
                expected_batch=actual_batch_size,
                expected_quantized_gradient=int(gradient),
            )
            raw_rows.append(values)
            invalid_rows.append(invalid)
        contribution_matrix = torch.stack(raw_rows).cpu()
        invalid_matrix = torch.stack(invalid_rows).cpu()
        invalid_positions = [
            index
            for index, invalid in enumerate(invalid_matrix.tolist())
            if invalid
        ]
        if invalid_positions:
            invalid_parameters = [
                self._circuit_plan[index][1] for index in invalid_positions
            ]
            raise ValueError(
                "captured per-example gradient contributions are inconsistent for "
                + ",".join(invalid_parameters)
            )
        if not bool(torch.isfinite(contribution_matrix).all()):
            raise ValueError("captured per-example gradient contributions are non-finite")

        encoded: list[tuple[int, ...]] = []
        for values, gradient in zip(contribution_matrix.tolist(), gradient_values):
            contributions = [
                quantize(float(value), scale=self.zk_config.scale, bits=self.zk_config.value_bits)
                for value in values
            ]
            if contributions:
                contributions[-1] += int(gradient) - sum(contributions)
            require_signed_range(contributions, bits=self.zk_config.value_bits)
            encoded.append(
                tuple(contributions)
                + (0,) * (self.zk_config.batch_terms - len(contributions))
            )
        return tuple(encoded)

    @staticmethod
    def _take(tensor: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        flat = tensor.detach().reshape(-1)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=flat.device)
        return flat.index_select(0, index_tensor).cpu().clone()

    def _sample_parameters(self, model: torch.nn.Module) -> Mapping[str, Any]:
        named = dict(model.named_parameters())
        return {
            name: {
                "indices": indices,
                "shape": tuple(named[name].shape),
                "values": self._take(named[name], indices),
            }
            for name, indices in self._sample_plan.items()
        }

    def _sample_gradients(self, model: torch.nn.Module) -> Mapping[str, Any]:
        named = dict(model.named_parameters())
        sampled = {}
        for name, indices in self._sample_plan.items():
            gradient = named[name].grad
            if gradient is None:
                raise ValueError(f"missing gradient for sampled parameter {name}")
            sampled[name] = {
                "indices": indices,
                "shape": tuple(gradient.shape),
                "values": self._take(gradient, indices),
            }
        return sampled

    def _sample_gradient_vector(self, model: torch.nn.Module) -> torch.Tensor:
        named = dict(model.named_parameters())
        pieces = []
        for name, indices in self._sample_plan.items():
            gradient = named[name].grad
            if gradient is None:
                raise ValueError(f"missing gradient for sampled parameter {name}")
            flat = gradient.detach().reshape(-1)
            cache_key = (name, str(flat.device))
            index_tensor = self._sample_index_cache.get(cache_key)
            if index_tensor is None:
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=flat.device)
                self._sample_index_cache[cache_key] = index_tensor
            pieces.append(flat.index_select(0, index_tensor))
        if not pieces:
            raise ValueError("gradient sample plan is empty")
        return torch.cat(pieces).cpu()

    def _sample_momentum(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Mapping[str, Any]:
        named = dict(model.named_parameters())
        sampled = {}
        for name, indices in self._sample_plan.items():
            parameter = named[name]
            buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if buffer is None:
                values = torch.zeros(len(indices), dtype=parameter.dtype)
            else:
                values = self._take(buffer, indices)
            sampled[name] = {"indices": indices, "values": values}
        return sampled

    def start(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        timestamp_ns: int | None = None,
    ) -> None:
        if self._started:
            raise RuntimeError("trace recorder already started")
        started_start = time.perf_counter()
        timestamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        self._sample_plan = self._build_sample_plan(model)
        self._raw_sample_plan_digest = hash_object(self._sample_plan)
        self._circuit_plan = self._build_circuit_plan(model)
        if self.zk_config is not None:
            if any(global_index >= 2**32 for global_index, _, _ in self._circuit_plan):
                raise OverflowError("ZK circuit sample indices must fit in 32 bits")
            if self.poseidon_bridge is None:
                raise AssertionError("ZK Poseidon bridge was not initialized")
            self._circuit_sample_plan_hash = self.poseidon_bridge.fold2(
                [global_index for global_index, _, _ in self._circuit_plan]
            )
            self._previous_circuit_weights = self._quantize_values(
                self._circuit_values_from_model(model, gradients=False)
            )
            self._install_auxiliary_hooks(model)
        self._previous_sampled_weights = (
            self._sample_parameters(model) if self.zk_config is None else {}
        )
        self._previous_sampled_momentum = (
            self._sample_momentum(model, optimizer) if self.zk_config is None else {}
        )
        self._last_model_state = (
            self._state_dict(model) if self.zk_config is None else {}
        )
        self._last_optimizer_state = _cpu_clone(optimizer.state_dict())
        self._last_model_ref = model
        self._last_optimizer_ref = optimizer
        self._append_checkpoint(
            step=0,
            epoch=0,
            timestamp_ns=timestamp,
            model_state=self._last_model_state,
            optimizer_state=self._last_optimizer_state,
            batch_data=[],
            batch_labels=[],
            batch_indices=(),
            interval_steps=(),
            interval_zk_steps=(),
        )
        self._last_step = 0
        self._started = True
        self._add_timing("start_seconds", time.perf_counter() - started_start)

    def record_optimizer_step(
        self,
        *,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_data: torch.Tensor,
        batch_labels: torch.Tensor,
        batch_indices: Sequence[int],
        activations: Any = None,
        timestamp_ns: int | None = None,
    ) -> None:
        if not self._started:
            raise RuntimeError("start() must be called before recording optimizer steps")
        started_step = time.perf_counter()
        if step != self._last_step + 1:
            raise ValueError("optimizer steps must be recorded without gaps")
        timestamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        indices = tuple(int(index) for index in batch_indices)
        if self._previous_sampled_weights is None or self._previous_sampled_momentum is None:
            raise AssertionError("gradient sampling state was not initialized")
        started_gradient_sample = time.perf_counter()
        gradients = (
            self._sample_gradients(model)
            if self.zk_config is None
            else {
                "sample_plan_digest": self._raw_sample_plan_digest,
                "values": self._sample_gradient_vector(model),
            }
        )
        self._add_timing(
            "gradient_sample_seconds",
            time.perf_counter() - started_gradient_sample,
        )
        sampled_weights_after = self._sample_parameters(model) if self.zk_config is None else {}
        sampled_momentum_after = (
            self._sample_momentum(model, optimizer) if self.zk_config is None else {}
        )
        zk_step: Mapping[str, Any] | None = None
        if self.zk_config is not None:
            started_transition = time.perf_counter()
            if self._previous_circuit_weights is None:
                raise AssertionError("ZK circuit start weights were not initialized")
            after_values = self._quantize_values(
                self._circuit_values_from_model(model, gradients=False)
            )
            gradient_values = self._quantize_values(
                self._circuit_values_from_model(model, gradients=True)
            )
            learning_rate = int(round(self.context.learning_rate * self.zk_config.learning_rate_scale))
            if not 0 < learning_rate < 2**32:
                raise ValueError("fixed-point learning rate must fit in 32 bits")
            rounding = tuple(
                self.zk_config.scale * (after - before) + learning_rate * gradient
                for before, after, gradient in zip(
                    self._previous_circuit_weights,
                    after_values,
                    gradient_values,
                )
            )
            require_signed_range(rounding, bits=self.zk_config.value_bits)
            if any(abs(value) > self.zk_config.max_rounding_error for value in rounding):
                raise ValueError("optimizer transition exceeds the strict pairwise ZK tolerance")
            padded_indices = pad_batch_indices(indices, batch_terms=self.zk_config.batch_terms)
            gradient_contributions = self._quantized_gradient_contributions(
                model,
                gradient_values,
                actual_batch_size=len(indices),
            )
            activation_factors = tuple(
                (self.zk_config.scale,) * len(indices)
                + (0,) * (self.zk_config.batch_terms - len(indices))
                for _ in range(self.zk_config.sample_count)
            )
            error_factors = gradient_contributions
            zk_step = {
                "schema_version": 1,
                "sample_indices": tuple(index for index, _, _ in self._circuit_plan),
                "weights_before": self._previous_circuit_weights,
                "weights_after": after_values,
                "gradients": gradient_values,
                "rounding": rounding,
                "data_indices": padded_indices,
                "actual_batch_size": len(indices),
                "activation_factors": activation_factors,
                "error_factors": error_factors,
            }
            self._module_inputs.clear()
            self._module_grad_outputs.clear()
            self._conv_padded_inputs.clear()
            self._batchnorm_normalized.clear()
            self._add_timing(
                "zk_transition_seconds",
                time.perf_counter() - started_transition,
            )
        started_hashes = time.perf_counter()
        batch_digest = hash_batch(batch_data, batch_labels)
        indices_digest = hash_batch_indices(indices)
        gradient_digest = hash_object(gradients)
        self._add_timing("step_hash_seconds", time.perf_counter() - started_hashes)
        payload = {
            "protocol_version": self.context.protocol_version,
            "context_digest": self.context.digest,
            "step": step,
            "epoch": epoch,
            "timestamp_ns": timestamp,
            "batch_indices": indices,
            "batch_digest": batch_digest,
            "indices_digest": indices_digest,
            "gradient_digest": gradient_digest,
            "gradient_sample_rate": self.gradient_sample_rate,
            "zk_witness": zk_step,
        }
        if self.zk_config is None:
            payload.update(
                {
                    "batch_data": _cpu_clone(batch_data),
                    "batch_labels": _cpu_clone(batch_labels),
                    "sample_plan": self._sample_plan,
                    "sampled_weights_before": self._previous_sampled_weights,
                    "sampled_weights_after": sampled_weights_after,
                    "gradients": gradients,
                    "sampled_momentum_before": self._previous_sampled_momentum,
                    "sampled_momentum_after": sampled_momentum_after,
                    "activations": _cpu_clone(activations),
                }
            )
        elif activations is not None:
            payload["activations_digest"] = hash_object(activations)
        started_store = time.perf_counter()
        blob = self.store.put(_torch_bytes(payload))
        self._add_timing("step_store_seconds", time.perf_counter() - started_store)
        evidence = StepEvidence(
            step,
            epoch,
            indices,
            batch_digest,
            indices_digest,
            gradient_digest,
            blob,
        )
        self.steps[step] = evidence
        self._pending_steps.append(evidence)
        if zk_step is not None:
            self._pending_zk_steps.append(zk_step)
        self._last_step = step
        self._last_epoch = epoch
        self._last_model_ref = model
        self._last_optimizer_ref = optimizer
        if self.zk_config is None:
            self._last_model_state = self._state_dict(model)
            self._last_optimizer_state = _cpu_clone(optimizer.state_dict())
        self._last_batch_data = _cpu_clone(batch_data)
        self._last_batch_labels = _cpu_clone(batch_labels)
        self._last_batch_indices = indices
        self._previous_sampled_weights = sampled_weights_after
        self._previous_sampled_momentum = sampled_momentum_after
        if zk_step is not None:
            self._previous_circuit_weights = tuple(int(value) for value in zk_step["weights_after"])

        if step % self.context.checkpoint_interval == 0:
            if self.zk_config is not None:
                is_final = (
                    self.context.expected_steps is None
                    or step == self.context.expected_steps
                )
                self._last_model_state = self._state_dict(model) if is_final else {}
                self._last_optimizer_state = _cpu_clone(optimizer.state_dict())
            started_checkpoint = time.perf_counter()
            self._append_checkpoint(
                step=step,
                epoch=epoch,
                timestamp_ns=timestamp,
                model_state=self._last_model_state,
                optimizer_state=self._last_optimizer_state,
                batch_data=self._last_batch_data,
                batch_labels=self._last_batch_labels,
                batch_indices=self._last_batch_indices,
                interval_steps=tuple(self._pending_steps),
                interval_zk_steps=tuple(self._pending_zk_steps),
            )
            self._pending_steps.clear()
            self._pending_zk_steps.clear()
            self._add_timing(
                "checkpoint_seconds",
                time.perf_counter() - started_checkpoint,
            )
        self._add_timing("record_step_seconds", time.perf_counter() - started_step)

    def _inactive_zk_step(self, weights: tuple[int, ...]) -> Mapping[str, Any]:
        if self.zk_config is None:
            raise AssertionError("ZK circuit configuration is not active")
        zeros = (0,) * self.zk_config.sample_count
        zero_terms = tuple(
            (0,) * self.zk_config.batch_terms for _ in range(self.zk_config.sample_count)
        )
        return {
            "schema_version": 1,
            "sample_indices": tuple(index for index, _, _ in self._circuit_plan),
            "weights_before": weights,
            "weights_after": weights,
            "gradients": zeros,
            "rounding": zeros,
            "data_indices": (PADDED_DATA_INDEX,) * self.zk_config.batch_terms,
            "actual_batch_size": 0,
            "activation_factors": zero_terms,
            "error_factors": zero_terms,
        }

    def _zk_checkpoint_metadata(
        self,
        *,
        current_weights: tuple[int, ...],
        interval_steps: tuple[StepEvidence, ...],
        interval_zk_steps: tuple[Mapping[str, Any], ...],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if self.zk_config is None or self.poseidon_bridge is None:
            raise AssertionError("ZK checkpoint metadata requested without a configured circuit")
        if self._circuit_sample_plan_hash is None:
            raise AssertionError("ZK circuit sample plan has not been committed")
        if self._raw_sample_plan_digest is None:
            raise AssertionError("raw gradient sample plan has not been committed")
        if len(interval_steps) != len(interval_zk_steps):
            raise ValueError("public step evidence and private ZK steps differ")
        if len(interval_zk_steps) > self.zk_config.steps:
            raise ValueError("checkpoint interval exceeds the ZK circuit capacity")

        weight_rows = [
            [global_index, value]
            for (global_index, _name, _local_index), value in zip(
                self._circuit_plan,
                current_weights,
            )
        ]
        operations: list[Mapping[str, Any]] = [
            {
                "kind": "fold3",
                "rows": weight_rows,
                "initial": str(digest_to_field(self.context.digest)),
            }
        ]

        padded_steps = list(interval_zk_steps)
        if interval_zk_steps:
            if self._last_zk_checkpoint_weights is None:
                raise AssertionError("missing preceding ZK checkpoint weights")
            if tuple(interval_zk_steps[0]["weights_before"]) != self._last_zk_checkpoint_weights:
                raise ValueError("ZK interval does not start at the preceding checkpoint")
            for left, right in zip(interval_zk_steps, interval_zk_steps[1:]):
                if tuple(left["weights_after"]) != tuple(right["weights_before"]):
                    raise ValueError("ZK step trajectory is not contiguous")
            if tuple(interval_zk_steps[-1]["weights_after"]) != current_weights:
                raise ValueError("ZK interval does not end at the checkpoint")
            while len(padded_steps) < self.zk_config.steps:
                padded_steps.append(self._inactive_zk_step(current_weights))

            gradients = [
                int(value)
                for zk_step in padded_steps
                for value in zk_step["gradients"]
            ]
            data_indices = [
                int(value)
                for zk_step in padded_steps
                for value in zk_step["data_indices"]
            ]
            auxiliary_rows = [
                [
                    int(zk_step["activation_factors"][sample][term]),
                    int(zk_step["error_factors"][sample][term]),
                ]
                for zk_step in padded_steps
                for sample in range(self.zk_config.sample_count)
                for term in range(self.zk_config.batch_terms)
            ]
            operations.extend(
                (
                    {
                        "kind": "fold2",
                        "values": gradients,
                        "initial": str(digest_to_field(self.context.digest)),
                    },
                    {
                        "kind": "fold2",
                        "values": data_indices,
                        "initial": str(digest_to_field(self.context.digest)),
                    },
                    {
                        "kind": "fold_pair_chunks",
                        "rows": auxiliary_rows,
                        "pairs_per_chunk": self.zk_config.auxiliary_pairs_per_chunk,
                        "initial": str(digest_to_field(self.context.digest)),
                    },
                )
            )

            start_weights = tuple(int(value) for value in interval_zk_steps[0]["weights_before"])
            distance_squared = sum(
                (end - start) ** 2 for start, end in zip(start_weights, current_weights)
            )
            if distance_squared > self.zk_config.max_distance_squared:
                raise ValueError("sampled checkpoint update exceeds the ZK L2 bound")
            cumulative_rounding_squared = sum(
                sum(int(step["rounding"][sample]) for step in padded_steps) ** 2
                for sample in range(self.zk_config.sample_count)
            )
            if cumulative_rounding_squared > self.zk_config.max_cumulative_rounding_error_squared:
                raise ValueError("ZK interval exceeds the cumulative final tolerance")

        started_poseidon = time.perf_counter()
        poseidon_results = self.poseidon_bridge.execute(operations)
        self._add_timing("poseidon_seconds", time.perf_counter() - started_poseidon)
        public = {
            "schema_version": 1,
            "context_hash": str(digest_to_field(self.context.digest)),
            "sample_plan_hash": self._circuit_sample_plan_hash,
            "raw_sample_plan_digest": self._raw_sample_plan_digest,
            "gradient_sample_rate": self.gradient_sample_rate,
            "sampled_weights_hash": poseidon_results[0],
            "circuit": {
                "sample_count": self.zk_config.sample_count,
                "steps": self.zk_config.steps,
                "batch_terms": self.zk_config.batch_terms,
                "value_bits": self.zk_config.value_bits,
            },
        }
        if interval_zk_steps:
            batch_commitment = interval_batch_commitment(
                evidence.to_dict() for evidence in interval_steps
            )
            learning_rate = int(round(self.context.learning_rate * self.zk_config.scale))
            public["interval"] = {
                "active_step_count": len(interval_zk_steps),
                "gradients_hash": poseidon_results[1],
                "data_indices_hash": poseidon_results[2],
                "auxiliary_hash": poseidon_results[3],
                "batch_commitment_hash": str(digest_to_field(batch_commitment)),
                "scale": self.zk_config.scale,
                "learning_rate": learning_rate,
                "max_distance_squared": self.zk_config.max_distance_squared,
                "max_rounding_error": self.zk_config.max_rounding_error,
                "max_cumulative_rounding_error_squared": (
                    self.zk_config.max_cumulative_rounding_error_squared
                ),
            }
        private = {
            "schema_version": 1,
            "sample_indices": tuple(index for index, _, _ in self._circuit_plan),
            "sampled_weights": current_weights,
        }
        self._last_zk_checkpoint_weights = current_weights
        return public, private

    def _append_checkpoint(
        self,
        *,
        step: int,
        epoch: int,
        timestamp_ns: int,
        model_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        batch_data: Any,
        batch_labels: Any,
        batch_indices: tuple[int, ...],
        interval_steps: tuple[StepEvidence, ...],
        interval_zk_steps: tuple[Mapping[str, Any], ...],
    ) -> None:
        optimizer_state_digest = hash_object(optimizer_state)
        auxiliary = {
            "interval_start_step": self.builder._records[-1].step if self.builder._records else 0,
            "interval_end_step": step,
            "optimizer_state_digest": optimizer_state_digest,
            "step_evidence": [evidence.to_dict() for evidence in interval_steps],
        }
        zk_private: Mapping[str, Any] | None = None
        precomputed_model_digest: str | None = None
        if self.zk_config is not None:
            if self._previous_circuit_weights is None:
                raise AssertionError("missing sampled checkpoint weights")
            zk_public, zk_private = self._zk_checkpoint_metadata(
                current_weights=self._previous_circuit_weights,
                interval_steps=interval_steps,
                interval_zk_steps=interval_zk_steps,
            )
            auxiliary["zk"] = zk_public
            final_full_state = bool(model_state) and (
                self.context.expected_steps is None
                or step == self.context.expected_steps
            )
            if final_full_state:
                auxiliary["model_commitment"] = "full_model_state_v1"
            else:
                auxiliary["model_commitment"] = "zk_sampled_parameters_v1"
                precomputed_model_digest = domain_hash(
                    "POLBFL_SAMPLED_MODEL_CHECKPOINT_V1",
                    bytes.fromhex(self.context.digest),
                    str(zk_public["sample_plan_hash"]),
                    str(zk_public["sampled_weights_hash"]),
                )
        checkpoint_payload = {
            "model_state": _cpu_clone(model_state) if self.zk_config is None else {},
            "optimizer_state": _cpu_clone(optimizer_state) if self.zk_config is None else {},
            "batch_data": _cpu_clone(batch_data) if self.zk_config is None else [],
            "batch_labels": _cpu_clone(batch_labels) if self.zk_config is None else [],
            "batch_indices": batch_indices,
            "auxiliary": auxiliary,
            "zk_private": zk_private,
        }
        blob = self.store.put(_torch_bytes(checkpoint_payload))
        self.builder.append_checkpoint(
            step=step,
            epoch=epoch,
            timestamp_ns=timestamp_ns,
            model_state=model_state,
            batch_data=batch_data,
            batch_labels=batch_labels,
            batch_indices=batch_indices,
            auxiliary=auxiliary,
            precomputed_model_digest=precomputed_model_digest,
        )
        index = len(self.builder._records) - 1
        self.checkpoints[index] = CheckpointMaterial(
            model_state=_cpu_clone(model_state) if self.zk_config is None else {},
            optimizer_state=_cpu_clone(optimizer_state) if self.zk_config is None else {},
            batch_data=_cpu_clone(batch_data) if self.zk_config is None else [],
            batch_labels=_cpu_clone(batch_labels) if self.zk_config is None else [],
            batch_indices=batch_indices,
            auxiliary=auxiliary,
            zk_private=zk_private,
            blob=blob,
        )

    def finalize(self, *, timestamp_ns: int | None = None) -> RecordedTrace:
        started_finalize = time.perf_counter()
        if not self._started or self._last_model_state is None or self._last_optimizer_state is None:
            raise RuntimeError("cannot finalize an unstarted trace")
        if self._pending_steps:
            if self.zk_config is not None:
                if self._last_model_ref is None or self._last_optimizer_ref is None:
                    raise AssertionError("missing live model for final ZK checkpoint")
                self._last_model_state = self._state_dict(self._last_model_ref)
                self._last_optimizer_state = _cpu_clone(self._last_optimizer_ref.state_dict())
            timestamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
            self._append_checkpoint(
                step=self._last_step,
                epoch=self._last_epoch,
                timestamp_ns=timestamp,
                model_state=self._last_model_state,
                optimizer_state=self._last_optimizer_state,
                batch_data=self._last_batch_data,
                batch_labels=self._last_batch_labels,
                batch_indices=self._last_batch_indices,
                interval_steps=tuple(self._pending_steps),
                interval_zk_steps=tuple(self._pending_zk_steps),
            )
            self._pending_steps.clear()
            self._pending_zk_steps.clear()
        trace = self.builder.finalize()
        if len(self.checkpoints) != trace.commitment.checkpoint_count:
            raise AssertionError("checkpoint material count does not match committed trace")
        if not self.steps:
            raise ValueError("a finalized training trace requires at least one optimizer step")
        data_root = MerkleTree(evidence.batch_digest for evidence in self.steps.values()).root
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._module_inputs.clear()
        self._module_grad_outputs.clear()
        self._auxiliary_modules.clear()
        self._auxiliary_parameters.clear()
        self._conv_padded_inputs.clear()
        self._batchnorm_normalized.clear()
        self._sample_index_cache.clear()
        recorded = RecordedTrace(
            trace=trace,
            data_root=data_root,
            steps=dict(self.steps),
            checkpoints=dict(self.checkpoints),
        )
        self._add_timing("finalize_seconds", time.perf_counter() - started_finalize)
        return recorded
