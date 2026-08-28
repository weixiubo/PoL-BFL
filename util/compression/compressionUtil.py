"""Model-state quantization and magnitude-pruning utilities."""

from collections import OrderedDict

import torch

from .baseCompression import ModelCompression


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        raise ValueError("quantization scale must be finite and positive")
    return scale


def quantify_encode(
    model_state_dict: OrderedDict,
    scale: float = 10.0,
) -> OrderedDict:
    """Return a fixed-scale signed-int8 copy of a model state dictionary."""
    scale = _validate_scale(scale)
    encoded = OrderedDict()
    for key, value in model_state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point():
            encoded[key] = torch.round(value.detach() * scale).clamp(-128, 127).to(
                torch.int8
            )
        else:
            encoded[key] = value.detach().clone() if torch.is_tensor(value) else value
    return encoded


def quantify_decode(
    model_compressed_state_dict: OrderedDict,
    scale: float = 10.0,
) -> OrderedDict:
    """Decode a fixed-scale int8 model state without mutating the input."""
    scale = _validate_scale(scale)
    decoded = OrderedDict()
    for key, value in model_compressed_state_dict.items():
        if torch.is_tensor(value) and value.dtype == torch.int8:
            decoded[key] = value.to(torch.float32) / scale
        else:
            decoded[key] = value.detach().clone() if torch.is_tensor(value) else value
    return decoded


def magnitude_prune(
    model_state_dict: OrderedDict,
    fraction: float,
) -> OrderedDict:
    """Zero the smallest-magnitude fraction of each floating tensor."""
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("pruning fraction must be in [0, 1)")
    pruned = OrderedDict()
    for key, value in model_state_dict.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            pruned[key] = value.detach().clone() if torch.is_tensor(value) else value
            continue
        result = value.detach().clone()
        remove = int(result.numel() * fraction)
        if remove > 0:
            flat = result.abs().reshape(-1)
            indices = torch.topk(flat, k=remove, largest=False).indices
            result.reshape(-1)[indices] = 0
        pruned[key] = result
    return pruned


class Int8Quantizer(ModelCompression):
    """Per-tensor symmetric int8 codec with retained decoding scales."""

    def __init__(self) -> None:
        self.scales = OrderedDict()

    def encode(self, model_state_dict: OrderedDict) -> OrderedDict:
        encoded = OrderedDict()
        self.scales = OrderedDict()
        for key, value in model_state_dict.items():
            if torch.is_tensor(value) and value.is_floating_point():
                maximum = float(value.detach().abs().max().item())
                scale = maximum / 127.0 if maximum > 0 else 1.0
                self.scales[key] = scale
                encoded[key] = torch.round(value.detach() / scale).clamp(-127, 127).to(
                    torch.int8
                )
            else:
                encoded[key] = value.detach().clone() if torch.is_tensor(value) else value
        return encoded

    def decode(self, compressed_state_dict: OrderedDict) -> OrderedDict:
        decoded = OrderedDict()
        for key, value in compressed_state_dict.items():
            if torch.is_tensor(value) and value.dtype == torch.int8:
                if key not in self.scales:
                    raise ValueError(f"missing quantization scale for {key!r}")
                decoded[key] = value.to(torch.float32) * self.scales[key]
            else:
                decoded[key] = value.detach().clone() if torch.is_tensor(value) else value
        return decoded
