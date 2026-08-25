"""
Shared ZKP hashing and quantization utilities.
- Deterministic quantization from float tensors to field integers
- Poseidon(2)-fold hashing via Node helper (circomlibjs)
"""
from __future__ import annotations
import atexit
import logging
from pathlib import Path
from typing import List, Sequence

import torch
from polbfl.zk import PoseidonBridge

logger = logging.getLogger(__name__)

# bn128 field prime
FR_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
DEFAULT_SCALE = 10**6  # fixed-point scale for weights

REPO_ROOT = Path(__file__).resolve().parents[1]
POSEIDON_JS = REPO_ROOT / "analysis" / "poseidon_fold.js"
_POSEIDON_BRIDGE: PoseidonBridge | None = None


def _bridge() -> PoseidonBridge:
    global _POSEIDON_BRIDGE
    if _POSEIDON_BRIDGE is None:
        _POSEIDON_BRIDGE = PoseidonBridge(persistent=True)
        atexit.register(_POSEIDON_BRIDGE.close)
    return _POSEIDON_BRIDGE


def flatten_first_n(state_dict: dict, n: int) -> torch.Tensor:
    """Flatten parameters in a deterministic order (sorted keys), take first n scalars."""
    parts: List[torch.Tensor] = []
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if isinstance(v, torch.Tensor):
            parts.append(v.reshape(-1))
    if not parts:
        return torch.empty(0)
    flat = torch.cat(parts)
    if flat.numel() < n:
        # pad with zeros to match circuit size
        pad = torch.zeros(n - flat.numel(), dtype=flat.dtype, device=flat.device)
        flat = torch.cat([flat, pad])
    return flat[:n].detach().cpu()


def quantize_to_field(x: torch.Tensor, scale: int = DEFAULT_SCALE) -> List[int]:
    """Deterministically quantize to signed integers suitable for Fr.
    Strategy: round(x * scale) as Python int (can be negative). The circuit and JS Poseidon
    both work over Fr and will implicitly reduce modulo p consistently.
    """
    y = torch.round(x * scale).to(dtype=torch.int64).cpu().numpy().astype(object)
    res = [int(v) for v in y.tolist()]
    return res


def poseidon_fold(values: Sequence[int]) -> str:
    """Compute the exact Circomlib Poseidon(2) fold through one persistent bridge."""

    return _bridge().fold2(tuple(map(int, values)))


def poseidon_fold_many(
    rows: Sequence[Sequence[int]],
) -> tuple[str, ...]:
    """Batch independent Poseidon folds through one bridge request."""

    return _bridge().execute(
        tuple(
            {
                "kind": "fold2",
                "values": list(map(int, row)),
                "initial": "0",
            }
            for row in rows
        )
    )


def fold_indices(indices: Sequence[int]) -> str:
    return poseidon_fold(indices)


def fold_weights_from_state(state_dict: dict, n: int = 100, scale: int = DEFAULT_SCALE) -> str:
    flat = flatten_first_n(state_dict, n)
    q = quantize_to_field(flat, scale)
    return poseidon_fold(q)

