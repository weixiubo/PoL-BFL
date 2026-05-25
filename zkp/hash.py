"""
Shared ZKP hashing and quantization utilities.
- Deterministic quantization from float tensors to field integers
- Poseidon(2)-fold hashing via Node helper (circomlibjs)
"""
from __future__ import annotations
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Sequence

import torch

logger = logging.getLogger(__name__)

# bn128 field prime
FR_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
DEFAULT_SCALE = 10**6  # fixed-point scale for weights

REPO_ROOT = Path(__file__).resolve().parents[1]
POSEIDON_JS = REPO_ROOT / 'analysis' / 'poseidon_fold.js'


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
    """Compute Poseidon(2) fold hash by invoking Node helper. Returns decimal string.
    Requires circomlibjs in Node environment.
    """
    if not POSEIDON_JS.exists():
        raise FileNotFoundError(f"Missing Node helper: {POSEIDON_JS}")
    try:
        proc = subprocess.run(
            ['node', str(POSEIDON_JS), json.dumps(list(map(int, values)))],
            check=True, capture_output=True, text=True
        )
        return proc.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"poseidon_fold failed: {e.stderr}")
        raise


def fold_indices(indices: Sequence[int]) -> str:
    return poseidon_fold(indices)


def fold_weights_from_state(state_dict: dict, n: int = 100, scale: int = DEFAULT_SCALE) -> str:
    flat = flatten_first_n(state_dict, n)
    q = quantize_to_field(flat, scale)
    return poseidon_fold(q)

