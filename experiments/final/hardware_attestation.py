"""Source-bound CUDA hardware attestation and numerical probes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch


def gpu_attestation(index: int) -> dict[str, Any]:
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("GPU attestation index is unavailable")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    fields = [value.strip() for value in query.split(",")]
    if len(fields) != 6:
        raise RuntimeError("nvidia-smi attestation output is incomplete")
    payload = {
        "index": int(fields[0]),
        "name": fields[1],
        "uuid": fields[2],
        "pci_bus_id": fields[3],
        "memory_mib": int(fields[4]),
        "driver_version": fields[5],
        "torch_name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "raw_nvidia_smi": query,
    }
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["attestation_digest"] = hashlib.sha256(body).hexdigest()
    return payload


def require_gpu_name(attestation: dict[str, Any], expected: str) -> None:
    normalized_expected = expected.lower().replace("nvidia", "").strip()
    normalized_observed = str(attestation["name"]).lower().replace(
        "nvidia", ""
    )
    if normalized_expected not in normalized_observed:
        raise RuntimeError(
            "expected GPU "
            + expected
            + ", observed "
            + str(attestation["name"])
        )


def cross_device_numerical_probe(
    vectors: Sequence[Sequence[float]],
    *,
    trainer_device: int,
    verifier_device: int,
) -> dict[str, Any]:
    matrix = torch.as_tensor(vectors, dtype=torch.float32, device="cpu")
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("cross-device probe requires a trajectory matrix")
    trainer = matrix.to("cuda:" + str(trainer_device))
    verifier = matrix.to("cuda:" + str(verifier_device))
    trainer_result = (trainer @ trainer.T).detach().cpu()
    verifier_result = (verifier @ verifier.T).detach().cpu()
    maximum_error = float(
        torch.max(torch.abs(trainer_result - verifier_result))
    )
    result = {
        "shape": [int(value) for value in matrix.shape],
        "trainer_device": trainer_device,
        "verifier_device": verifier_device,
        "maximum_absolute_error": maximum_error,
        "within_final_tolerance": maximum_error <= 1e-3,
    }
    result["probe_digest"] = hashlib.sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return result


__all__ = [
    "cross_device_numerical_probe",
    "gpu_attestation",
    "require_gpu_name",
]
