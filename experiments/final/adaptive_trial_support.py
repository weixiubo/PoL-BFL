"""Measured adaptive-attack primitives over real CIFAR/PoL traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.final.adaptive_attacks import (
    checkpoint_interpolation,
    combined_adaptive_trajectory,
    gradient_mimicry,
    measure_attack,
    partial_replay,
)
from experiments.final.client_worker import train_client_task
from polbfl.communication import decompress_update_4bit
from polbfl.crypto import domain_hash
from polbfl.zk import Groth16Artifacts, Groth16Backend


VARIANTS = (
    "BaselineNT",
    "CheckpointInterpolation",
    "GradientMimicry",
    "PartialReplay",
    "CombinedAdaptive",
)

# Algorithmic reference inputs, never paper timing values. The attacker
# estimates endpoints or gradient distributions from independent trajectories.
REFERENCE_TRAJECTORIES = {
    "BaselineNT": 0,
    "CheckpointInterpolation": 3,
    "GradientMimicry": 4,
    "PartialReplay": 2,
    "CombinedAdaptive": 5,
}


def production_backend(
    *,
    root: Path,
    build: Path,
    icicle_root: Path,
    rapidsnark_prover: Path,
    rapidsnark_verifier: Path,
) -> Groth16Backend:
    return Groth16Backend(
        Groth16Artifacts(
            wasm=build
            / "sampled_sgd_reference_js"
            / "sampled_sgd_reference.wasm",
            proving_key=build / "sampled_sgd_reference_final.zkey",
            verification_key=build / "verification_key.json",
            r1cs=build / "sampled_sgd_reference.r1cs",
        ),
        snarkjs_cli=root / "node_modules" / "snarkjs" / "cli.js",
        witness_binary=(
            build
            / "sampled_sgd_reference_cpp"
            / "sampled_sgd_reference"
        ),
        prover_binary=rapidsnark_prover,
        verifier_binary=rapidsnark_verifier,
        icicle_binary=icicle_root / "bin" / "icicle-snark",
        icicle_backend_directory=icicle_root / "backend",
        icicle_library_directories=(
            icicle_root / "lib",
            icicle_root / "backend" / "cuda",
        ),
        icicle_devices=(0, 1),
        timeout_seconds=300,
    )


def load_worker_result(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def training_task(
    *,
    output: Path,
    global_state_path: Path,
    data_root: Path,
    poseidon_binary: Path,
    seed: int,
    client_index: int,
    local_epochs: int,
    device: str,
) -> dict[str, Any]:
    name = "reference-" + str(client_index) + "-e" + str(local_epochs)
    return {
        "dataset": "CIFAR10",
        "data_root": str(data_root),
        "num_clients": 50,
        "seed": seed,
        "partition_alpha": None,
        "study": "adaptive",
        "client_index": client_index,
        "source_index": client_index,
        "round_number": 0,
        "device": device,
        "classes": 10,
        "model_name": "ResNet18",
        "global_state_path": str(global_state_path),
        "local_epochs": local_epochs,
        "evidence_root": str(output / "traces" / name),
        "round_randomness": hashlib.sha256(
            ("adaptive:" + str(seed)).encode()
        ).hexdigest(),
        "sampler_seed": seed * 1009 + client_index,
        "poseidon_binary": str(poseidon_binary),
        "packed_evidence": True,
        "record_pol": True,
        "result_path": str(
            output / "worker-results" / (name + ".pt")
        ),
    }


def train_reference(
    *,
    output: Path,
    global_state_path: Path,
    data_root: Path,
    poseidon_binary: Path,
    seed: int,
    client_index: int,
    local_epochs: int,
    device: str,
) -> dict[str, Any]:
    result_path = Path(
        train_client_task(
            training_task(
                output=output,
                global_state_path=global_state_path,
                data_root=data_root,
                poseidon_binary=poseidon_binary,
                seed=seed,
                client_index=client_index,
                local_epochs=local_epochs,
                device=device,
            )
        )
    )
    payload = load_worker_result(result_path)
    if (
        payload.get("recorded") is None
        or payload.get("commitment") is None
        or payload.get("fingerprint") is None
        or not payload.get("store_root")
    ):
        raise RuntimeError("adaptive reference training lacks a real trace")
    payload["_result_path"] = str(result_path)
    return payload


def sample_update_vector(
    payload: Mapping[str, Any],
    global_state: Mapping[str, torch.Tensor],
    *,
    maximum: int = 4096,
) -> torch.Tensor:
    update = decompress_update_4bit(
        payload["compressed_update"],
        global_state,
    )
    vector = torch.cat(
        [
            value.detach().to("cpu", torch.float32).reshape(-1)
            for value in update.values()
            if value.is_floating_point()
        ]
    )
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("adaptive reference update is non-finite")
    return vector[:maximum].clone()


def checkpoint_vectors(
    payload: Mapping[str, Any],
) -> tuple[torch.Tensor, ...]:
    vectors = tuple(
        torch.tensor(value, dtype=torch.float32)
        for value in payload["fingerprint"].checkpoint_vectors
    )
    if len(vectors) < 2:
        raise ValueError("adaptive reference lacks checkpoints")
    return vectors


def build_adaptive_material(
    variant: str,
    *,
    honest: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    global_state: Mapping[str, torch.Tensor],
    seed: int,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    checkpoints = checkpoint_vectors(honest)
    updates = [
        sample_update_vector(payload, global_state)
        for payload in references
    ]
    reference_checkpoints = [
        checkpoint_vectors(payload)
        for payload in references
    ]
    if variant == "BaselineNT":
        operation = lambda: torch.randn(
            (4096,), generator=generator
        )
    elif variant == "CheckpointInterpolation":
        first = torch.stack(
            [values[0] for values in reference_checkpoints]
        ).mean(dim=0)
        final = torch.stack(
            [values[-1] for values in reference_checkpoints]
        ).mean(dim=0)
        interpolation_inputs = (first, *checkpoints[1:-1], final)
        operation = lambda: checkpoint_interpolation(
            interpolation_inputs
        )
    elif variant == "GradientMimicry":
        operation = lambda: gradient_mimicry(
            updates, generator=generator
        )
    elif variant == "PartialReplay":
        operation = lambda: partial_replay(
            reference_checkpoints[0], honest_fraction=0.3
        )
    elif variant == "CombinedAdaptive":
        operation = lambda: combined_adaptive_trajectory(
            reference_checkpoints[0],
            updates,
            generator=generator,
            honest_fraction=0.3,
        )
    else:
        raise ValueError("unknown adaptive variant: " + variant)
    measured = measure_attack(operation)

    def serializable(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            flat = value.detach().to("cpu", torch.float32).reshape(-1)
            return [float(item) for item in flat[:4096]]
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        return value

    digest = domain_hash(
        "POLBFL_ADAPTIVE_MATERIAL_V1",
        variant,
        json.dumps(
            serializable(measured.output),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return measured, digest


def forge_bundle(bundle, *, variant: str, material_digest: str):
    if variant == "BaselineNT":
        return replace(
            bundle,
            challenge=replace(
                bundle.challenge,
                client_id="adaptive-no-training-attacker",
            ),
        )
    if variant == "CheckpointInterpolation":
        return replace(
            bundle,
            commitment=replace(
                bundle.commitment,
                merkle_root=material_digest,
            ),
            challenge=replace(
                bundle.challenge,
                commitment_root=material_digest,
            ),
        )
    if variant == "GradientMimicry":
        return replace(
            bundle,
            uploaded_final_model_digest=material_digest,
        )
    if variant == "PartialReplay":
        replacement_pair = 0 if bundle.pair_index != 0 else 1
        return replace(
            bundle,
            challenge=replace(
                bundle.challenge,
                pair_indices=(replacement_pair,),
            ),
        )
    if variant == "CombinedAdaptive":
        return replace(
            bundle,
            commitment=replace(
                bundle.commitment,
                merkle_root=material_digest,
            ),
            challenge=replace(
                bundle.challenge,
                commitment_root=material_digest,
            ),
            uploaded_final_model_digest=material_digest,
        )
    raise ValueError("unknown adaptive variant: " + variant)


def adaptive_profit(
    *,
    reward: float,
    base_cost: float,
    slash: float,
    forge_train_ratio: float,
) -> float:
    return reward - base_cost * max(
        0.0, forge_train_ratio
    ) - slash


__all__ = [
    "REFERENCE_TRAJECTORIES",
    "VARIANTS",
    "adaptive_profit",
    "build_adaptive_material",
    "forge_bundle",
    "production_backend",
    "train_reference",
]
