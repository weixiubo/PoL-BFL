"""Spawn-safe, cached-dataset client training worker for formal cells."""

from __future__ import annotations

import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts" / "utils"))

import numpy as np
import torch

from client.trainer.ProtocolPoLTrainer import ProtocolPoLTrainer
from data_utils import create_dataloaders, load_dataset, partition_data_by_user, partition_data_iid
from experiments.final.data_attacks import DeterministicLabelPoison, clone_indexed_loader
from experiments.final.partitions import partition_dataset_dirichlet
from experiments.scripts.utils.models import create_model
from polbfl.communication import compress_update_4bit
from polbfl.sybil import TraceFingerprint


_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_GLOBAL_STATE_CACHE: tuple[tuple[str, int, int], dict[str, Any]] | None = None


def _seed(seed: int) -> None:
    full_seed = int(seed)
    numpy_seed = full_seed % (2**32)
    torch_seed = full_seed % (2**63 - 1)
    random.seed(full_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _cached_loaders(task: dict[str, Any]):
    key = (
        task["dataset"],
        task["data_root"],
        task["num_clients"],
        task["seed"],
        task.get("partition_alpha"),
        task.get("study", "main"),
    )
    cached = _CACHE.get(key)
    if cached is not None:
        return cached["loaders"]
    _seed(int(task["seed"]))
    train = load_dataset(
        task["dataset"],
        data_dir=str(Path(task["data_root"]) / task["dataset"]),
        train=True,
    )
    partition_alpha = task.get("partition_alpha")
    if partition_alpha is not None:
        partitions = partition_dataset_dirichlet(
            train,
            num_clients=int(task["num_clients"]),
            alpha=float(partition_alpha),
            seed=int(task["seed"]),
        )
    else:
        partitions = (
            partition_data_by_user(train, int(task["num_clients"]))
            if task["dataset"] == "FEMNIST" and task.get("study", "main") == "main"
            else partition_data_iid(train, int(task["num_clients"]))
        )
    loaders = create_dataloaders(partitions, batch_size=32, num_workers=0)
    _CACHE[key] = {"loaders": loaders}
    return loaders


def _cpu_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().cpu().clone()) for name, value in model.state_dict().items())


def _cached_global_state(path_value: str):
    global _GLOBAL_STATE_CACHE
    path = Path(path_value)
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    if _GLOBAL_STATE_CACHE is not None and _GLOBAL_STATE_CACHE[0] == key:
        return _GLOBAL_STATE_CACHE[1]
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload["global_state"]
    _GLOBAL_STATE_CACHE = (key, state)
    return state


def _delta(local, global_state):
    return OrderedDict(
        (
            name,
            local[name].to(torch.float32) - value.to(torch.float32)
            if value.is_floating_point()
            else torch.zeros_like(value),
        )
        for name, value in global_state.items()
    )


def train_client_task(task: dict[str, Any]) -> str:
    started_task = time.perf_counter()
    timings: dict[str, float] = {}
    seed = int(task["seed"])
    client_index = int(task["client_index"])
    round_number = int(task["round_number"])
    device = str(task["device"])
    task_seed = seed * 1_000_003 + round_number * 10_007 + client_index
    _seed(task_seed)
    torch.cuda.set_device(int(device.split(":")[-1]))
    started_loaders = time.perf_counter()
    loaders = _cached_loaders(task)
    # A cache miss seeds the shared partition construction. Restore the
    # client/round RNG domain before any local training operation.
    _seed(task_seed)
    timings["loaders_seconds"] = time.perf_counter() - started_loaders
    source_index = int(task.get("source_index", client_index))
    loader = loaders[source_index]
    sampler_seed = int(task.get("sampler_seed", seed * 1009 + client_index)) + round_number * 10_007
    if task.get("poison_labels"):
        dataset = DeterministicLabelPoison(
            loader.dataset,
            num_classes=int(task["classes"]),
            poison_ratio=float(task.get("poison_ratio", 1.0)),
            seed=seed + client_index,
        )
        loader = clone_indexed_loader(loader, seed=sampler_seed, dataset=dataset)
    else:
        loader = clone_indexed_loader(loader, seed=sampler_seed)

    started_model = time.perf_counter()
    global_state = _cached_global_state(task["global_state_path"])
    model = create_model(
        task["model_name"],
        num_classes=int(task["classes"]),
        input_channels=1 if task["model_name"] == "TwoLayerCNN" else 3,
    )
    model.load_state_dict(global_state, strict=True)
    timings["model_setup_seconds"] = time.perf_counter() - started_model
    client_id = f"client-{client_index}"
    trainer_args = {
        "enable_pol": True,
        "enable_zkp": True,
        "client_id": client_id,
        "round_num": round_number,
        "round_id": f"round-{round_number}",
        "model_id": f"{task['model_name']}-{task['dataset']}",
        "device": device,
        "optimizer": "SGD",
        "lr": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "pol_save_freq": 5,
        "pol_save_dir": task["evidence_root"],
        "round_randomness": task["round_randomness"],
        "gradient_sample_rate": 0.01,
        "batch_size": 32,
        "pair_tolerance": float(task.get("pair_tolerance", 1e-5)),
        "final_tolerance": float(task.get("final_tolerance", 1e-3)),
        "max_update_l2": 10.0,
        "node_binary": "node",
        "poseidon_binary": task.get("poseidon_binary"),
        "packed_evidence": bool(task.get("packed_evidence", True)),
        "clip_norm": None,
    }
    local_epochs = int(task["local_epochs"])
    trainer = None
    recorded = None
    if task.get("record_pol", True):
        trainer = ProtocolPoLTrainer(
            model,
            loader,
            torch.nn.CrossEntropyLoss(),
            args=trainer_args,
        )
        started_training = time.perf_counter()
        trainer.train(local_epochs)
        timings["local_training_seconds"] = time.perf_counter() - started_training
        started_finalize = time.perf_counter()
        trainer.finalize_pol(epoch=local_epochs - 1)
        timings["trace_finalize_seconds"] = time.perf_counter() - started_finalize
        recorded = trainer.recorded_trace
        if trainer.trace_recorder is not None:
            timings.update(
                {
                    f"recorder_{name}": float(value)
                    for name, value in trainer.trace_recorder.timings.items()
                }
            )
    else:
        started_training = time.perf_counter()
        model.to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(local_epochs):
            setter = getattr(loader.dataset, "set_replay_context", None)
            if callable(setter):
                setter(round_num=round_number, epoch=epoch)
            for batch in loader:
                data, labels = batch[:2]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(
                    model(data.to(device)), labels.to(device)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("client produced a non-finite loss")
                loss.backward()
                optimizer.step()
        timings["local_training_seconds"] = time.perf_counter() - started_training
    started_transport = time.perf_counter()
    update = _delta(_cpu_state(model), global_state)
    compressed_payload = compress_update_4bit(update)
    timings["transport_seconds"] = time.perf_counter() - started_transport
    storage_bytes = (
        sum(item.blob.size for item in recorded.steps.values())
        + sum(item.blob.size for item in recorded.checkpoints.values())
        if recorded is not None
        else 0
    )
    result = {
        "client_id": client_id,
        "compressed_update": compressed_payload,
        "recorded": recorded,
        "commitment": None if recorded is None else recorded.trace.commitment,
        "fingerprint": None if recorded is None else TraceFingerprint.from_recorded(recorded),
        "storage_bytes": storage_bytes,
        "communication_bytes": len(compressed_payload),
        "store_root": None if trainer is None else str(trainer.trace_recorder.store.root),
        "computation_valid": recorded is not None,
        "timings": timings,
    }
    timings["worker_seconds_before_result_write"] = time.perf_counter() - started_task
    destination = Path(task["result_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(result, temporary)
    os.replace(temporary, destination)
    del trainer, model, recorded, result, global_state
    return str(destination)
