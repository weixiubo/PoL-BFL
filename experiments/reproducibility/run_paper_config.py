#!/usr/bin/env python3
"""Run formal paper reproduction configs with resumable cell manifests.

The paper configs in this directory describe target matrices. This launcher
turns them into concrete runner invocations, keeps every result under
``experiments/results/reproduction/formal``, and skips cells that already have
successful manifests. It never edits the paper.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = sys.executable
VALIDATION_RESULT_NAMES = {
    "rq1_results.json",
    "rq5_results.json",
    "table2_layer_contribution_summary.json",
    "table6_noniid_summary.json",
    "table9_adaptive_summary.json",
}


@dataclass
class Job:
    job_id: str
    command: List[str]
    output_dir: Path
    expected_files: List[Path]
    config: Dict[str, Any]
    env: Dict[str, str] = field(default_factory=dict)


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(value: Any) -> str:
    text = str(value)
    text = text.replace("CIFAR10", "cifar10").replace("CIFAR100", "cifar100")
    text = text.replace("PoL_FL", "pol_bfl")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "cell"


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (CODE_ROOT / path)


def _as_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _auto_model(dataset: str) -> str:
    if dataset in {"MNIST", "FEMNIST"}:
        return "SimpleCNN"
    if dataset == "CIFAR100":
        return "ResNet34"
    return "ResNet18"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return payload


def _execution_value(config: Dict[str, Any], key: str, default: Any) -> Any:
    execution = config.get("execution") or {}
    if key in execution:
        return execution[key]
    if key in config:
        return config[key]
    return default


def _output_root(config: Dict[str, Any]) -> Path:
    return _resolve(config.get("output_root", "experiments/results/reproduction/formal"))


def _dataset_items(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for item in _as_list(config.get("datasets")):
        if isinstance(item, dict):
            row = dict(item)
        else:
            row = {"name": str(item)}
        row.setdefault("model", _auto_model(str(row["name"])))
        items.append(row)
    if not items and config.get("dataset"):
        items.append(
            {
                "name": str(config["dataset"]),
                "model": str(config.get("model") or _auto_model(str(config["dataset"]))),
                "data_distribution": str(config.get("data_distribution", "IID")),
            }
        )
    return items


def _build_rq1_jobs(config: Dict[str, Any], config_path: Path, args: argparse.Namespace) -> List[Job]:
    runner = str(config["runner"])
    output_root = _output_root(config)
    rounds = int(_execution_value(config, "rounds", 200))
    num_clients = int(_execution_value(config, "num_clients", 50))
    clients_per_round = int(_execution_value(config, "clients_per_round", num_clients))
    local_epochs = int(_execution_value(config, "local_epochs", 5))
    default_batch_size = _execution_value(config, "batch_size", None)
    default_learning_rate = _execution_value(config, "learning_rate", None)
    default_momentum = _execution_value(config, "momentum", None)
    default_weight_decay = _execution_value(config, "weight_decay", None)
    default_verification_rate = _execution_value(config, "verification_rate", None)
    default_pol_delta = _execution_value(config, "pol_delta", None)
    seeds = [int(x) for x in _as_list(_execution_value(config, "seeds", [42]))]
    default_dist = str(_execution_value(config, "data_distribution", "IID"))
    default_alpha = _execution_value(config, "dirichlet_alpha", None)
    attacks = [str(x) for x in _as_list(config.get("attacks"))]
    baselines = [str(x) for x in _as_list(config.get("baselines"))]
    jobs: List[Job] = []

    for dataset in _dataset_items(config):
        ds_name = str(dataset["name"])
        model = str(dataset.get("model") or _auto_model(ds_name))
        data_distribution = str(dataset.get("data_distribution") or default_dist)
        dirichlet_alpha = dataset.get("dirichlet_alpha", default_alpha)
        batch_size = dataset.get("batch_size", default_batch_size)
        learning_rate = dataset.get("learning_rate", default_learning_rate)
        momentum = dataset.get("momentum", default_momentum)
        weight_decay = dataset.get("weight_decay", default_weight_decay)
        verification_rate = dataset.get("verification_rate", default_verification_rate)
        pol_delta = dataset.get("pol_delta", default_pol_delta)
        for attack in attacks:
            for baseline in baselines:
                for seed in seeds:
                    job_id = "__".join(
                        [
                            _slug(ds_name),
                            _slug(attack),
                            _slug(baseline),
                            f"seed{seed}",
                        ]
                    )
                    output_dir = output_root / job_id
                    result_dir = output_dir / "rq1_output"
                    cmd = [
                        args.python,
                        runner,
                        "--dataset",
                        ds_name,
                        "--model",
                        model,
                        "--num_rounds",
                        str(rounds),
                        "--num_clients",
                        str(num_clients),
                        "--clients_per_round",
                        str(clients_per_round),
                        "--data_distribution",
                        data_distribution,
                        "--local_epochs",
                        str(local_epochs),
                        "--attacks",
                        attack,
                        "--baselines",
                        baseline,
                        "--output_dir",
                        str(result_dir),
                    ]
                    if dirichlet_alpha is not None and data_distribution == "NonIID_Dirichlet":
                        cmd.extend(["--dirichlet_alpha", str(dirichlet_alpha)])
                    if batch_size is not None:
                        cmd.extend(["--batch_size", str(batch_size)])
                    if learning_rate is not None:
                        cmd.extend(["--learning_rate", str(learning_rate)])
                    if momentum is not None:
                        cmd.extend(["--momentum", str(momentum)])
                    if weight_decay is not None:
                        cmd.extend(["--weight_decay", str(weight_decay)])
                    if verification_rate is not None:
                        cmd.extend(["--verification_rate", str(verification_rate)])
                    if pol_delta is not None:
                        cmd.extend(["--pol_delta", str(pol_delta)])
                    job_config = {
                        "source_config": str(config_path),
                        "runner": runner,
                        "dataset": ds_name,
                        "model": model,
                        "rounds": rounds,
                        "num_clients": num_clients,
                        "clients_per_round": clients_per_round,
                        "local_epochs": local_epochs,
                        "data_distribution": data_distribution,
                        "dirichlet_alpha": dirichlet_alpha,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "momentum": momentum,
                        "weight_decay": weight_decay,
                        "verification_rate": verification_rate,
                        "pol_delta": pol_delta,
                        "attack": attack,
                        "baseline": baseline,
                        "seed": seed,
                    }
                    env = {
                        "POL_REPRO_RUN_ID": job_id,
                        "PYTHONHASHSEED": str(seed),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "POL_INTEGRITY": str((config.get("protocol") or {}).get("pol_integrity", 1)),
                        "POL_ATTACK_CONTEXT": attack,
                    }
                    if baseline == "PoL_FL":
                        env["POL_DETERMINISTIC_AUG"] = "1"
                        env["POL_SAVE_CHECKPOINTS_TO_DISK"] = "0"
                        env["POL_SAVE_FREQ"] = "20"
                        env["POL_MEMORY_CHECKPOINT_LIMIT"] = os.getenv("POL_MEMORY_CHECKPOINT_LIMIT", "2")
                        env["POL_CHALLENGE_SELECTED_PAIRS"] = "1"
                        env["POL_ALWAYS_VERIFY_LAST_K"] = "1"
                        env["POL_RANDOM_Q"] = "0"
                        env["POL_COMPRESS_CHECKPOINTS"] = "0"
                        env["POL_COMPACT_REMOTE_RESPONSE"] = os.getenv("POL_COMPACT_REMOTE_RESPONSE", "1")
                        env["POL_ENABLE_PARALLEL_CLIENT_TRAINING"] = os.getenv("POL_ENABLE_PARALLEL_CLIENT_TRAINING", "1")
                        env["POL_CLIENT_TRAIN_WORKERS_PER_DEVICE"] = os.getenv("POL_CLIENT_TRAIN_WORKERS_PER_DEVICE", "2")
                        env["POL_CLIENT_TRAIN_MAX_WORKERS"] = os.getenv("POL_CLIENT_TRAIN_MAX_WORKERS", "0")
                        env["POL_SUPPRESS_MODEL_INFO"] = "1"
                        env["NUM_WORKERS_OVERRIDE"] = os.getenv("NUM_WORKERS_OVERRIDE", "0")
                        sybil_active = "sybil" in attack.lower()
                        env["POL_ENABLE_SYBIL_DETECTOR"] = "1" if sybil_active else "0"
                        env["POL_SYBIL_TRAJECTORY_ONLY"] = "0"
                    jobs.append(
                        Job(
                            job_id=job_id,
                            command=cmd,
                            output_dir=output_dir,
                            expected_files=[result_dir / "rq1_results.json", result_dir / "config.json"],
                            config=job_config,
                            env=env,
                        )
                    )
    return jobs


def _build_single_runner_job(
    config: Dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
    job_id: str,
    extra_args: List[str],
    expected_name: str,
) -> Job:
    runner = str(config["runner"])
    output_root = _output_root(config)
    output_dir = output_root / job_id
    cmd = [args.python, runner] + extra_args + ["--output_dir", str(output_dir)]
    job_config = {
        "source_config": str(config_path),
        "runner": runner,
        "job_id": job_id,
        "rounds": int(_execution_value(config, "rounds", 200)),
        "num_clients": int(_execution_value(config, "num_clients", 50)),
        "clients_per_round": int(_execution_value(config, "clients_per_round", 50)),
        "local_epochs": int(_execution_value(config, "local_epochs", 5)),
    }
    env = {
        "POL_REPRO_RUN_ID": job_id,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "POL_INTEGRITY": str((config.get("protocol") or {}).get("pol_integrity", 1)),
    }
    return Job(
        job_id=job_id,
        command=cmd,
        output_dir=output_dir,
        expected_files=[output_dir / expected_name],
        config=job_config,
        env=env,
    )


def _build_table_runner_jobs(config: Dict[str, Any], config_path: Path, args: argparse.Namespace) -> List[Job]:
    runner = str(config.get("runner", ""))
    rounds = str(int(_execution_value(config, "rounds", 200)))
    num_clients = str(int(_execution_value(config, "num_clients", 50)))
    clients_per_round = str(int(_execution_value(config, "clients_per_round", 50)))
    local_epochs = str(int(_execution_value(config, "local_epochs", 5)))

    if runner.endswith("run_rq2_layer_contribution.py"):
        datasets = ",".join(str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in _as_list(config.get("datasets")))
        attacks = ",".join(str(x) for x in _as_list(config.get("attacks")))
        variants = ",".join(str(x) for x in _as_list(config.get("variants"), ["l1_only", "l1_l2", "l1_l3", "full"]))
        return [
            _build_single_runner_job(
                config,
                config_path,
                args,
                "table2_full_matrix",
                [
                    "--datasets",
                    datasets,
                    "--attacks",
                    attacks,
                    "--variants",
                    variants,
                    "--num_rounds",
                    rounds,
                    "--num_clients",
                    num_clients,
                    "--clients_per_round",
                    clients_per_round,
                    "--local_epochs",
                    local_epochs,
                    "--verification_rate",
                    "1.0",
                ],
                "table2_layer_contribution_summary.json",
            )
        ]

    if runner.endswith("run_rq6_noniid.py"):
        dataset = str(config.get("dataset", "CIFAR10"))
        alphas = ",".join(str(x) for x in _as_list(config.get("dirichlet_alphas"), ["0.1", "0.5", "1.0", "IID"]))
        return [
            _build_single_runner_job(
                config,
                config_path,
                args,
                "table6_full_matrix",
                [
                    "--datasets",
                    dataset,
                    "--alphas",
                    alphas,
                    "--num_rounds",
                    rounds,
                    "--num_clients",
                    num_clients,
                    "--clients_per_round",
                    clients_per_round,
                    "--local_epochs",
                    local_epochs,
                    "--verification_rate",
                    "1.0",
                ],
                "table6_noniid_summary.json",
            )
        ]

    if runner.endswith("run_rq9_adaptive.py"):
        variants = ",".join(str(x) for x in _as_list(config.get("variants"), []))
        if not variants:
            variants = "baseline_nt,checkpoint_interpolation,gradient_mimicry,partial_replay,combined_adaptive"
        return [
            _build_single_runner_job(
                config,
                config_path,
                args,
                "table9_full_matrix",
                [
                    "--variants",
                    variants,
                    "--num_rounds",
                    rounds,
                    "--num_clients",
                    num_clients,
                    "--clients_per_round",
                    clients_per_round,
                    "--local_epochs",
                    local_epochs,
                    "--verification_rate",
                    "1.0",
                ],
                "table9_adaptive_summary.json",
            )
        ]

    if runner.endswith("run_rq5_composability.py"):
        attacks = ",".join(str(x) for x in _as_list(config.get("attacks"), ["byzantine_alie", "free_riding_no_training"]))
        baselines = ",".join(str(x) for x in _as_list(config.get("baselines"), ["Krum", "PoL_Krum", "Trimmed_Mean", "PoL_Trimmed_Mean", "Median", "PoL_Median"]))
        output_root = _output_root(config)
        output_dir = output_root / "table5_full_matrix"
        cmd = [
            args.python,
            runner,
            "--dataset",
            str(config.get("dataset", "CIFAR10")),
            "--model",
            str(config.get("model", "ResNet18")),
            "--num_rounds",
            rounds,
            "--num_clients",
            num_clients,
            "--clients_per_round",
            clients_per_round,
            "--local_epochs",
            local_epochs,
            "--attacks",
            attacks,
            "--baselines",
            baselines,
            "--output_dir",
            str(output_dir),
        ]
        return [
            Job(
                job_id="table5_full_matrix",
                command=cmd,
                output_dir=output_dir,
                expected_files=[output_dir / "rq5_composability" / "rq5_results.json"],
                config={
                    "source_config": str(config_path),
                    "runner": runner,
                    "rounds": int(rounds),
                    "num_clients": int(num_clients),
                    "clients_per_round": int(clients_per_round),
                    "local_epochs": int(local_epochs),
                },
                env={"POL_REPRO_RUN_ID": "table5_full_matrix", "POL_INTEGRITY": str((config.get("protocol") or {}).get("pol_integrity", 1))},
            )
        ]

    if runner.endswith("run_rq3_overhead.py"):
        dataset = str(_as_list(config.get("datasets"), ["CIFAR10"])[0])
        model = str(_as_list(config.get("models"), [_auto_model(dataset)])[0])
        output_root = _output_root(config)
        output_dir = output_root / "rq3_overhead_measurement"
        cmd = [
            args.python,
            runner,
            "--dataset",
            dataset,
            "--model",
            model,
            "--rounds",
            str(int(_execution_value(config, "rounds", 20))),
            "--num_clients",
            str(int(_execution_value(config, "num_clients", 20))),
            "--clients_per_round",
            str(int(_execution_value(config, "clients_per_round", 10))),
            "--output_dir",
            str(output_dir),
        ]
        return [
            Job(
                job_id="rq3_overhead_measurement",
                command=cmd,
                output_dir=output_dir,
                expected_files=[output_dir / "rq3_overhead" / "rq3_results.json"],
                config={"source_config": str(config_path), "runner": runner, "dataset": dataset, "model": model},
                env={"POL_REPRO_RUN_ID": "rq3_overhead_measurement", "POL_INTEGRITY": str((config.get("protocol") or {}).get("pol_integrity", 1))},
            )
        ]

    if runner.endswith("run_rq4_incentive.py"):
        dataset = str(config.get("dataset", "CIFAR10"))
        model = str(config.get("model", _auto_model(dataset)))
        output_root = _output_root(config)
        output_dir = output_root / "rq4_incentive_measurement"
        cmd = [
            args.python,
            runner,
            "--dataset",
            dataset,
            "--model",
            model,
            "--num_rounds",
            str(int(_execution_value(config, "rounds", 50))),
            "--num_clients",
            str(int(_execution_value(config, "num_clients", 20))),
            "--clients_per_round",
            str(int(_execution_value(config, "clients_per_round", 10))),
            "--output_dir",
            str(output_dir),
        ]
        return [
            Job(
                job_id="rq4_incentive_measurement",
                command=cmd,
                output_dir=output_dir,
                expected_files=[output_dir / "rq4_incentive" / "rq4_results.json"],
                config={"source_config": str(config_path), "runner": runner, "dataset": dataset, "model": model},
                env={"POL_REPRO_RUN_ID": "rq4_incentive_measurement", "POL_INTEGRITY": str((config.get("protocol") or {}).get("pol_integrity", 1))},
            )
        ]

    raise ValueError(f"Unsupported paper config runner: {runner}")


def build_jobs(config: Dict[str, Any], config_path: Path, args: argparse.Namespace) -> List[Job]:
    runner = str(config.get("runner", ""))
    if runner.endswith("run_rq1_security.py"):
        return _build_rq1_jobs(config, config_path, args)
    return _build_table_runner_jobs(config, config_path, args)


def _manifest_path(job: Job) -> Path:
    return job.output_dir / "run_manifest.json"


def _job_completed(job: Job) -> bool:
    manifest_path = _manifest_path(job)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("returncode") != 0 or manifest.get("status") != "completed":
        return False
    return all(path.exists() for path in job.expected_files)


def _filter_jobs(jobs: List[Job], args: argparse.Namespace) -> List[Job]:
    selected = jobs
    if args.only:
        needles = [item.lower() for item in args.only]
        selected = [job for job in selected if all(_job_id_matches_filter(job.job_id, needle) for needle in needles)]
    if args.resume:
        selected = [job for job in selected if not _job_completed(job)]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    return selected


def _job_id_matches_filter(job_id: str, needle: str) -> bool:
    """Match filters on job-id tokens without letting cifar10 select cifar100."""
    normalized = str(needle).strip().lower()
    if not normalized:
        return True
    tokens = [token for token in job_id.lower().split("__") if token]
    for token in tokens:
        if token == normalized:
            return True
        if token.startswith(f"{normalized}_") or token.endswith(f"_{normalized}"):
            return True
        if f"_{normalized}_" in f"_{token}_":
            return True
    return False


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _start_verifiers(args: argparse.Namespace, output_root: Path) -> Dict[str, Dict[str, Any]]:
    if not args.start_verifiers:
        return {}
    verifiers: Dict[str, Dict[str, Any]] = {}
    log_dir = output_root / "_verifiers"
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, gpu in enumerate(args.gpus):
        port = int(args.verifier_port_base) + index
        endpoint = f"http://127.0.0.1:{port}"
        if _port_open(port):
            verifiers[gpu] = {"endpoint": endpoint, "process": None, "already_running": True}
            continue
        log_path = log_dir / f"verifier_gpu{_slug(gpu)}_{port}.log"
        log_fh = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{CODE_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        env["POL_REMOTE_MODE"] = "strict_replay"
        env["CUDA_VISIBLE_DEVICES"] = "" if str(gpu).lower() == "cpu" else str(gpu)
        env["POL_VERIFIER_DEVICE"] = "cpu" if str(gpu).lower() == "cpu" else "cuda"
        proc = subprocess.Popen(
            [args.python, "-m", "server.committee.VerifierNode", "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(CODE_ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        verifiers[gpu] = {"endpoint": endpoint, "process": proc, "log": str(log_path), "already_running": False}
    time.sleep(float(args.verifier_startup_sleep))
    return verifiers


def _gpu_free_memory_mb(gpu: str) -> Optional[int]:
    if gpu.lower() == "cpu":
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(gpu),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(CODE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        return int(first_line.strip())
    except ValueError:
        return None


def _select_gpu(args: argparse.Namespace, start_index: int, active: List[Dict[str, Any]]) -> str:
    if int(args.min_gpu_free_mb) <= 0:
        return args.gpus[start_index % len(args.gpus)]

    active_gpus = {str(state.get("gpu")) for state in active if str(state.get("gpu", "")).lower() != "cpu"}
    deadline = None
    if float(args.gpu_wait_timeout_sec) > 0:
        deadline = time.time() + float(args.gpu_wait_timeout_sec)

    while True:
        observed: List[str] = []
        for offset in range(len(args.gpus)):
            gpu = args.gpus[(start_index + offset) % len(args.gpus)]
            if gpu.lower() == "cpu":
                return gpu
            if gpu in active_gpus:
                observed.append(f"{gpu}:active")
                continue
            free_mb = _gpu_free_memory_mb(gpu)
            observed.append(f"{gpu}:{'unknown' if free_mb is None else free_mb}")
            if free_mb is None or free_mb >= int(args.min_gpu_free_mb):
                return gpu

        message = (
            f"Waiting for GPU free memory >= {int(args.min_gpu_free_mb)} MiB; "
            f"observed {', '.join(observed)}"
        )
        if deadline is not None and time.time() >= deadline:
            raise RuntimeError(message)
        print(message, flush=True)
        time.sleep(float(args.gpu_wait_sleep_sec))


def _write_plan(config_path: Path, config: Dict[str, Any], jobs: List[Job], selected: List[Job], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _dt.datetime.now().isoformat(),
        "code_root": str(CODE_ROOT),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "output_root": str(output_root),
        "total_jobs": len(jobs),
        "selected_jobs": len(selected),
        "jobs": [
            {
                "job_id": job.job_id,
                "output_dir": str(job.output_dir),
                "expected_files": [str(path) for path in job.expected_files],
                "command": job.command,
                "config": job.config,
            }
            for job in selected
        ],
        "raw_config": config,
    }
    plan_path = output_root / f"paper_run_plan_{_stamp()}.json"
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_path


def _start_job(job: Job, args: argparse.Namespace, gpu: str, verifier: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(job.env)
    env["PYTHONPATH"] = f"{CODE_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu.lower() == "cpu" else gpu
    if verifier:
        env["POL_DECENT_MODE"] = "1"
        env["POL_REQUIRE_REMOTE_VERIFIER"] = "1"
        env["POL_REMOTE_MODE"] = "strict_replay"
        env["POL_REMOTE_TIMEOUT_SEC"] = str(args.remote_timeout_sec)
        env["POL_VERIFIER_ENDPOINTS"] = str(verifier["endpoint"])

    log_path = job.output_dir / "runner.log"
    manifest = {
        "schema_version": 1,
        "job_id": job.job_id,
        "status": "running",
        "started_at": _dt.datetime.now().isoformat(),
        "code_root": str(CODE_ROOT),
        "output_dir": str(job.output_dir),
        "command": job.command,
        "config": job.config,
        "expected_files": [str(path) for path in job.expected_files],
        "assigned_gpu": gpu,
        "gpu_free_memory_mb_at_start": _gpu_free_memory_mb(gpu),
        "environment_overrides": {key: env.get(key) for key in sorted(set(job.env) | {"CUDA_VISIBLE_DEVICES", "POL_DECENT_MODE", "POL_REQUIRE_REMOTE_VERIFIER", "POL_REMOTE_MODE", "POL_VERIFIER_ENDPOINTS", "POL_INTEGRITY", "POL_DETERMINISTIC_AUG", "POL_ENABLE_SYBIL_DETECTOR", "POL_SYBIL_TRAJECTORY_ONLY", "POL_ATTACK_CONTEXT", "POL_REMOTE_TIMEOUT_SEC", "POL_SAVE_CHECKPOINTS_TO_DISK", "POL_SAVE_FREQ", "POL_MEMORY_CHECKPOINT_LIMIT", "POL_CHALLENGE_SELECTED_PAIRS", "POL_ALWAYS_VERIFY_LAST_K", "POL_RANDOM_Q", "POL_COMPRESS_CHECKPOINTS", "POL_COMPACT_REMOTE_RESPONSE", "POL_ENABLE_PARALLEL_CLIENT_TRAINING", "POL_CLIENT_TRAIN_WORKERS_PER_DEVICE", "POL_CLIENT_TRAIN_MAX_WORKERS", "POL_SUPPRESS_MODEL_INFO", "NUM_WORKERS_OVERRIDE"})},
    }
    _manifest_path(job).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log_fh = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(job.command, cwd=str(CODE_ROOT), env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    return {"job": job, "process": proc, "log_fh": log_fh, "log_path": log_path, "manifest": manifest}


def _finish_job(state: Dict[str, Any]) -> int:
    proc = state["process"]
    returncode = int(proc.wait())
    state["log_fh"].close()
    job: Job = state["job"]
    manifest = dict(state["manifest"])
    manifest.update(
        {
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "completed_at": _dt.datetime.now().isoformat(),
            "log_path": str(state["log_path"]),
            "expected_files_present": {str(path): path.exists() for path in job.expected_files},
        }
    )
    _manifest_path(job).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return returncode


def _path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _default_validation_results_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    parts = list(resolved.parts)
    if "formal" in parts:
        idx = len(parts) - 1 - parts[::-1].index("formal")
        return Path(*parts[: idx + 1])
    return resolved


def _job_should_have_validation_rows(job: Job) -> bool:
    return any(path.name in VALIDATION_RESULT_NAMES for path in job.expected_files)


def _as_count(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _format_comparison(item: Dict[str, Any]) -> str:
    labels = [
        str(item.get("table", "")),
        str(item.get("dataset", "")),
        str(item.get("attack", "")),
        str(item.get("method", item.get("variant", item.get("aggregation", "")))),
        str(item.get("metric", "")),
    ]
    left = " / ".join(label for label in labels if label and label != "None")
    return (
        f"- {item.get('status')}: {left}; "
        f"observed={item.get('observed')}, target={item.get('target')}, "
        f"delta={item.get('delta')}"
    )


def _write_validation_gate_report(
    job: Job,
    *,
    validation_dir: Path,
    validation_manifest_path: Path,
    validation_report_path: Path,
    validation_stdout: str,
    validation_stderr: str,
    returncode: int,
    current_rows: List[Dict[str, Any]],
    blocking_rows: List[Dict[str, Any]],
    blocking_reasons: List[str],
) -> Path:
    report_path = job.output_dir / "validation_gate_report.md"
    lines = [
        f"# Validation Gate: {job.job_id}",
        "",
        f"- validation_dir: `{validation_dir}`",
        f"- validation_manifest: `{validation_manifest_path}`",
        f"- validation_report: `{validation_report_path}`",
        f"- validation_returncode: `{returncode}`",
        f"- gate_status: `{'blocked' if blocking_reasons else 'pass'}`",
        "",
    ]
    if blocking_reasons:
        lines.append("## Blocking Reasons")
        lines.extend(f"- {reason}" for reason in blocking_reasons)
        lines.append("")

    lines.append("## Current Job Comparisons")
    if current_rows:
        lines.extend(_format_comparison(item) for item in current_rows[:40])
    else:
        lines.append("- none")
    lines.append("")

    if blocking_rows:
        lines.append("## Root-Cause Entry Points")
        lines.extend(
            [
                "- Check `run_manifest.json` environment overrides against the known-good formal settings.",
                "- Check whether the attack actually affected malicious clients and whether malicious clients were verified.",
                "- Check strict replay pass/fail evidence before changing numeric thresholds.",
                "- Check aggregation weighting and validation target mapping before treating a result as a paper discrepancy.",
                "- Write a discrepancy report only after implementation/configuration issues are ruled out.",
                "",
            ]
        )

    if validation_stdout.strip():
        lines.extend(["## Validation Stdout", "```text", validation_stdout[-4000:], "```", ""])
    if validation_stderr.strip():
        lines.extend(["## Validation Stderr", "```text", validation_stderr[-4000:], "```", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _run_validation_gate(job: Job, args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    if not args.validate_after_job:
        return {"enabled": False}

    results_root = (
        args.validation_results_root.expanduser().resolve()
        if args.validation_results_root
        else _default_validation_results_root(output_root)
    )
    validation_base = (
        args.validation_output_root.expanduser().resolve()
        if args.validation_output_root
        else results_root / "_validation_gates"
    )
    validation_dir = validation_base / f"{_slug(job.job_id)}_{_stamp()}"
    cmd = [
        args.python,
        "experiments/reproducibility/validate_reproduction.py",
        "--results-root",
        str(results_root),
        "--output-dir",
        str(validation_dir),
        "--tolerance-ma",
        str(args.validation_tolerance_ma),
        "--tolerance-detection",
        str(args.validation_tolerance_detection),
        "--tolerance-other",
        str(args.validation_tolerance_other),
        "--min-rounds-rq1",
        str(args.validation_min_rounds_rq1),
        "--min-clients-rq1",
        str(args.validation_min_clients_rq1),
        "--min-clients-per-round-rq1",
        str(args.validation_min_clients_per_round_rq1),
        "--min-local-epochs-rq1",
        str(args.validation_min_local_epochs_rq1),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(CODE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    validation_manifest_path = validation_dir / "validation_manifest.json"
    validation_report_path = validation_dir / "validation_report.md"
    current_rows: List[Dict[str, Any]] = []
    blocking_rows: List[Dict[str, Any]] = []
    blocking_reasons: List[str] = []
    summary: Dict[str, Any] = {}

    if proc.returncode != 0:
        blocking_reasons.append(f"validation command returned {proc.returncode}")
    if not validation_manifest_path.exists():
        blocking_reasons.append("validation_manifest.json was not produced")
    else:
        try:
            manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
            summary = manifest.get("summary", {})
            overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
            if _as_count(overall.get("fail")) > 0:
                blocking_reasons.append(f"validation overall fail count is {_as_count(overall.get('fail'))}")
            job_root = job.output_dir.resolve()
            for item in manifest.get("comparisons", []):
                source = item.get("source")
                if source and _path_is_relative_to(Path(source), job_root):
                    current_rows.append(item)
                    if item.get("status") in {"fail", "protocol_mismatch"}:
                        blocking_rows.append(item)
            if blocking_rows:
                blocking_reasons.append(
                    f"{len(blocking_rows)} comparison(s) for this job failed validation/protocol checks"
                )
            if _job_should_have_validation_rows(job) and not current_rows:
                blocking_reasons.append("completed job produced no validation-mapped comparison rows")
        except Exception as exc:
            blocking_reasons.append(f"could not parse validation manifest: {exc}")

    gate_report = _write_validation_gate_report(
        job,
        validation_dir=validation_dir,
        validation_manifest_path=validation_manifest_path,
        validation_report_path=validation_report_path,
        validation_stdout=proc.stdout,
        validation_stderr=proc.stderr,
        returncode=proc.returncode,
        current_rows=current_rows,
        blocking_rows=blocking_rows,
        blocking_reasons=blocking_reasons,
    )
    gate = {
        "enabled": True,
        "blocking": bool(blocking_reasons),
        "validation_command": cmd,
        "validation_dir": str(validation_dir),
        "validation_manifest": str(validation_manifest_path),
        "validation_report": str(validation_report_path),
        "gate_report": str(gate_report),
        "returncode": proc.returncode,
        "summary": summary,
        "current_job_comparisons": len(current_rows),
        "blocking_reasons": blocking_reasons,
    }
    manifest_path = _manifest_path(job)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest["validation_gate"] = gate
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return gate


def run_jobs(selected: List[Job], args: argparse.Namespace, output_root: Path) -> int:
    if args.dry_run:
        return 0
    verifiers = _start_verifiers(args, output_root)
    active: List[Dict[str, Any]] = []
    failures = 0
    cursor = 0
    stop_launching = False
    try:
        while (cursor < len(selected) and not stop_launching) or active:
            while cursor < len(selected) and not stop_launching and len(active) < int(args.parallel):
                job = selected[cursor]
                gpu = _select_gpu(args, cursor, active)
                verifier = verifiers.get(gpu)
                active.append(_start_job(job, args, gpu, verifier))
                active[-1]["gpu"] = gpu
                cursor += 1
            time.sleep(float(args.poll_interval))
            still_active: List[Dict[str, Any]] = []
            for state in active:
                if state["process"].poll() is None:
                    still_active.append(state)
                    continue
                returncode = _finish_job(state)
                if returncode != 0:
                    failures += 1
                    if not args.continue_on_job_failure:
                        stop_launching = True
                    continue
                gate = _run_validation_gate(state["job"], args, output_root)
                if gate.get("blocking"):
                    failures += 1
                    if not args.continue_on_validation_fail:
                        stop_launching = True
            active = still_active
    finally:
        for state in active:
            if state["process"].poll() is None:
                state["process"].terminate()
                try:
                    state["process"].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    state["process"].kill()
                failures += 1
            state["log_fh"].close()
        for meta in verifiers.values():
            proc = meta.get("process")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
    return 0 if failures == 0 else 1


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", action="append", default=[], help="Keep jobs whose id contains all supplied substrings")
    parser.add_argument("--resume", action="store_true", help="Skip jobs with completed run_manifest.json and expected outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-verifiers", action="store_true", help="Start local strict replay VerifierNode processes for each GPU worker")
    parser.add_argument("--verifier-port-base", type=int, default=19088)
    parser.add_argument("--verifier-startup-sleep", type=float, default=3.0)
    parser.add_argument("--remote-timeout-sec", type=int, default=900)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--min-gpu-free-mb", type=int, default=int(os.getenv("POL_MIN_GPU_FREE_MB", "0")))
    parser.add_argument("--gpu-wait-timeout-sec", type=float, default=float(os.getenv("POL_GPU_WAIT_TIMEOUT_SEC", "0")))
    parser.add_argument("--gpu-wait-sleep-sec", type=float, default=float(os.getenv("POL_GPU_WAIT_SLEEP_SEC", "60")))
    parser.add_argument("--validate-after-job", action="store_true", help="Run paper validation after each completed job and write a gate report")
    parser.add_argument("--validation-results-root", type=Path, default=None, help="Results root passed to validate_reproduction.py; defaults to the formal root")
    parser.add_argument("--validation-output-root", type=Path, default=None, help="Directory for per-job validation gate outputs")
    parser.add_argument("--continue-on-job-failure", action="store_true", help="Keep launching queued jobs after a runner exits non-zero")
    parser.add_argument("--continue-on-validation-fail", action="store_true", help="Keep launching queued jobs after a validation gate blocks")
    parser.add_argument("--validation-tolerance-ma", type=float, default=1.0)
    parser.add_argument("--validation-tolerance-detection", type=float, default=1.0)
    parser.add_argument("--validation-tolerance-other", type=float, default=5.0)
    parser.add_argument("--validation-min-rounds-rq1", type=int, default=200)
    parser.add_argument("--validation-min-clients-rq1", type=int, default=50)
    parser.add_argument("--validation-min-clients-per-round-rq1", type=int, default=50)
    parser.add_argument("--validation-min-local-epochs-rq1", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.config_file = args.config_file.expanduser().resolve()
    args.gpus = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()] or ["0"]
    args.python = str(Path(args.python).expanduser()) if "/" in args.python else args.python

    config = _load_config(args.config_file)
    jobs = build_jobs(config, args.config_file, args)
    selected = _filter_jobs(jobs, args)
    output_root = _output_root(config)
    plan_path = _write_plan(args.config_file, config, jobs, selected, output_root)

    print(json.dumps({"plan": str(plan_path), "total_jobs": len(jobs), "selected_jobs": len(selected), "dry_run": args.dry_run}, indent=2))
    if not selected:
        return 0
    return run_jobs(selected, args, output_root)


if __name__ == "__main__":
    raise SystemExit(main())
