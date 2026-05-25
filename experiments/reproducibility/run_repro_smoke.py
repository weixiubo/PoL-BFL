#!/usr/bin/env python3
"""Create and optionally run a reproducible PoL-BFL smoke experiment."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "experiments" / "results" / "reproduction" / "smoke"


def _run_capture(cmd: List[str], timeout: int = 20) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(CODE_ROOT), capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-8000:],
        }
    except Exception as exc:  # pragma: no cover - defensive snapshot
        return {"cmd": cmd, "returncode": None, "error": repr(exc)}


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_default(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _load_config_from_argv(argv: Optional[List[str]]) -> tuple[Dict[str, Any], Optional[Path]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-file", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    if not known.config_file:
        return {}, None

    config_path = known.config_file.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return config, config_path


def _auto_model(dataset: str) -> str:
    if dataset in ("MNIST", "FEMNIST"):
        return "SimpleCNN"
    if dataset == "CIFAR100":
        return "ResNet34"
    return "ResNet18"


def _env_snapshot(python_bin: str) -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_launcher": python_bin,
        "python_version": _run_capture([python_bin, "-V"]),
        "torch": _run_capture(
            [
                python_bin,
                "-c",
                "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())",
            ],
            timeout=45,
        ),
        "git_status": _run_capture(["git", "status", "--short", "--branch"]),
        "git_diff_stat": _run_capture(["git", "diff", "--stat"]),
        "node": _run_capture(["node", "--version"]),
    }


def _build_rq1_command(args: argparse.Namespace, output_dir: Path) -> List[str]:
    model = args.model or _auto_model(args.dataset)
    cmd = [
        args.python,
        "experiments/scripts/runners/run_rq1_security.py",
        "--dataset",
        args.dataset,
        "--model",
        model,
        "--num_rounds",
        str(args.rounds),
        "--num_clients",
        str(args.num_clients),
        "--clients_per_round",
        str(args.clients_per_round),
        "--data_distribution",
        args.data_distribution,
        "--attacks",
        args.attacks,
        "--baselines",
        args.baselines,
        "--output_dir",
        str(output_dir / "rq1_output"),
    ]
    if args.dirichlet_alpha is not None:
        cmd.extend(["--dirichlet_alpha", str(args.dirichlet_alpha)])
    if args.local_epochs is not None:
        cmd.extend(["--local_epochs", str(args.local_epochs)])
    return cmd


def main(argv: Optional[Iterable[str]] = None) -> int:
    argv_list = list(argv) if argv is not None else None
    file_config, config_path = _load_config_from_argv(argv_list)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=config_path)
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10", "CIFAR100", "FEMNIST"], default=file_config.get("dataset", "MNIST"))
    parser.add_argument("--model", default=file_config.get("model"))
    parser.add_argument("--rounds", type=int, default=int(file_config.get("rounds", 1)))
    parser.add_argument("--num-clients", type=int, default=int(file_config.get("num_clients", 3)))
    parser.add_argument("--clients-per-round", type=int, default=int(file_config.get("clients_per_round", 3)))
    parser.add_argument("--data-distribution", default=file_config.get("data_distribution", "NonIID_Dirichlet"), choices=["IID", "NonIID_Dirichlet", "Natural_Writer", "FEMNIST_Natural", "LEAF_Natural"])
    parser.add_argument("--dirichlet-alpha", type=float, default=file_config.get("dirichlet_alpha"))
    parser.add_argument("--local-epochs", type=int, default=file_config.get("local_epochs"))
    parser.add_argument("--attacks", default=_csv_default(file_config.get("attacks"), "no_attack,free_riding_no_training"))
    parser.add_argument("--baselines", default=_csv_default(file_config.get("baselines"), "Vanilla_FL,PoL_FL"))
    parser.add_argument("--python", default=file_config.get("python", os.getenv("PYTHON_BIN", sys.executable)))
    parser.add_argument("--gpu", default=str(file_config.get("gpu", os.getenv("GPU_ID", "cpu"))))
    parser.add_argument("--output-root", type=Path, default=Path(file_config.get("output_root", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--run-id", default=file_config.get("run_id"))
    parser.add_argument("--dry-run", action="store_true", default=bool(file_config.get("dry_run", False)))
    args = parser.parse_args(argv_list)

    run_id = args.run_id or f"{_stamp()}_{args.dataset.lower()}_{args.rounds}r"
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    command = _build_rq1_command(args, output_dir)
    env = os.environ.copy()
    if args.gpu == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.setdefault("NUM_WORKERS_OVERRIDE", "0")
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("POL_REPRO_RUN_ID", run_id)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _dt.datetime.now().isoformat(),
        "code_root": str(CODE_ROOT),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "command": command,
        "config_source": {
            "path": str(args.config_file.resolve()) if args.config_file else None,
            "sha256": _sha256(args.config_file.resolve()) if args.config_file else None,
            "raw": file_config if args.config_file else None,
        },
        "environment_overrides": {
            key: env.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUM_WORKERS_OVERRIDE",
                "POL_REPRO_RUN_ID",
            ]
        },
        "config": {
            "dataset": args.dataset,
            "model": args.model or _auto_model(args.dataset),
            "rounds": args.rounds,
            "num_clients": args.num_clients,
            "clients_per_round": args.clients_per_round,
            "data_distribution": args.data_distribution,
            "dirichlet_alpha": args.dirichlet_alpha,
            "local_epochs": args.local_epochs,
            "attacks": args.attacks.split(","),
            "baselines": args.baselines.split(","),
        },
        "environment_snapshot": _env_snapshot(args.python),
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"run_manifest": str(manifest_path), "dry_run": True}, indent=2))
        return 0

    log_path = output_dir / "runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(CODE_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)

    manifest["returncode"] = proc.returncode
    manifest["log_path"] = str(log_path)
    manifest["completed_at"] = _dt.datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"run_manifest": str(manifest_path), "log": str(log_path), "returncode": proc.returncode}, indent=2))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
