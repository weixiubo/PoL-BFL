"""Source-, environment-, dataset-, and artifact-bound experiment manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(args: Sequence[str], *, cwd: Path) -> str | None:
    try:
        process = subprocess.run(
            list(args),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def _version(command: Sequence[str], *, cwd: Path) -> str | None:
    output = _command(command, cwd=cwd)
    return None if not output else output.splitlines()[0]


def source_identity(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    prefix: tuple[str, ...] = ("git",)
    git_marker = root / ".git"
    if git_marker.is_file():
        marker = git_marker.read_text(encoding="utf-8").strip()
        if marker.startswith("gitdir:"):
            raw = marker.split(":", 1)[1].strip().replace("\\", "/")
            if raw.startswith("//wsl.localhost/"):
                parts = raw.split("/", 4)
                raw = "/" + parts[4] if len(parts) == 5 else raw
            git_dir = Path(raw)
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()
            prefix = ("git", f"--git-dir={git_dir}", f"--work-tree={root}")
    commit = _command((*prefix, "rev-parse", "HEAD"), cwd=root)
    status = _command((*prefix, "status", "--porcelain=v1", "--untracked-files=all"), cwd=root)
    diff = _command((*prefix, "diff", "--binary", "HEAD"), cwd=root)
    deployment_archive = False
    if commit is None and (root / ".polbfl-source-commit").is_file():
        commit = (root / ".polbfl-source-commit").read_text(encoding="utf-8").strip()
        status = ""
        diff = ""
        deployment_archive = True
    return {
        "commit": commit,
        "dirty": bool(status),
        "deployment_archive": deployment_archive,
        "status_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
        "diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
    }


def environment_identity(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        torch_info = {"error": f"{type(exc).__name__}:{exc}"}
    return {
        "captured_at_ns": time.time_ns(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu": platform.processor(),
        "torch": torch_info,
        "nvidia_smi": _command(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            cwd=root,
        ),
        "tools": {
            "circom": _version((os.getenv("CIRCOM_BIN", "circom"), "--version"), cwd=root),
            "node": _version((os.getenv("NODE_BIN", "node"), "--version"), cwd=root),
            "solc": _version((str(root / "node_modules" / ".bin" / "solcjs"), "--version"), cwd=root),
            "rapidsnark": os.getenv("RAPIDSNARK_COMMIT"),
        },
        "integrity_environment": {
            key: os.getenv(key)
            for key in (
                "POL_INTEGRITY",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "LD_PRELOAD",
                "POLBFL_SYSTEM_CXX_PRELOADED",
            )
        },
    }


def write_manifest_atomic(path: str | Path, manifest: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_run_manifest(
    *,
    root: str | Path,
    run_id: str,
    seed: int,
    configuration_files: Sequence[str | Path],
    dataset: Mapping[str, Any],
    artifacts: Sequence[str | Path] = (),
    runtime_artifacts: Sequence[str | Path] = (),
    run_parameters: Mapping[str, Any] | None = None,
    state: str = "created",
) -> dict[str, Any]:
    root = Path(root).resolve()
    if not run_id or state not in {"created", "running", "completed", "failed"}:
        raise ValueError("manifest run ID or state is invalid")
    def label(path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            return str(resolved)

    configs = {label(path): sha256_file(path) for path in configuration_files}
    artifact_hashes = {label(path): sha256_file(path) for path in artifacts}
    runtime_hashes = {label(path): sha256_file(path) for path in runtime_artifacts}
    body = {
        "schema_version": 2,
        "run_id": run_id,
        "state": state,
        "seed": int(seed),
        "source": source_identity(root),
        "environment": environment_identity(root),
        "configuration_sha256": configs,
        "runtime_artifact_sha256": runtime_hashes,
        "run_parameters": dict(run_parameters or {}),
        "dataset": dict(dataset),
        "artifact_sha256": artifact_hashes,
    }
    body["manifest_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body
