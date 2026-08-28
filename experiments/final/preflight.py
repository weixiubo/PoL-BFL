#!/usr/bin/env python3
"""Fail-closed preflight for formal paper experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, source_identity
from experiments.final.evidence import seal_evidence
from experiments.final.target_provenance import (
    AUTHORITY_TARGET_FILES,
    target_paths,
    validate_all_target_files,
)
from experiments.final.trust_setup import validate_trust_setup
from polbfl.zk import PoseidonBridge


PAPER_SHA256 = "0b013e58d4f99f91470c61a891a4ee89dfd09eff58e131abbe730d1f6f91e6d4"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_preflight(
    *,
    root: Path,
    paper: Path,
    data_root: Path,
    zk_build: Path,
    require_clean: bool,
    require_idle_gpus: bool = True,
    poseidon_binary: Path | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"host": platform.node()}
    checks["paper_digest"] = paper.is_file() and sha256_file(paper) == PAPER_SHA256
    checks["cifar10_archive"] = (
        (data_root / "CIFAR10" / "cifar-10-python.tar.gz").is_file()
        and md5_file(data_root / "CIFAR10" / "cifar-10-python.tar.gz") == CIFAR10_MD5
    )
    checks["cifar100_archive"] = (
        (data_root / "CIFAR100" / "cifar-100-python.tar.gz").is_file()
        and md5_file(data_root / "CIFAR100" / "cifar-100-python.tar.gz") == CIFAR100_MD5
    )
    train_shards = sorted((data_root / "FEMNIST" / "train").glob("*.json"))
    test_shards = sorted((data_root / "FEMNIST" / "test").glob("*.json"))
    checks["femnist_leaf_shards"] = len(train_shards) == 36 and len(test_shards) == 36
    details["femnist_train_shards"] = len(train_shards)
    details["femnist_test_shards"] = len(test_shards)

    required_zk = (
        zk_build / "sampled_sgd_reference.r1cs",
        zk_build / "sampled_sgd_reference_final.zkey",
        zk_build / "verification_key.json",
        zk_build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
    )
    checks["reference_zk_artifacts"] = all(path.is_file() and path.stat().st_size > 0 for path in required_zk)
    details["reference_zk_sha256"] = {
        str(path.relative_to(zk_build)): sha256_file(path)
        for path in required_zk
        if path.is_file()
    }

    poseidon_binary = (
        root / ".tools" / "poseidon-native" / "polbfl-poseidon-native"
        if poseidon_binary is None
        else poseidon_binary.resolve()
    )
    checks["native_poseidon_binary"] = poseidon_binary.is_file() and os.access(
        poseidon_binary, os.X_OK
    )
    toolchain_lock = json.loads(
        (root / "config" / "toolchain.lock.json").read_text(encoding="utf-8")
    )
    trust_setup = validate_trust_setup(build=zk_build, toolchain=toolchain_lock)
    checks["production_trust_setup"] = trust_setup["passed"]
    details["trust_setup"] = trust_setup
    rapidsnark_lock = toolchain_lock["rapidsnark"]
    rapidsnark_root = root / ".tools" / "rapidsnark" / "package"
    rapidsnark_files = {
        "prover": (
            rapidsnark_root / "bin" / "prover",
            rapidsnark_lock["linux_x86_64_prover_sha256"],
        ),
        "verifier": (
            rapidsnark_root / "bin" / "verifier",
            rapidsnark_lock["linux_x86_64_verifier_sha256"],
        ),
        "shared_library": (
            rapidsnark_root / "lib" / "librapidsnark.so",
            rapidsnark_lock["linux_x86_64_shared_library_sha256"],
        ),
    }
    details["rapidsnark_sha256"] = {
        name: sha256_file(path)
        for name, (path, _expected) in rapidsnark_files.items()
        if path.is_file()
    }
    checks["rapidsnark_locked_artifacts"] = all(
        path.is_file() and details["rapidsnark_sha256"].get(name) == expected
        for name, (path, expected) in rapidsnark_files.items()
    )
    icicle_lock = toolchain_lock["icicle_snark"]
    icicle_root = root / ".tools" / "icicle-snark"
    icicle_files = {
        name: (icicle_root / name, expected)
        for name, expected in icicle_lock["artifacts"].items()
    }
    details["icicle_snark_sha256"] = {
        name: sha256_file(path)
        for name, (path, _expected) in icicle_files.items()
        if path.is_file()
    }
    checks["icicle_snark_locked_artifacts"] = all(
        path.is_file() and details["icicle_snark_sha256"].get(name) == expected
        for name, (path, expected) in icicle_files.items()
    )
    icicle_cargo_lock = root / "tools" / "icicle_snark" / "Cargo.lock"
    checks["icicle_snark_locked_sources"] = (
        icicle_cargo_lock.is_file()
        and sha256_file(icicle_cargo_lock) == icicle_lock["cargo_lock_sha256"]
    )
    icicle_memory_patch = (
        root / "tools" / "icicle_snark" / "0001-limit-msm-memory.patch"
    )
    checks["icicle_snark_memory_patch"] = icicle_memory_patch.is_file() and sha256_file(
        icicle_memory_patch
    ) == icicle_lock["memory_patch_sha256"]
    native_lock = toolchain_lock["native_poseidon"]
    cargo_lock = root / "tools" / "poseidon_native" / "Cargo.lock"
    if checks["native_poseidon_binary"]:
        details["native_poseidon_sha256"] = sha256_file(poseidon_binary)
        checks["native_poseidon_locked_binary"] = (
            details["native_poseidon_sha256"] == native_lock["linux_x86_64_sha256"]
        )
        checks["native_poseidon_locked_sources"] = (
            cargo_lock.is_file()
            and sha256_file(cargo_lock) == native_lock["cargo_lock_sha256"]
        )
        try:
            self_test = subprocess.run(
                [str(poseidon_binary), "--self-test"],
                text=True,
                capture_output=True,
                timeout=30,
                check=True,
            )
            details["native_poseidon_self_test"] = self_test.stdout.strip()
            checks["native_poseidon_self_test"] = bool(self_test.stdout.strip())
        except Exception as exc:
            checks["native_poseidon_self_test"] = False
            details["native_poseidon_self_test_error"] = f"{type(exc).__name__}:{exc}"
        operations = (
            {"kind": "fold2", "values": [-3, 0, 2**48 - 1], "initial": "9"},
            {"kind": "fold3", "rows": [[-1, 2], [3, -4]], "initial": "9"},
            {
                "kind": "fold_pair_chunks",
                "rows": [[index - 4, 2 * index + 1] for index in range(8)],
                "pairs_per_chunk": 4,
                "initial": "9",
            },
        )
        try:
            expected = PoseidonBridge().execute(operations)
            native = PoseidonBridge(native_binary=poseidon_binary).execute(operations)
            details["native_poseidon_crosscheck"] = {
                "circomlibjs": expected,
                "native": native,
            }
            checks["native_poseidon_crosscheck"] = native == expected
        except Exception as exc:
            checks["native_poseidon_crosscheck"] = False
            details["native_poseidon_crosscheck_error"] = f"{type(exc).__name__}:{exc}"
    else:
        checks["native_poseidon_locked_binary"] = False
        checks["native_poseidon_locked_sources"] = False
        checks["native_poseidon_self_test"] = False
        checks["native_poseidon_crosscheck"] = False

    try:
        import torch

        if os.getenv("POL_INTEGRITY") == "1":
            torch.use_deterministic_algorithms(True)
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        checks["dual_rtx4090"] = len(names) == 2 and all("4090" in name for name in names)
        checks["deterministic_torch"] = torch.are_deterministic_algorithms_enabled()
        details["gpus"] = names
    except Exception as exc:
        checks["dual_rtx4090"] = False
        checks["deterministic_torch"] = False
        details["torch_error"] = f"{type(exc).__name__}:{exc}"
    checks["integrity_mode"] = os.getenv("POL_INTEGRITY") == "1"
    try:
        busy = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.strip().splitlines()
        usage = [tuple(int(part.strip()) for part in line.split(",")) for line in busy]
        details["gpu_usage"] = [
            {"utilization_percent": utilization, "memory_used_mib": memory}
            for utilization, memory in usage
        ]
        checks["gpu_idle"] = (
            all(utilization <= 10 and memory <= 1024 for utilization, memory in usage)
            if require_idle_gpus
            else True
        )
    except Exception as exc:
        checks["gpu_idle"] = not require_idle_gpus
        details["gpu_usage_error"] = f"{type(exc).__name__}:{exc}"

    source = source_identity(root)
    checks["source_commit"] = bool(source["commit"])
    checks["clean_source"] = not source["dirty"] if require_clean else True
    details["source"] = source
    checks["paper_matrix"] = (root / "experiments" / "final" / "paper_matrix.json").is_file()
    checks["paper_targets"] = (root / "config" / "paper_targets.json").is_file()
    target_validation = validate_all_target_files(root)
    checks["dedicated_paper_targets"] = target_validation["passed"]
    details["dedicated_paper_targets"] = target_validation
    checks["contract_runtime"] = (
        (root / "chainEnv" / "contracts" / "PoLBFLProtocol.sol").is_file()
        and (root / "node_modules" / "solc").is_dir()
        and (root / "node_modules" / "ganache").is_dir()
    )
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def main() -> None:
    root = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--zk-build", type=Path, default=root / "circuits" / "final" / "build" / "production")
    parser.add_argument(
        "--poseidon-binary",
        type=Path,
        default=root / ".tools" / "poseidon-native" / "polbfl-poseidon-native",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_preflight(
        root=root,
        paper=args.paper,
        data_root=args.data_root,
        zk_build=args.zk_build,
        poseidon_binary=args.poseidon_binary,
        require_clean=not args.allow_dirty,
        require_idle_gpus=not args.allow_busy_gpus,
    )
    source_commit = report.get("details", {}).get("source", {}).get("commit")
    declared_inputs = (
        args.paper.resolve(),
        root / "config" / "paper_targets.json",
        *target_paths(root, AUTHORITY_TARGET_FILES),
        root / "config" / "toolchain.lock.json",
        root / "experiments" / "final" / "paper_matrix.json",
        (args.zk_build / "trust_setup.json").resolve(),
    )
    report["input_sha256"] = {
        (
            str(path.relative_to(root))
            if path.is_relative_to(root)
            else str(path)
        ): sha256_file(path)
        for path in declared_inputs
        if path.is_file()
    }
    report = seal_evidence(
        report,
        source_commit=source_commit,
        analysis_source=report["details"]["source"],
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
