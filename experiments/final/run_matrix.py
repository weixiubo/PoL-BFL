#!/usr/bin/env python3
"""Plan and execute the source-bound PoL-BFL paper security matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, source_identity
from experiments.final.supervision import supervised_gpu_command


@dataclass(frozen=True)
class MatrixCell:
    dataset: str
    attack: str
    method: str
    seed: int
    run_id: str


def _slug(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def plan_cells(
    matrix: Mapping[str, object],
    *,
    datasets: Sequence[str] | None = None,
    attacks: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[MatrixCell, ...]:
    selected_datasets = tuple(datasets or matrix["datasets"].keys())
    selected_attacks = tuple(attacks or matrix["attacks"])
    reference_methods = tuple(str(method) for method in matrix["baselines"])
    selected_methods = tuple(
        methods
        or ("PoLBFL", *(method for method in reference_methods if method != "PoLBFL"))
    )
    selected_seeds = tuple(int(seed) for seed in (seeds or matrix["seeds"]))
    unknown_datasets = set(selected_datasets) - set(matrix["datasets"])
    unknown_attacks = set(selected_attacks) - set(matrix["attacks"])
    unknown_methods = set(selected_methods) - set(reference_methods)
    unknown_seeds = set(selected_seeds) - {int(seed) for seed in matrix["seeds"]}
    if unknown_datasets or unknown_attacks or unknown_methods or unknown_seeds:
        raise ValueError(
            "matrix filter contains unknown values: "
            f"datasets={sorted(unknown_datasets)}, "
            f"attacks={sorted(unknown_attacks)}, methods={sorted(unknown_methods)}, "
            f"seeds={sorted(unknown_seeds)}"
        )
    return tuple(
        MatrixCell(
            dataset=dataset,
            attack=attack,
            method=method,
            seed=seed,
            run_id=f"formal-{_slug(dataset)}-{_slug(attack)}-{_slug(method)}-s{seed}",
        )
        for dataset in selected_datasets
        for attack in selected_attacks
        for method in selected_methods
        for seed in selected_seeds
    )


def cell_command(
    cell: MatrixCell,
    *,
    python: Path,
    output: Path,
    data_root: Path,
    zk_build: Path,
    train_processes_per_gpu: int = 8,
    proof_workers: int = 8,
    resume: bool = False,
) -> list[str]:
    command = [
        str(python),
        "-u",
        "-m",
        "experiments.final.run_security_cell",
        "--dataset",
        cell.dataset,
        "--attack",
        cell.attack,
        "--method",
        cell.method,
        "--seed",
        str(cell.seed),
        "--run-id",
        cell.run_id,
        "--output",
        str(output),
        "--data-root",
        str(data_root),
        "--zk-build",
        str(zk_build),
        "--process-training",
        "--train-processes-per-gpu",
        str(train_processes_per_gpu),
        "--proof-workers",
        str(proof_workers),
    ]
    if resume:
        command.append("--resume")
    return command


def _accepted_result(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("formal_accepted") is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=ROOT / "experiments" / "final" / "paper_matrix.json")
    parser.add_argument("--results-root", type=Path, default=ROOT / "experiments" / "results" / "final")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--zk-build", type=Path, default=ROOT / "circuits" / "final" / "build" / "production")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--attack", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--train-processes-per-gpu", type=int, default=8)
    parser.add_argument("--proof-workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cells = plan_cells(
        matrix,
        datasets=args.dataset,
        attacks=args.attack,
        methods=args.method,
        seeds=args.seed,
    )
    plan = {
        "schema_version": 1,
        "source": source_identity(ROOT),
        "matrix_sha256": sha256_file(args.matrix),
        "cells": [asdict(cell) for cell in cells],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if plan["source"]["dirty"] or not plan["source"]["commit"]:
        raise RuntimeError(
            "formal matrix execution requires a clean, identified source tree"
        )
    environment = os.environ.copy()
    environment.update(
        {
            "POL_INTEGRITY": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "OMP_NUM_THREADS": "2",
        }
    )
    failures = []
    for cell in cells:
        output = args.results_root.resolve() / cell.run_id
        result_path = output / "result.json"
        if _accepted_result(result_path):
            continue
        if result_path.exists():
            raise RuntimeError(f"refusing to overwrite a rejected result: {result_path}")
        resume = (output / "checkpoint.pt").is_file()
        command = cell_command(
            cell,
            python=args.python.resolve(),
            output=output,
            data_root=args.data_root.resolve(),
            zk_build=args.zk_build.resolve(),
            train_processes_per_gpu=args.train_processes_per_gpu,
            proof_workers=args.proof_workers,
            resume=resume,
        )
        command = supervised_gpu_command(
            command,
            python=args.python,
            root=ROOT,
            run_dir=output,
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
        )
        accepted = completed.returncode == 0 and _accepted_result(result_path)
        if not accepted:
            failures.append(
                {
                    "run_id": cell.run_id,
                    "returncode": completed.returncode,
                    "result": str(result_path),
                }
            )
            if not args.continue_on_failure:
                break
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
