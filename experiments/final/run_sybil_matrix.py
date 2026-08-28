#!/usr/bin/env python3
"""Plan and execute final-paper Figure 6 Sybil-identity scaling cells."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import source_identity
from experiments.final.supervision import supervised_gpu_command


@dataclass(frozen=True)
class SybilCell:
    dataset: str
    identity_count: int
    seed: int

    @property
    def num_clients(self) -> int:
        return 40 + self.identity_count

    @property
    def run_id(self) -> str:
        dataset = (
            "" if self.dataset == "CIFAR10" else self.dataset.lower() + "-"
        )
        return (
            "formal-figure6-"
            + dataset
            + "sybil-"
            + str(self.identity_count)
            + "-s"
            + str(self.seed)
        )


def plan_sybil_cells(
    matrix: Mapping[str, object],
    *,
    datasets: Sequence[str] | None = None,
    identity_counts: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[SybilCell, ...]:
    study = matrix["studies"]["figure_6_sybil_scalability"]
    allowed_datasets = tuple(
        study.get("datasets", (study.get("dataset", "CIFAR10"),))
    )
    selected_datasets = tuple(datasets or allowed_datasets)
    selected_counts = tuple(
        int(value)
        for value in (
            identity_counts or study["identities_per_attacker"]
        )
    )
    selected_seeds = tuple(
        int(seed) for seed in (seeds or matrix["seeds"])
    )
    if (
        set(selected_datasets) - set(allowed_datasets)
        or set(selected_counts)
        - {int(value) for value in study["identities_per_attacker"]}
        or set(selected_seeds)
        - {int(seed) for seed in matrix["seeds"]}
    ):
        raise ValueError("Figure 6 matrix filter contains unknown values")
    return tuple(
        SybilCell(dataset, identity_count, seed)
        for dataset in selected_datasets
        for identity_count in selected_counts
        for seed in selected_seeds
    )


def sybil_command(
    cell: SybilCell,
    *,
    python: Path,
    output: Path,
    data_root: Path,
    zk_build: Path,
    resume: bool = False,
) -> list[str]:
    command = [
        str(python),
        "-u",
        "-m",
        "experiments.final.run_security_cell",
        "--study",
        "sybil_scalability",
        "--dataset",
        cell.dataset,
        "--attack",
        "Sybil",
        "--method",
        "PoLBFL",
        "--sybil-identities",
        str(cell.identity_count),
        "--num-clients",
        str(cell.num_clients),
        "--num-malicious",
        str(cell.identity_count),
        "--clients-per-round",
        str(cell.num_clients),
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
        "8",
        "--proof-workers",
        "8",
    ]
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "experiments" / "final" / "paper_matrix.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "experiments" / "results" / "final" / "figure6",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--zk-build",
        type=Path,
        default=ROOT / "circuits" / "final" / "build" / "production",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--identity-count", action="append", type=int)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cells = plan_sybil_cells(
        matrix,
        datasets=args.dataset,
        identity_counts=args.identity_count,
        seeds=args.seed,
    )
    source = source_identity(ROOT)
    plan = {
        "source": source,
        "cells": [
            cell.__dict__
            | {"num_clients": cell.num_clients, "run_id": cell.run_id}
            for cell in cells
        ],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if source["dirty"] or not source["commit"]:
        raise RuntimeError(
            "formal Figure 6 execution requires a clean, identified source"
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
    for cell in cells:
        output = args.results_root.resolve() / cell.run_id
        result_path = output / "result.json"
        if result_path.is_file():
            if json.loads(
                result_path.read_text(encoding="utf-8")
            ).get("formal_accepted") is True:
                continue
            raise RuntimeError(
                "refusing to overwrite rejected Figure 6 result: "
                + str(result_path)
            )
        command = sybil_command(
                cell,
                python=args.python.resolve(),
                output=output,
                data_root=args.data_root.resolve(),
                zk_build=args.zk_build.resolve(),
                resume=(output / "checkpoint.pt").is_file(),
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
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        if json.loads(
            result_path.read_text(encoding="utf-8")
        ).get("formal_accepted") is not True:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
