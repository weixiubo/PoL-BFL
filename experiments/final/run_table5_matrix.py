#!/usr/bin/env python3
"""Plan and execute all four final-paper Table 5 incentive methods."""

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


METHODS = {
    "Vanilla": "VanillaFL",
    "FedCoin": "FedCoin",
    "ShapleyFL": "ShapleyFL",
    "PoLBFL": "PoLBFL",
}


@dataclass(frozen=True)
class Table5Cell:
    method: str
    seed: int

    @property
    def runner_method(self) -> str:
        return METHODS[self.method]

    @property
    def run_id(self) -> str:
        return (
            "formal-table5-"
            + self.method.lower()
            + "-s"
            + str(self.seed)
        )


def plan_table5_cells(
    matrix: Mapping[str, object],
    *,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[Table5Cell, ...]:
    study = matrix["studies"]["table_5_incentive_effectiveness"]
    selected_methods = tuple(methods or study["methods"])
    selected_seeds = tuple(
        int(seed) for seed in (seeds or matrix["seeds"])
    )
    if (
        set(selected_methods) - set(METHODS)
        or set(selected_seeds)
        - {int(seed) for seed in matrix["seeds"]}
    ):
        raise ValueError("Table 5 matrix filter contains unknown values")
    return tuple(
        Table5Cell(method, seed)
        for method in selected_methods
        for seed in selected_seeds
    )


def table5_command(
    cell: Table5Cell,
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
        "incentive",
        "--dataset",
        "CIFAR10",
        "--attack",
        "FreeRidingNT",
        "--method",
        cell.runner_method,
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
        default=ROOT / "experiments" / "results" / "final" / "table5",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--zk-build",
        type=Path,
        default=ROOT
        / "circuits"
        / "final"
        / "build"
        / "production",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cells = plan_table5_cells(
        matrix,
        methods=args.method,
        seeds=args.seed,
    )
    source = source_identity(ROOT)
    plan = {
        "source": source,
        "cells": [
            cell.__dict__
            | {
                "runner_method": cell.runner_method,
                "run_id": cell.run_id,
            }
            for cell in cells
        ],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if source["dirty"] or not source["commit"]:
        raise RuntimeError(
            "formal Table 5 execution requires a clean, identified source"
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
                "refusing to overwrite rejected Table 5 result: "
                + str(result_path)
            )
        command = table5_command(
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
