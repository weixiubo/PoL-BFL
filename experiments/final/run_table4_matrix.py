#!/usr/bin/env python3
"""Plan and execute both standalone and PoL-prefilter Table 4 modes."""

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


AGGREGATION_METHODS = {
    "Krum": "krum",
    "TrimmedMean": "trimmed_mean",
    "Median": "median",
}
MODES = ("Standalone", "PoLBFLPrefilter")


@dataclass(frozen=True)
class Table4Cell:
    aggregation: str
    attack: str
    mode: str
    seed: int

    @property
    def method(self) -> str:
        if self.mode == "PoLBFLPrefilter":
            return "PoLBFL"
        return {
            "krum": "Krum",
            "trimmed_mean": "TrimmedMean",
            "median": "Median",
        }[self.aggregation]

    @property
    def run_id(self) -> str:
        return (
            f"formal-table4-{self.mode.lower()}-{self.aggregation}-"
            f"{self.attack.lower()}-s{self.seed}"
        )


def plan_table4_cells(
    matrix: Mapping[str, object],
    *,
    aggregations: Sequence[str] | None = None,
    attacks: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[Table4Cell, ...]:
    study = matrix["studies"]["table_4_composability"]
    selected_aggregations = tuple(aggregations or AGGREGATION_METHODS.values())
    selected_attacks = tuple(attacks or study["attacks"])
    selected_modes = tuple(modes or MODES)
    selected_seeds = tuple(int(seed) for seed in (seeds or matrix["seeds"]))
    if (
        set(selected_aggregations) - set(AGGREGATION_METHODS.values())
        or set(selected_attacks) - set(study["attacks"])
        or set(selected_modes) - set(MODES)
        or set(selected_seeds) - {int(seed) for seed in matrix["seeds"]}
    ):
        raise ValueError("Table 4 matrix filter contains unknown values")
    return tuple(
        Table4Cell(aggregation, attack, mode, seed)
        for aggregation in selected_aggregations
        for attack in selected_attacks
        for mode in selected_modes
        for seed in selected_seeds
    )


def table4_command(
    cell: Table4Cell,
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
        "composability",
        "--dataset",
        "CIFAR10",
        "--attack",
        cell.attack,
        "--method",
        cell.method,
        "--aggregation-method",
        cell.aggregation,
        "--composition-mode",
        cell.mode,
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
        default=ROOT / "experiments" / "results" / "final" / "table4",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--zk-build",
        type=Path,
        default=ROOT / "circuits" / "final" / "build" / "production",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--aggregation", action="append")
    parser.add_argument("--attack", action="append")
    parser.add_argument("--mode", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cells = plan_table4_cells(
        matrix,
        aggregations=args.aggregation,
        attacks=args.attack,
        modes=args.mode,
        seeds=args.seed,
    )
    plan = {
        "source": source_identity(ROOT),
        "cells": [cell.__dict__ | {"method": cell.method, "run_id": cell.run_id} for cell in cells],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if plan["source"]["dirty"] or not plan["source"]["commit"]:
        raise RuntimeError("formal Table 4 execution requires a clean, identified source")
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
            if json.loads(result_path.read_text(encoding="utf-8")).get("formal_accepted") is True:
                continue
            raise RuntimeError(f"refusing to overwrite rejected Table 4 result: {result_path}")
        command = table4_command(
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
        if json.loads(result_path.read_text(encoding="utf-8")).get("formal_accepted") is not True:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
