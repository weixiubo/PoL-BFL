#!/usr/bin/env python3
"""Plan and execute final-paper Figure 4 spot-check sensitivity cells."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import source_identity
from experiments.final.supervision import supervised_gpu_command


PAPER_PROBABILITIES = (
    Decimal("0.05"),
    Decimal("0.10"),
    Decimal("0.15"),
    Decimal("0.20"),
    Decimal("0.25"),
    Decimal("0.30"),
    Decimal("0.50"),
    Decimal("1.00"),
)


@dataclass(frozen=True)
class SensitivityCell:
    audit_probability: Decimal
    seed: int

    @property
    def run_id(self) -> str:
        probability = str(int(self.audit_probability * 100)).zfill(3)
        return f"formal-sensitivity-cifar10-p{probability}-s{self.seed}"


def plan_sensitivity_cells(
    matrix: Mapping[str, object],
    *,
    probabilities: Sequence[Decimal] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[SensitivityCell, ...]:
    selected_probabilities = tuple(probabilities or PAPER_PROBABILITIES)
    selected_seeds = tuple(int(value) for value in (seeds or matrix["seeds"]))
    if set(selected_probabilities) - set(PAPER_PROBABILITIES) or set(selected_seeds) - {
        int(value) for value in matrix["seeds"]
    }:
        raise ValueError("sensitivity matrix filter contains unknown values")
    return tuple(
        SensitivityCell(probability, seed)
        for probability in selected_probabilities
        for seed in selected_seeds
    )


def sensitivity_command(
    cell: SensitivityCell,
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
        "sensitivity",
        "--dataset",
        "CIFAR10",
        "--attack",
        "FreeRidingNT",
        "--audit-probability",
        str(cell.audit_probability),
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
    parser.add_argument("--matrix", type=Path, default=ROOT / "experiments" / "final" / "paper_matrix.json")
    parser.add_argument("--results-root", type=Path, default=ROOT / "experiments" / "results" / "final" / "sensitivity")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--zk-build", type=Path, default=ROOT / "circuits" / "final" / "build" / "production")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--probability", action="append", type=Decimal)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cells = plan_sensitivity_cells(
        matrix,
        probabilities=args.probability,
        seeds=args.seed,
    )
    plan = {
        "source": source_identity(ROOT),
        "cells": [
            {
                "audit_probability": str(cell.audit_probability),
                "seed": cell.seed,
                "run_id": cell.run_id,
            }
            for cell in cells
        ],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if plan["source"]["dirty"] or not plan["source"]["commit"]:
        raise RuntimeError("formal sensitivity execution requires a clean, identified source")
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
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("formal_accepted") is True:
                continue
            raise RuntimeError(f"refusing to overwrite rejected sensitivity result: {result_path}")
        command = sensitivity_command(
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
