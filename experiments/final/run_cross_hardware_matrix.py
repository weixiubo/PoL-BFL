#!/usr/bin/env python3
"""Plan Table 11 and execute pairs backed by an explicit hardware map."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import source_identity
from experiments.final.supervision import supervised_gpu_command


@dataclass(frozen=True)
class HardwareCell:
    hardware_pair: str
    seed: int

    @property
    def run_id(self) -> str:
        return (
            "formal-table11-"
            + self.hardware_pair.lower()
            + "-s"
            + str(self.seed)
        )


def plan_cross_hardware_cells(
    matrix: Mapping[str, object],
    *,
    hardware_pairs: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[HardwareCell, ...]:
    study = matrix["studies"]["table_11_cross_hardware"]
    selected_pairs = tuple(
        hardware_pairs or study["hardware_pairs"]
    )
    selected_seeds = tuple(
        int(seed) for seed in (seeds or matrix["seeds"])
    )
    if (
        set(selected_pairs) - set(study["hardware_pairs"])
        or set(selected_seeds)
        - {int(seed) for seed in matrix["seeds"]}
    ):
        raise ValueError("Table 11 matrix filter contains unknown values")
    return tuple(
        HardwareCell(pair, seed)
        for pair in selected_pairs
        for seed in selected_seeds
    )


def hardware_command(
    cell: HardwareCell,
    hardware: Mapping[str, Any],
    *,
    python: Path,
    output: Path,
    data_root: Path,
    zk_build: Path,
) -> list[str]:
    if cell.hardware_pair == "Kaizen_RTX4090_V100":
        raise ValueError(
            "the Kaizen hardware pair requires its controlled baseline runner"
        )
    required = {
        "trainer_device",
        "verifier_device",
        "expected_trainer",
        "expected_verifier",
    }
    if not required.issubset(hardware):
        raise ValueError("hardware map entry is incomplete")
    return [
        str(python),
        "-u",
        "-m",
        "experiments.final.run_cross_hardware_trial",
        "--hardware-pair",
        cell.hardware_pair,
        "--trainer-device",
        str(int(hardware["trainer_device"])),
        "--verifier-device",
        str(int(hardware["verifier_device"])),
        "--expected-trainer",
        str(hardware["expected_trainer"]),
        "--expected-verifier",
        str(hardware["expected_verifier"]),
        "--seed",
        str(cell.seed),
        "--output",
        str(output),
        "--data-root",
        str(data_root),
        "--zk-build",
        str(zk_build),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "experiments" / "final" / "paper_matrix.json",
    )
    parser.add_argument("--hardware-map", type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "experiments" / "results" / "final" / "table11",
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
    parser.add_argument("--hardware-pair", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cells = plan_cross_hardware_cells(
        matrix,
        hardware_pairs=args.hardware_pair,
        seeds=args.seed,
    )
    hardware_map = (
        {}
        if args.hardware_map is None
        else json.loads(
            args.hardware_map.read_text(encoding="utf-8")
        ).get("hardware_pairs", {})
    )
    source = source_identity(ROOT)
    plan = {
        "source": source,
        "cells": [
            cell.__dict__
            | {
                "run_id": cell.run_id,
                "available": bool(
                    hardware_map.get(
                        cell.hardware_pair, {}
                    ).get("available", False)
                ),
            }
            for cell in cells
        ],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if source["dirty"] or not source["commit"]:
        raise RuntimeError(
            "formal Table 11 execution requires a clean source"
        )
    unavailable = sorted(
        {
            cell.hardware_pair
            for cell in cells
            if not hardware_map.get(
                cell.hardware_pair, {}
            ).get("available", False)
        }
    )
    if unavailable:
        raise RuntimeError(
            "Table 11 hardware is unavailable: " + ", ".join(unavailable)
        )
    environment = os.environ.copy()
    environment.update(
        {
            "POL_INTEGRITY": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
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
                "refusing to overwrite rejected Table 11 result: "
                + str(result_path)
            )
        command = hardware_command(
                cell,
                hardware_map[cell.hardware_pair],
                python=args.python.resolve(),
                output=output,
                data_root=args.data_root.resolve(),
                zk_build=args.zk_build.resolve(),
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
