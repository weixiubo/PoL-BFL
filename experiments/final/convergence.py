#!/usr/bin/env python3
"""Derive final-paper Figure 2 from accepted CIFAR-10 security cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.capture_formal_evidence import verify_completed_cell
from experiments.final.evidence import require_single_source_commit, seal_evidence
from experiments.final.target_provenance import (
    FIGURE2_TARGET_FILES,
    load_merged_targets,
)


METHODS = ("VanillaFL", "Krum", "SDEA", "ShapleyFL", "FoolsGold", "PoLBFL")
ATTACKS = ("FreeRidingNT", "ALIE", "Sybil")
SAMPLE_ROUNDS = (0, 50, 100, 150, 200)


def aggregate_convergence(
    trials: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
    sample_rounds: Sequence[int] = SAMPLE_ROUNDS,
) -> dict[str, Any]:
    trials = tuple(trials)
    if "figure_2_convergence" not in targets:
        raise ValueError("Figure 2 aggregation requires PDF-derived vector targets")
    source_commit = require_single_source_commit(trials, context="Figure 2")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for trial in trials:
        required = {
            "method",
            "attack",
            "seed",
            "initial_accuracy",
            "rounds",
            "source_commit",
            "manifest_digest",
        }
        if not required.issubset(trial) or trial.get("formal_accepted") is not True:
            raise ValueError("convergence input is not an accepted formal trial")
        method = str(trial["method"])
        attack = str(trial["attack"])
        if method not in METHODS or attack not in ATTACKS:
            raise ValueError("convergence method or attack is outside Figure 2")
        round_rows = list(trial["rounds"])
        if len(round_rows) != 200 or [int(row["round"]) for row in round_rows] != list(
            range(200)
        ):
            raise ValueError("convergence trial must contain all 200 rounds")
        if any("accuracy" not in row for row in round_rows):
            raise ValueError("convergence trial lacks measured round accuracy")
        groups[(attack, method)].append(trial)
    expected = {(attack, method) for attack in ATTACKS for method in METHODS}
    if set(groups) != expected:
        raise ValueError("Figure 2 does not cover every attack and method")
    if tuple(sample_rounds) != SAMPLE_ROUNDS:
        raise ValueError("Figure 2 sampling points differ from the final paper")

    figure: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    provenance: dict[str, Any] = {}
    for attack in ATTACKS:
        for method in METHODS:
            rows = groups[(attack, method)]
            seeds = [int(row["seed"]) for row in rows]
            if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
                raise ValueError(f"Figure 2 seed set is incomplete: {attack}/{method}")
            points = []
            for round_number in sample_rounds:
                values = (
                    [float(row["initial_accuracy"]) for row in rows]
                    if round_number == 0
                    else [
                        float(row["rounds"][round_number - 1]["accuracy"])
                        for row in rows
                    ]
                )
                points.append({"round": int(round_number), "MA": statistics.fmean(values)})
            figure.setdefault(attack, {})[method] = points
            vector = targets["figure_2_convergence"][attack][method]
            if [int(point["round"]) for point in vector] != list(sample_rounds):
                raise ValueError(f"Figure 2 target rounds differ: {attack}/{method}")
            for observed, expected in zip(points, vector):
                checks[
                    f"{attack}.{method}.round_{observed['round']}.MA"
                ] = float(observed["MA"]) >= float(expected["MA"])
            target = targets["table_2_all_methods"]["CIFAR10"][attack][method]
            checks[f"{attack}.{method}.final_MA"] = points[-1]["MA"] >= float(target["MA"])
            provenance[f"{attack}.{method}"] = {
                "seeds": sorted(seeds),
                "source_commits": sorted({str(row["source_commit"]) for row in rows}),
                "manifest_digests": sorted(str(row["manifest_digest"]) for row in rows),
            }
    return {
        "source_commit": source_commit,
        "figure_2_convergence": figure,
        "provenance": provenance,
        "acceptance": {
            "passed": bool(checks) and all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
    }


def _trial_from_result(result_path: Path) -> dict[str, Any]:
    evidence = verify_completed_cell(result_path, root=ROOT)
    result = evidence["result"]
    rounds_path = result_path.with_name("rounds.jsonl")
    round_rows = [
        json.loads(line)
        for line in rounds_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        **result,
        "rounds": round_rows,
        "manifest_digest": evidence["manifest_digest"],
        "rounds_sha256": hashlib.sha256(rounds_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    targets = (
        json.loads(args.targets.read_text(encoding="utf-8"))
        if args.targets is not None
        else load_merged_targets(ROOT, FIGURE2_TARGET_FILES)
    )
    aggregate = aggregate_convergence(
        [_trial_from_result(path.resolve()) for path in args.results],
        targets,
    )
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.results
    }
    aggregate["formal_result_paths"] = sorted(
        str(path.resolve()) for path in args.results
    )
    aggregate = seal_evidence(aggregate, analysis_root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
