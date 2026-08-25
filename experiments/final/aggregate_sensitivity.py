#!/usr/bin/env python3
"""Aggregate Figure 4 spot-check sensitivity observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.run_sensitivity_matrix import PAPER_PROBABILITIES
from experiments.final.evidence import require_single_source_commit, seal_evidence


def aggregate_sensitivity_cells(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Figure 4")
    groups: dict[Decimal, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("formal_accepted") is not True or result.get("study") != "sensitivity":
            raise ValueError("sensitivity aggregate requires accepted formal sensitivity cells")
        groups[Decimal(str(result["audit_probability"]))].append(result)
    if set(groups) != set(PAPER_PROBABILITIES):
        raise ValueError("sensitivity aggregate does not cover every paper plot probability")
    table: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    provenance: dict[str, Any] = {}
    prior: Mapping[str, float] | None = None
    for probability in PAPER_PROBABILITIES:
        rows = groups[probability]
        seeds = [int(row["seed"]) for row in rows]
        if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(f"sensitivity seed set is incomplete: p={probability}")
        observed = {
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in ("MA", "DR", "FPR", "runtime_seconds")
        }
        key = format(probability, "f")
        table[key] = observed
        provenance[key] = {
            "seeds": sorted(seeds),
            "source_commits": sorted({str(row["source_commit"]) for row in rows}),
        }
        if prior is not None:
            checks[f"{key}.DR_nondecreasing"] = observed["DR"] + 1e-9 >= prior["DR"]
            checks[f"{key}.runtime_nondecreasing_with_2s_tolerance"] = (
                observed["runtime_seconds"] + 2.0 >= prior["runtime_seconds"]
            )
        prior = observed
    default = table["0.20"]
    target = targets["table_2_pol_bfl"]["CIFAR10"]["FreeRidingNT"]
    overhead = targets["table_7_overhead"]
    checks.update(
        {
            "default.MA": default["MA"] >= float(target["MA"]),
            "default.DR": default["DR"] >= float(target["DR"]),
            "default.FPR": default["FPR"] <= float(target["FPR"]),
            "default.runtime_seconds": default["runtime_seconds"]
            <= float(overhead["runtime_seconds"]),
        }
    )
    return {
        "source_commit": source_commit,
        "figure_4_spot_check_sensitivity": table,
        "provenance": provenance,
        "acceptance": {
            "passed": bool(checks) and all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--targets", type=Path, default=root / "config" / "paper_targets.json")
    parser.add_argument("--required-seed-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_sensitivity_cells(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
        required_seed_count=args.required_seed_count,
    )
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.results
    }
    aggregate["formal_result_paths"] = sorted(
        str(path.resolve()) for path in args.results
    )
    aggregate = seal_evidence(aggregate, analysis_root=root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
