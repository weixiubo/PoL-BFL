#!/usr/bin/env python3
"""Aggregate source-bound cell results into paper-table observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


def aggregate_security_cells(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    results = tuple(results)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        required = {"dataset", "attack", "seed", "MA", "DR", "FPR", "source_commit"}
        if not required.issubset(result):
            raise ValueError("security cell result is missing required fields")
        if result.get("formal_accepted") is not True:
            raise ValueError("security cell result did not pass its formal acceptance gate")
        grouped[(str(result["dataset"]), str(result["attack"]))].append(result)
    table: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for (dataset, attack), rows in sorted(grouped.items()):
        seeds = [int(row["seed"]) for row in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seed for {dataset}/{attack}")
        metrics = {}
        for metric in ("MA", "DR", "FPR"):
            values = [float(row[metric]) for row in rows]
            mean = statistics.fmean(values)
            stderr = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            metrics[metric] = mean
            metrics[f"{metric}_ci95"] = 1.96 * stderr
        table.setdefault(dataset, {})[attack] = metrics
        provenance[f"{dataset}.{attack}"] = {
            "seeds": sorted(seeds),
            "source_commits": sorted({str(row["source_commit"]) for row in rows}),
        }
    source_commit = require_single_source_commit(
        results,
        context="PoL-BFL security aggregate",
    )
    return {
        "source_commit": source_commit,
        "table_2_pol_bfl": table,
        "provenance": provenance,
    }


def validate_security_aggregate(
    aggregate: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for dataset, attacks in aggregate["table_2_pol_bfl"].items():
        for attack, observed in attacks.items():
            target = targets["table_2_pol_bfl"][dataset][attack]
            prefix = f"{dataset}.{attack}"
            checks[f"{prefix}.MA"] = float(observed["MA"]) >= float(target["MA"])
            checks[f"{prefix}.DR"] = float(observed["DR"]) >= float(target["DR"])
            checks[f"{prefix}.FPR"] = float(observed["FPR"]) <= float(target["FPR"])
            seeds = aggregate["provenance"][prefix]["seeds"]
            checks[f"{prefix}.seed_count"] = len(seeds) == int(required_seed_count)
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "paper_targets.json",
    )
    parser.add_argument("--required-seed-count", type=int, default=3)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    aggregate = aggregate_security_cells(rows)
    aggregate["acceptance"] = validate_security_aggregate(
        aggregate,
        json.loads(args.targets.read_text(encoding="utf-8")),
        required_seed_count=args.required_seed_count,
    )
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.results
    }
    aggregate["formal_result_paths"] = sorted(
        str(path.resolve()) for path in args.results
    )
    aggregate = seal_evidence(
        aggregate,
        analysis_root=Path(__file__).resolve().parents[2],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
