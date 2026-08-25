#!/usr/bin/env python3
"""Aggregate all six methods into the complete final-paper Table 2."""

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


def aggregate_table2(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
    require_complete: bool = True,
) -> dict[str, Any]:
    results = tuple(results)
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        required = {
            "dataset",
            "attack",
            "method",
            "seed",
            "MA",
            "DR",
            "FPR",
            "source_commit",
        }
        if not required.issubset(result):
            raise ValueError("Table 2 result is missing required fields")
        if result.get("formal_accepted") is not True or result.get("study") != "main":
            raise ValueError("Table 2 aggregate requires accepted main-study cells")
        key = (
            str(result["dataset"]),
            str(result["attack"]),
            str(result["method"]),
        )
        groups[key].append(result)
    target_table = targets["table_2_all_methods"]
    target_cells = {
        (dataset, attack, method)
        for dataset, attacks in target_table.items()
        for attack, methods in attacks.items()
        for method in methods
    }
    if require_complete and set(groups) != target_cells:
        missing = sorted(target_cells - set(groups))
        extra = sorted(set(groups) - target_cells)
        raise ValueError(f"Table 2 cell coverage differs: missing={missing}, extra={extra}")
    source_commit = require_single_source_commit(results, context="Table 2")

    table: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for (dataset, attack, method), rows in sorted(groups.items()):
        if (dataset, attack, method) not in target_cells:
            raise ValueError(f"unexpected Table 2 cell: {dataset}/{attack}/{method}")
        seeds = [int(row["seed"]) for row in rows]
        if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(f"Table 2 seed set is incomplete: {dataset}/{attack}/{method}")
        observed = {}
        for metric in ("MA", "DR", "FPR"):
            values = [float(row[metric]) for row in rows]
            mean = statistics.fmean(values)
            stderr = (
                statistics.stdev(values) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            observed[metric] = mean
            observed[f"{metric}_ci95"] = 1.96 * stderr
        table.setdefault(dataset, {}).setdefault(attack, {})[method] = observed
        target = target_table[dataset][attack][method]
        prefix = f"{dataset}.{attack}.{method}"
        checks[f"{prefix}.MA"] = observed["MA"] >= float(target["MA"])
        if "DR" in target:
            checks[f"{prefix}.DR"] = observed["DR"] >= float(target["DR"])
            checks[f"{prefix}.FPR"] = observed["FPR"] <= float(target["FPR"])
        provenance[prefix] = {
            "seeds": sorted(seeds),
            "source_commits": sorted({str(row["source_commit"]) for row in rows}),
        }
    checks["complete_table_2"] = not require_complete or set(groups) == target_cells
    return {
        "source_commit": source_commit,
        "table_2_all_methods": table,
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
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table2_all_methods.json",
    )
    parser.add_argument("--required-seed-count", type=int, default=3)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_table2(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
        required_seed_count=args.required_seed_count,
        require_complete=not args.allow_partial,
    )
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.results
    }
    aggregate["formal_result_paths"] = sorted(
        str(path.resolve()) for path in args.results
    )
    aggregate = seal_evidence(aggregate, analysis_root=root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
