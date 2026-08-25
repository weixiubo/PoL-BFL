#!/usr/bin/env python3
"""Aggregate both standalone and PoL-prefilter cells into Table 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


AGGREGATION_LABELS = {
    "krum": "Krum",
    "trimmed_mean": "TrimmedMean",
    "median": "Median",
}


def aggregate_table4(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 4")
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        required = {
            "aggregation_method",
            "attack",
            "composition_mode",
            "method",
            "seed",
            "MA",
            "DR",
            "FPR",
            "source_commit",
        }
        if not required.issubset(result):
            raise ValueError("Table 4 result is incomplete")
        if result.get("formal_accepted") is not True or result.get("study") != "composability":
            raise ValueError("Table 4 aggregate requires accepted composability cells")
        aggregation = AGGREGATION_LABELS[str(result["aggregation_method"])]
        attack = str(result["attack"])
        mode = str(result["composition_mode"])
        expected_method = "PoLBFL" if mode == "PoLBFLPrefilter" else aggregation
        if str(result["method"]) != expected_method:
            raise ValueError("Table 4 mode and executed method differ")
        groups[(aggregation, attack, mode)].append(result)
    target_table = targets["table_4_all_modes"]
    expected = {
        (aggregation, attack, mode)
        for aggregation, attacks in target_table.items()
        for attack, modes in attacks.items()
        for mode in modes
    }
    if set(groups) != expected:
        raise ValueError("Table 4 aggregate does not cover every mode/cell")

    table: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    provenance: dict[str, Any] = {}
    for (aggregation, attack, mode), rows in sorted(groups.items()):
        seeds = [int(row["seed"]) for row in rows]
        if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(f"Table 4 seed set is incomplete: {aggregation}/{attack}/{mode}")
        observed = {
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in ("MA", "DR", "FPR")
        }
        table.setdefault(aggregation, {}).setdefault(attack, {})[mode] = observed
        target = target_table[aggregation][attack][mode]
        prefix = f"{aggregation}.{attack}.{mode}"
        checks[f"{prefix}.MA"] = observed["MA"] + 1e-9 >= float(target["MA"])
        checks[f"{prefix}.DR"] = observed["DR"] + 1e-9 >= float(target["DR"])
        checks[f"{prefix}.FPR"] = observed["FPR"] <= float(target["FPR"]) + 1e-9
        provenance[prefix] = {
            "seeds": sorted(seeds),
            "source_commits": sorted({str(row["source_commit"]) for row in rows}),
        }
    return {
        "source_commit": source_commit,
        "table_4_all_modes": table,
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
        default=root / "config" / "paper_table4_all_modes.json",
    )
    parser.add_argument("--required-seed-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_table4(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
        required_seed_count=args.required_seed_count,
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
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
