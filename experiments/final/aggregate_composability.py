#!/usr/bin/env python3
"""Aggregate accepted composition cells into final-paper Table 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


AGGREGATION_LABELS = {"krum": "Krum", "trimmed_mean": "TrimmedMean", "median": "Median"}


def aggregate_composability_cells(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 4 composition")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("formal_accepted") is not True or result.get("study") != "composability":
            raise ValueError("composition aggregate requires accepted formal cells")
        groups[(str(result["aggregation_method"]), str(result["attack"]))].append(result)
    table: dict[str, Any] = {}
    checks = {}
    provenance = {}
    for (aggregation, attack), rows in sorted(groups.items()):
        if len(rows) != required_seed_count or len({int(row["seed"]) for row in rows}) != len(rows):
            raise ValueError(f"composition seed set is incomplete: {aggregation}/{attack}")
        aggregation_label = AGGREGATION_LABELS[aggregation]
        attack_label = "FreeRiding" if attack == "FreeRidingNT" else attack
        observed = {
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in ("MA", "DR", "FPR")
        }
        table.setdefault(aggregation_label, {})[attack_label] = observed
        target = targets["table_4_composability_cifar10"][aggregation_label][attack_label]
        prefix = f"{aggregation_label}.{attack_label}"
        checks[f"{prefix}.MA"] = observed["MA"] >= float(target["MA"])
        checks[f"{prefix}.DR"] = observed["DR"] >= float(target["DR"])
        checks[f"{prefix}.FPR"] = observed["FPR"] <= float(target["FPR"])
        provenance[prefix] = {
            "seeds": sorted(int(row["seed"]) for row in rows),
            "source_commits": sorted({str(row["source_commit"]) for row in rows}),
        }
    return {
        "source_commit": source_commit,
        "table_4_composability_cifar10": table,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = aggregate_composability_cells(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    output["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.results
    }
    output["formal_result_paths"] = sorted(
        str(path.resolve()) for path in args.results
    )
    output = seal_evidence(output, analysis_root=root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not output["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
