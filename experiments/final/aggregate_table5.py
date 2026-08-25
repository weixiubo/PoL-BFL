#!/usr/bin/env python3
"""Aggregate complete incentive-method evidence into final-paper Table 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


METHODS = ("Vanilla", "FedCoin", "ShapleyFL", "PoLBFL")


def aggregate_table5(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 5")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        required = {
            "method",
            "seed",
            "ParticipationRate",
            "AttackSuccessRate",
            "ModelAccuracy",
            "source_commit",
            "evidence_digest",
        }
        if not required.issubset(result) or result.get("formal_accepted") is not True:
            raise ValueError("Table 5 result is not accepted or is incomplete")
        method = str(result.get("table5_method", result["method"]))
        if method not in METHODS or len(str(result["evidence_digest"])) != 64:
            raise ValueError("Table 5 method or evidence digest is invalid")
        if method == "PoLBFL":
            if result.get("real_contract_rounds") is not True or int(
                result.get("contract_rounds", 0)
            ) != 200:
                raise ValueError("PoL-BFL incentive evidence requires 200 real contract rounds")
        else:
            if result.get("real_training") is not True or int(
                result.get("training_rounds", 0)
            ) != 200:
                raise ValueError("baseline incentive evidence requires 200 real training rounds")
            if len(str(result.get("baseline_source_lock_digest", ""))) != 64:
                raise ValueError("baseline incentive evidence lacks its public source lock")
        groups[method].append(result)
    if set(groups) != set(METHODS):
        raise ValueError("Table 5 aggregate does not cover all four methods")

    table = {}
    provenance = {}
    checks = {}
    for method in METHODS:
        rows = groups[method]
        seeds = [int(row["seed"]) for row in rows]
        if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(f"Table 5 seed set is incomplete: {method}")
        observed = {
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in ("ParticipationRate", "AttackSuccessRate", "ModelAccuracy")
        }
        table[method] = observed
        target = targets["table_5_all_methods"][method]
        checks[f"{method}.ParticipationRate"] = observed["ParticipationRate"] + 1e-9 >= float(
            target["ParticipationRate"]
        )
        checks[f"{method}.AttackSuccessRate"] = observed["AttackSuccessRate"] <= float(
            target["AttackSuccessRate"]
        ) + 1e-9
        checks[f"{method}.ModelAccuracy"] = observed["ModelAccuracy"] + 1e-9 >= float(
            target["ModelAccuracy"]
        )
        provenance[method] = {
            "seeds": sorted(seeds),
            "evidence_digests": sorted(str(row["evidence_digest"]) for row in rows),
        }
    return {
        "source_commit": source_commit,
        "table_5_all_methods": table,
        "provenance": {
            "source_commit": source_commit,
            "methods": provenance,
        },
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
        default=root / "config" / "paper_table5_all_methods.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_table5(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
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
