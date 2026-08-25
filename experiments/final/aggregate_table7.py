#!/usr/bin/env python3
"""Aggregate measured four-method system overhead into Table 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


METHODS = ("Vanilla", "VeriblockFL", "Kaizen", "PoLBFL")
METRICS = ("runtime_seconds", "communication_mb", "gas_usd", "storage_mb_per_client")


def aggregate_table7(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        required = {"method", "seed", "source_commit", "evidence_digest", *METRICS}
        if not required.issubset(result) or result.get("formal_accepted") is not True:
            raise ValueError("Table 7 result is not accepted or is incomplete")
        method = str(result["method"])
        if method not in METHODS or result.get("real_measurement") is not True:
            raise ValueError("Table 7 method is invalid or not measured")
        if int(result.get("training_rounds", 0)) != 200:
            raise ValueError("Table 7 evidence requires 200 training rounds")
        if len(str(result["evidence_digest"])) != 64:
            raise ValueError("Table 7 evidence digest is invalid")
        if method == "PoLBFL":
            if (
                len(str(result.get("trust_setup_record_digest", ""))) != 64
                or result.get("real_contract_gas") is not True
            ):
                raise ValueError("PoL-BFL overhead lacks production proof or gas evidence")
        elif len(str(result.get("baseline_source_lock_digest", ""))) != 64:
            raise ValueError("overhead baseline lacks its source lock")
        groups[method].append(result)
    if set(groups) != set(METHODS):
        raise ValueError("Table 7 aggregate does not cover all four methods")
    source_commits = {str(row["source_commit"]) for rows in groups.values() for row in rows}
    if len(source_commits) != 1:
        raise ValueError("Table 7 methods were not executed from one source commit")

    table = {}
    checks = {}
    provenance = {}
    for method in METHODS:
        rows = groups[method]
        seeds = [int(row["seed"]) for row in rows]
        if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(f"Table 7 seed set is incomplete: {method}")
        observed = {
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in METRICS
        }
        table[method] = observed
        target = targets["table_7_all_methods"][method]
        for metric in METRICS:
            checks[f"{method}.{metric}"] = observed[metric] <= float(target[metric]) + 1e-9
        provenance[method] = {
            "seeds": sorted(seeds),
            "evidence_digests": sorted(str(row["evidence_digest"]) for row in rows),
        }
    return {
        "source_commit": next(iter(source_commits)),
        "table_7_all_methods": table,
        "provenance": {
            "source_commit": next(iter(source_commits)),
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
        default=root / "config" / "paper_table7_all_methods.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_table7(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.results
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
