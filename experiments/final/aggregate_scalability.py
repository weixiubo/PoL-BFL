#!/usr/bin/env python3
"""Aggregate accepted client-count cells into final-paper Table 8."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


PAPER_GAS_PRICE_GWEI = Decimal("1.5")
PAPER_ETH_USD = Decimal("2500")


def _gas_usd(*, gas: int, client_count: int) -> float:
    if gas <= 0 or client_count <= 0:
        raise ValueError("gas and client count must be positive")
    eth = Decimal(gas) * PAPER_GAS_PRICE_GWEI * Decimal("1e-9")
    return float(eth * PAPER_ETH_USD * Decimal(client_count) / Decimal(50))


def aggregate_scalability_cells(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    gas_evidence: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 8")
    if gas_evidence.get("passed") is not True:
        raise ValueError("scalability aggregation requires accepted real-contract gas evidence")
    observed_gas = gas_evidence.get("observed_gas", {})
    honest_round_gas = int(observed_gas.get("honest_round_total", 0))
    gas_source = gas_evidence.get("source", {})
    if gas_source.get("dirty") or not gas_source.get("commit"):
        raise ValueError("gas evidence must be bound to a clean source commit")

    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("formal_accepted") is not True or result.get("study") != "scalability":
            raise ValueError("scalability aggregate requires accepted formal scalability cells")
        groups[int(result["num_clients"])].append(result)
    if not groups:
        raise ValueError("scalability aggregate is empty")

    table: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    previous: dict[str, float] | None = None
    for client_count, rows in sorted(groups.items()):
        seeds = [int(row["seed"]) for row in rows]
        if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(f"scalability seed set is incomplete: N={client_count}")
        source_commits = {str(row["source_commit"]) for row in rows}
        if source_commits != {source_commit} or source_commit != str(gas_source["commit"]):
            raise ValueError("scalability cells and gas evidence use different source commits")
        observed = {
            "runtime_seconds": statistics.fmean(float(row["runtime_seconds"]) for row in rows),
            "communication_mb": statistics.fmean(float(row["communication_mb"]) for row in rows),
            "gas_usd": _gas_usd(gas=honest_round_gas, client_count=client_count),
            "seconds_per_client": statistics.fmean(float(row["seconds_per_client"]) for row in rows),
            "MA": statistics.fmean(float(row["MA"]) for row in rows),
            "DR": statistics.fmean(float(row["DR"]) for row in rows),
            "FPR": statistics.fmean(float(row["FPR"]) for row in rows),
        }
        if previous is not None:
            observed["delta_previous_percent"] = {
                name: 100.0 * (observed[name] - previous[name]) / previous[name]
                for name in ("runtime_seconds", "communication_mb", "gas_usd", "seconds_per_client", "MA", "DR", "FPR")
                if previous[name] != 0
            }
        table[str(client_count)] = observed
        target = targets["table_8_scalability"][str(client_count)]
        prefix = str(client_count)
        for metric in ("runtime_seconds", "communication_mb", "gas_usd", "seconds_per_client", "FPR"):
            checks[f"{prefix}.{metric}"] = observed[metric] <= float(target[metric])
        for metric in ("MA", "DR"):
            checks[f"{prefix}.{metric}"] = observed[metric] >= float(target[metric])
        provenance[prefix] = {
            "seeds": sorted(seeds),
            "source_commit": next(iter(source_commits)),
        }
        previous = observed
    expected_counts = {int(value) for value in targets["table_8_scalability"]}
    checks["all_client_counts"] = set(groups) == expected_counts
    return {
        "source_commit": source_commit,
        "table_8_scalability": table,
        "provenance": {
            "cells": provenance,
            "gas_evidence_digest": str(gas_evidence.get("evidence_digest", "")),
            "honest_round_gas": honest_round_gas,
            "gas_price_gwei": str(PAPER_GAS_PRICE_GWEI),
            "eth_usd": str(PAPER_ETH_USD),
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
    parser.add_argument("--gas-evidence", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=root / "config" / "paper_targets.json")
    parser.add_argument("--required-seed-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_scalability_cells(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
        json.loads(args.gas_evidence.read_text(encoding="utf-8")),
        required_seed_count=args.required_seed_count,
    )
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (*args.results, args.gas_evidence)
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
