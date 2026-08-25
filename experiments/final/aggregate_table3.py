#!/usr/bin/env python3
"""Aggregate complete three-seed layer cells into final-paper Table 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


DATASETS = ("CIFAR10", "FEMNIST", "CIFAR100")
ATTACKS = ("FreeRidingNT", "ALIE", "Sybil")
VARIANTS = ("L1", "L1L2", "L1L3", "Full")


def aggregate_table3(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 3")
    groups: dict[
        tuple[str, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for result in results:
        required = {
            "study",
            "dataset",
            "attack",
            "layer_variant",
            "seed",
            "DR",
            "FPR",
            "source_commit",
            "result_digest",
            "real_groth16",
            "real_robust_aggregation",
            "real_contract_transition",
        }
        if (
            not required.issubset(result)
            or result.get("formal_accepted") is not True
            or result["study"] != "layer"
        ):
            raise ValueError("Table 3 result is not accepted or is incomplete")
        dataset = str(result["dataset"])
        attack = str(result["attack"])
        variant = str(result["layer_variant"])
        if (
            dataset not in DATASETS
            or attack not in ATTACKS
            or variant not in VARIANTS
            or len(str(result["result_digest"])) != 64
            or result["real_groth16"] is not True
        ):
            raise ValueError("Table 3 cell identity or proof evidence is invalid")
        expected_robust = variant in {"L1L2", "Full"}
        expected_contract = variant in {"L1L3", "Full"}
        if (
            bool(result["real_robust_aggregation"]) != expected_robust
            or bool(result["real_contract_transition"]) != expected_contract
        ):
            raise ValueError("Table 3 cell used the wrong layer profile")
        groups[(dataset, attack, variant)].append(result)
    expected_groups = {
        (dataset, attack, variant)
        for dataset in DATASETS
        for attack in ATTACKS
        for variant in VARIANTS
    }
    if set(groups) != expected_groups:
        raise ValueError("Table 3 aggregate does not cover all 36 cells")
    table: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for key in sorted(groups):
        dataset, attack, variant = key
        rows = groups[key]
        seeds = [int(row["seed"]) for row in rows]
        if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(
                "Table 3 seed set is incomplete: " + ".".join(key)
            )
        observed_dr = statistics.fmean(float(row["DR"]) for row in rows)
        observed_fpr = statistics.fmean(float(row["FPR"]) for row in rows)
        table.setdefault(dataset, {}).setdefault(attack, {})[
            variant
        ] = observed_dr
        target = float(
            targets["table_3_layer_dr"][dataset][attack][variant]
        )
        gate = dataset + "." + attack + "." + variant
        checks[gate + ".DR"] = observed_dr + 1e-9 >= target
        checks[gate + ".FPR_range"] = 0.0 <= observed_fpr <= 100.0
        provenance[gate] = {
            "seeds": sorted(seeds),
            "DR": observed_dr,
            "FPR": observed_fpr,
            "result_digests": sorted(
                str(row["result_digest"]) for row in rows
            ),
        }
    return {
        "source_commit": source_commit,
        "table_3_layer_dr": table,
        "provenance": {
            "source_commit": source_commit,
            "cells": provenance,
        },
        "acceptance": {
            "passed": bool(checks) and all(checks.values()),
            "checks": checks,
            "failed": sorted(
                name for name, passed in checks.items() if not passed
            ),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_targets.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.results
    ]
    aggregate = aggregate_table3(
        rows,
        json.loads(args.targets.read_text(encoding="utf-8")),
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
