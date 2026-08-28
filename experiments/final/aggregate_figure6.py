#!/usr/bin/env python3
"""Aggregate three-seed real Sybil cells into final-paper Figure 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


IDENTITY_COUNTS = (5, 10, 15, 20)
DATASETS = ("CIFAR10", "FEMNIST", "CIFAR100")


def aggregate_figure6(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Figure 6")
    if "figure_6_vector_targets" not in targets:
        raise ValueError("Figure 6 aggregate requires PDF-derived vector targets")
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        required = {
            "study",
            "dataset",
            "attack",
            "seed",
            "sybil_identity_count",
            "sybil_stake_eth",
            "MA",
            "DR",
            "FPR",
            "source_commit",
            "result_digest",
            "real_groth16",
            "real_contract_transition",
        }
        if (
            not required.issubset(result)
            or result.get("formal_accepted") is not True
            or result["study"] != "sybil_scalability"
            or result["dataset"] not in DATASETS
            or result["attack"] != "Sybil"
            or result["real_groth16"] is not True
            or result["real_contract_transition"] is not True
            or len(str(result["result_digest"])) != 64
        ):
            raise ValueError("Figure 6 result is not accepted or is incomplete")
        count = int(result["sybil_identity_count"])
        if count not in IDENTITY_COUNTS:
            raise ValueError("Figure 6 identity count is invalid")
        groups[(str(result["dataset"]), count)].append(result)
    expected_groups = {
        (dataset, count) for dataset in DATASETS for count in IDENTITY_COUNTS
    }
    if set(groups) != expected_groups:
        raise ValueError("Figure 6 aggregate lacks a dataset/identity-count group")
    figure: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    checks = {}
    for dataset in DATASETS:
        figure[dataset] = {}
        provenance[dataset] = {}
        for count in IDENTITY_COUNTS:
            rows = groups[(dataset, count)]
            seeds = [int(row["seed"]) for row in rows]
            if len(rows) != required_seed_count or len(set(seeds)) != len(seeds):
                raise ValueError(
                    f"Figure 6 seed set is incomplete for {dataset}/{count}"
                )
            observed = {
                "MA": statistics.fmean(float(row["MA"]) for row in rows),
                "DR": statistics.fmean(float(row["DR"]) for row in rows),
                "FPR": statistics.fmean(float(row["FPR"]) for row in rows),
                "stake_eth": statistics.fmean(
                    float(row["sybil_stake_eth"]) for row in rows
                ),
            }
            figure[dataset][str(count)] = observed
            target = targets["figure_6_vector_targets"][dataset][str(count)]
            prefix = f"{dataset}.{count}"
            checks.update(
                {
                    prefix + ".MA": observed["MA"] >= float(target["MA"]),
                    prefix + ".DR": observed["DR"] >= float(target["DR"]),
                    prefix + ".FPR": observed["FPR"] <= float(target["FPR"]),
                    prefix + ".stake_eth": observed["stake_eth"] + 1e-12
                    >= float(target["stake_eth"]),
                }
            )
            provenance[dataset][str(count)] = {
                "seeds": sorted(seeds),
                "result_digests": sorted(
                    str(row["result_digest"]) for row in rows
                ),
            }
        checks[dataset + ".DR_non_increasing_with_identity_count"] = all(
            figure[dataset][str(left)]["DR"] + 1e-9
            >= figure[dataset][str(right)]["DR"]
            for left, right in zip(IDENTITY_COUNTS, IDENTITY_COUNTS[1:])
        )
    return {
        "source_commit": source_commit,
        "figure_6_sybil_scalability": figure,
        "provenance": {
            "source_commit": source_commit,
            "datasets": provenance,
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
        default=root / "config" / "paper_figure6_targets.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_figure6(
        [
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.results
        ],
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
