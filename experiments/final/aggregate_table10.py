#!/usr/bin/env python3
"""Aggregate all measured adaptive cells into final-paper Table 10."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.adaptive_evaluation import (
    aggregate_adaptive_trials,
)
from experiments.final.adaptive_trial_support import VARIANTS
from experiments.final.evidence import require_single_source_commit, seal_evidence


def aggregate_table10(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 10")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    trials = []
    for result in results:
        required = {
            "study",
            "dataset",
            "variant",
            "seed",
            "source_commit",
            "evidence_digest",
            "result_digest",
            "trials",
        }
        if (
            not required.issubset(result)
            or result.get("formal_accepted") is not True
            or result["study"] != "adaptive"
            or result["dataset"] != "CIFAR10"
            or result["variant"] not in VARIANTS
            or len(str(result["evidence_digest"])) != 64
            or len(str(result["result_digest"])) != 64
            or len(result["trials"]) != 2
        ):
            raise ValueError("Table 10 result is not accepted or is incomplete")
        groups[str(result["variant"])].append(result)
        trials.extend(result["trials"])
    if set(groups) != set(VARIANTS):
        raise ValueError("Table 10 aggregate does not cover every variant")
    provenance = {}
    for variant in VARIANTS:
        rows = groups[variant]
        seeds = [int(row["seed"]) for row in rows]
        if (
            len(rows) != required_seed_count
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(
                "Table 10 seed set is incomplete: " + variant
            )
        provenance[variant] = {
            "seeds": sorted(seeds),
            "evidence_digests": sorted(
                str(row["evidence_digest"]) for row in rows
            ),
            "result_digests": sorted(
                str(row["result_digest"]) for row in rows
            ),
        }
    aggregate = aggregate_adaptive_trials(trials, targets)
    aggregate["source_commit"] = source_commit
    aggregate["provenance"] = {
        "source_commit": source_commit,
        "variants": provenance,
        "trials": aggregate["provenance"],
    }
    return aggregate


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table10_adaptive.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_table10(
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
    aggregate["formal_result_paths"] = [
        str(path.resolve()) for path in args.results
    ]
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
