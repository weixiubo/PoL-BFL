#!/usr/bin/env python3
"""Aggregate attested three-seed hardware trials into Table 11."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.cross_hardware import aggregate_cross_hardware
from experiments.final.evidence import require_single_source_commit, seal_evidence


def aggregate_table11(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 11")
    expected_pairs = set(targets["table_11_cross_hardware"])
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observations = []
    for result in results:
        required = {
            "study",
            "hardware_pair",
            "seed",
            "source_commit",
            "evidence_digest",
            "result_digest",
            "observations",
        }
        if (
            not required.issubset(result)
            or result.get("formal_accepted") is not True
            or result["study"] != "cross_hardware"
            or result["hardware_pair"] not in expected_pairs
            or len(str(result["evidence_digest"])) != 64
            or len(str(result["result_digest"])) != 64
            or len(result["observations"]) != 2
        ):
            raise ValueError("Table 11 result is not accepted or is incomplete")
        pair = str(result["hardware_pair"])
        groups[pair].append(result)
        observations.extend(result["observations"])
    if set(groups) != expected_pairs:
        raise ValueError("Table 11 aggregate lacks a hardware pair")
    provenance = {}
    for pair in sorted(expected_pairs):
        rows = groups[pair]
        seeds = [int(row["seed"]) for row in rows]
        if (
            len(rows) != required_seed_count
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(
                "Table 11 seed set is incomplete: " + pair
            )
        provenance[pair] = {
            "seeds": sorted(seeds),
            "evidence_digests": sorted(
                str(row["evidence_digest"]) for row in rows
            ),
            "result_digests": sorted(
                str(row["result_digest"]) for row in rows
            ),
        }
    aggregate = aggregate_cross_hardware(observations, targets)
    aggregate["source_commit"] = source_commit
    aggregate["provenance"] = {
        "source_commit": source_commit,
        "hardware_pairs": provenance,
        "observations": aggregate["provenance"],
    }
    return aggregate


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table11_cross_hardware.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_table11(
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
