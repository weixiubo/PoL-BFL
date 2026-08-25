#!/usr/bin/env python3
"""Aggregate accepted non-IID cells into final-paper Table 9."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import require_single_source_commit, seal_evidence


REQUIRED_ATTACKS = ("NoAttack", "FreeRidingNT", "ALIE")


def aggregate_noniid_cells(
    results: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    results = tuple(results)
    source_commit = require_single_source_commit(results, context="Table 9")
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("formal_accepted") is not True or result.get("study") != "noniid":
            raise ValueError("non-IID aggregate requires accepted non-IID formal cells")
        key = (
            str(result["dataset"]),
            str(result["partition_label"]),
            str(result["attack"]),
        )
        if key[2] not in REQUIRED_ATTACKS:
            raise ValueError(f"unexpected non-IID attack: {key[2]}")
        groups[key].append(result)
    cells = sorted({(dataset, partition) for dataset, partition, _attack in groups})
    table: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for dataset, partition in cells:
        attack_rows = {
            attack: groups.get((dataset, partition, attack), [])
            for attack in REQUIRED_ATTACKS
        }
        if any(len(rows) != required_seed_count for rows in attack_rows.values()):
            raise ValueError(f"non-IID cell seed count is incomplete: {dataset}/{partition}")
        for attack, rows in attack_rows.items():
            seeds = [int(row["seed"]) for row in rows]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"duplicate non-IID seed: {dataset}/{partition}/{attack}")
        observed = {
            "NoAttackMA": statistics.fmean(float(row["MA"]) for row in attack_rows["NoAttack"]),
            "FreeRidingDR": statistics.fmean(float(row["DR"]) for row in attack_rows["FreeRidingNT"]),
            "ALIEDR": statistics.fmean(float(row["DR"]) for row in attack_rows["ALIE"]),
            "FPR": statistics.fmean(float(row["FPR"]) for row in attack_rows["FreeRidingNT"]),
        }
        table.setdefault(dataset, {})[partition] = observed
        target = targets["table_9_noniid"][dataset][partition]
        prefix = f"{dataset}.{partition}"
        checks[f"{prefix}.NoAttackMA"] = observed["NoAttackMA"] >= float(target["NoAttackMA"])
        checks[f"{prefix}.FreeRidingDR"] = observed["FreeRidingDR"] >= float(target["FreeRidingDR"])
        checks[f"{prefix}.ALIEDR"] = observed["ALIEDR"] >= float(target["ALIEDR"])
        checks[f"{prefix}.FPR"] = observed["FPR"] <= float(target["FPR"])
        provenance[prefix] = {
            attack: {
                "seeds": sorted(int(row["seed"]) for row in rows),
                "source_commits": sorted({str(row["source_commit"]) for row in rows}),
            }
            for attack, rows in attack_rows.items()
        }
    return {
        "source_commit": source_commit,
        "table_9_noniid": table,
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
    parser.add_argument("--required-seed-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_noniid_cells(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results],
        json.loads(args.targets.read_text(encoding="utf-8")),
        required_seed_count=args.required_seed_count,
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
