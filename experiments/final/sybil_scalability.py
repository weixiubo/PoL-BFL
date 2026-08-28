#!/usr/bin/env python3
"""Aggregate real Sybil identity trials for final-paper Figure 6."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping


def aggregate_sybil_scalability(
    observations: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for observation in observations:
        required = {
            "identity_count",
            "identity_id",
            "detected",
            "stake_eth",
            "real_trace",
            "real_groth16",
            "trace_digest",
        }
        if not required.issubset(observation):
            raise ValueError("Sybil scalability observation is incomplete")
        if observation["real_trace"] is not True or observation["real_groth16"] is not True:
            raise ValueError("Sybil scalability requires real trace and Groth16 evidence")
        count = int(observation["identity_count"])
        identity = (count, str(observation["identity_id"]), str(observation["trace_digest"]))
        if identity in seen:
            raise ValueError("duplicate Sybil identity observation")
        seen.add(identity)
        groups[count].append(observation)
    figure = {}
    checks = {}
    provenance = {}
    for count, rows in sorted(groups.items()):
        if len(rows) != count:
            raise ValueError(f"Sybil group size differs from committed identity count: {count}")
        dr = 100.0 * sum(bool(row["detected"]) for row in rows) / count
        total_stake = float(
            sum((Decimal(str(row["stake_eth"])) for row in rows), Decimal("0"))
        )
        observed = {"DR": dr, "stake_eth": total_stake}
        figure[str(count)] = observed
        target = targets["figure_6_sybil_scalability"][str(count)]
        checks[f"{count}.DR"] = observed["DR"] >= float(target["DR"])
        checks[f"{count}.stake_eth"] = observed["stake_eth"] >= float(target["stake_eth"])
        provenance[str(count)] = {
            "identities": sorted(str(row["identity_id"]) for row in rows),
            "trace_digests": sorted(str(row["trace_digest"]) for row in rows),
        }
    return {
        "figure_6_sybil_scalability": figure,
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
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_figure6_targets.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_sybil_scalability(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.observations],
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
