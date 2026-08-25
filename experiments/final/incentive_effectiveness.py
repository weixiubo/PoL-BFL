#!/usr/bin/env python3
"""Aggregate real round/contract observations for final-paper Table 5."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


def aggregate_incentive_rounds(
    rounds: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    required_rounds: int = 200,
) -> dict[str, Any]:
    rows = sorted(rounds, key=lambda row: int(row["round"]))
    if len(rows) != required_rounds or [int(row["round"]) for row in rows] != list(
        range(required_rounds)
    ):
        raise ValueError("incentive evidence must contain every required round exactly once")
    required = {
        "registered_clients",
        "valid_submissions",
        "malicious_submissions",
        "malicious_attack_successes",
        "model_accuracy",
        "real_contract_transition",
        "settlement_digest",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("incentive round evidence is incomplete")
    if any(row["real_contract_transition"] is not True for row in rows):
        raise ValueError("incentive evidence requires real contract transitions")
    participation = []
    malicious_submissions = 0
    malicious_included = 0
    for row in rows:
        registered = int(row["registered_clients"])
        valid = int(row["valid_submissions"])
        malicious = int(row["malicious_submissions"])
        successful = int(row["malicious_attack_successes"])
        if registered <= 0 or not 0 <= valid <= registered or not 0 <= successful <= malicious:
            raise ValueError("incentive round counts are invalid")
        participation.append(100.0 * valid / registered)
        malicious_submissions += malicious
        malicious_included += successful
    observed = {
        "ParticipationRate": statistics.fmean(participation),
        "AttackSuccessRate": (
            0.0
            if malicious_submissions == 0
            else 100.0 * malicious_included / malicious_submissions
        ),
        "ModelAccuracy": float(rows[-1]["model_accuracy"]),
    }
    target = targets["table_5_incentive"]
    checks = {
        "ParticipationRate": observed["ParticipationRate"]
        >= float(target["ParticipationRate"]),
        "AttackSuccessRate": observed["AttackSuccessRate"]
        <= float(target["AttackSuccessRate"]),
        "ModelAccuracy": observed["ModelAccuracy"] >= float(target["ModelAccuracy"]),
    }
    return {
        "table_5_incentive": observed,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
        "provenance": {
            "rounds": len(rows),
            "settlement_digests": [str(row["settlement_digest"]) for row in rows],
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("rounds", type=Path)
    parser.add_argument("--targets", type=Path, default=root / "config" / "paper_targets.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.rounds.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aggregate = aggregate_incentive_rounds(
        rows,
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
