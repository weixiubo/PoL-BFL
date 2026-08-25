#!/usr/bin/env python3
"""Derive final-paper Figure 3 from accepted LT and NT round ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.capture_formal_evidence import verify_completed_cell
from experiments.final.evidence import seal_evidence


SAMPLE_ROUNDS = (0, 20, 50, 100, 150, 200)


def _validated_rounds(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_rounds: int,
) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["round"]))
    if len(ordered) != required_rounds or [int(row["round"]) for row in ordered] != list(
        range(required_rounds)
    ):
        raise ValueError("reputation evidence must contain every round exactly once")
    required = {
        "honest_reputation_mean",
        "malicious_reputation_mean",
        "reputation_by_client",
        "effective_reputation_by_client",
        "stake_by_client",
        "settlement_digest",
    }
    if any(not required.issubset(row) for row in ordered):
        raise ValueError("reputation round evidence is incomplete")
    if any(row["malicious_reputation_mean"] is None for row in ordered):
        raise ValueError("reputation study requires malicious behavior observations")
    digests = [str(row["settlement_digest"]) for row in ordered]
    if any(len(value) != 64 for value in digests) or len(set(digests)) != len(digests):
        raise ValueError("reputation settlements must be unique canonical digests")
    for row in ordered:
        values = (
            float(row["honest_reputation_mean"]),
            float(row["malicious_reputation_mean"]),
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("reputation observations must be normalized")
    return ordered


def aggregate_reputation_evolution(
    rational_rows: Iterable[Mapping[str, Any]],
    malicious_rows: Iterable[Mapping[str, Any]],
    *,
    required_rounds: int = 200,
    sample_rounds: Sequence[int] = SAMPLE_ROUNDS,
) -> dict[str, Any]:
    rational = _validated_rounds(rational_rows, required_rounds=required_rounds)
    malicious = _validated_rounds(malicious_rows, required_rounds=required_rounds)
    if tuple(sample_rounds) != SAMPLE_ROUNDS or sample_rounds[-1] != required_rounds:
        raise ValueError("Figure 3 sampling points must match the final paper")

    points = []
    for round_number in sample_rounds:
        if round_number == 0:
            honest = rational_value = malicious_value = 50.0
            rational_digest = malicious_digest = None
        else:
            rational_row = rational[round_number - 1]
            malicious_row = malicious[round_number - 1]
            honest = 50.0 * (
                float(rational_row["honest_reputation_mean"])
                + float(malicious_row["honest_reputation_mean"])
            )
            rational_value = 100.0 * float(rational_row["malicious_reputation_mean"])
            malicious_value = 100.0 * float(malicious_row["malicious_reputation_mean"])
            rational_digest = str(rational_row["settlement_digest"])
            malicious_digest = str(malicious_row["settlement_digest"])
        points.append(
            {
                "round": int(round_number),
                "Honest": honest,
                "Rational": rational_value,
                "Malicious": malicious_value,
                "rational_settlement_digest": rational_digest,
                "malicious_settlement_digest": malicious_digest,
            }
        )
    final = points[-1]
    checks = {
        "initial_neutral": points[0]["Honest"]
        == points[0]["Rational"]
        == points[0]["Malicious"]
        == 50.0,
        "honest_final": float(final["Honest"]) >= 95.0,
        "rational_final": float(final["Rational"]) <= 76.0,
        "malicious_final": float(final["Malicious"]) <= 5.0,
        "honest_dominates": float(final["Honest"])
        > max(float(final["Rational"]), float(final["Malicious"])),
    }
    return {
        "figure_3_reputation_evolution": points,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rational-result", type=Path, required=True)
    parser.add_argument("--malicious-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rational_evidence = verify_completed_cell(args.rational_result, root=ROOT)
    malicious_evidence = verify_completed_cell(args.malicious_result, root=ROOT)
    rational_result = rational_evidence["result"]
    malicious_result = malicious_evidence["result"]
    if rational_evidence["source_commit"] != malicious_evidence["source_commit"]:
        raise ValueError("Figure 3 inputs must use one source commit")
    if (
        rational_result.get("dataset") != "CIFAR10"
        or rational_result.get("attack") != "FreeRidingLT"
        or malicious_result.get("dataset") != "CIFAR10"
        or malicious_result.get("attack") != "FreeRidingNT"
    ):
        raise ValueError("Figure 3 requires accepted CIFAR10 LT and NT cells")
    rational_rounds = args.rational_result.with_name("rounds.jsonl")
    malicious_rounds = args.malicious_result.with_name("rounds.jsonl")
    aggregate = aggregate_reputation_evolution(
        _read_jsonl(rational_rounds),
        _read_jsonl(malicious_rounds),
    )
    aggregate["source_commit"] = rational_evidence["source_commit"]
    aggregate["provenance"] = {
        "rational_manifest_digest": rational_evidence["manifest_digest"],
        "malicious_manifest_digest": malicious_evidence["manifest_digest"],
        "rational_source_commit": rational_evidence["source_commit"],
        "malicious_source_commit": malicious_evidence["source_commit"],
        "trust_setup_record_digests": sorted(
            {
                rational_evidence["trust_setup_record_digest"],
                malicious_evidence["trust_setup_record_digest"],
            }
        ),
        "input_sha256": {
            str(rational_rounds): hashlib.sha256(rational_rounds.read_bytes()).hexdigest(),
            str(malicious_rounds): hashlib.sha256(malicious_rounds.read_bytes()).hexdigest(),
        },
    }
    aggregate["input_sha256"] = dict(aggregate["provenance"]["input_sha256"])
    aggregate["formal_result_paths"] = sorted(
        {
            str(args.rational_result.resolve()),
            str(args.malicious_result.resolve()),
        }
    )
    aggregate = seal_evidence(aggregate, analysis_root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
