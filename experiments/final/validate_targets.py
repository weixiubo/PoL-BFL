#!/usr/bin/env python3
"""Strict directional comparison against the final paper's reported values."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Comparison:
    path: str
    observed: Any
    target: Any
    rule: str
    passed: bool


def _leaves(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _leaves(child, (*prefix, str(key)))
    else:
        yield prefix, value


def _lookup(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(".".join(path))
        current = current[part]
    return current


def _rule(path: tuple[str, ...]) -> str:
    metric = path[-1]
    joined = ".".join(path)
    if metric == "profitable" or isinstance(metric, bool):
        return "exact"
    if metric == "circuit_constraints":
        return "approximately"
    if metric == "profit":
        return "max" if "MaliciousNT" in joined else "min"
    if metric == "reward":
        return "max" if "MaliciousNT" in joined else "min"
    if metric == "slash":
        return "max" if "Honest" not in joined else "exact"
    if metric == "cost":
        return "min"
    if metric in {
        "MA",
        "DR",
        "PR",
        "ParticipationRate",
        "ModelAccuracy",
        "NoAttackMA",
        "FreeRidingDR",
        "ALIEDR",
        "honest_pass_rate",
        "block_rate",
        "forge_train_ratio",
        "stake_eth",
    } or metric in {"L1", "L1L2", "L1L3", "Full"}:
        return "min"
    if metric in {
        "FPR",
        "ASR",
        "AttackSuccessRate",
        "runtime_seconds",
        "communication_mb",
        "gas_usd",
        "storage_mb_per_client",
        "seconds_per_client",
        "proof_generation_seconds",
        "witness_seconds",
        "prover_memory_gb",
        "proof_bytes",
        "verification_ms",
        "merkle_proof_kb",
        "total_verification_ms",
        "commitment",
        "proof_receipt",
        "reward_claim",
        "honest_round_total",
    }:
        return "max"
    raise ValueError(f"no target direction is defined for {joined}")


def validate(
    observed: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    tables: set[str] | None = None,
) -> dict[str, Any]:
    comparisons: list[Comparison] = []
    missing: list[str] = []
    selected = {
        key: value
        for key, value in targets.items()
        if key.startswith("table_") or key.startswith("figure_")
        if tables is None or key in tables
    }
    for path, target in _leaves(selected):
        text_path = ".".join(path)
        try:
            actual = _lookup(observed, path)
        except KeyError:
            missing.append(text_path)
            continue
        rule = _rule(path)
        if rule == "exact":
            passed = actual == target
        elif not isinstance(actual, (int, float)) or not isinstance(target, (int, float)):
            passed = False
        elif rule == "min":
            passed = actual >= target
        elif rule == "max":
            passed = actual <= target
        else:
            passed = 0.8 * target <= actual <= 1.2 * target
        comparisons.append(Comparison(text_path, actual, target, rule, bool(passed)))
    return {
        "passed": not missing and all(item.passed for item in comparisons),
        "missing": missing,
        "comparisons": [asdict(item) for item in comparisons],
        "failed": [asdict(item) for item in comparisons if not item.passed],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("observed", type=Path)
    parser.add_argument("--targets", type=Path, default=root / "config" / "paper_targets.json")
    parser.add_argument("--table", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    observed = json.loads(args.observed.read_text(encoding="utf-8"))
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    report = validate(observed, targets, tables=None if not args.table else set(args.table))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
