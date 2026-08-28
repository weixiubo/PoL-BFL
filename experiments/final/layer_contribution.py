#!/usr/bin/env python3
"""Aggregate real layer-ablation trials into final-paper Table 3."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VARIANTS = ("L1", "L1L2", "L1L3", "Full")


def aggregate_layer_trials(
    trials: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for trial in trials:
        required = {
            "dataset",
            "attack",
            "variant",
            "trial_id",
            "behavior",
            "detected",
            "real_groth16",
            "real_robust_aggregation",
            "real_contract_transition",
            "evidence_digest",
        }
        if not required.issubset(trial):
            raise ValueError("layer-contribution trial is incomplete")
        variant = str(trial["variant"])
        if variant not in VARIANTS or trial["behavior"] not in {"honest", "malicious"}:
            raise ValueError("layer-contribution variant or behavior is invalid")
        if trial["real_groth16"] is not True:
            raise ValueError("every layer-contribution trial requires real Groth16")
        if variant in {"L1L2", "Full"} and trial["real_robust_aggregation"] is not True:
            raise ValueError(f"{variant} requires real robust aggregation")
        if variant in {"L1L3", "Full"} and trial["real_contract_transition"] is not True:
            raise ValueError(f"{variant} requires a real contract transition")
        identity = (
            str(trial["dataset"]),
            str(trial["attack"]),
            variant,
            str(trial["trial_id"]),
            str(trial["evidence_digest"]),
        )
        if identity in seen:
            raise ValueError("duplicate layer-contribution trial")
        seen.add(identity)
        groups[(identity[0], identity[1], variant)].append(trial)
    table: dict[str, Any] = {}
    checks = {}
    provenance = {}
    for (dataset, attack, variant), rows in sorted(groups.items()):
        honest = [row for row in rows if row["behavior"] == "honest"]
        malicious = [row for row in rows if row["behavior"] == "malicious"]
        if not honest or not malicious:
            raise ValueError("layer trial group lacks honest or malicious evidence")
        dr = 100.0 * sum(bool(row["detected"]) for row in malicious) / len(malicious)
        fpr = 100.0 * sum(bool(row["detected"]) for row in honest) / len(honest)
        table.setdefault(dataset, {}).setdefault(attack, {})[variant] = dr
        target = targets["table_3_layer_dr"][dataset][attack][variant]
        key = f"{dataset}.{attack}.{variant}"
        checks[key] = dr >= float(target)
        provenance[key] = {
            "DR": dr,
            "FPR": fpr,
            "honest_trials": len(honest),
            "malicious_trials": len(malicious),
            "evidence_digests": sorted(str(row["evidence_digest"]) for row in rows),
        }
    return {
        "table_3_layer_dr": table,
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
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table3_layer_dr.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_layer_trials(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.trials],
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
