#!/usr/bin/env python3
"""Aggregate measured adaptive-attack trials into final-paper Table 10."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VARIANTS = (
    "BaselineNT",
    "CheckpointInterpolation",
    "GradientMimicry",
    "PartialReplay",
    "CombinedAdaptive",
)


def aggregate_adaptive_trials(
    trials: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for trial in trials:
        required = {
            "variant",
            "trial_id",
            "behavior",
            "detected",
            "expected_profit_usd",
            "real_trace",
            "real_groth16",
            "proof_digest",
        }
        if not required.issubset(trial):
            raise ValueError("adaptive trial is incomplete")
        variant = str(trial["variant"])
        if variant not in VARIANTS or trial["behavior"] not in {"honest", "malicious"}:
            raise ValueError("adaptive variant or behavior is invalid")
        if trial["real_trace"] is not True or trial["real_groth16"] is not True:
            raise ValueError("adaptive evidence must use real trace and Groth16 paths")
        identity = (variant, str(trial["trial_id"]), str(trial["proof_digest"]))
        if identity in seen:
            raise ValueError("duplicate adaptive trial")
        seen.add(identity)
        groups[variant].append(trial)
    table = {}
    checks = {}
    provenance = {}
    for variant, rows in sorted(groups.items()):
        honest = [row for row in rows if row["behavior"] == "honest"]
        malicious = [row for row in rows if row["behavior"] == "malicious"]
        if not honest or not malicious:
            raise ValueError(f"adaptive variant lacks honest or malicious trials: {variant}")
        dr = 100.0 * sum(bool(row["detected"]) for row in malicious) / len(malicious)
        fpr = 100.0 * sum(bool(row["detected"]) for row in honest) / len(honest)
        observed: dict[str, Any] = {
            "DR": dr,
            "FPR": fpr,
            "profitable": any(float(row["expected_profit_usd"]) > 0 for row in malicious),
        }
        if variant != "BaselineNT":
            ratios = []
            for row in malicious:
                forge = float(row["forge_seconds"])
                honest_time = float(row["honest_train_seconds"])
                if forge <= 0 or honest_time <= 0:
                    raise ValueError("adaptive timing measurements must be positive")
                ratios.append(forge / honest_time)
            observed["forge_train_ratio"] = statistics.fmean(ratios)
        table[variant] = observed
        target = targets["table_10_adaptive"][variant]
        checks[f"{variant}.DR"] = observed["DR"] >= float(target["DR"])
        checks[f"{variant}.FPR"] = observed["FPR"] <= float(target["FPR"])
        checks[f"{variant}.profitable"] = observed["profitable"] == bool(target["profitable"])
        if variant != "BaselineNT":
            checks[f"{variant}.forge_train_ratio"] = observed["forge_train_ratio"] >= float(
                target["forge_train_ratio"]
            )
        provenance[variant] = {
            "honest_trials": len(honest),
            "malicious_trials": len(malicious),
            "proof_digests": sorted(str(row["proof_digest"]) for row in rows),
        }
    return {
        "table_10_adaptive": table,
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
        default=root / "config" / "paper_table10_adaptive.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_adaptive_trials(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.trials],
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
