#!/usr/bin/env python3
"""Reproduce the final paper's reward/profit table from disclosed equations."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, source_identity, write_manifest_atomic
from experiments.final.evidence import seal_evidence
from polbfl.incentives import EconomicParameters, IncentiveEngine


def reproduce(config: dict) -> dict:
    decimal = lambda value: Decimal(str(value))
    engine = IncentiveEngine(
        EconomicParameters(
            base_reward=decimal(config["base_reward_usd"]),
            beta_work=decimal(config["beta_work"]),
            beta_reputation=decimal(config["beta_reputation"]),
            reputation_decay=decimal(config["reputation_decay"]),
            slashing_ratio=decimal(config["slashing_ratio"]),
            challenge_probability=decimal(config["challenge_probability"]),
            detection_probability=decimal(config["detection_probability"]),
            base_minimum_stake=decimal(config["minimum_stake_eth"]),
        )
    )
    rows = {}
    for name, behavior in config["behaviors"].items():
        reward = engine.reward(
            normalized_work=decimal(behavior["work"]),
            reputation=decimal(behavior["reputation"]),
        )
        cost = decimal(behavior["cost_usd"])
        slash = decimal(behavior["expected_slash_usd"])
        rows[name] = {
            "reward": float(reward),
            "cost": float(-cost),
            "slash": float(-slash),
            "profit": float(reward - cost - slash),
        }
    rows["incentive_compatible"] = engine.honest_dominates(
        stake=decimal(config["minimum_stake_eth"]),
        expected_reward=Decimal("0.172"),
        saved_cost=Decimal("0.007"),
    )
    return {"table_6_profit_usd": rows}


def validate_economics(result: dict, targets: dict) -> dict:
    observed = result["table_6_profit_usd"]
    expected = targets["table_6_profit_usd"]
    checks = {}
    for behavior, target_row in expected.items():
        for metric, target in target_row.items():
            checks[f"{behavior}.{metric}"] = abs(
                float(observed[behavior][metric]) - float(target)
            ) <= 1e-9
    checks.update(
        {
            "incentive_compatible": observed["incentive_compatible"] is True,
            "honest_dominates_rational": observed["Honest"]["profit"]
            > observed["RationalLT"]["profit"],
            "rational_dominates_malicious": observed["RationalLT"]["profit"]
            > observed["MaliciousNT"]["profit"],
            "malicious_unprofitable": observed["MaliciousNT"]["profit"] < 0,
        }
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "economics_reference.json")
    parser.add_argument("--targets", type=Path, default=ROOT / "config" / "paper_targets.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    result = reproduce(json.loads(args.config.read_text(encoding="utf-8")))
    source = source_identity(ROOT)
    if not args.allow_dirty and (source["dirty"] or not source["commit"]):
        raise RuntimeError("formal economics reproduction requires a clean source commit")
    result["acceptance"] = validate_economics(
        result,
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    result["source"] = source
    result["input_sha256"] = {
        str(args.config.resolve().relative_to(ROOT)): sha256_file(args.config),
        str(args.targets.resolve().relative_to(ROOT)): sha256_file(args.targets),
    }
    result["formal_accepted"] = bool(
        not args.allow_dirty and result["acceptance"]["passed"]
    )
    result = seal_evidence(
        result,
        source_commit=source["commit"],
        analysis_source=source,
    )
    write_manifest_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
