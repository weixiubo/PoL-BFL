#!/usr/bin/env python3
"""Reconstruct the Figure 5 economic regimes from final-paper vector data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import source_identity, write_manifest_atomic
from experiments.final.evidence import seal_evidence
from experiments.final.preflight import PAPER_SHA256


ETH_PRICE_USD = 2500.0
HIGH_REGIME_START_GWEI = 100


def _fit_affine(points: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    selected = [
        (float(point["gas_price_gwei"]), float(point["payoff_usd"]))
        for point in points
        if int(point["gas_price_gwei"]) >= HIGH_REGIME_START_GWEI
    ]
    if len(selected) < 3 or len({x for x, _y in selected}) != len(selected):
        raise ValueError("gas stress fit requires at least three unique high-regime points")
    mean_x = sum(x for x, _y in selected) / len(selected)
    mean_y = sum(y for _x, y in selected) / len(selected)
    denominator = sum((x - mean_x) ** 2 for x, _y in selected)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in selected) / denominator
    intercept = mean_y - slope * mean_x
    residuals = [abs((intercept + slope * x) - y) for x, y in selected]
    return {
        "intercept_usd": intercept,
        "slope_usd_per_gwei": slope,
        "max_absolute_residual_usd": max(residuals),
    }


def reconstruct_gas_stress(curves: Mapping[str, Any]) -> dict[str, Any]:
    if curves.get("authority_pdf_sha256") != PAPER_SHA256:
        raise ValueError("Figure 5 targets are not bound to the authoritative PDF")
    figure = curves["figure_5_gas_price_stress"]
    honest_points = list(figure["HonestNetProfit"])
    attack_points = list(figure["AttackExpectedPayoff"])
    if [point["gas_price_gwei"] for point in honest_points] != [
        point["gas_price_gwei"] for point in attack_points
    ]:
        raise ValueError("Figure 5 curves use different gas-price coordinates")
    honest_fit = _fit_affine(honest_points)
    attack_fit = _fit_affine(attack_points)
    mean_slope = 0.5 * (
        honest_fit["slope_usd_per_gwei"]
        + attack_fit["slope_usd_per_gwei"]
    )
    inferred_client_gas = -mean_slope / (ETH_PRICE_USD * 1e-9)

    def predict(fit: Mapping[str, float], gas_price: float) -> float:
        return fit["intercept_usd"] + fit["slope_usd_per_gwei"] * gas_price

    honest_300 = predict(honest_fit, 300.0)
    attack_300 = predict(attack_fit, 300.0)
    indexed_honest = {
        int(point["gas_price_gwei"]): float(point["payoff_usd"])
        for point in honest_points
    }
    indexed_attack = {
        int(point["gas_price_gwei"]): float(point["payoff_usd"])
        for point in attack_points
    }
    checks = {
        "curve_lengths": len(honest_points) == len(attack_points) == 8,
        "honest_positive_at_10_gwei": indexed_honest[10] > 0,
        "attack_negative_at_10_gwei": indexed_attack[10] < 0,
        "high_regime_parallel": abs(
            honest_fit["slope_usd_per_gwei"]
            - attack_fit["slope_usd_per_gwei"]
        )
        <= 0.0001,
        "honest_high_fit": honest_fit["max_absolute_residual_usd"] <= 0.15,
        "attack_high_fit": attack_fit["max_absolute_residual_usd"] <= 0.15,
        "inferred_100_gwei_cost": abs(
            inferred_client_gas * 100e-9 * ETH_PRICE_USD - 2.81
        )
        <= 0.03,
        "honest_300_gwei": abs(honest_300 - (-8.06)) <= 0.10,
        "attack_300_gwei": abs(attack_300 - (-8.53)) <= 0.10,
        "high_cost_region_negative": indexed_honest[1000] < 0
        and indexed_attack[1000] < 0,
        "text_vector_attack_10_consistent_with_rounding": abs(
            indexed_attack[10] - (-0.08)
        )
        <= 0.05,
    }
    if not all(math.isfinite(value) for value in (
        inferred_client_gas,
        honest_300,
        attack_300,
    )):
        raise ValueError("Figure 5 reconstruction produced non-finite values")
    return {
        "figure_5_gas_price_stress": figure,
        "derived_economics": {
            "eth_price_usd": ETH_PRICE_USD,
            "high_regime_start_gwei": HIGH_REGIME_START_GWEI,
            "honest_affine_fit": honest_fit,
            "attack_affine_fit": attack_fit,
            "inferred_client_operations_gas": inferred_client_gas,
            "predicted_honest_profit_at_300_gwei": honest_300,
            "predicted_attack_payoff_at_300_gwei": attack_300,
        },
        "paper_internal_rounding": {
            "narrative_attack_payoff_at_10_gwei": -0.08,
            "vector_curve_attack_payoff_at_10_gwei": indexed_attack[10],
            "absolute_difference_usd": abs(indexed_attack[10] - (-0.08)),
        },
        "acceptance": {
            "passed": bool(checks) and all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-targets",
        type=Path,
        default=ROOT / "config" / "paper_figure5_targets.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = source_identity(ROOT)
    if source["dirty"] or not source["commit"]:
        raise RuntimeError(
            "formal Figure 5 reproduction requires a clean source commit"
        )
    curves = json.loads(args.curve_targets.read_text(encoding="utf-8"))
    result = reconstruct_gas_stress(curves)
    result["source"] = source
    result["source_commit"] = source["commit"]
    result["formal_accepted"] = bool(result["acceptance"]["passed"])
    result["input_sha256"] = {
        str(args.curve_targets): hashlib.sha256(
            args.curve_targets.read_bytes()
        ).hexdigest()
    }
    result = seal_evidence(result, analysis_source=source)
    write_manifest_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
