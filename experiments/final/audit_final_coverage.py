#!/usr/bin/env python3
"""Audit that every final-paper study has an explicit non-simulated route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


COVERAGE = {
    "table_2_main_security": ("executable_matrix", "experiments/final/run_matrix.py"),
    "table_3_layer_contribution": ("executable_matrix", "experiments/final/run_layer_matrix.py"),
    "table_4_composability": ("executable_two_mode_matrix", "experiments/final/run_table4_matrix.py"),
    "table_5_incentive_effectiveness": ("executable_four_method_matrix", "experiments/final/run_table5_matrix.py"),
    "table_6_client_profit": ("equation_reproduction", "experiments/final/run_economics.py"),
    "table_7_system_overhead": ("measured_four_method_matrix", "experiments/final/run_table7_matrix.py"),
    "table_8_scalability": ("executable_matrix", "experiments/final/run_scalability_matrix.py"),
    "table_9_noniid": ("executable_matrix", "experiments/final/run_noniid_matrix.py"),
    "table_10_adaptive": ("executable_real_trace_matrix", "experiments/final/run_adaptive_matrix.py"),
    "table_11_cross_hardware": ("attested_hardware_matrix", "experiments/final/run_cross_hardware_matrix.py"),
    "table_12_zk_cost": ("production_and_controlled_benchmarks", "experiments/final/aggregate_table12.py"),
    "table_13_gas": ("real_contract_benchmark", "scripts/contract_gas_benchmark.py"),
    "figure_2_convergence": ("accepted_cell_derivation", "experiments/final/convergence.py"),
    "figure_3_reputation_evolution": ("real_cell_derivation", "experiments/final/reputation_evolution.py"),
    "figure_4_spot_check_sensitivity": ("executable_matrix", "experiments/final/run_sensitivity_matrix.py"),
    "figure_5_gas_price_stress": ("paper_curve_equation_reconstruction", "experiments/final/gas_price_stress.py"),
    "figure_6_sybil_scalability": ("executable_matrix", "experiments/final/run_sybil_matrix.py"),
}


def audit_coverage(matrix: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    studies = set(matrix["studies"])
    checks = {
        "all_studies_routed": studies == set(COVERAGE),
    }
    routes = {}
    for study in sorted(studies | set(COVERAGE)):
        route = COVERAGE.get(study)
        exists = bool(route and (root / route[1]).is_file())
        checks[f"route:{study}"] = exists
        routes[study] = {
            "mode": None if route is None else route[0],
            "owner": None if route is None else route[1],
            "exists": exists,
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "routes": routes,
        "measurement_complete": False,
        "note": "Code-path coverage does not substitute for accepted result evidence.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=root / "experiments" / "final" / "paper_matrix.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_coverage(
        json.loads(args.matrix.read_text(encoding="utf-8")),
        root=root,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
