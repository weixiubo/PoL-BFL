#!/usr/bin/env python3
"""Build one source-bound plan for every unique formal training cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.final.evidence import seal_evidence
from experiments.final.manifest import source_identity, write_manifest_atomic
from experiments.final.run_adaptive_matrix import plan_adaptive_cells
from experiments.final.run_cross_hardware_matrix import plan_cross_hardware_cells
from experiments.final.run_layer_matrix import plan_layer_cells
from experiments.final.run_matrix import plan_cells
from experiments.final.run_noniid_matrix import plan_noniid_cells
from experiments.final.run_scalability_matrix import plan_scalability_cells
from experiments.final.run_sensitivity_matrix import plan_sensitivity_cells
from experiments.final.run_sybil_matrix import plan_sybil_cells
from experiments.final.run_table4_matrix import plan_table4_cells
from experiments.final.run_table5_matrix import plan_table5_cells


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROUTE_COUNTS = {
    "table_2_main_security": 432,
    "table_3_layer_contribution": 108,
    "table_4_composability": 36,
    "table_5_incentive_effectiveness": 12,
    "table_8_scalability": 9,
    "table_9_noniid": 108,
    "table_10_adaptive": 15,
    "figure_4_spot_check_sensitivity": 24,
    "figure_6_sybil_scalability": 36,
    "table_11_cross_hardware": 21,
}
DERIVED_OR_BENCHMARK_ROUTES = (
    "table_6_client_profit",
    "table_7_system_overhead",
    "table_12_zk_cost",
    "table_13_gas",
    "figure_2_convergence",
    "figure_3_reputation_evolution",
    "figure_5_gas_price_stress",
)


def _route_cells(matrix: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    return {
        "table_2_main_security": plan_cells(matrix),
        "table_3_layer_contribution": plan_layer_cells(matrix),
        "table_4_composability": plan_table4_cells(matrix),
        "table_5_incentive_effectiveness": plan_table5_cells(matrix),
        "table_8_scalability": plan_scalability_cells(matrix),
        "table_9_noniid": plan_noniid_cells(matrix),
        "table_10_adaptive": plan_adaptive_cells(matrix),
        "figure_4_spot_check_sensitivity": plan_sensitivity_cells(matrix),
        "figure_6_sybil_scalability": plan_sybil_cells(matrix),
        "table_11_cross_hardware": plan_cross_hardware_cells(matrix),
    }


def build_formal_cell_plan(
    matrix: Mapping[str, Any],
    *,
    available_hardware_pairs: Iterable[str] = (),
) -> dict[str, Any]:
    available_pairs = set(available_hardware_pairs)
    routes = _route_cells(matrix)
    route_counts = {route: len(cells) for route, cells in routes.items()}
    records = []
    for route, cells in routes.items():
        for cell in cells:
            hardware_pair = getattr(cell, "hardware_pair", None)
            available = (
                hardware_pair in available_pairs
                if route == "table_11_cross_hardware"
                else True
            )
            records.append(
                {
                    "route": route,
                    "run_id": str(cell.run_id),
                    "available_on_host": available,
                    "hardware_pair": hardware_pair,
                }
            )
    run_ids = [record["run_id"] for record in records]
    unavailable = [record for record in records if not record["available_on_host"]]
    checks = {
        "route_counts": route_counts == EXPECTED_ROUTE_COUNTS,
        "paper_unique_cells": len(records) == 801,
        "run_ids_unique": len(set(run_ids)) == len(run_ids),
        "all_routes_nonempty": all(route_counts.values()),
        "figure_6_three_datasets": len(routes["figure_6_sybil_scalability"]) == 36,
        "unavailable_only_table_11": all(
            record["route"] == "table_11_cross_hardware" for record in unavailable
        ),
    }
    return {
        "schema_version": 1,
        "kind": "formal_cell_execution_plan",
        "paper_unique_formal_cells": len(records),
        "available_unique_formal_cells": sum(
            bool(record["available_on_host"]) for record in records
        ),
        "unavailable_unique_formal_cells": len(unavailable),
        "available_hardware_pairs": sorted(available_pairs),
        "unavailable_hardware_pairs": sorted(
            {
                str(record["hardware_pair"])
                for record in unavailable
                if record["hardware_pair"] is not None
            }
        ),
        "route_counts": route_counts,
        "derived_or_benchmark_routes": list(DERIVED_OR_BENCHMARK_ROUTES),
        "cells": records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "experiments" / "final" / "paper_matrix.json",
    )
    parser.add_argument("--available-hardware-pair", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = source_identity(ROOT)
    if source["dirty"] or not source["commit"]:
        raise RuntimeError("formal cell planning requires a clean, identified source")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    plan = build_formal_cell_plan(
        matrix,
        available_hardware_pairs=args.available_hardware_pair,
    )
    plan["source_commit"] = source["commit"]
    plan["input_sha256"] = {
        str(args.matrix.resolve().relative_to(ROOT)): hashlib.sha256(
            args.matrix.read_bytes()
        ).hexdigest()
    }
    plan = seal_evidence(plan, analysis_root=ROOT)
    write_manifest_atomic(args.output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not plan["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
