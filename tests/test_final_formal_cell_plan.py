import json
from pathlib import Path

from experiments.final.plan_formal_cells import (
    EXPECTED_ROUTE_COUNTS,
    build_formal_cell_plan,
)


ROOT = Path(__file__).parents[1]


def test_unified_formal_plan_has_801_unique_cells_and_current_host_availability():
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    plan = build_formal_cell_plan(
        matrix,
        available_hardware_pairs=("RTX4090_RTX4090",),
    )
    assert plan["passed"]
    assert plan["route_counts"] == EXPECTED_ROUTE_COUNTS
    assert plan["paper_unique_formal_cells"] == 801
    assert plan["available_unique_formal_cells"] == 783
    assert plan["unavailable_unique_formal_cells"] == 18
    assert len({cell["run_id"] for cell in plan["cells"]}) == 801
    figure6 = [
        cell
        for cell in plan["cells"]
        if cell["route"] == "figure_6_sybil_scalability"
    ]
    assert len(figure6) == 36
    assert any("femnist" in cell["run_id"] for cell in figure6)
    assert any("cifar100" in cell["run_id"] for cell in figure6)
