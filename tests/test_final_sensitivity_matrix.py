import json
from decimal import Decimal
from pathlib import Path

from experiments.final.aggregate_sensitivity import aggregate_sensitivity_cells
from experiments.final.run_sensitivity_matrix import (
    PAPER_PROBABILITIES,
    plan_sensitivity_cells,
    sensitivity_command,
)


ROOT = Path(__file__).parents[1]


def test_sensitivity_matrix_covers_plot_probabilities_and_aggregates_three_seeds(tmp_path):
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(encoding="utf-8")
    )
    cells = plan_sensitivity_cells(matrix)
    assert len(cells) == len(PAPER_PROBABILITIES) * 3
    default = next(cell for cell in cells if cell.audit_probability == Decimal("0.20"))
    command = sensitivity_command(
        default,
        python=Path("/python"),
        output=tmp_path / "out",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--audit-probability") + 1] == "0.20"

    rows = []
    for index, probability in enumerate(PAPER_PROBABILITIES):
        for seed in (1337, 2026, 3817739):
            rows.append(
                {
                    "formal_accepted": True,
                    "study": "sensitivity",
                    "audit_probability": float(probability),
                    "seed": seed,
                    "MA": 88.0,
                    "DR": 97.0 + index * 0.1,
                    "FPR": 2.0,
                    "runtime_seconds": 70.0 + index,
                    "source_commit": "a" * 40,
                }
            )
    targets = json.loads(
        (ROOT / "config" / "paper_targets.json").read_text(encoding="utf-8")
    )
    assert aggregate_sensitivity_cells(rows, targets)["acceptance"]["passed"]
