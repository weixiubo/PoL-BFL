import json
from pathlib import Path

from experiments.final.run_scalability_matrix import (
    plan_scalability_cells,
    scalability_command,
)
from experiments.final.aggregate_scalability import aggregate_scalability_cells


ROOT = Path(__file__).parents[1]


def test_scalability_matrix_uses_all_clients_and_twenty_percent_attackers(tmp_path):
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(encoding="utf-8")
    )
    cells = plan_scalability_cells(matrix)
    assert len(cells) == 3 * 3
    cell = next(value for value in cells if value.num_clients == 200)
    command = scalability_command(
        cell,
        python=Path("/python"),
        output=tmp_path / "out",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--num-clients") + 1] == "200"
    assert command[command.index("--num-malicious") + 1] == "40"
    assert command[command.index("--clients-per-round") + 1] == "200"


def test_scalability_aggregate_requires_three_seeds_and_real_gas():
    rows = []
    for count, metrics in {
        50: (78.0, 178.0, 86.9, 97.0, 2.0),
        100: (152.0, 348.0, 85.8, 96.0, 2.2),
        200: (298.0, 685.0, 84.5, 95.0, 2.7),
    }.items():
        for seed in (1337, 2026, 3817739):
            rows.append(
                {
                    "formal_accepted": True,
                    "study": "scalability",
                    "num_clients": count,
                    "seed": seed,
                    "runtime_seconds": metrics[0],
                    "communication_mb": metrics[1],
                    "seconds_per_client": metrics[0] / count,
                    "MA": metrics[2],
                    "DR": metrics[3],
                    "FPR": metrics[4],
                    "source_commit": "a" * 40,
                }
            )
    targets = json.loads(
        (ROOT / "config" / "paper_table8_scalability.json").read_text(encoding="utf-8")
    )
    gas = {
        "passed": True,
        "observed_gas": {"honest_round_total": 151651},
        "source": {"commit": "a" * 40, "dirty": False},
        "evidence_digest": "b" * 64,
    }
    result = aggregate_scalability_cells(rows, targets, gas)
    assert result["acceptance"]["passed"]
    assert result["table_8_scalability"]["50"]["gas_usd"] < 0.85
