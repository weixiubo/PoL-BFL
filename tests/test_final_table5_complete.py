import json
from pathlib import Path

from experiments.final.aggregate_table5 import aggregate_table5
from experiments.final.run_security_cell import table5_metrics
from experiments.final.run_table5_matrix import (
    plan_table5_cells,
    table5_command,
)
from scripts.extract_table5_targets import parse_table5


ROOT = Path(__file__).parents[1]


def test_table5_extractor_uses_the_left_hand_table_columns_only():
    text = "\n".join(
        [
            "Table 5: Incentive Mechanism Comparison on CIFAR-10.",
            " Participation Rate (%) 62.5 75.5 78.5 94.2 52.3 1850.2 265.8 78.5",
            " Attack Success Rate (%) 35.2 20.2 18.5 3.2 98.5 520.5 312.5 178.2",
            " Model Accuracy (%) 67.2 76.8 78.5 86.8 0.0 5.2 2.8 0.8",
            "Table 6: Client Profit Analysis in PoL-BFL.",
        ]
    )
    table = parse_table5(text)["table_5_all_methods"]
    assert table["Vanilla"]["ParticipationRate"] == 62.5
    assert table["PoLBFL"]["AttackSuccessRate"] == 3.2


def test_table5_aggregate_requires_real_method_specific_evidence():
    targets = json.loads(
        (ROOT / "config" / "paper_table5_all_methods.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for method, metrics in targets["table_5_all_methods"].items():
        for seed in (1337, 2026, 3817739):
            row = {
                "formal_accepted": True,
                "method": method,
                "seed": seed,
                **metrics,
                "source_commit": "a" * 40,
                "evidence_digest": f"{seed + len(method):064x}",
            }
            if method == "PoLBFL":
                row.update(
                    {
                        "real_contract_rounds": True,
                        "contract_rounds": 200,
                    }
                )
            else:
                row.update(
                    {
                        "real_training": True,
                        "training_rounds": 200,
                        "baseline_source_lock_digest": "b" * 64,
                    }
                )
            rows.append(row)
    result = aggregate_table5(rows, targets)
    assert result["acceptance"]["passed"]
    assert set(result["table_5_all_methods"]) == {
        "Vanilla",
        "FedCoin",
        "ShapleyFL",
        "PoLBFL",
    }


def test_table5_metrics_use_complete_measured_counterfactual_rounds():
    rows = [
        {
            "round": round_number,
            "registered_clients": 50,
            "honest_registered_clients": 40,
            "honest_participating_clients": 32,
            "valid_submissions": 40,
            "attack_success": round_number < 20,
            "malicious_submissions": 10,
            "malicious_attack_successes": 10 if round_number < 20 else 0,
            "accuracy": 80.0 + round_number / 1000,
        }
        for round_number in range(200)
    ]
    metrics = table5_metrics(rows)
    assert metrics["ParticipationRate"] == 80.0
    assert metrics["AttackSuccessRate"] == 10.0
    assert metrics["ModelAccuracy"] == rows[-1]["accuracy"]


def test_table5_matrix_covers_four_methods_and_three_seeds(tmp_path):
    matrix = {
        "seeds": [1337, 2026, 3817739],
        "studies": {
            "table_5_incentive_effectiveness": {
                "methods": ["Vanilla", "FedCoin", "ShapleyFL", "PoLBFL"]
            }
        },
    }
    cells = plan_table5_cells(matrix)
    assert len(cells) == 12
    fedcoin = next(cell for cell in cells if cell.method == "FedCoin")
    command = table5_command(
        fedcoin,
        python=Path("/python"),
        output=tmp_path / "cell",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--study") + 1] == "incentive"
    assert command[command.index("--method") + 1] == "FedCoin"
