from pathlib import Path

from experiments.final.aggregate_figure6 import aggregate_figure6
from experiments.final.run_sybil_matrix import (
    plan_sybil_cells,
    sybil_command,
)
from experiments.final.sybil_scalability import aggregate_sybil_scalability


def test_sybil_scalability_uses_real_identity_evidence_and_total_stake():
    observations = [
        {
            "identity_count": 5,
            "identity_id": f"sybil-{index}",
            "detected": True,
            "stake_eth": 0.05,
            "real_trace": True,
            "real_groth16": True,
            "trace_digest": f"{index:064x}",
        }
        for index in range(5)
    ]
    targets = {
        "figure_6_sybil_scalability": {
            "5": {"DR": 94.5, "stake_eth": 0.25}
        }
    }
    aggregate = aggregate_sybil_scalability(observations, targets)
    assert aggregate["figure_6_sybil_scalability"]["5"]["DR"] == 100.0
    assert aggregate["figure_6_sybil_scalability"]["5"]["stake_eth"] == 0.25
    assert aggregate["acceptance"]["passed"]


def test_figure6_matrix_covers_four_identity_counts_and_three_seeds(tmp_path):
    matrix = {
        "seeds": [1337, 2026, 3817739],
        "studies": {
            "figure_6_sybil_scalability": {
                "identities_per_attacker": [5, 10, 15, 20]
            }
        },
    }
    cells = plan_sybil_cells(matrix)
    assert len(cells) == 12
    assert {cell.num_clients for cell in cells} == {45, 50, 55, 60}
    command = sybil_command(
        cells[-1],
        python=Path("/python"),
        output=tmp_path / "cell",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--study") + 1] == "sybil_scalability"
    assert command[command.index("--num-clients") + 1] == "60"


def test_complete_figure6_aggregate_requires_real_proofs_contracts_and_seeds():
    results = []
    detection = {5: 100.0, 10: 98.0, 15: 96.0, 20: 94.0}
    counter = 0
    for count in (5, 10, 15, 20):
        for seed in (1337, 2026, 3817739):
            results.append(
                {
                    "study": "sybil_scalability",
                    "dataset": "CIFAR10",
                    "attack": "Sybil",
                    "seed": seed,
                    "sybil_identity_count": count,
                    "sybil_stake_eth": 0.05 * count,
                    "DR": detection[count],
                    "FPR": 1.0,
                    "source_commit": "a" * 40,
                    "result_digest": f"{counter:064x}",
                    "real_groth16": True,
                    "real_contract_transition": True,
                    "formal_accepted": True,
                }
            )
            counter += 1
    targets = {
        "figure_6_sybil_scalability": {
            "5": {"DR": 94.5, "stake_eth": 0.25},
            "20": {"DR": 87.5, "stake_eth": 1.0},
        }
    }
    aggregate = aggregate_figure6(results, targets)
    assert aggregate["acceptance"]["passed"]
    assert aggregate["figure_6_sybil_scalability"]["20"]["stake_eth"] == 1.0
