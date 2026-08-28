import json
from pathlib import Path

from experiments.final.aggregate_figure6 import aggregate_figure6
from experiments.final.run_sybil_matrix import (
    plan_sybil_cells,
    sybil_command,
)
from experiments.final.sybil_scalability import aggregate_sybil_scalability


ROOT = Path(__file__).parents[1]


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


def test_figure6_matrix_covers_three_datasets_four_identity_counts_and_three_seeds(tmp_path):
    matrix = {
        "seeds": [1337, 2026, 3817739],
        "studies": {
            "figure_6_sybil_scalability": {
                "datasets": ["CIFAR10", "FEMNIST", "CIFAR100"],
                "identities_per_attacker": [5, 10, 15, 20]
            }
        },
    }
    cells = plan_sybil_cells(matrix)
    assert len(cells) == 36
    assert {cell.dataset for cell in cells} == {"CIFAR10", "FEMNIST", "CIFAR100"}
    assert {cell.num_clients for cell in cells} == {45, 50, 55, 60}
    command = sybil_command(
        cells[-1],
        python=Path("/python"),
        output=tmp_path / "cell",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--study") + 1] == "sybil_scalability"
    assert command[command.index("--dataset") + 1] == "CIFAR100"
    assert command[command.index("--num-clients") + 1] == "60"


def test_complete_figure6_aggregate_requires_real_proofs_contracts_and_seeds():
    results = []
    targets = json.loads(
        (ROOT / "config" / "paper_figure6_targets.json").read_text(encoding="utf-8")
    )
    counter = 0
    for dataset, dataset_targets in targets["figure_6_vector_targets"].items():
        for count in (5, 10, 15, 20):
            target = dataset_targets[str(count)]
            for seed in (1337, 2026, 3817739):
                results.append(
                    {
                        "study": "sybil_scalability",
                        "dataset": dataset,
                        "attack": "Sybil",
                        "seed": seed,
                        "sybil_identity_count": count,
                        "sybil_stake_eth": 0.05 * count,
                        "MA": target["MA"] + 0.1,
                        "DR": target["DR"] + 0.1,
                        "FPR": max(0.0, target["FPR"] - 0.1),
                        "source_commit": "a" * 40,
                        "result_digest": f"{counter:064x}",
                        "real_groth16": True,
                        "real_contract_transition": True,
                        "formal_accepted": True,
                    }
                )
                counter += 1
    aggregate = aggregate_figure6(results, targets)
    assert aggregate["acceptance"]["passed"]
    assert aggregate["figure_6_sybil_scalability"]["CIFAR10"]["20"]["stake_eth"] == 1.0
