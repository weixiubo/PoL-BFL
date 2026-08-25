from pathlib import Path

from experiments.final.aggregate_table3 import (
    ATTACKS,
    DATASETS,
    VARIANTS,
    aggregate_table3,
)
from experiments.final.layer_contribution import aggregate_layer_trials
from experiments.final.run_layer_matrix import (
    layer_command,
    plan_layer_cells,
)


def _trial(behavior, index, detected):
    return {
        "dataset": "CIFAR10",
        "attack": "ALIE",
        "variant": "Full",
        "trial_id": f"{behavior}-{index}",
        "behavior": behavior,
        "detected": detected,
        "real_groth16": True,
        "real_robust_aggregation": True,
        "real_contract_transition": True,
        "evidence_digest": f"{index + (100 if behavior == 'malicious' else 0):064x}",
    }


def test_layer_contribution_requires_the_real_components_for_each_variant():
    trials = [
        *[_trial("honest", index, False) for index in range(10)],
        *[_trial("malicious", index, index < 9) for index in range(10)],
    ]
    targets = {
        "table_3_layer_dr": {
            "CIFAR10": {"ALIE": {"Full": 85.6}}
        }
    }
    aggregate = aggregate_layer_trials(trials, targets)
    assert aggregate["table_3_layer_dr"]["CIFAR10"]["ALIE"]["Full"] == 90.0
    assert aggregate["acceptance"]["passed"]


def test_layer_matrix_covers_every_dataset_attack_variant_and_seed(tmp_path):
    matrix = {
        "seeds": [1337, 2026, 3817739],
        "studies": {
            "table_3_layer_contribution": {
                "datasets": list(DATASETS),
                "attacks": list(ATTACKS),
                "variants": list(VARIANTS),
            }
        },
    }
    cells = plan_layer_cells(matrix)
    assert len(cells) == 3 * 3 * 4 * 3
    command = layer_command(
        cells[0],
        python=Path("/python"),
        output=tmp_path / "cell",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--study") + 1] == "layer"
    assert command[command.index("--layer-variant") + 1] == cells[0].variant


def test_complete_table3_aggregate_requires_all_profiles_and_three_seeds():
    targets = {"table_3_layer_dr": {}}
    results = []
    counter = 0
    for dataset in DATASETS:
        targets["table_3_layer_dr"][dataset] = {}
        for attack in ATTACKS:
            targets["table_3_layer_dr"][dataset][attack] = {}
            for variant in VARIANTS:
                targets["table_3_layer_dr"][dataset][attack][variant] = 90.0
                for seed in (1337, 2026, 3817739):
                    results.append(
                        {
                            "study": "layer",
                            "dataset": dataset,
                            "attack": attack,
                            "layer_variant": variant,
                            "seed": seed,
                            "DR": 95.0,
                            "FPR": 2.0,
                            "source_commit": "a" * 40,
                            "result_digest": f"{counter:064x}",
                            "real_groth16": True,
                            "real_robust_aggregation": variant in {"L1L2", "Full"},
                            "real_contract_transition": variant in {"L1L3", "Full"},
                            "formal_accepted": True,
                        }
                    )
                    counter += 1
    aggregate = aggregate_table3(results, targets)
    assert aggregate["acceptance"]["passed"]
    assert aggregate["table_3_layer_dr"]["CIFAR10"]["ALIE"]["Full"] == 95.0
