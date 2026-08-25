from pathlib import Path

from experiments.final.adaptive_evaluation import aggregate_adaptive_trials
from experiments.final.adaptive_trial_support import VARIANTS
from experiments.final.aggregate_table10 import aggregate_table10
from experiments.final.run_adaptive_matrix import (
    adaptive_command,
    plan_adaptive_cells,
)


def _trial(behavior, index, *, detected, profit, forge=2.9, honest=1.0):
    return {
        "variant": "CombinedAdaptive",
        "trial_id": f"{behavior}-{index}",
        "behavior": behavior,
        "detected": detected,
        "expected_profit_usd": profit,
        "forge_seconds": forge,
        "honest_train_seconds": honest,
        "real_trace": True,
        "real_groth16": True,
        "proof_digest": f"{index:064x}" if behavior == "honest" else f"{index + 100:064x}",
    }


def test_adaptive_aggregate_uses_measured_costs_and_real_proofs():
    trials = [
        *[_trial("honest", index, detected=False, profit=0.1) for index in range(10)],
        *[
            _trial("malicious", index, detected=index < 9, profit=-0.1)
            for index in range(10)
        ],
    ]
    targets = {
        "table_10_adaptive": {
            "CombinedAdaptive": {
                "DR": 85.2,
                "FPR": 4.0,
                "forge_train_ratio": 2.8,
                "profitable": False,
            }
        }
    }
    aggregate = aggregate_adaptive_trials(trials, targets)
    assert aggregate["table_10_adaptive"]["CombinedAdaptive"] == {
        "DR": 90.0,
        "FPR": 0.0,
        "profitable": False,
        "forge_train_ratio": 2.9,
    }
    assert aggregate["acceptance"]["passed"]


def test_adaptive_matrix_covers_five_variants_and_three_seeds(tmp_path):
    matrix = {
        "seeds": [1337, 2026, 3817739],
        "studies": {"table_10_adaptive": {"variants": list(VARIANTS)}},
    }
    cells = plan_adaptive_cells(matrix)
    assert len(cells) == 15
    command = adaptive_command(
        cells[0],
        python=Path("/python"),
        output=tmp_path / "cell",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--variant") + 1] == cells[0].variant


def test_complete_table10_aggregate_requires_measured_three_seed_trials():
    targets = {
        "table_10_adaptive": {
            "BaselineNT": {"DR": 96.5, "FPR": 2.1, "profitable": False},
            "CheckpointInterpolation": {"DR": 94.2, "FPR": 2.8, "forge_train_ratio": 1.8, "profitable": False},
            "GradientMimicry": {"DR": 91.5, "FPR": 3.2, "forge_train_ratio": 2.3, "profitable": False},
            "PartialReplay": {"DR": 88.8, "FPR": 3.5, "forge_train_ratio": 1.2, "profitable": False},
            "CombinedAdaptive": {"DR": 85.2, "FPR": 4.0, "forge_train_ratio": 2.8, "profitable": False},
        }
    }
    results = []
    counter = 0
    for variant in VARIANTS:
        target = targets["table_10_adaptive"][variant]
        for seed in (1337, 2026, 3817739):
            honest = {
                "variant": variant,
                "trial_id": f"{variant}-honest-{seed}",
                "behavior": "honest",
                "detected": False,
                "expected_profit_usd": 0.15,
                "real_trace": True,
                "real_groth16": True,
                "proof_digest": f"{counter:064x}",
            }
            malicious = {
                "variant": variant,
                "trial_id": f"{variant}-malicious-{seed}",
                "behavior": "malicious",
                "detected": True,
                "expected_profit_usd": -0.1,
                "real_trace": True,
                "real_groth16": True,
                "proof_digest": f"{counter + 1000:064x}",
            }
            if variant != "BaselineNT":
                malicious.update(
                    {
                        "forge_seconds": float(target["forge_train_ratio"]) + 0.5,
                        "honest_train_seconds": 1.0,
                    }
                )
            results.append(
                {
                    "study": "adaptive",
                    "dataset": "CIFAR10",
                    "variant": variant,
                    "seed": seed,
                    "source_commit": "a" * 40,
                    "evidence_digest": f"{counter + 2000:064x}",
                    "result_digest": f"{counter + 3000:064x}",
                    "trials": [honest, malicious],
                    "formal_accepted": True,
                }
            )
            counter += 1
    aggregate = aggregate_table10(results, targets)
    assert aggregate["acceptance"]["passed"]
    assert aggregate["table_10_adaptive"]["CombinedAdaptive"]["DR"] == 100.0
