import pytest

from experiments.final.aggregate_table2 import aggregate_table2


def _row(method, seed, ma, dr=0.0, fpr=0.0):
    return {
        "formal_accepted": True,
        "study": "main",
        "dataset": "CIFAR10",
        "attack": "FreeRidingNT",
        "method": method,
        "seed": seed,
        "MA": ma,
        "DR": dr,
        "FPR": fpr,
        "source_commit": "a" * 40,
    }


def test_complete_table2_aggregate_handles_vanilla_and_detection_methods():
    targets = {
        "table_2_all_methods": {
            "CIFAR10": {
                "FreeRidingNT": {
                    "VanillaFL": {"MA": 67.2},
                    "PoLBFL": {"MA": 86.8, "DR": 96.5, "FPR": 2.1},
                }
            }
        }
    }
    rows = [
        *[_row("VanillaFL", seed, 68.0) for seed in (1337, 2026, 3817739)],
        *[
            _row("PoLBFL", seed, 87.0, dr=97.0, fpr=2.0)
            for seed in (1337, 2026, 3817739)
        ],
    ]
    result = aggregate_table2(rows, targets)
    assert result["acceptance"]["passed"]
    assert result["table_2_all_methods"]["CIFAR10"]["FreeRidingNT"]["PoLBFL"][
        "DR"
    ] == 97.0


def test_complete_table2_aggregate_rejects_missing_method_cell():
    targets = {
        "table_2_all_methods": {
            "CIFAR10": {"FreeRidingNT": {"VanillaFL": {"MA": 67.2}}}
        }
    }
    with pytest.raises(ValueError, match="coverage differs"):
        aggregate_table2([], targets)
