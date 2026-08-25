import pytest

from experiments.final.aggregate_noniid import aggregate_noniid_cells


def _row(attack, seed, *, ma=83.0, dr=95.0, fpr=4.0):
    return {
        "study": "noniid",
        "formal_accepted": True,
        "dataset": "CIFAR10",
        "partition_label": "0.1",
        "attack": attack,
        "seed": seed,
        "MA": ma,
        "DR": dr,
        "FPR": fpr,
        "source_commit": "a" * 40,
    }


def test_noniid_aggregate_requires_three_attacks_and_three_seeds():
    rows = [
        _row(attack, seed)
        for attack in ("NoAttack", "FreeRidingNT", "ALIE")
        for seed in (1, 2, 3)
    ]
    targets = {
        "table_9_noniid": {
            "CIFAR10": {
                "0.1": {
                    "NoAttackMA": 82.5,
                    "FreeRidingDR": 94.2,
                    "ALIEDR": 80.5,
                    "FPR": 4.8,
                }
            }
        }
    }
    aggregate = aggregate_noniid_cells(rows, targets)
    assert aggregate["acceptance"]["passed"]
    observed = aggregate["table_9_noniid"]["CIFAR10"]["0.1"]
    assert observed == {
        "NoAttackMA": 83.0,
        "FreeRidingDR": 95.0,
        "ALIEDR": 95.0,
        "FPR": 4.0,
    }


def test_noniid_aggregate_rejects_mixed_source_commits():
    rows = [
        _row(attack, seed)
        for attack in ("NoAttack", "FreeRidingNT", "ALIE")
        for seed in (1, 2, 3)
    ]
    rows[-1]["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="one valid source commit"):
        aggregate_noniid_cells(rows, {"table_9_noniid": {}})
