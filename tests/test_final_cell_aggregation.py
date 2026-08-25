import pytest

from experiments.final.aggregate_cells import (
    aggregate_security_cells,
    validate_security_aggregate,
)


def test_security_cell_aggregation_reports_mean_ci_and_provenance():
    rows = [
        {
            "dataset": "CIFAR10",
            "attack": "FreeRidingNT",
            "seed": seed,
            "MA": accuracy,
            "DR": detection,
            "FPR": fpr,
            "source_commit": "a" * 40,
            "formal_accepted": True,
        }
        for seed, accuracy, detection, fpr in (
            (1, 86.0, 96.0, 2.0),
            (2, 87.0, 97.0, 2.2),
            (3, 88.0, 98.0, 1.8),
        )
    ]
    result = aggregate_security_cells(rows)
    metrics = result["table_2_pol_bfl"]["CIFAR10"]["FreeRidingNT"]
    assert metrics["MA"] == 87.0
    assert metrics["DR"] == 97.0
    assert metrics["FPR"] == 2.0
    assert metrics["MA_ci95"] > 0
    assert result["provenance"]["CIFAR10.FreeRidingNT"]["seeds"] == [1, 2, 3]
    targets = {
        "table_2_pol_bfl": {
            "CIFAR10": {"FreeRidingNT": {"MA": 86.8, "DR": 96.5, "FPR": 2.1}}
        }
    }
    assert validate_security_aggregate(result, targets)["passed"]


def test_security_aggregate_requires_all_seeds_and_target_directions():
    result = aggregate_security_cells(
        [
            {
                "dataset": "CIFAR10",
                "attack": "FreeRidingNT",
                "seed": 1,
                "MA": 80.0,
                "DR": 90.0,
                "FPR": 3.0,
                "source_commit": "a" * 40,
                "formal_accepted": True,
            }
        ]
    )
    targets = {
        "table_2_pol_bfl": {
            "CIFAR10": {"FreeRidingNT": {"MA": 86.8, "DR": 96.5, "FPR": 2.1}}
        }
    }
    report = validate_security_aggregate(result, targets)
    assert not report["passed"]
    assert set(report["failed"]) == {
        "CIFAR10.FreeRidingNT.MA",
        "CIFAR10.FreeRidingNT.DR",
        "CIFAR10.FreeRidingNT.FPR",
        "CIFAR10.FreeRidingNT.seed_count",
    }


def test_security_cell_aggregation_rejects_duplicate_seed():
    row = {
        "dataset": "CIFAR10",
        "attack": "ALIE",
        "seed": 1,
        "MA": 85,
        "DR": 90,
        "FPR": 2,
        "source_commit": "a" * 40,
        "formal_accepted": True,
    }
    with pytest.raises(ValueError, match="duplicate seed"):
        aggregate_security_cells([row, row])


def test_security_cell_aggregation_rejects_mixed_source_commits():
    rows = [
        {
            "dataset": "CIFAR10",
            "attack": "ALIE",
            "seed": seed,
            "MA": 85,
            "DR": 90,
            "FPR": 2,
            "source_commit": commit * 40,
            "formal_accepted": True,
        }
        for seed, commit in ((1, "a"), (2, "b"))
    ]
    with pytest.raises(ValueError, match="one valid source commit"):
        aggregate_security_cells(rows)
