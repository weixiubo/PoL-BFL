from pathlib import Path

from experiments.final.convergence import ATTACKS, METHODS, aggregate_convergence
from experiments.final.target_provenance import (
    FIGURE2_TARGET_FILES,
    load_merged_targets,
)


ROOT = Path(__file__).parents[1]


def test_convergence_requires_all_methods_attacks_seeds_and_rounds():
    trials = []
    targets = load_merged_targets(ROOT, FIGURE2_TARGET_FILES)
    for attack in ATTACKS:
        for method in METHODS:
            for seed in (1337, 2026, 3817739):
                trials.append(
                    {
                        "formal_accepted": True,
                        "method": method,
                        "attack": attack,
                        "seed": seed,
                        "initial_accuracy": 10.0,
                        "rounds": [
                            {
                                "round": round_number,
                                "accuracy": 100.0,
                            }
                            for round_number in range(200)
                        ],
                        "source_commit": "a" * 40,
                        "manifest_digest": f"{seed:064x}",
                    }
                )
    result = aggregate_convergence(trials, targets)
    assert result["acceptance"]["passed"]
    assert result["figure_2_convergence"]["ALIE"]["PoLBFL"][-1] == {
        "round": 200,
        "MA": 100.0,
    }
