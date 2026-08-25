from experiments.final.convergence import ATTACKS, METHODS, aggregate_convergence


def test_convergence_requires_all_methods_attacks_seeds_and_rounds():
    trials = []
    targets = {"table_2_all_methods": {"CIFAR10": {}}}
    for attack in ATTACKS:
        targets["table_2_all_methods"]["CIFAR10"][attack] = {}
        for method in METHODS:
            targets["table_2_all_methods"]["CIFAR10"][attack][method] = {"MA": 80.0}
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
                                "accuracy": 10.0 + 71.0 * (round_number + 1) / 200,
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
        "MA": 81.0,
    }
