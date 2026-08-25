from experiments.final.incentive_effectiveness import aggregate_incentive_rounds


def test_incentive_effectiveness_requires_real_contiguous_contract_rounds():
    rows = [
        {
            "round": round_number,
            "registered_clients": 50,
            "valid_submissions": 48,
            "malicious_submissions": 10,
            "malicious_attack_successes": 0,
            "model_accuracy": 90.0,
            "real_contract_transition": True,
            "settlement_digest": f"{round_number:064x}",
        }
        for round_number in range(200)
    ]
    targets = {
        "table_5_incentive": {
            "ParticipationRate": 94.2,
            "AttackSuccessRate": 3.2,
            "ModelAccuracy": 86.8,
        }
    }
    aggregate = aggregate_incentive_rounds(rows, targets)
    assert aggregate["table_5_incentive"] == {
        "ParticipationRate": 96.0,
        "AttackSuccessRate": 0.0,
        "ModelAccuracy": 90.0,
    }
    assert aggregate["acceptance"]["passed"]
