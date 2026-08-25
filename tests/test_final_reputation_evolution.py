from experiments.final.reputation_evolution import aggregate_reputation_evolution


def _rows(kind):
    output = []
    for round_number in range(200):
        progress = (round_number + 1) / 200
        malicious = (0.5 - (0.26 if kind == "rational" else 0.5) * progress)
        output.append(
            {
                "round": round_number,
                "honest_reputation_mean": 0.5 + 0.5 * progress,
                "malicious_reputation_mean": max(0.0, malicious),
                "reputation_by_client": {"client-0": 0.5},
                "effective_reputation_by_client": {"client-0": 0.5},
                "stake_by_client": {"client-0": "0.05"},
                "settlement_digest": f"{round_number + (1000 if kind == 'rational' else 2000):064x}",
            }
        )
    return output


def test_reputation_evolution_uses_every_real_round_and_paper_sample_points():
    result = aggregate_reputation_evolution(_rows("rational"), _rows("malicious"))
    assert result["acceptance"]["passed"]
    points = result["figure_3_reputation_evolution"]
    assert [point["round"] for point in points] == [0, 20, 50, 100, 150, 200]
    assert points[0]["Honest"] == points[0]["Rational"] == points[0]["Malicious"] == 50.0
