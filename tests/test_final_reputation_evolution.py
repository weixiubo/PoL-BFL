import json
from pathlib import Path

from experiments.final.reputation_evolution import aggregate_reputation_evolution


ROOT = Path(__file__).parents[1]


def _rows(kind):
    output = []
    for round_number in range(200):
        output.append(
            {
                "round": round_number,
                "honest_reputation_mean": 1.0,
                "malicious_reputation_mean": 0.0,
                "reputation_by_client": {"client-0": 0.5},
                "effective_reputation_by_client": {"client-0": 0.5},
                "stake_by_client": {"client-0": "0.05"},
                "settlement_digest": f"{round_number + (1000 if kind == 'rational' else 2000):064x}",
            }
        )
    return output


def test_reputation_evolution_uses_every_real_round_and_paper_sample_points():
    targets = json.loads(
        (ROOT / "config" / "paper_figure3_targets.json").read_text(encoding="utf-8")
    )
    result = aggregate_reputation_evolution(
        _rows("rational"), _rows("malicious"), targets
    )
    assert result["acceptance"]["passed"]
    points = result["figure_3_reputation_evolution"]
    assert [point["round"] for point in points] == [0, 20, 50, 100, 150, 200]
    assert points[0]["Honest"] == points[0]["Rational"] == points[0]["Malicious"] == 50.0
