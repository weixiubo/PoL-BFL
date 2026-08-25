import json
from pathlib import Path

from experiments.final.run_economics import reproduce, validate_economics


ROOT = Path(__file__).parents[1]


def test_inferred_economic_parameters_reproduce_published_profit_rows_exactly():
    config = json.loads((ROOT / "config" / "economics_reference.json").read_text())
    result = reproduce(config)["table_6_profit_usd"]
    assert result["Honest"] == {"reward": 0.172, "cost": -0.022, "slash": 0.0, "profit": 0.15}
    assert result["RationalLT"] == {"reward": 0.098, "cost": -0.016, "slash": -0.008, "profit": 0.074}
    assert result["MaliciousNT"] == {"reward": 0.025, "cost": -0.015, "slash": -0.145, "profit": -0.135}
    assert result["incentive_compatible"] is True
    targets = json.loads(
        (ROOT / "config" / "paper_targets.json").read_text(encoding="utf-8")
    )
    report = validate_economics({"table_6_profit_usd": result}, targets)
    assert report["passed"]
