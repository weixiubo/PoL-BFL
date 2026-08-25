import json
from pathlib import Path

from experiments.final.gas_price_stress import reconstruct_gas_stress
from scripts.extract_figure5_targets import parse_figure5_svg


ROOT = Path(__file__).parents[1]


def test_figure5_svg_extraction_distinguishes_solid_and_dashed_curves():
    transform = "matrix(1,0,0,-1,0,100)"
    grid = "\n".join(
        f'<path style="fill:none;stroke-opacity:0.3" d="M 1 {value} L 8 {value}" transform="{transform}"/>'
        for value in (70, 60, 50, 40, 30, 20, 10)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg">
      {grid}
      <path style="fill:none;stroke-width:3;stroke-linejoin:round"
        d="M 1 70 L 2 69 L 3 68 L 4 67 L 5 66 L 6 65 L 7 64 L 8 63"
        transform="{transform}"/>
      <path style="fill:none;stroke-width:3;stroke-linejoin:round;stroke-dasharray:11,4"
        d="M 1 69 L 2 68 L 3 67 L 4 66 L 5 65 L 6 64 L 7 63 L 8 62"
        transform="{transform}"/>
    </svg>'''
    payload = parse_figure5_svg(svg)
    curves = payload["figure_5_gas_price_stress"]
    assert curves["HonestNetProfit"][0] == {
        "gas_price_gwei": 5,
        "payoff_usd": 0.0,
    }
    assert curves["AttackExpectedPayoff"][0]["payoff_usd"] == -0.5


def test_figure5_reconstruction_matches_paper_regions_and_narrative_anchors():
    curves = json.loads(
        (ROOT / "config" / "paper_figure5_targets.json").read_text(
            encoding="utf-8"
        )
    )
    result = reconstruct_gas_stress(curves)
    assert result["acceptance"]["passed"]
    economics = result["derived_economics"]
    assert 11_000 < economics["inferred_client_operations_gas"] < 11_500
    assert economics["predicted_honest_profit_at_300_gwei"] < 0
    assert economics["predicted_attack_payoff_at_300_gwei"] < 0
