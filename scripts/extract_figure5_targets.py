#!/usr/bin/env python3
"""Extract Figure 5 curve coordinates from the authoritative vector PDF."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, write_manifest_atomic
from experiments.final.preflight import PAPER_SHA256


GAS_PRICES_GWEI = (5, 10, 30, 50, 100, 200, 500, 1000)
FIGURE_PAGE = 11
POINT_PATTERN = re.compile(
    r"[ML]\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
)


def _points(path_data: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in POINT_PATTERN.findall(path_data)]


def parse_figure5_svg(svg_text: str) -> dict[str, object]:
    root = ET.fromstring(svg_text)
    paths = [element for element in root.iter() if element.tag.endswith("path")]
    curves = []
    for element in paths:
        style = element.attrib.get("style", "")
        points = _points(element.attrib.get("d", ""))
        if (
            "fill:none" in style
            and "stroke-width:3" in style
            and "stroke-linejoin:round" in style
            and len(points) == len(GAS_PRICES_GWEI)
        ):
            curves.append(
                {
                    "points": points,
                    "style": style,
                    "transform": element.attrib.get("transform", ""),
                }
            )
    if len(curves) != 2 or curves[0]["transform"] != curves[1]["transform"]:
        raise ValueError(f"expected two Figure 5 curves, observed {len(curves)}")
    solid = [curve for curve in curves if "stroke-dasharray" not in curve["style"]]
    dashed = [curve for curve in curves if "stroke-dasharray" in curve["style"]]
    if len(solid) != 1 or len(dashed) != 1:
        raise ValueError("Figure 5 solid/dashed curve identity is ambiguous")

    x_start = solid[0]["points"][0][0]
    x_end = solid[0]["points"][-1][0]
    transform = solid[0]["transform"]
    grid_values = []
    for element in paths:
        if element.attrib.get("transform", "") != transform:
            continue
        style = element.attrib.get("style", "")
        points = _points(element.attrib.get("d", ""))
        if (
            "stroke-opacity:0.3" in style
            and len(points) == 2
            and math.isclose(points[0][0], x_start, abs_tol=1e-3)
            and math.isclose(points[1][0], x_end, abs_tol=1e-3)
            and math.isclose(points[0][1], points[1][1], abs_tol=1e-6)
        ):
            grid_values.append(points[0][1])
    grid_values = sorted(set(grid_values), reverse=True)
    if len(grid_values) != 7:
        raise ValueError(f"expected seven Figure 5 y-grid lines, observed {len(grid_values)}")
    grid_steps = [
        left - right for left, right in zip(grid_values, grid_values[1:])
    ]
    grid_step = sum(grid_steps) / len(grid_steps)
    if grid_step <= 0 or any(abs(value - grid_step) > 0.02 for value in grid_steps):
        raise ValueError("Figure 5 y-grid is not the expected linear five-dollar scale")
    zero_grid = grid_values[0]

    def decode(curve: dict[str, object]) -> list[dict[str, object]]:
        raw_points = curve["points"]
        if any(left[0] >= right[0] for left, right in zip(raw_points, raw_points[1:])):
            raise ValueError("Figure 5 x coordinates are not strictly increasing")
        return [
            {
                "gas_price_gwei": gas_price,
                "payoff_usd": round(5.0 * (raw_y - zero_grid) / grid_step, 2),
            }
            for gas_price, (_raw_x, raw_y) in zip(GAS_PRICES_GWEI, raw_points)
        ]

    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "physical_pdf_page": FIGURE_PAGE,
        "figure_5_gas_price_stress": {
            "HonestNetProfit": decode(solid[0]),
            "AttackExpectedPayoff": decode(dashed[0]),
        },
        "extraction": {
            "x_values_gwei": list(GAS_PRICES_GWEI),
            "y_grid_usd": [0, -5, -10, -15, -20, -25, -30],
            "curve_identity": {
                "HonestNetProfit": "solid",
                "AttackExpectedPayoff": "dashed",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_figure5_targets.json",
    )
    parser.add_argument("--pdftocairo", default="pdftocairo")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Figure 5 extraction requires the authoritative final-paper PDF")
    with tempfile.TemporaryDirectory(prefix="polbfl-figure5-") as directory:
        svg_path = Path(directory) / "page.svg"
        completed = subprocess.run(
            [
                args.pdftocairo,
                "-f",
                str(FIGURE_PAGE),
                "-l",
                str(FIGURE_PAGE),
                "-svg",
                str(args.paper.resolve()),
                str(svg_path),
            ],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not svg_path.is_file():
            raise RuntimeError(completed.stderr.strip() or "pdftocairo failed")
        payload = parse_figure5_svg(svg_path.read_text(encoding="utf-8"))
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
