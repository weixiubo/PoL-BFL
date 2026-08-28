#!/usr/bin/env python3
"""Extract Figure 2 convergence coordinates from the authoritative vector PDF."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, write_manifest_atomic
from experiments.final.preflight import PAPER_SHA256


FIGURE_PAGE = 7
SAMPLE_ROUNDS = (0, 50, 100, 150, 200)
METHOD_COLORS = {
    "VanillaFL": "59.959412%,59.959412%,59.959412%",
    "Krum": "90.234375%,62.304688%,0%",
    "SDEA": "0%,61.914062%,45.092773%",
    "ShapleyFL": "33.714294%,70.506287%,91.40625%",
    "FoolsGold": "80.076599%,47.459412%,65.429688%",
    "PoLBFL": "83.59375%,36.863708%,0%",
}
ATTACKS = ("FreeRidingNT", "ALIE", "Sybil")
POINT_PATTERN = re.compile(r"[ML]\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _points(element: ET.Element) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in POINT_PATTERN.findall(element.attrib.get("d", ""))
    ]


def _all_coordinate_pairs(element: ET.Element) -> list[tuple[float, float]]:
    values = [float(value) for value in NUMBER_PATTERN.findall(element.attrib.get("d", ""))]
    if len(values) % 2:
        raise ValueError("Figure 2 marker path has an odd coordinate count")
    return list(zip(values[::2], values[1::2]))


def parse_figure2_svg(svg_text: str) -> dict[str, object]:
    paths = [element for element in ET.fromstring(svg_text).iter() if element.tag.endswith("path")]

    def method_for_style(style: str) -> str | None:
        for method, color in METHOD_COLORS.items():
            if color in style:
                return method
        return None

    curves = []
    for element in paths:
        style = element.attrib.get("style", "")
        points = _points(element)
        method = method_for_style(style)
        if (
            method is not None
            and "fill:none" in style
            and any(width in style for width in ("stroke-width:15", "stroke-width:18", "stroke-width:30"))
            and len(points) >= 4
        ):
            curves.append((element, points, method))
    if len(curves) != len(ATTACKS) * len(METHOD_COLORS):
        raise ValueError(f"expected 18 Figure 2 curves, observed {len(curves)}")
    transforms = {element.attrib.get("transform", "") for element, _points_value, _method in curves}
    if len(transforms) != 1:
        raise ValueError("Figure 2 plot transform is ambiguous")
    transform = next(iter(transforms))

    def attack_for_x(value: float) -> str:
        return ATTACKS[0] if value < 3500 else (ATTACKS[1] if value < 7000 else ATTACKS[2])

    grouped = {
        (attack_for_x(points[0][0]), method): (element, points)
        for element, points, method in curves
    }
    if len(grouped) != len(ATTACKS) * len(METHOD_COLORS):
        raise ValueError("Figure 2 curve identity is ambiguous")

    grid_y = []
    for element in paths:
        if element.attrib.get("transform", "") != transform:
            continue
        style = element.attrib.get("style", "")
        points = _points(element)
        if (
            "stroke-opacity:0.3" in style
            and len(points) == 2
            and abs(points[0][1] - points[1][1]) < 1e-6
            and abs(points[1][0] - points[0][0]) > 2500
        ):
            grid_y.append(points[0][1])
    grid_y = sorted(set(grid_y))
    if len(grid_y) != 4:
        raise ValueError("Figure 2 y-grid geometry is ambiguous")
    steps = [right - left for left, right in zip(grid_y, grid_y[1:])]
    step = statistics.fmean(steps)
    if any(abs(value - step) > 0.2 for value in steps):
        raise ValueError("Figure 2 y-grid is not linear")

    figure: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for attack in ATTACKS:
        expected_x = sorted(point[0] for point in grouped[(attack, "PoLBFL")][1])
        if len(expected_x) != len(SAMPLE_ROUNDS):
            raise ValueError("Figure 2 sample-round x coordinates are incomplete")
        figure[attack] = {}
        for method, color in METHOD_COLORS.items():
            _element, line_points = grouped[(attack, method)]
            values_by_x = {point[0]: point[1] for point in line_points}
            if len(values_by_x) < len(expected_x):
                for marker in paths:
                    if marker.attrib.get("transform", "") != transform:
                        continue
                    style = marker.attrib.get("style", "")
                    if color not in style or "stroke-width:10" not in style:
                        continue
                    coordinates = _all_coordinate_pairs(marker)
                    if not coordinates:
                        continue
                    xs = [point[0] for point in coordinates]
                    ys = [point[1] for point in coordinates]
                    center_x = (min(xs) + max(xs)) / 2.0
                    center_y = (min(ys) + max(ys)) / 2.0
                    closest = min(expected_x, key=lambda value: abs(value - center_x))
                    if abs(closest - center_x) < 1.0:
                        values_by_x.setdefault(closest, center_y)
            raw_y = []
            for expected in expected_x:
                candidates = [value for x_value, value in values_by_x.items() if abs(x_value - expected) < 1.0]
                if len(candidates) != 1:
                    raise ValueError(f"Figure 2 point is missing or duplicated: {attack}/{method}")
                raw_y.append(candidates[0])
            decoded = [20.0 + (value - grid_y[0]) * 20.0 / step for value in raw_y]
            rounded = [round(value, 1) for value in decoded]
            if any(abs(value - result) > 0.06 for value, result in zip(decoded, rounded)):
                raise ValueError("Figure 2 coordinates do not resolve to one-decimal paper values")
            figure[attack][method] = [
                {"round": round_number, "MA": rounded[index]}
                for index, round_number in enumerate(SAMPLE_ROUNDS)
            ]
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "physical_pdf_page": FIGURE_PAGE,
        "figure_2_convergence": figure,
        "extraction": {
            "attacks": list(ATTACKS),
            "methods": list(METHOD_COLORS),
            "sample_rounds": list(SAMPLE_ROUNDS),
            "accuracy_tick_values": [20, 40, 60, 80],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "config" / "paper_figure2_targets.json"
    )
    parser.add_argument("--pdftocairo", default="pdftocairo")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Figure 2 extraction requires the authoritative final-paper PDF")
    with tempfile.TemporaryDirectory(prefix="polbfl-figure2-") as directory:
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
        payload = parse_figure2_svg(svg_path.read_text(encoding="utf-8"))
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
