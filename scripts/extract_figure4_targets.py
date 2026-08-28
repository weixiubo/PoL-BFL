#!/usr/bin/env python3
"""Extract Figure 4 sensitivity coordinates from the authoritative vector PDF."""

from __future__ import annotations

import argparse
import json
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


FIGURE_PAGE = 8
PROBABILITIES = ("0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.50", "1.00")
POINT_PATTERN = re.compile(r"[ML]\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def _points(element: ET.Element) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in POINT_PATTERN.findall(element.attrib.get("d", ""))
    ]


def _decode(values: list[float], raw_min: float, raw_max: float, low: float, high: float) -> list[float]:
    decoded = [low + (value - raw_min) * (high - low) / (raw_max - raw_min) for value in values]
    rounded = [round(value, 1) for value in decoded]
    if any(abs(value - result) > 0.06 for value, result in zip(decoded, rounded)):
        raise ValueError("Figure 4 coordinates do not resolve to one-decimal paper values")
    return rounded


def parse_figure4_svg(svg_text: str) -> dict[str, object]:
    paths = [element for element in ET.fromstring(svg_text).iter() if element.tag.endswith("path")]
    dr_candidates = []
    for element in paths:
        style = element.attrib.get("style", "")
        points = _points(element)
        if (
            "fill:none" in style
            and "stroke-width:30" in style
            and "stroke:rgb(0%,61.914062%,45.092773%)" in style
            and len(points) == len(PROBABILITIES)
        ):
            dr_candidates.append((element, points))
    if len(dr_candidates) != 1:
        raise ValueError(f"expected one Figure 4 DR curve, observed {len(dr_candidates)}")
    curve, dr_points = dr_candidates[0]
    transform = curve.attrib.get("transform", "")
    x_values = [point[0] for point in dr_points]

    grid_y = []
    border_y = []
    for element in paths:
        if element.attrib.get("transform", "") != transform:
            continue
        style = element.attrib.get("style", "")
        points = _points(element)
        if (
            len(points) == 2
            and abs(points[0][1] - points[1][1]) < 1e-6
            and abs(points[1][0] - points[0][0]) > 6000
        ):
            if "stroke-opacity:0.3" in style:
                grid_y.append(points[0][1])
            if (
                "stroke-width:8" in style
                and "stroke-linecap:square" in style
                and "stroke:rgb(0%,0%,0%)" in style
            ):
                border_y.append(points[0][1])
    grid_y = sorted(set(grid_y))
    border_y = sorted(set(border_y))
    if len(grid_y) != 6 or len(border_y) != 2:
        raise ValueError("Figure 4 axis geometry is ambiguous")

    def marker_values(token: str, point_count: int) -> list[float]:
        markers = []
        for element in paths:
            if element.attrib.get("transform", "") != transform:
                continue
            style = element.attrib.get("style", "")
            points = _points(element)
            if token not in style or len(points) != point_count:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            center_x = (min(xs) + max(xs)) / 2.0
            if min(abs(center_x - expected) for expected in x_values) < 1.0:
                markers.append((center_x, (min(ys) + max(ys)) / 2.0))
        markers.sort()
        if len(markers) != len(PROBABILITIES):
            raise ValueError(f"Figure 4 marker count is incomplete for {token}")
        return [value for _x, value in markers]

    raw = {
        "DR": [point[1] for point in dr_points],
        "FPR": marker_values("fill:rgb(83.59375%,36.863708%,0%)", 5),
        "MA": marker_values("stroke:rgb(33.714294%,70.506287%,91.40625%)", 4),
        "runtime_seconds": marker_values("stroke:rgb(90.234375%,62.304688%,0%)", 5),
    }
    decoded = {
        "DR": _decode(raw["DR"], grid_y[0], grid_y[-1], 0.0, 100.0),
        "FPR": _decode(raw["FPR"], grid_y[0], grid_y[-1], 0.0, 100.0),
        "MA": _decode(raw["MA"], border_y[0], border_y[-1], 80.0, 90.0),
        "runtime_seconds": _decode(
            raw["runtime_seconds"], border_y[0], border_y[-1], 40.0, 200.0
        ),
    }
    table = {
        probability: {metric: values[index] for metric, values in decoded.items()}
        for index, probability in enumerate(PROBABILITIES)
    }
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "physical_pdf_page": FIGURE_PAGE,
        "figure_4_spot_check_sensitivity": table,
        "extraction": {
            "probabilities": list(PROBABILITIES),
            "left_axis": {"minimum": 0.0, "maximum": 100.0},
            "ma_axis": {"minimum": 80.0, "maximum": 90.0},
            "runtime_axis": {"minimum": 40.0, "maximum": 200.0},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "config" / "paper_figure4_targets.json"
    )
    parser.add_argument("--pdftocairo", default="pdftocairo")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Figure 4 extraction requires the authoritative final-paper PDF")
    with tempfile.TemporaryDirectory(prefix="polbfl-figure4-") as directory:
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
        payload = parse_figure4_svg(svg_path.read_text(encoding="utf-8"))
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
