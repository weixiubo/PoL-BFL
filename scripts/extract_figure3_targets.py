#!/usr/bin/env python3
"""Extract Figure 3 reputation coordinates from the authoritative vector PDF."""

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
SAMPLE_ROUNDS = (0, 20, 50, 100, 150, 200)
BEHAVIOR_COLORS = {
    "Honest": "0%,61.914062%,45.092773%",
    "Rational": "90.234375%,62.304688%,0%",
    "Malicious": "83.59375%,36.863708%,0%",
}
POINT_PATTERN = re.compile(r"[ML]\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def _points(element: ET.Element) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in POINT_PATTERN.findall(element.attrib.get("d", ""))
    ]


def parse_figure3_svg(svg_text: str) -> dict[str, object]:
    paths = [element for element in ET.fromstring(svg_text).iter() if element.tag.endswith("path")]
    curves: dict[str, tuple[ET.Element, list[tuple[float, float]]]] = {}
    for element in paths:
        style = element.attrib.get("style", "")
        points = _points(element)
        if "fill:none" not in style or "stroke-width:30" not in style or len(points) != 6:
            continue
        for behavior, color in BEHAVIOR_COLORS.items():
            if color in style:
                curves[behavior] = (element, points)
    if set(curves) != set(BEHAVIOR_COLORS):
        raise ValueError("Figure 3 behavior curves are incomplete")
    transforms = {element.attrib.get("transform", "") for element, _points_value in curves.values()}
    if len(transforms) != 1:
        raise ValueError("Figure 3 plot transform is ambiguous")
    transform = next(iter(transforms))
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
            and abs(points[1][0] - points[0][0]) > 6000
        ):
            grid_y.append(points[0][1])
    grid_y = sorted(set(grid_y))
    if len(grid_y) != 6:
        raise ValueError("Figure 3 y-grid geometry is ambiguous")
    decoded = {}
    for behavior, (_element, points) in curves.items():
        values = [
            100.0 * (point[1] - grid_y[0]) / (grid_y[-1] - grid_y[0])
            for point in points
        ]
        rounded = [round(value, 1) for value in values]
        if any(abs(value - result) > 0.06 for value, result in zip(values, rounded)):
            raise ValueError("Figure 3 coordinates do not resolve to one-decimal paper values")
        decoded[behavior] = rounded
    figure = [
        {
            "round": round_number,
            **{behavior: values[index] for behavior, values in decoded.items()},
        }
        for index, round_number in enumerate(SAMPLE_ROUNDS)
    ]
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "physical_pdf_page": FIGURE_PAGE,
        "figure_3_reputation_evolution": figure,
        "extraction": {
            "behaviors": list(BEHAVIOR_COLORS),
            "sample_rounds": list(SAMPLE_ROUNDS),
            "reputation_axis": {"minimum": 0.0, "maximum": 100.0},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "config" / "paper_figure3_targets.json"
    )
    parser.add_argument("--pdftocairo", default="pdftocairo")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Figure 3 extraction requires the authoritative final-paper PDF")
    with tempfile.TemporaryDirectory(prefix="polbfl-figure3-") as directory:
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
        payload = parse_figure3_svg(svg_path.read_text(encoding="utf-8"))
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
