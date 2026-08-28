#!/usr/bin/env python3
"""Extract Figure 6 Sybil-scaling coordinates from the authoritative vector PDF."""

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


FIGURE_PAGE = 11
IDENTITY_COUNTS = (5, 10, 15, 20)
COLORS = {
    "CIFAR10": "83.59375%,36.863708%,0%",
    "FEMNIST": "0%,61.914062%,45.092773%",
    "CIFAR100": "33.714294%,70.506287%,91.40625%",
}
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
        raise ValueError("Figure 6 coordinates do not resolve to one-decimal paper values")
    return rounded


def parse_figure6_svg(svg_text: str) -> dict[str, object]:
    paths = [element for element in ET.fromstring(svg_text).iter() if element.tag.endswith("path")]
    line_curves = []
    for element in paths:
        style = element.attrib.get("style", "")
        points = _points(element)
        if "fill:none" in style and "stroke-width:40" in style and len(points) == 4:
            line_curves.append((element, points))
    transforms = {element.attrib.get("transform", "") for element, _points_value in line_curves}
    if len(transforms) != 1:
        raise ValueError("Figure 6 plot transform is ambiguous")
    transform = next(iter(transforms))

    grid_groups: dict[str, list[float]] = {"MA": [], "DR": [], "FPR": []}
    for element in paths:
        if element.attrib.get("transform", "") != transform:
            continue
        style = element.attrib.get("style", "")
        points = _points(element)
        if (
            "stroke-opacity:0.3" not in style
            or len(points) != 2
            or abs(points[0][1] - points[1][1]) > 1e-6
            or abs(points[1][0] - points[0][0]) < 3000
        ):
            continue
        start_x = points[0][0]
        panel = "MA" if start_x < 4500 else ("DR" if start_x < 9000 else "FPR")
        grid_groups[panel].append(points[0][1])
    grid_groups = {name: sorted(set(values)) for name, values in grid_groups.items()}
    if {name: len(values) for name, values in grid_groups.items()} != {"MA": 8, "DR": 9, "FPR": 8}:
        raise ValueError("Figure 6 axis geometry is ambiguous")

    def color_name(style: str) -> str | None:
        for dataset, color in COLORS.items():
            if color in style:
                return dataset
        return None

    raw: dict[str, dict[str, list[float]]] = {
        dataset: {} for dataset in COLORS
    }
    fpr_x_values: list[float] | None = None
    for element, points in line_curves:
        dataset = color_name(element.attrib.get("style", ""))
        if dataset is None:
            continue
        start_x = points[0][0]
        panel = "MA" if start_x < 4500 else ("DR" if start_x < 9000 else "FPR")
        raw[dataset][panel] = [point[1] for point in points]
        if dataset == "FEMNIST" and panel == "FPR":
            fpr_x_values = [point[0] for point in points]

    if fpr_x_values is None:
        raise ValueError("Figure 6 FPR x coordinates are missing")

    brown_markers = []
    for element in paths:
        if element.attrib.get("transform", "") != transform:
            continue
        style = element.attrib.get("style", "")
        points = _points(element)
        if COLORS["CIFAR10"] not in style or len(points) != 4:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        center_x = (min(xs) + max(xs)) / 2.0
        if min(abs(center_x - expected) for expected in fpr_x_values) < 1.0:
            brown_markers.append((center_x, (min(ys) + max(ys)) / 2.0))
    brown_markers.sort()
    if len(brown_markers) != 4:
        raise ValueError("Figure 6 CIFAR-10 FPR markers are incomplete")
    raw["CIFAR10"]["FPR"] = [value for _x, value in brown_markers]
    if any(set(metrics) != {"MA", "DR", "FPR"} for metrics in raw.values()):
        raise ValueError("Figure 6 dataset curves are incomplete")

    decoded: dict[str, dict[str, list[float]]] = {}
    for dataset, metrics in raw.items():
        decoded[dataset] = {
            "MA": _decode(metrics["MA"], grid_groups["MA"][0], grid_groups["MA"][-1], 55.0, 90.0),
            "DR": _decode(metrics["DR"], grid_groups["DR"][0], grid_groups["DR"][-1], 80.0, 100.0),
            "FPR": _decode(metrics["FPR"], grid_groups["FPR"][0], grid_groups["FPR"][-1], 2.0, 3.75),
        }
    vector = {
        dataset: {
            str(count): {
                "MA": metrics["MA"][index],
                "DR": metrics["DR"][index],
                "FPR": metrics["FPR"][index],
                "stake_eth": round(0.05 * count, 2),
            }
            for index, count in enumerate(IDENTITY_COUNTS)
        }
        for dataset, metrics in decoded.items()
    }
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "physical_pdf_page": FIGURE_PAGE,
        "figure_6_sybil_scalability": {
            "5": {"DR": vector["CIFAR10"]["5"]["DR"], "stake_eth": 0.25},
            "20": {"DR": vector["CIFAR10"]["20"]["DR"], "stake_eth": 1.0},
        },
        "figure_6_vector_targets": vector,
        "extraction": {
            "datasets": list(COLORS),
            "identity_counts": list(IDENTITY_COUNTS),
            "stake_rule_eth_per_identity": 0.05,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "config" / "paper_figure6_targets.json"
    )
    parser.add_argument("--pdftocairo", default="pdftocairo")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Figure 6 extraction requires the authoritative final-paper PDF")
    with tempfile.TemporaryDirectory(prefix="polbfl-figure6-") as directory:
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
        payload = parse_figure6_svg(svg_path.read_text(encoding="utf-8"))
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
