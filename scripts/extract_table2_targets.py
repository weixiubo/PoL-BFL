#!/usr/bin/env python3
"""Transcribe Table 2 directly from the authoritative final-paper PDF."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, write_manifest_atomic
from experiments.final.preflight import PAPER_SHA256


ATTACK_LABELS = (
    "Free-riding (NT)",
    "Free-riding (LT)",
    "Byzantine (Random)",
    "Model Replacement",
    "ALIE",
    "MinMax",
    "Data Poisoning",
    "Sybil",
)
ATTACK_NAMES = {
    "Free-riding (NT)": "FreeRidingNT",
    "Free-riding (LT)": "FreeRidingLT",
    "Byzantine (Random)": "ByzantineRandom",
    "Model Replacement": "ModelReplacement",
    "ALIE": "ALIE",
    "MinMax": "MinMax",
    "Data Poisoning": "DataPoisoning",
    "Sybil": "Sybil",
}
DATASETS = ("CIFAR10", "FEMNIST", "CIFAR100")
METHODS = ("Krum", "SDEA", "ShapleyFL", "FoolsGold", "PoLBFL")


def parse_table2(text: str) -> dict[str, object]:
    marker = "Table 2: Main Security Results across 8 attacks and 3 datasets."
    end_marker = "Table 3: Layer Contribution Analysis"
    if marker not in text or end_marker not in text:
        raise ValueError("final-paper Table 2 boundaries were not found")
    block = text[text.index(marker) : text.index(end_marker, text.index(marker))]
    lines = [
        line
        for line in block.splitlines()
        if any(line.lstrip().startswith(label) for label in ATTACK_LABELS)
    ]
    expected_rows = len(DATASETS) * len(ATTACK_LABELS)
    if len(lines) != expected_rows:
        raise ValueError(f"expected {expected_rows} Table 2 rows, observed {len(lines)}")
    table: dict[str, dict[str, object]] = {}
    for position, line in enumerate(lines):
        label = next(
            label for label in ATTACK_LABELS if line.lstrip().startswith(label)
        )
        values = [
            float(value)
            for value in re.findall(r"(?<![A-Za-z])\d+\.\d", line)
        ]
        if len(values) != 16:
            raise ValueError(
                f"Table 2 row {position} has {len(values)} numeric values, expected 16"
            )
        dataset = DATASETS[position // len(ATTACK_LABELS)]
        attack = ATTACK_NAMES[label]
        row: dict[str, object] = {"VanillaFL": {"MA": values[0]}}
        offset = 1
        for method in METHODS:
            row[method] = {
                "MA": values[offset],
                "DR": values[offset + 1],
                "FPR": values[offset + 2],
            }
            offset += 3
        table.setdefault(dataset, {})[attack] = row
    if any(
        tuple(attacks) != tuple(ATTACK_NAMES[label] for label in ATTACK_LABELS)
        for attacks in table.values()
    ):
        raise ValueError("Table 2 attack order or membership changed")
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_2_all_methods": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table2_all_methods.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 2 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table2(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
