#!/usr/bin/env python3
"""Transcribe both modes of Table 4 from the authoritative final PDF."""

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


AGGREGATIONS = ("Krum", "TrimmedMean", "Median")
ATTACKS = ("ALIE", "FreeRidingNT")


def parse_table4(text: str) -> dict[str, object]:
    marker = "Table 4: PoL-BFL + Robust Aggregation on CIFAR-10."
    end_marker = "Table 5: Incentive Mechanism Comparison on CIFAR-10."
    if marker not in text or end_marker not in text:
        raise ValueError("final-paper Table 4 boundaries were not found")
    block = text[text.index(marker) : text.index(end_marker, text.index(marker))]
    rows = [
        line
        for line in block.splitlines()
        if line.lstrip().startswith("ALIE")
        or line.lstrip().startswith("Free-riding")
    ]
    if len(rows) != 6:
        raise ValueError(f"expected six Table 4 rows, observed {len(rows)}")
    table: dict[str, object] = {}
    for position, line in enumerate(rows):
        values = [
            float(value)
            for value in re.findall(r"(?<![A-Za-z])\d+\.\d", line)
        ]
        if len(values) < 6:
            raise ValueError(f"Table 4 row {position} has {len(values)} values")
        values = values[:6]
        aggregation = AGGREGATIONS[position // 2]
        attack = ATTACKS[position % 2]
        table.setdefault(aggregation, {})[attack] = {
            "Standalone": {
                "MA": values[0],
                "DR": values[1],
                "FPR": values[2],
            },
            "PoLBFLPrefilter": {
                "MA": values[3],
                "DR": values[4],
                "FPR": values[5],
            },
        }
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_4_all_modes": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table4_all_modes.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 4 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table4(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
