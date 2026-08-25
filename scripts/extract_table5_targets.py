#!/usr/bin/env python3
"""Transcribe the complete incentive comparison from final-paper Table 5."""

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


METHODS = ("Vanilla", "FedCoin", "ShapleyFL", "PoLBFL")
METRICS = {
    "Participation Rate (%)": "ParticipationRate",
    "Attack Success Rate (%)": "AttackSuccessRate",
    "Model Accuracy (%)": "ModelAccuracy",
}


def parse_table5(text: str) -> dict[str, object]:
    marker = "Table 5: Incentive Mechanism Comparison on CIFAR-10."
    end_marker = "Table 6: Client Profit Analysis in PoL-BFL."
    if marker not in text or end_marker not in text:
        raise ValueError("final-paper Table 5 boundaries were not found")
    block = text[text.index(marker) : text.index(end_marker, text.index(marker))]
    table = {method: {} for method in METHODS}
    for label, metric in METRICS.items():
        lines = [line for line in block.splitlines() if line.lstrip().startswith(label)]
        if len(lines) != 1:
            raise ValueError(f"Table 5 metric row is missing or duplicated: {label}")
        values = [float(value) for value in re.findall(r"(?<![A-Za-z])\d+\.\d", lines[0])]
        if len(values) < len(METHODS):
            raise ValueError(f"Table 5 metric {label} has {len(values)} values")
        values = values[: len(METHODS)]
        for method, value in zip(METHODS, values):
            table[method][metric] = value
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_5_all_methods": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table5_all_methods.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 5 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table5(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
