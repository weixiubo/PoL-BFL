#!/usr/bin/env python3
"""Transcribe the complete system-overhead comparison from Table 7."""

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


METHODS = ("Vanilla", "VeriblockFL", "Kaizen", "PoLBFL")
METRICS = {
    "Time/round (s)": "runtime_seconds",
    "Comm (MB/round)": "communication_mb",
    "Gas (USD/round)": "gas_usd",
    "Storage (MB/client)": "storage_mb_per_client",
}


def parse_table7(text: str) -> dict[str, object]:
    marker = "Table 7: System Overhead Comparison on CIFAR-10."
    end_marker = "Table 8: Scalability with Increasing Clients on CIFAR-10."
    if marker not in text or end_marker not in text:
        raise ValueError("final-paper Table 7 boundaries were not found")
    block = text[text.index(marker) : text.index(end_marker, text.index(marker))]
    table = {method: {} for method in METHODS}
    for label, metric in METRICS.items():
        lines = [line for line in block.splitlines() if label in line]
        if len(lines) != 1:
            raise ValueError(f"Table 7 metric row is missing or duplicated: {label}")
        right = lines[0][lines[0].index(label) + len(label) :]
        values = [float(value) for value in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", right)]
        if len(values) < len(METHODS):
            raise ValueError(f"Table 7 metric {label} has {len(values)} values")
        for method, value in zip(METHODS, values[: len(METHODS)]):
            table[method][metric] = value
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_7_all_methods": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table7_all_methods.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 7 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table7(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
