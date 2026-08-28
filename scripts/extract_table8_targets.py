#!/usr/bin/env python3
"""Transcribe the complete scalability matrix from final-paper Table 8."""

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


CLIENT_COUNTS = ("50", "100", "200")
METRICS = {
    "Time/round (s)": "runtime_seconds",
    "Comm (MB/round)": "communication_mb",
    "Gas (USD/round)": "gas_usd",
    "Time/client (s)": "seconds_per_client",
    "MA (%)": "MA",
    "DR (%)": "DR",
    "FPR (%)": "FPR",
}


def parse_table8(text: str) -> dict[str, object]:
    marker = "Table 8: Scalability with Increasing Clients on CIFAR-10."
    boundary = "Figure 3: Reputation evolution on CIFAR-10."
    if marker not in text or boundary not in text[text.index(marker) :]:
        raise ValueError("final-paper Table 8 boundaries were not found")
    block = text[text.index(marker) : text.index(boundary, text.index(marker))]
    table = {count: {} for count in CLIENT_COUNTS}
    for label, metric in METRICS.items():
        rows = [line for line in block.splitlines() if label in line]
        if len(rows) != 1:
            raise ValueError(f"Table 8 metric row is missing or duplicated: {label}")
        right = rows[0][rows[0].index(label) + len(label) :]
        values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", right)]
        if len(values) < len(CLIENT_COUNTS):
            raise ValueError(f"Table 8 metric has too few client-count values: {label}")
        for count, value in zip(CLIENT_COUNTS, values[: len(CLIENT_COUNTS)]):
            table[count][metric] = value
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_8_scalability": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table8_scalability.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 8 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table8(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
