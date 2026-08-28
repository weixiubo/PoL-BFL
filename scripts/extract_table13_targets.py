#!/usr/bin/env python3
"""Transcribe the settlement-gas bounds from final-paper Table 13."""

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


OPERATIONS = {
    "Submit Commitment": "commitment",
    "Submit Proof Receipt": "proof_receipt",
    "Claim Reward": "reward_claim",
    "Slash Penalty": "slash",
    "Average/round": "honest_round_total",
}


def parse_table13(text: str) -> dict[str, object]:
    marker = "Table 13: Gas Cost Breakdown per Round (@1.5 gwei,"
    if marker not in text:
        raise ValueError("final-paper Table 13 boundaries were not found")
    tail = text[text.index(marker) :]
    boundary = re.search(r"A\.3\s+Protocol Pseudocode", tail)
    if boundary is None:
        raise ValueError("final-paper Table 13 boundaries were not found")
    block = tail[: boundary.start()]
    table: dict[str, int] = {}
    for label, operation in OPERATIONS.items():
        rows = [line for line in block.splitlines() if label in line]
        if len(rows) != 1:
            raise ValueError(f"Table 13 operation row is missing or duplicated: {label}")
        right = rows[0][rows[0].index(label) + len(label) :]
        matches = re.findall(r"\d{1,3}(?:,\d{3})+", right)
        if len(matches) != 1:
            raise ValueError(f"Table 13 gas value is missing or ambiguous: {label}")
        table[operation] = int(matches[0].replace(",", ""))
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_13_gas": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table13_gas.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 13 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table13(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
