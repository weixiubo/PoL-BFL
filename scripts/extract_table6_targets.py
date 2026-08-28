#!/usr/bin/env python3
"""Transcribe the client-profit rows from final-paper Table 6."""

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


CLIENT_LABELS = {
    "Honest": "Honest",
    "Rational (LT)": "RationalLT",
    "Malicious (NT)": "MaliciousNT",
}
FIELDS = ("reward", "cost", "slash", "profit")
MONEY = re.compile(r"([+\-−–]?)\s*\$(\d+(?:\.\d+)?)")


def _money_values(text: str) -> list[float]:
    values = []
    for sign, number in MONEY.findall(text):
        value = float(number)
        values.append(-value if sign in {"-", "−", "–"} else value)
    return values


def parse_table6(text: str) -> dict[str, object]:
    marker = "Table 6: Client Profit Analysis in PoL-BFL."
    boundary = "Figure 3: Reputation evolution on CIFAR-10."
    if marker not in text or boundary not in text[text.index(marker) :]:
        raise ValueError("final-paper Table 6 boundaries were not found")
    block = text[text.index(marker) : text.index(boundary, text.index(marker))]
    table: dict[str, dict[str, float]] = {}
    for label, client in CLIENT_LABELS.items():
        rows = [line for line in block.splitlines() if label in line]
        if len(rows) != 1:
            raise ValueError(f"Table 6 client row is missing or duplicated: {label}")
        right = rows[0][rows[0].index(label) + len(label) :]
        values = _money_values(right)
        if len(values) != len(FIELDS):
            raise ValueError(f"Table 6 client row has an invalid money column count: {label}")
        table[client] = dict(zip(FIELDS, values))
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_6_profit_usd": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table6_profit.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 6 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table6(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
