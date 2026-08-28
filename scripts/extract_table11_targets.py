#!/usr/bin/env python3
"""Transcribe the cross-hardware matrix from final-paper Table 11."""

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


EXPECTED_ROWS = (
    ("RTX 4090 -> RTX 4090", "RTX4090_RTX4090"),
    ("V100 -> V100", "V100_V100"),
    ("RTX 4090 -> RTX 3080", "RTX4090_RTX3080"),
    ("RTX 4090 -> V100", "RTX4090_V100"),
    ("RTX 4090 -> A100", "RTX4090_A100"),
    ("V100 -> A100", "V100_A100"),
    ("RTX 4090 -> V100", "Kaizen_RTX4090_V100"),
)
FIELDS = ("FPR", "honest_pass_rate", "DR", "block_rate")


def parse_table11(text: str) -> dict[str, object]:
    marker = "Table 11: Cross-Hardware Verification Performance on"
    if marker not in text:
        raise ValueError("final-paper Table 11 boundaries were not found")
    tail = text[text.index(marker) :]
    boundary = re.search(r"\n\s*A\.2\s+ZK Proof Cost", tail)
    if boundary is None:
        raise ValueError("final-paper Table 11 boundaries were not found")
    block = tail[: boundary.start()]
    observed: list[tuple[str, list[float]]] = []
    labels = tuple(dict.fromkeys(label for label, _key in EXPECTED_ROWS))
    for line in block.splitlines():
        normalized = " ".join(line.replace("→", "->").split())
        for label in labels:
            if label not in normalized:
                continue
            right = normalized[normalized.index(label) + len(label) :]
            values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", right)]
            if len(values) >= len(FIELDS):
                observed.append((label, values[: len(FIELDS)]))
            break
    if [label for label, _values in observed] != [label for label, _key in EXPECTED_ROWS]:
        raise ValueError("Table 11 hardware rows are missing, duplicated, or out of order")
    table = {
        key: dict(zip(FIELDS, values))
        for (_label, key), (_observed_label, values) in zip(EXPECTED_ROWS, observed)
    }
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_11_cross_hardware": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table11_cross_hardware.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 11 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table11(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
