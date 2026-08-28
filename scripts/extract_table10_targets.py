#!/usr/bin/env python3
"""Transcribe the adaptive-attacker matrix from final-paper Table 10."""

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


VARIANT_LABELS = {
    "Baseline (NT)": "BaselineNT",
    "Checkpoint Interpolation": "CheckpointInterpolation",
    "Gradient Mimicry": "GradientMimicry",
    "Partial Replay": "PartialReplay",
    "Combined Adaptive": "CombinedAdaptive",
}


def parse_table10(text: str) -> dict[str, object]:
    marker = "Table 10: Adaptive Attacker Evaluation on CIFAR-10."
    boundary = "A.1.5"
    if marker not in text or boundary not in text[text.index(marker) :]:
        raise ValueError("final-paper Table 10 boundaries were not found")
    block = text[text.index(marker) : text.index(boundary, text.index(marker))]
    table: dict[str, dict[str, object]] = {}
    for label, variant in VARIANT_LABELS.items():
        rows = [line for line in block.splitlines() if label in line]
        if len(rows) != 1:
            raise ValueError(f"Table 10 variant row is missing or duplicated: {label}")
        right = rows[0][rows[0].index(label) + len(label) :]
        values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", right)]
        expected = 2 if variant == "BaselineNT" else 3
        if len(values) != expected:
            raise ValueError(f"Table 10 variant has an invalid metric count: {label}")
        row: dict[str, object] = {
            "DR": values[0],
            "FPR": values[1],
            "profitable": False,
        }
        if variant != "BaselineNT":
            row["forge_train_ratio"] = values[2]
        table[variant] = row
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_10_adaptive": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table10_adaptive.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 10 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table10(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
