#!/usr/bin/env python3
"""Transcribe the complete layer-contribution matrix from final-paper Table 3."""

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


DATASETS = ("CIFAR10", "FEMNIST", "CIFAR100")
ATTACK_LABELS = (
    ("Free-riding (NT)", "FreeRidingNT"),
    ("ALIE", "ALIE"),
    ("Sybil", "Sybil"),
)
PROFILES = ("L1", "L1L2", "L1L3", "Full")


def parse_table3(text: str) -> dict[str, object]:
    marker = "Table 3: Layer Contribution Analysis (DR %)."
    boundary = "Table 4: PoL-BFL + Robust Aggregation on CIFAR-10."
    if marker not in text or boundary not in text[text.index(marker) :]:
        raise ValueError("final-paper Table 3 boundaries were not found")
    block = text[text.index(marker) : text.index(boundary, text.index(marker))]
    rows: list[tuple[str, list[float]]] = []
    for line in block.splitlines():
        for label, attack in ATTACK_LABELS:
            if label not in line:
                continue
            right = line[line.index(label) + len(label) :]
            values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", right)]
            if len(values) < len(PROFILES):
                raise ValueError(f"Table 3 row has too few profile values: {label}")
            rows.append((attack, values[: len(PROFILES)]))
            break
    expected_attacks = [attack for _dataset in DATASETS for _label, attack in ATTACK_LABELS]
    if [attack for attack, _values in rows] != expected_attacks:
        raise ValueError("Table 3 attack rows are missing, duplicated, or out of order")
    table: dict[str, dict[str, dict[str, float]]] = {}
    offset = 0
    for dataset in DATASETS:
        table[dataset] = {}
        for _label, attack in ATTACK_LABELS:
            observed_attack, values = rows[offset]
            offset += 1
            if observed_attack != attack:
                raise ValueError("Table 3 attack order changed during transcription")
            table[dataset][attack] = dict(zip(PROFILES, values))
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_3_layer_dr": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table3_layer_dr.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 3 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table3(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
