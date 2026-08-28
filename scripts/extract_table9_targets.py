#!/usr/bin/env python3
"""Transcribe the complete non-IID sensitivity matrix from final-paper Table 9."""

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
PARTITIONS = ("0.1", "0.5", "1.0", "IID")
METRICS = {
    "No Attack MA (%)": "NoAttackMA",
    "Free-riding DR (%)": "FreeRidingDR",
    "ALIE DR (%)": "ALIEDR",
    "FPR (%)": "FPR",
}


def _metric_rows(block: str, label: str) -> list[list[float]]:
    rows = [line for line in block.splitlines() if label in line]
    if len(rows) != len(DATASETS):
        raise ValueError(
            f"Table 9 metric row count differs from the three datasets: {label}"
        )
    values = []
    for line in rows:
        right = line[line.index(label) + len(label) :]
        parsed = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", right)]
        if len(parsed) != len(PARTITIONS):
            raise ValueError(
                f"Table 9 metric {label} does not contain four partition values"
            )
        values.append(parsed)
    return values


def parse_table9(text: str) -> dict[str, object]:
    marker = "Table 9: Non-IID Sensitivity using Dirichlet distribution"
    if marker not in text:
        raise ValueError("final-paper Table 9 marker was not found")
    tail = text[text.index(marker) :]
    boundaries = [
        position
        for token in ("A.1.2", "\f")
        if (position := tail.find(token)) > 0
    ]
    if not boundaries:
        raise ValueError("final-paper Table 9 boundary was not found")
    block = tail[: min(boundaries)]
    metric_rows = {
        metric: _metric_rows(block, label) for label, metric in METRICS.items()
    }
    table: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        table[dataset] = {}
        for partition_index, partition in enumerate(PARTITIONS):
            table[dataset][partition] = {
                metric: rows[dataset_index][partition_index]
                for metric, rows in metric_rows.items()
            }
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_9_noniid": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table9_noniid.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 9 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table9(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
