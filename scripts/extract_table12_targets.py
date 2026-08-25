#!/usr/bin/env python3
"""Transcribe both proof-cost columns from final-paper Table 12."""

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


def _row(block: str, label: str) -> list[float]:
    lines = [line for line in block.splitlines() if line.lstrip().startswith(label)]
    if len(lines) != 1:
        raise ValueError(f"Table 12 row is missing or duplicated: {label}")
    values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", lines[0])]
    if not values:
        raise ValueError(f"Table 12 row has too few values: {label}")
    return values


def parse_table12(text: str) -> dict[str, object]:
    marker = "Table 12: ZK Proof Technical Specifications on CIFAR-10."
    if marker not in text:
        raise ValueError("final-paper Table 12 marker was not found")
    block = text[text.index(marker) :].split("\f", 1)[0]
    proof = _row(block, "Proof Gen Time")
    circuit = _row(block, "Circuit Size")
    witness = _row(block, "Witness Computation")
    memory = _row(block, "Prover Memory")
    proof_size = _row(block, "Proof Size")
    verification = _row(block, "Verification Time")
    merkle = _row(block, "Merkle Proof Size")
    total = _row(block, "Total Verification")
    table = {
        "PoLBFL": {
            "proof_generation_seconds": proof[0],
            "circuit_constraints": int(round(circuit[0] * 1_000_000)),
            "witness_seconds": witness[0],
            "prover_memory_gb": memory[0],
            "proof_bytes": int(proof_size[0]),
            "verification_ms": verification[0],
            "merkle_proof_kb": merkle[0],
            "total_verification_ms": total[0],
        },
        "Kaizen": {
            "proof_generation_seconds": proof[1],
            "circuit_constraints": int(round(circuit[1] * 1_000_000)),
            "witness_seconds": witness[1],
            "prover_memory_gb": memory[1],
            "proof_bytes": int(proof_size[1]),
            "verification_ms": verification[1],
            "merkle_proof_kb": None,
            "total_verification_ms": total[1],
        },
    }
    return {
        "schema_version": 1,
        "authority_pdf_sha256": PAPER_SHA256,
        "table_12_all_methods": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "paper_table12_all_methods.json",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args()
    if not args.paper.is_file() or sha256_file(args.paper) != PAPER_SHA256:
        raise ValueError("Table 12 extraction requires the authoritative final-paper PDF")
    completed = subprocess.run(
        [args.pdftotext, "-layout", str(args.paper.resolve()), "-"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    payload = parse_table12(completed.stdout)
    write_manifest_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
