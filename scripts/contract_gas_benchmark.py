#!/usr/bin/env python3
"""Run the real Ganache protocol path and emit source-bound gas evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, source_identity, write_manifest_atomic
from experiments.final.evidence import seal_evidence


def build_evidence(raw: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    commitment = [int(value) for value in raw["commitment_gas"]]
    receipts = [int(value) for value in raw["receipt_gas"]]
    observed = {
        "commitment": max(commitment),
        "proof_receipt": max(receipts),
        "reward_claim": int(raw["reward_claim_gas"]),
        "slash": int(raw["slash_gas"]),
    }
    observed["honest_round_total"] = int(
        Decimal(observed["commitment"])
        + Decimal("0.2") * Decimal(observed["proof_receipt"])
        + Decimal(observed["reward_claim"])
    )
    target = targets["table_13_gas"]
    checks = {
        name: observed[name] <= int(target[name])
        for name in observed
    }
    return {
        "schema_version": 1,
        "raw_transactions": dict(raw),
        "observed_gas": observed,
        "paper_targets": {
            name: int(target[name])
            for name in observed
        },
        "checks": checks,
        "passed": all(checks.values()),
        "honest_round_formula": "commitment + 0.2 * proof_receipt + reward_claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="node")
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "tests" / "contracts" / "polbfl_protocol_e2e.cjs",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "config" / "paper_targets.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    process = subprocess.run(
        [args.node, str(args.test.resolve())],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "contract gas benchmark failed: "
            + (process.stderr.strip() or process.stdout.strip())
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("contract gas benchmark produced no JSON result")
    raw = json.loads(lines[-1])
    evidence = build_evidence(
        raw,
        json.loads(args.targets.read_text(encoding="utf-8")),
    )
    source = source_identity(ROOT)
    if source["dirty"] or not source["commit"]:
        raise RuntimeError("formal gas benchmark requires a clean source commit")
    evidence.update(
        {
            "source": source,
            "input_sha256": {
                str(args.test.resolve().relative_to(ROOT)): sha256_file(args.test),
                str(args.targets.resolve().relative_to(ROOT)): sha256_file(args.targets),
                "chainEnv/contracts/PoLBFLProtocol.sol": sha256_file(
                    ROOT / "chainEnv" / "contracts" / "PoLBFLProtocol.sol"
                ),
                "scripts/contract_gas_benchmark.py": sha256_file(Path(__file__)),
                "package-lock.json": sha256_file(ROOT / "package-lock.json"),
            },
            "node_version": subprocess.run(
                [args.node, "--version"],
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            ).stdout.strip(),
        }
    )
    evidence = seal_evidence(
        evidence,
        source_commit=source["commit"],
        analysis_source=source,
    )
    write_manifest_atomic(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
