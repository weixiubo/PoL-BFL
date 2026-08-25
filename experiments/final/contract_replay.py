#!/usr/bin/env python3
"""Replay measured round decisions through the real Solidity protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from experiments.final.manifest import (
    sha256_file,
    source_identity,
    write_manifest_atomic,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_rounds(path: Path, *, expected_rounds: int) -> list[Mapping[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row["round"]))
    if len(rows) != expected_rounds or [
        int(row["round"]) for row in rows
    ] != list(range(expected_rounds)):
        raise ValueError("contract replay requires every round exactly once")
    required = {
        "active_clients",
        "participating_clients",
        "trace_commitments",
        "audited_clients",
        "audit_evidence",
        "proof_outcomes",
        "statistically_rejected_clients",
        "sybil_flagged_clients",
        "settlement_digest",
        "stake_by_client",
        "reputation_by_client",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValueError("contract replay round evidence is incomplete")
        participants = {str(value) for value in row["participating_clients"]}
        if (
            len(participants) != int(row["active_clients"])
            or set(map(str, row["trace_commitments"])) != participants
            or set(map(str, row["proof_outcomes"])) != participants
            or set(map(str, row["audited_clients"])) - participants
        ):
            raise ValueError("contract replay round identities are inconsistent")
    return rows


def replay_contract_rounds(
    *,
    rounds_path: Path,
    seed: int,
    num_clients: int,
    expected_rounds: int,
    output: Path,
    node: str = "node",
    formal: bool = False,
    timeout_seconds: int = 7_200,
    root: Path = ROOT,
) -> dict[str, Any]:
    rounds_path = rounds_path.resolve()
    output = output.resolve()
    root = root.resolve()
    if not rounds_path.is_file():
        raise FileNotFoundError(rounds_path)
    if num_clients <= 0 or expected_rounds <= 0 or timeout_seconds <= 0:
        raise ValueError("contract replay dimensions and timeout must be positive")
    _read_rounds(rounds_path, expected_rounds=expected_rounds)
    source = source_identity(root)
    if formal:
        if expected_rounds != 200:
            raise ValueError("formal contract replay requires 200 rounds")
        if source["dirty"] or not source["commit"]:
            raise RuntimeError(
                "formal contract replay requires a clean, identified source"
            )
    script = root / "scripts" / "contract_round_replay.cjs"
    process = subprocess.run(
        [
            node,
            str(script),
            "--rounds-jsonl",
            str(rounds_path),
            "--seed",
            str(int(seed)),
            "--num-clients",
            str(int(num_clients)),
            "--expected-rounds",
            str(int(expected_rounds)),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Solidity round replay failed: "
            + (process.stderr.strip() or process.stdout.strip())
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Solidity round replay emitted no evidence")
    raw = json.loads(lines[-1])
    required_output = {
        "passed",
        "real_contract_transitions",
        "contract_rounds",
        "num_clients",
        "transaction_count",
        "total_gas",
        "transition_digest",
        "rounds",
    }
    if (
        not required_output.issubset(raw)
        or raw["passed"] is not True
        or raw["real_contract_transitions"] is not True
        or int(raw["contract_rounds"]) != expected_rounds
        or int(raw["num_clients"]) != num_clients
        or len(raw["rounds"]) != expected_rounds
        or len(str(raw["transition_digest"])) != 64
    ):
        raise RuntimeError("Solidity round replay evidence is incomplete")
    evidence: dict[str, Any] = {
        **raw,
        "source": source,
        "formal": bool(formal),
        "formal_accepted": bool(formal),
        "input_sha256": {
            str(rounds_path): sha256_file(rounds_path),
            "scripts/contract_round_replay.cjs": sha256_file(script),
            "experiments/final/contract_replay.py": sha256_file(Path(__file__)),
            "chainEnv/contracts/PoLBFLProtocol.sol": sha256_file(
                root / "chainEnv" / "contracts" / "PoLBFLProtocol.sol"
            ),
            "chainEnv/contracts/MockAuthenticatedRandomness.sol": sha256_file(
                root
                / "chainEnv"
                / "contracts"
                / "MockAuthenticatedRandomness.sol"
            ),
            "package-lock.json": sha256_file(root / "package-lock.json"),
        },
    }
    body = dict(evidence)
    evidence["evidence_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_atomic(output, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--expected-rounds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    args = parser.parse_args()
    evidence = replay_contract_rounds(
        rounds_path=args.rounds_jsonl,
        seed=args.seed,
        num_clients=args.num_clients,
        expected_rounds=args.expected_rounds,
        output=args.output,
        node=args.node,
        formal=args.formal,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
