#!/usr/bin/env python3
"""Compose all 12 Table 7 cells from measured same-source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.final.aggregate_table7 import aggregate_table7
from experiments.final.compose_table7_result import compose_table7_result
from experiments.final.evidence import seal_evidence
from experiments.final.manifest import source_identity


METHODS = ("Vanilla", "VeriblockFL", "Kaizen", "PoLBFL")


def _load(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return json.loads(resolved.read_text(encoding="utf-8"))


def compose_table7_matrix(
    evidence_map: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    source_lock_digest: str,
) -> tuple[dict[str, Any], ...]:
    expected_seeds = (1337, 2026, 3817739)
    declared = evidence_map.get("seeds", {})
    if set(map(str, expected_seeds)) != set(map(str, declared)):
        raise ValueError("Table 7 evidence map lacks a paper seed")
    results = []
    for seed in expected_seeds:
        inputs = declared[str(seed)]
        required = {
            "vanilla_training",
            "pol_training",
            "contract_gas",
            "kaizen_proof",
            "veriblock_benchmark",
        }
        if not required.issubset(inputs):
            raise ValueError(
                "Table 7 evidence map is incomplete for seed "
                + str(seed)
            )
        vanilla = _load(inputs["vanilla_training"])
        pol = _load(inputs["pol_training"])
        gas = _load(inputs["contract_gas"])
        kaizen = _load(inputs["kaizen_proof"])
        veriblock = _load(inputs["veriblock_benchmark"])
        for method in METHODS:
            training = pol if method == "PoLBFL" else vanilla
            results.append(
                compose_table7_result(
                    method=method,
                    seed=seed,
                    training=training,
                    targets=targets,
                    source_lock_digest=source_lock_digest,
                    gas_evidence=(
                        gas if method in {"PoLBFL", "Kaizen"} else None
                    ),
                    proof_evidence=(
                        kaizen if method == "Kaizen" else None
                    ),
                    veriblock_evidence=(
                        veriblock
                        if method == "VeriblockFL"
                        else None
                    ),
                )
            )
    return tuple(results)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-map", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "config" / "paper_table7_all_methods.json",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=root / "config" / "baseline_sources.lock.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiments" / "results" / "final" / "table7",
    )
    args = parser.parse_args()
    source = source_identity(root)
    if source["dirty"] or not source["commit"]:
        raise RuntimeError(
            "formal Table 7 composition requires a clean source"
        )
    evidence_map = json.loads(
        args.evidence_map.read_text(encoding="utf-8")
    )
    if evidence_map.get("source_commit") != source["commit"]:
        raise ValueError(
            "Table 7 evidence map source differs from the deployed source"
        )
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    source_lock_digest = hashlib.sha256(
        args.source_lock.read_bytes()
    ).hexdigest()
    results = compose_table7_matrix(
        evidence_map,
        targets,
        source_lock_digest=source_lock_digest,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for result in results:
        path = (
            args.output_root
            / (
                "table7-"
                + str(result["method"]).lower()
                + "-s"
                + str(result["seed"])
                + ".json"
            )
        )
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("evidence_digest") != result["evidence_digest"]:
                raise RuntimeError(
                    "refusing to overwrite different Table 7 evidence: "
                    + str(path)
                )
        else:
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        paths.append(path)
    aggregate = aggregate_table7(results, targets)
    aggregate["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    aggregate["formal_result_paths"] = sorted(
        {
            str(Path(inputs[key]).resolve())
            for inputs in evidence_map["seeds"].values()
            for key in ("vanilla_training", "pol_training")
        }
    )
    aggregate = seal_evidence(aggregate, analysis_root=root)
    aggregate_path = args.output_root / "table7-aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if not aggregate["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
