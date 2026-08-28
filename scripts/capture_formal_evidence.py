#!/usr/bin/env python3
"""Verify completed cell manifests and capture compact formal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.manifest import sha256_file, write_manifest_atomic
from experiments.final.run_security_cell import table5_metrics
from polbfl.crypto import domain_hash
from polbfl.protocol import TraceCommitment, select_audit_clients


def _canonical_field_digest(payload: Mapping[str, Any], field: str) -> bool:
    declared = str(payload.get(field, ""))
    if len(declared) != 64:
        return False
    body = dict(payload)
    body.pop(field, None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() == declared


def verify_standalone_trial_evidence(
    result_path: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> int:
    study = str(result.get("study", ""))
    evidence_name = {
        "adaptive": "adaptive-evidence.json",
        "cross_hardware": "hardware-evidence.json",
    }.get(study)
    if evidence_name is None:
        raise ValueError("unsupported standalone formal trial")
    evidence_path = result_path.with_name(evidence_name)
    declared_artifact_names = {
        Path(str(label)).name
        for label in manifest.get("artifact_sha256", {})
    }
    if {"result.json", evidence_name} - declared_artifact_names:
        raise ValueError("standalone trial artifacts are not manifest-bound")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        not _canonical_field_digest(result, "result_digest")
        or not _canonical_field_digest(evidence, "evidence_digest")
        or result.get("evidence_digest") != evidence.get("evidence_digest")
        or result.get("source_commit") != manifest["source"]["commit"]
        or evidence.get("source_commit") != manifest["source"]["commit"]
        or result.get("seed") != evidence.get("seed")
        or evidence.get("acceptance", {}).get("passed") is not True
        or int(evidence.get("proof_bytes", 0)) != 192
    ):
        raise ValueError("standalone formal trial digest or provenance is invalid")
    honest = evidence.get("honest_report", {})
    malicious = evidence.get("malicious_report", {})
    if (
        honest.get("valid") is not True
        or honest.get("proof_valid") is not True
        or malicious.get("valid") is not False
        or malicious.get("proof_valid") is not True
    ):
        raise ValueError("standalone formal trial proof controls are invalid")
    if study == "adaptive":
        if (
            result.get("variant") != evidence.get("variant")
            or result.get("trials") != evidence.get("trials")
            or len(result.get("trials", ())) != 2
        ):
            raise ValueError("adaptive formal evidence differs from its result")
    else:
        if (
            result.get("hardware_pair") != evidence.get("hardware_pair")
            or result.get("observations") != evidence.get("observations")
            or len(result.get("observations", ())) != 2
            or not evidence.get("trainer_attestation")
            or not evidence.get("verifier_attestation")
            or evidence.get("numerical_decision", {}).get("passed") is not True
        ):
            raise ValueError("cross-hardware formal evidence differs from its result")
    return len(manifest.get("artifact_sha256", {}))


def verify_completed_cell(result_path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    result_path = result_path.resolve()
    manifest_path = result_path.with_name("manifest.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("formal_accepted") is not True:
        raise ValueError(f"formal result did not pass acceptance: {result_path}")
    if manifest.get("state") != "completed" or manifest.get("source", {}).get("dirty"):
        raise ValueError(f"formal manifest is not clean and completed: {manifest_path}")
    method = str(result.get("method", "PoLBFL"))
    trust_digest = manifest.get("run_parameters", {}).get(
        "trust_setup_record_digest"
    )
    has_trust_record = any(
        Path(str(label)).name == "trust_setup.json"
        for label in manifest.get("runtime_artifact_sha256", {})
    )
    has_baseline_lock = any(
        Path(str(label)).name == "baseline_sources.lock.json"
        for label in manifest.get("configuration_sha256", {})
    )
    if method == "PoLBFL":
        if not isinstance(trust_digest, str) or len(trust_digest) != 64:
            raise ValueError(
                f"formal manifest lacks production trust provenance: {manifest_path}"
            )
        if not has_trust_record:
            raise ValueError(
                f"formal manifest does not bind trust_setup.json: {manifest_path}"
            )
    elif not has_baseline_lock:
        raise ValueError(
            f"formal baseline manifest lacks source locks: {manifest_path}"
        )
    body = dict(manifest)
    declared_manifest_digest = body.pop("manifest_digest")
    calculated_manifest_digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if calculated_manifest_digest != declared_manifest_digest:
        raise ValueError(f"manifest digest mismatch: {manifest_path}")
    verified_artifacts = {}
    for label, expected in manifest["artifact_sha256"].items():
        path = Path(label)
        if not path.is_absolute():
            path = root / path
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"formal artifact hash mismatch: {path}")
        verified_artifacts[label] = observed
    retained_evidence_count = 0
    if result.get("study") in {"adaptive", "cross_hardware"}:
        retained_evidence_count = verify_standalone_trial_evidence(
            result_path, result, manifest
        )
    elif method == "PoLBFL":
        rounds_path = result_path.with_name("rounds.jsonl")
        round_rows = [
            json.loads(line)
            for line in rounds_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(round_rows) != int(result["rounds"]) or [
            int(row["round"]) for row in round_rows
        ] != list(range(int(result["rounds"]))):
            raise ValueError("formal round log is incomplete or non-contiguous")
        parameters = manifest.get("run_parameters", {})
        for row in round_rows:
            predictions = [int(value) for value in row.get("test_predictions", ())]
            labels = [int(value) for value in row.get("test_labels", ())]
            if not predictions or len(predictions) != len(labels):
                raise ValueError("formal round prediction trace is incomplete")
            observed_accuracy = 100.0 * sum(
                prediction == label
                for prediction, label in zip(predictions, labels)
            ) / len(labels)
            if abs(observed_accuracy - float(row["accuracy"])) > 1e-9:
                raise ValueError("formal round accuracy differs from raw predictions")
            expected_prediction_digest = domain_hash(
                "POLBFL_TEST_PREDICTIONS_V1",
                int(row["round"]),
                json.dumps(tuple(predictions)),
                json.dumps(tuple(labels)),
            )
            if row.get("prediction_digest") != expected_prediction_digest:
                raise ValueError("formal prediction digest mismatch")
            audited = set(str(value) for value in row.get("audited_clients", ()))
            participants = tuple(str(value) for value in row.get("participating_clients", ()))
            commitment_payloads = row.get("trace_commitments", {})
            proof_outcomes = row.get("proof_outcomes", {})
            if (
                len(participants) != int(row["active_clients"])
                or len(set(participants)) != len(participants)
                or set(commitment_payloads) != set(participants)
                or set(proof_outcomes) != set(participants)
            ):
                raise ValueError("formal round decision identities are incomplete")
            commitments = [
                TraceCommitment(**dict(commitment_payloads[client_id]))
                for client_id in participants
            ]
            audit_seed = hashlib.sha256(
                f"audit:{int(manifest['seed'])}:{int(row['round'])}".encode()
            ).digest()
            selection = select_audit_clients(
                commitments,
                vrf_output=audit_seed,
                probability=Decimal(str(parameters["audit_probability"])),
            )
            if audited != set(selection.selected_clients):
                raise ValueError("formal Python audit selection does not reproduce")
            declared_selection = row.get("audit_selection", {})
            if (
                int(declared_selection.get("population_size", -1))
                != selection.population_size
                or str(declared_selection.get("probability"))
                != str(selection.probability)
                or declared_selection.get("randomness_digest")
                != selection.randomness_digest
                or declared_selection.get("transcript_digest")
                != selection.transcript_digest
            ):
                raise ValueError("formal audit-selection transcript mismatch")
            audit_evidence = row.get("audit_evidence", {})
            if audited != set(audit_evidence):
                raise ValueError("formal round audit evidence is incomplete")
            for client_id, evidence in audit_evidence.items():
                if evidence.get("diagnostic_bypass") is not False:
                    raise ValueError("formal audit used a diagnostic proof bypass")
                if len(str(evidence.get("proof_set_digest", ""))) != 64:
                    raise ValueError("formal audit proof-set digest is invalid")
                receipts = evidence.get("receipt_digests", ())
                if len(receipts) != 3 or any(len(str(value)) != 64 for value in receipts):
                    raise ValueError("formal audit lacks the signed three-receipt quorum")
                if any(int(value) != 192 for value in evidence.get("proof_bytes", ())):
                    raise ValueError("formal Groth16 proof transport is not 192 bytes")
            for _client_id, files in row.get("retained_evidence_sha256", {}).items():
                for label, expected in files.items():
                    path = (result_path.parent / str(label)).resolve()
                    if not path.is_relative_to(result_path.parent.resolve()):
                        raise ValueError("retained evidence path escaped its run directory")
                    if not path.is_file() or sha256_file(path) != expected:
                        raise ValueError(f"retained evidence hash mismatch: {path}")
                    retained_evidence_count += 1
        requires_contract = bool(parameters.get("economic_enforcement", True))
        if requires_contract:
            contract_path = result_path.with_name("contract-evidence.json")
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract_body = dict(contract)
            declared_contract_digest = contract_body.pop("evidence_digest", None)
            calculated_contract_digest = hashlib.sha256(
                json.dumps(
                    contract_body, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            contract_rounds = contract.get("rounds", ())
            if (
                contract.get("formal_accepted") is not True
                or contract.get("real_contract_transitions") is not True
                or int(contract.get("contract_rounds", 0)) != int(result["rounds"])
                or len(contract_rounds) != len(round_rows)
                or declared_contract_digest != calculated_contract_digest
                or result.get("contract_evidence_digest") != declared_contract_digest
                or result.get("real_contract_rounds") is not True
                or int(result.get("contract_rounds", 0)) != int(result["rounds"])
                or contract.get("source", {}).get("commit")
                != manifest["source"]["commit"]
            ):
                raise ValueError("formal contract replay evidence is invalid")
            input_round_hashes = [
                value
                for label, value in contract.get("input_sha256", {}).items()
                if Path(str(label)).name == "rounds.jsonl"
            ]
            if input_round_hashes != [sha256_file(result_path.with_name("rounds.jsonl"))]:
                raise ValueError("contract replay is not bound to the formal rounds")
            observed_transactions = 0
            for row, chain_round in zip(round_rows, contract_rounds, strict=True):
                transactions = chain_round.get("transactions", ())
                if (
                    int(chain_round.get("round", -1)) != int(row["round"])
                    or set(map(str, chain_round.get("audited_clients", ())))
                    != set(map(str, row["audited_clients"]))
                    or chain_round.get("python_settlement_digest")
                    != row["settlement_digest"]
                    or int(chain_round.get("transaction_count", -1))
                    != len(transactions)
                    or float(chain_round.get("runtime_seconds", 0.0)) <= 0.0
                ):
                    raise ValueError("formal contract round differs from training evidence")
                for transaction in transactions:
                    if (
                        len(str(transaction.get("transaction_hash", ""))) != 66
                        or int(transaction.get("gas_used", 0)) <= 0
                    ):
                        raise ValueError("formal contract transaction evidence is invalid")
                observed_transactions += len(transactions)
            if observed_transactions != int(contract.get("transaction_count", -1)):
                raise ValueError("formal contract transaction count mismatch")
        elif (
            result.get("real_contract_rounds") is True
            or int(result.get("contract_rounds", 0)) != 0
            or result_path.with_name("contract-evidence.json").exists()
        ):
            raise ValueError("non-economic ablation contains contract evidence")
    if result.get("study") == "incentive":
        table5_rows = [
            json.loads(line)
            for line in result_path.with_name("rounds.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        observed_table5 = table5_metrics(
            table5_rows, required_rounds=int(result["rounds"])
        )
        for metric, observed in observed_table5.items():
            if abs(float(result[metric]) - float(observed)) > 1e-9:
                raise ValueError("formal Table 5 aggregate differs from raw rounds")
        malicious = set(map(str, result.get("malicious_clients", ())))
        for row in table5_rows:
            predictions = tuple(
                int(value)
                for value in row.get("counterfactual_honest_predictions", ())
            )
            labels = tuple(
                int(value)
                for value in row.get("counterfactual_honest_labels", ())
            )
            if not predictions or len(predictions) != len(labels):
                raise ValueError("formal Table 5 counterfactual is incomplete")
            accuracy = 100.0 * sum(
                prediction == label
                for prediction, label in zip(predictions, labels)
            ) / len(labels)
            if abs(
                accuracy - float(row["counterfactual_honest_accuracy"])
            ) > 1e-9:
                raise ValueError("formal Table 5 counterfactual accuracy mismatch")
            expected_digest = domain_hash(
                "POLBFL_TABLE5_COUNTERFACTUAL_V1",
                int(row["round"]),
                json.dumps(predictions),
                json.dumps(labels),
            )
            if row.get("counterfactual_honest_digest") != expected_digest:
                raise ValueError("formal Table 5 counterfactual digest mismatch")
            expected_effect = (
                float(row["accuracy"]) + 1e-9
                < float(row["counterfactual_honest_accuracy"])
            )
            if row.get("attack_success") is not expected_effect:
                raise ValueError("formal Table 5 aggregate-effect decision mismatch")
            successful = set(
                map(str, row.get("malicious_successful_clients", ()))
            )
            marginal = {
                str(client_id): float(value)
                for client_id, value in row.get(
                    "marginal_accuracy_by_client", {}
                ).items()
            }
            included = set(
                map(str, row.get("aggregation_included_clients", ()))
            )
            if (
                successful - malicious
                or successful - included
                or any(marginal.get(client_id, 0.0) <= 0 for client_id in successful)
                or len(successful)
                != int(row.get("malicious_attack_successes", -1))
            ):
                raise ValueError("formal Table 5 malicious-success evidence mismatch")
    return {
        "result": result,
        "result_path": str(result_path),
        "manifest_path": str(manifest_path),
        "manifest_digest": declared_manifest_digest,
        "method": method,
        "source_commit": manifest["source"]["commit"],
        "trust_setup_record_digest": trust_digest,
        "retained_evidence_files": retained_evidence_count,
        "contract_evidence_digest": (
            result.get("contract_evidence_digest") if method == "PoLBFL" else None
        ),
        "verified_artifact_sha256": verified_artifacts,
        "configuration_sha256": manifest["configuration_sha256"],
        "runtime_artifact_sha256": manifest["runtime_artifact_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        "schema_version": 1,
        "cells": [verify_completed_cell(path) for path in args.results],
    }
    body = dict(evidence)
    evidence["evidence_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_manifest_atomic(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
