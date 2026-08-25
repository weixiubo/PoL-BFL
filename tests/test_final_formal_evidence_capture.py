import hashlib
import json
from decimal import Decimal

import pytest

from experiments.final.manifest import (
    create_run_manifest,
    sha256_file,
    write_manifest_atomic,
)
from polbfl.crypto import domain_hash
from polbfl.protocol import TraceCommitment, select_audit_clients
from scripts.capture_formal_evidence import verify_completed_cell


def test_formal_evidence_capture_revalidates_manifest_and_artifacts(tmp_path):
    result_path = tmp_path / "cell" / "result.json"
    result_path.parent.mkdir()
    rounds_path = result_path.with_name("rounds.jsonl")
    commitments = []
    for index in range(5):
        digest = lambda label, i=index: hashlib.sha256(
            f"{label}-{i}".encode()
        ).hexdigest()
        commitments.append(
            TraceCommitment(
                protocol_version="1",
                round_id="round-0",
                client_id=f"client-{index}",
                context_digest=digest("context"),
                merkle_root=digest("root"),
                checkpoint_count=2,
                first_step=0,
                final_step=5,
                final_model_digest=digest("model"),
                trace_digest=digest("trace"),
            )
        )
    for seed in range(1, 100):
        audit_seed = hashlib.sha256(f"audit:{seed}:0".encode()).digest()
        selection = select_audit_clients(
            commitments,
            vrf_output=audit_seed,
            probability=Decimal("0.2"),
        )
        if selection.selected_clients:
            break
    audited = set(selection.selected_clients)
    settlement_digest = hashlib.sha256(b"settlement").hexdigest()
    round_row = {
        "round": 0,
        "accuracy": 50.0,
        "test_predictions": [1, 0],
        "test_labels": [1, 1],
        "prediction_digest": domain_hash(
            "POLBFL_TEST_PREDICTIONS_V1",
            0,
            json.dumps((1, 0)),
            json.dumps((1, 1)),
        ),
        "active_clients": 5,
        "participating_clients": [item.client_id for item in commitments],
        "trace_commitments": {
            item.client_id: item.to_dict() for item in commitments
        },
        "audited_clients": sorted(audited),
        "audit_selection": {
            "probability": str(selection.probability),
            "population_size": selection.population_size,
            "randomness_digest": selection.randomness_digest,
            "transcript_digest": selection.transcript_digest,
        },
        "audit_evidence": {
            client_id: {
                "diagnostic_bypass": False,
                "proof_set_digest": hashlib.sha256(client_id.encode()).hexdigest(),
                "proof_bytes": [],
                "receipt_digests": ["1" * 64, "2" * 64, "3" * 64],
            }
            for client_id in audited
        },
        "proof_outcomes": {
            item.client_id: (
                "accept" if item.client_id in audited else "not_audited"
            )
            for item in commitments
        },
        "retained_evidence_sha256": {},
        "settlement_digest": settlement_digest,
    }
    rounds_path.write_text(json.dumps(round_row) + "\n", encoding="utf-8")
    contract_path = result_path.with_name("contract-evidence.json")
    contract = {
        "formal_accepted": True,
        "real_contract_transitions": True,
        "contract_rounds": 1,
        "transaction_count": 1,
        "source": {"commit": "a" * 40, "dirty": False},
        "input_sha256": {str(rounds_path): sha256_file(rounds_path)},
        "rounds": [
            {
                "round": 0,
                "audited_clients": sorted(audited),
                "python_settlement_digest": settlement_digest,
                "transaction_count": 1,
                "runtime_seconds": 0.1,
                "transactions": [
                    {
                        "transaction_hash": "0x" + "1" * 64,
                        "gas_used": "21000",
                    }
                ],
            }
        ],
    }
    body = dict(contract)
    contract["evidence_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_manifest_atomic(contract_path, contract)
    result_path.write_text(
        json.dumps(
            {
                "formal_accepted": True,
                "method": "PoLBFL",
                "dataset": "CIFAR10",
                "attack": "FreeRidingNT",
                "seed": seed,
                "rounds": 1,
                "MA": 90.0,
                "DR": 100.0,
                "FPR": 0.0,
                "real_contract_rounds": True,
                "contract_rounds": 1,
                "contract_evidence_digest": contract["evidence_digest"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    runtime = tmp_path / "runtime.bin"
    runtime.write_bytes(b"runtime")
    trust_setup = tmp_path / "trust_setup.json"
    trust_setup.write_text("{}\n", encoding="utf-8")
    manifest = create_run_manifest(
        root=tmp_path,
        run_id="cell",
        seed=seed,
        configuration_files=(config,),
        dataset={"name": "CIFAR10"},
        artifacts=(result_path, rounds_path, contract_path),
        runtime_artifacts=(runtime, trust_setup),
        run_parameters={
            "trust_setup_record_digest": "b" * 64,
            "num_clients": 5,
            "audit_probability": "0.2",
        },
        state="completed",
    )
    manifest["source"]["commit"] = "a" * 40
    manifest["source"]["dirty"] = False
    body = dict(manifest)
    body.pop("manifest_digest")
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_manifest_atomic(result_path.with_name("manifest.json"), manifest)
    evidence = verify_completed_cell(result_path, root=tmp_path)
    assert evidence["result"]["formal_accepted"]
    assert evidence["source_commit"]
    assert evidence["contract_evidence_digest"] == contract["evidence_digest"]

    result_path.write_text(
        json.dumps({"formal_accepted": False}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        verify_completed_cell(result_path, root=tmp_path)

def test_formal_baseline_evidence_requires_the_locked_public_sources(tmp_path):
    result_path = tmp_path / "baseline" / "result.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "formal_accepted": True,
                "method": "Krum",
                "study": "main",
                "dataset": "CIFAR10",
                "attack": "ALIE",
                "seed": 1337,
                "rounds": 200,
                "MA": 70.0,
                "DR": 50.0,
                "FPR": 5.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_lock = tmp_path / "baseline_sources.lock.json"
    source_lock.write_text("{}\n", encoding="utf-8")
    manifest = create_run_manifest(
        root=tmp_path,
        run_id="baseline",
        seed=1337,
        configuration_files=(source_lock,),
        dataset={"name": "CIFAR10"},
        artifacts=(result_path,),
        runtime_artifacts=(),
        run_parameters={"trust_setup_record_digest": None},
        state="completed",
    )
    manifest["source"]["commit"] = "a" * 40
    manifest["source"]["dirty"] = False
    body = dict(manifest)
    body.pop("manifest_digest")
    import hashlib

    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_manifest_atomic(result_path.with_name("manifest.json"), manifest)
    evidence = verify_completed_cell(result_path, root=tmp_path)
    assert evidence["method"] == "Krum"
    assert evidence["trust_setup_record_digest"] is None
