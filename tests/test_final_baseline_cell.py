from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

torch = pytest.importorskip("torch")

from experiments.final.run_security_cell import (
    SecurityCell,
    evaluate_cell_acceptance,
)
from polbfl.committee import QuorumDecision


def _cell(method):
    cell = SecurityCell.__new__(SecurityCell)
    cell.method = method
    cell.args = SimpleNamespace(
        num_malicious=1,
        seed=1337,
        shapley_permutations=3,
    )
    cell.foolsgold_history = None
    cell.shapley_history = None
    return cell


def _artifacts(values):
    return [
        SimpleNamespace(
            client_id=f"client-{index}",
            update={"w": torch.tensor([float(value), 0.0])},
        )
        for index, value in enumerate(values)
    ]


def test_baseline_cell_krum_and_vanilla_paths_emit_bound_evidence():
    values = (0.0, 0.1, -0.1, 0.05, -0.05, 0.02, 10.0)
    krum = _cell("Krum")._baseline_decision(_artifacts(values), round_number=0)
    assert "client-6" in krum.flagged_clients
    assert krum.included_clients != ("client-6",)
    assert len(krum.execution_digest) == 64
    vanilla = _cell("VanillaFL")._baseline_decision(
        _artifacts(values),
        round_number=0,
    )
    assert len(vanilla.included_clients) == len(values)
    assert not vanilla.flagged_clients


def test_baseline_acceptance_uses_the_complete_table2_target():
    result = {
        "study": "main",
        "dataset": "CIFAR10",
        "attack": "FreeRidingNT",
        "method": "Krum",
        "MA": 73.0,
        "DR": 16.0,
        "FPR": 8.0,
    }
    targets = {
        "table_2_all_methods": {
            "CIFAR10": {
                "FreeRidingNT": {
                    "Krum": {"MA": 72.5, "DR": 15.2, "FPR": 8.5}
                }
            }
        }
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]


def test_baseline_scratch_cleanup_is_scoped_to_the_run(tmp_path):
    cell = _cell("VanillaFL")
    cell.run_dir = tmp_path / "run"
    round_root = cell.run_dir / "scratch" / "round-0"
    round_root.mkdir(parents=True)
    (round_root / "payload").write_bytes(b"x")
    cell._remove_baseline_scratch(round_root)
    assert not round_root.exists()
    with pytest.raises(RuntimeError, match="unexpected scratch"):
        cell._remove_baseline_scratch(tmp_path)


def test_pol_scratch_retains_only_audited_hash_bound_stores(tmp_path):
    cell = _cell("PoLBFL")
    cell.run_dir = tmp_path / "run"
    round_root = cell.run_dir / "scratch" / "round-0"
    retained_store = round_root / "client_client-0" / "round-0"
    discarded_store = round_root / "client_client-1" / "round-0"
    retained_store.mkdir(parents=True)
    discarded_store.mkdir(parents=True)
    (retained_store / "pack.bin").write_bytes(b"retained")
    (discarded_store / "pack.bin").write_bytes(b"discarded")
    worker_results = round_root / "worker-results"
    worker_results.mkdir()
    (worker_results / "client.pt").write_bytes(b"temporary")
    (round_root / "global-state.pt").write_bytes(b"temporary")
    artifacts = [
        SimpleNamespace(client_id="client-0", store_root=str(retained_store)),
        SimpleNamespace(client_id="client-1", store_root=str(discarded_store)),
    ]
    hashes, retained_roots = cell._retained_pol_evidence(
        round_root,
        {"client-0"},
        artifacts,
    )
    assert list(hashes) == ["client-0"]
    assert len(next(iter(hashes.values()))) == 1
    cell._prune_pol_scratch(round_root, retained_roots)
    assert retained_store.is_dir()
    assert not discarded_store.exists()
    assert not worker_results.exists()
    assert not (round_root / "global-state.pt").exists()


def test_audit_evidence_records_proof_set_and_receipt_digests():
    cell = _cell("PoLBFL")
    cell._audit_evidence = {}
    cell._audit_evidence_lock = threading.Lock()
    challenge = SimpleNamespace(
        challenge_id="a" * 64,
        commitment_root="b" * 64,
        pair_indices=(1, 2, 3),
    )
    receipts = [
        SimpleNamespace(receipt_digest=f"{index:064x}", verifier_id=f"v-{index}")
        for index in range(3)
    ]
    cell._record_audit_evidence(
        artifact=SimpleNamespace(client_id="client-0"),
        challenge=challenge,
        proof_set_digest_value="c" * 64,
        receipts=receipts,
        decision=QuorumDecision.ACCEPT,
    )
    evidence = cell._audit_evidence["client-0"]
    assert evidence["proof_set_digest"] == "c" * 64
    assert len(evidence["receipt_digests"]) == 3
    assert evidence["diagnostic_bypass"] is False
