import json
from pathlib import Path

from experiments.final.manifest import create_run_manifest, sha256_file, write_manifest_atomic


ROOT = Path(__file__).parents[1]


def test_paper_scope_and_trace_availability_requirements_are_explicit():
    protocol = json.loads(
        (ROOT / "config" / "paper_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["system_scope"] == {
        "federation_setting": "cross-silo",
        "persistent_organizational_identities": True,
        "adversary_knowledge": "full",
        "corrupted_client_bound": "f < N/2",
        "deterministic_training_required": True,
        "meaningful_stake_required": True,
        "gas_oracle_authentication_required": True,
        "gas_oracle_deployment_scope": ["permissioned", "semi-permissioned"],
    }
    assert protocol["storage"] == {
        "trace_location": "content-addressed off-chain storage",
        "on_chain_anchor": "32-byte Merkle root",
        "availability_challenge": "post-commit authenticated-random audit",
        "availability_failure": "aggregation exclusion and full stake slashing",
    }


    alignment = protocol["implementation_choices"]["sybil_trajectory_alignment"]
    assert "normalized-training-progress grid" in alignment
    assert "non-comparable for cosine" in alignment
def test_experiment_manifest_binds_source_config_dataset_seed_and_artifacts(tmp_path):
    artifact = tmp_path / "raw.json"
    artifact.write_text('{"accuracy": 90.0}\n', encoding="utf-8")
    config = ROOT / "config" / "paper_protocol.json"
    targets = ROOT / "config" / "paper_targets.json"
    manifest = create_run_manifest(
        root=ROOT,
        run_id="manifest-unit",
        seed=1337,
        configuration_files=(config, targets),
        dataset={"name": "unit", "sha256": "a" * 64, "partition_sha256": "b" * 64},
        runtime_artifacts=(artifact,),
        run_parameters={"rounds": 200, "transport": "signed_4bit_packed_zlib1"},
        artifacts=(artifact,),
        state="completed",
    )
    assert manifest["source"]["commit"]
    assert manifest["seed"] == 1337
    assert manifest["configuration_sha256"]["config/paper_targets.json"] == sha256_file(targets)
    assert manifest["runtime_artifact_sha256"][str(artifact)] == sha256_file(artifact)
    assert manifest["run_parameters"]["rounds"] == 200
    assert len(manifest["manifest_digest"]) == 64
    output = tmp_path / "manifest.json"
    write_manifest_atomic(output, manifest)
    assert json.loads(output.read_text(encoding="utf-8"))["manifest_digest"] == manifest["manifest_digest"]
