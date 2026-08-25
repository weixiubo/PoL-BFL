import json
from pathlib import Path

from experiments.final.aggregate_table12 import aggregate_table12
from experiments.final.compose_table12_pol import compose_pol_table12
from scripts.extract_table12_targets import parse_table12
from scripts.zk_kaizen_controlled_benchmark import evaluate_kaizen_metrics, validate_controlled_setup


ROOT = Path(__file__).parents[1]


def test_table12_extractor_converts_units_for_both_methods():
    text = "\n".join(
        [
            "Table 12: ZK Proof Technical Specifications on CIFAR-10.",
            " Proof Gen Time 4.2 s 12.5 s 3.0x",
            " Circuit Size 1.1M 4.5M 4.1x",
            " Witness Computation 1.8 s 5.2 s 2.9x",
            " Prover Memory 2.5 GB 8.2 GB 3.3x",
            " Proof Size 192 B 192 B 1x",
            " Verification Time 8.5 ms 8.5 ms 1x",
            " Merkle Proof Size 1.2 KB - -",
            " Total Verification 52 ms 8.5 ms 0.16x",
            "\f",
        ]
    )
    table = parse_table12(text)["table_12_all_methods"]
    assert table["PoLBFL"]["circuit_constraints"] == 1_100_000
    assert table["Kaizen"]["circuit_constraints"] == 4_500_000
    assert table["Kaizen"]["merkle_proof_kb"] is None


def test_table12_aggregate_requires_real_same_source_benchmarks():
    targets = json.loads(
        (ROOT / "config" / "paper_table12_all_methods.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for method, metrics in targets["table_12_all_methods"].items():
        row = {
            "formal_accepted": True,
            "real_benchmark": True,
            "proof_system": "Groth16",
            "method": method,
            "metrics": metrics,
            "source_commit": "a" * 40,
            "evidence_digest": ("b" if method == "PoLBFL" else "c") * 64,
        }
        if method == "PoLBFL":
            row["trust_setup_record_digest"] = "d" * 64
        else:
            row["controlled_baseline_digest"] = "e" * 64
        rows.append(row)
    result = aggregate_table12(rows, targets)
    assert result["acceptance"]["passed"]
    assert set(result["table_12_all_methods"]) == {"PoLBFL", "Kaizen"}


def test_pol_table12_composition_requires_matching_source_and_trust():
    targets = json.loads(
        (ROOT / "config" / "paper_table12_all_methods.json").read_text(
            encoding="utf-8"
        )
    )
    core = {
        "formal_accepted": True,
        "method": "PoLBFL",
        "source_commit": "a" * 40,
        "trust_setup_record_digest": "b" * 64,
        "evidence_digest": "c" * 64,
        "metrics": {
            "proof_seconds_median": 1.0,
            "circuit_constraints": 1_090_382,
            "witness_seconds_median": 0.6,
            "prover_memory_gb_max": 1.8,
            "proof_bytes": 192,
        },
    }
    bundle = {
        "formal_accepted": True,
        "method": "PoLBFL",
        "source_commit": "a" * 40,
        "trust_setup_record_digest": "b" * 64,
        "evidence_digest": "d" * 64,
        "metrics": {
            "verification_ms": 5.0,
            "merkle_proof_kb": 0.1,
            "total_verification_ms": 7.0,
        },
    }
    result = compose_pol_table12(core, bundle, targets)
    assert result["formal_accepted"]
    assert result["metrics"]["circuit_constraints"] == 1_090_382


def test_controlled_kaizen_setup_and_metrics_are_hash_bound(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"controlled")
    import hashlib

    record = {
        "schema_version": 1,
        "classification": "controlled_kaizen_style_cost_baseline",
        "constraints": 4_000_000,
        "artifacts": {"artifact.bin": hashlib.sha256(b"controlled").hexdigest()},
    }
    body = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["record_digest"] = hashlib.sha256(body).hexdigest()
    (tmp_path / "controlled_setup.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    assert validate_controlled_setup(tmp_path)["constraints"] == 4_000_000
    targets = json.loads(
        (ROOT / "config" / "paper_table12_all_methods.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = dict(targets["table_12_all_methods"]["Kaizen"])
    assert evaluate_kaizen_metrics(metrics, targets)["passed"]


def test_controlled_setup_supports_hashable_resume_and_native_witness():
    script = (ROOT / "scripts" / "kaizen_controlled_setup.sh").read_text(
        encoding="utf-8"
    )
    assert "--max-old-space-size=32768" in script
    assert "REUSE_COMPILED" in script
    assert "REUSE_ZKEY0" in script
    assert "kaizen_controlled_cost_cpp/kaizen_controlled_cost" in script
    assert "generate_witness.js" not in script


def test_ceremony_entropy_never_appears_in_process_arguments():
    scripts = [
        ROOT / "scripts" / "kaizen_controlled_setup.sh",
        ROOT / "scripts" / "zk_reference_continue.sh",
        ROOT / "scripts" / "zk_reference_setup.sh",
        ROOT / "scripts" / "zk_smoke.sh",
    ]
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "-e=" not in text
        assert "openssl rand -hex 64 |" in text
