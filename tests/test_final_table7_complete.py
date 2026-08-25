import json
from pathlib import Path

from experiments.final.aggregate_table7 import aggregate_table7
from experiments.final.run_table7_matrix import compose_table7_matrix
from experiments.final.compose_table7_result import (
    compose_table7_result,
    gas_usd,
)
from scripts.extract_table7_targets import parse_table7
from scripts.veriblock_controlled_benchmark import generate_input


ROOT = Path(__file__).parents[1]


def test_table7_extractor_reads_the_right_hand_comparison_table():
    text = "\n".join(
        [
            "Table 7: System Overhead Comparison on CIFAR-10.",
            "Participation 62.5 75.5 Time/round (s) 52.3 1850.2 265.8 78.5",
            "Attack 35.2 20.2 Comm (MB/round) 98.5 520.5 312.5 178.2",
            "Accuracy 67.2 76.8 Gas (USD/round) 0 5.20 2.85 0.85",
            "Profit 0.15 Storage (MB/client) 0 45.8 12.8 2.5",
            "Table 8: Scalability with Increasing Clients on CIFAR-10.",
        ]
    )
    table = parse_table7(text)["table_7_all_methods"]
    assert table["Vanilla"]["runtime_seconds"] == 52.3
    assert table["PoLBFL"]["storage_mb_per_client"] == 2.5


def test_table7_aggregate_requires_real_locked_three_seed_measurements():
    targets = json.loads(
        (ROOT / "config" / "paper_table7_all_methods.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for method, metrics in targets["table_7_all_methods"].items():
        for seed in (1337, 2026, 3817739):
            row = {
                "formal_accepted": True,
                "real_measurement": True,
                "training_rounds": 200,
                "method": method,
                "seed": seed,
                **metrics,
                "source_commit": "a" * 40,
                "evidence_digest": f"{seed + len(method):064x}",
            }
            if method == "PoLBFL":
                row.update(
                    {
                        "trust_setup_record_digest": "c" * 64,
                        "real_contract_gas": True,
                    }
                )
            else:
                row["baseline_source_lock_digest"] = "b" * 64
            rows.append(row)
    result = aggregate_table7(rows, targets)
    assert result["acceptance"]["passed"]
    assert set(result["table_7_all_methods"]) == {
        "Vanilla",
        "VeriblockFL",
        "Kaizen",
        "PoLBFL",
    }


def test_table7_composition_uses_measured_training_proof_and_gas():
    targets = json.loads(
        (ROOT / "config" / "paper_table7_all_methods.json").read_text(
            encoding="utf-8"
        )
    )
    training = {
        "formal_accepted": True,
        "rounds": 200,
        "dataset": "CIFAR10",
        "source_commit": "a" * 40,
        "protocol_runtime_seconds": 40.0,
        "runtime_seconds": 40.0,
        "communication_mb": 90.0,
        "storage_mb_per_client": 1.0,
        "num_clients": 50,
        "result_digest": "b" * 64,
    }
    gas = {
        "passed": True,
        "source": {"commit": "a" * 40},
        "observed_gas": {"honest_round_total": 151_757},
        "evidence_digest": "c" * 64,
    }
    proof = {
        "formal_accepted": True,
        "real_benchmark": True,
        "method": "Kaizen",
        "source_commit": "a" * 40,
        "evidence_digest": "d" * 64,
        "metrics": {
            "witness_seconds": 0.5,
            "proof_generation_seconds": 2.0,
            "verification_ms": 5.0,
            "proof_bytes": 192,
        },
    }
    result = compose_table7_result(
        method="Kaizen",
        seed=1337,
        training=training,
        targets=targets,
        source_lock_digest="e" * 64,
        gas_evidence=gas,
        proof_evidence=proof,
    )
    assert result["formal_accepted"]
    assert result["gas_usd"] == gas_usd(151_757)
    assert result["runtime_seconds"] == 102.75


def test_veriblock_table7_composition_scales_measured_per_client_proofs():
    targets = json.loads(
        (ROOT / "config" / "paper_table7_all_methods.json").read_text(
            encoding="utf-8"
        )
    )
    training = {
        "formal_accepted": True,
        "rounds": 200,
        "dataset": "CIFAR10",
        "source_commit": "a" * 40,
        "protocol_runtime_seconds": 40.0,
        "runtime_seconds": 40.0,
        "communication_mb": 90.0,
        "num_clients": 50,
        "result_digest": "b" * 64,
    }
    benchmark = {
        "classification": "controlled_veriblockfl_full_verification",
        "real_benchmark": True,
        "formal_accepted": True,
        "source_commit": "a" * 40,
        "evidence_digest": "c" * 64,
        "metrics": {
            "witness_seconds": 6.0,
            "proof_seconds": 18.0,
            "verification_ms": 1.0,
            "proof_bytes": 1145,
            "public_input_bytes": 160,
            "persistent_bytes_per_client": 1305,
            "gas_usd": 0.8,
        },
    }
    result = compose_table7_result(
        method="VeriblockFL",
        seed=1337,
        training=training,
        targets=targets,
        source_lock_digest="d" * 64,
        veriblock_evidence=benchmark,
    )
    assert result["formal_accepted"]
    assert result["runtime_seconds"] == 1240.05


def test_veriblock_controlled_input_satisfies_locked_circuit_shapes():
    constants = ",".join(str(value) for value in range(64))
    source = (
        "const u32 ac = 6; const u32 fe = 9; const u32 bs = 10; "
        + "field[64] round_constants = ["
        + constants
        + "];"
    )
    values = generate_input(source)
    assert len(values) == 13
    assert len(values[0]) == 6 and len(values[0][0]) == 9
    assert len(values[4]) == 10 and len(values[4][0]) == 9
    assert values[-1] == values[-2]


def test_table7_matrix_composes_twelve_same_source_results(tmp_path):
    targets = json.loads(
        (ROOT / "config" / "paper_table7_all_methods.json").read_text(
            encoding="utf-8"
        )
    )

    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    seeds = {}
    for seed in (1337, 2026, 3817739):
        base = {
            "formal_accepted": True,
            "rounds": 200,
            "dataset": "CIFAR10",
            "source_commit": "a" * 40,
            "protocol_runtime_seconds": 40.0,
            "runtime_seconds": 40.0,
            "communication_mb": 90.0,
            "storage_mb_per_client": 0.0,
            "num_clients": 50,
            "result_digest": f"{seed:064x}",
        }
        pol = {
            **base,
            "runtime_seconds": 70.0,
            "communication_mb": 170.0,
            "storage_mb_per_client": 2.0,
            "trust_setup_record_digest": "b" * 64,
            "real_contract_rounds": True,
            "contract_rounds": 200,
            "contract_evidence_digest": "c" * 64,
        }
        gas = {
            "passed": True,
            "source": {"commit": "a" * 40},
            "observed_gas": {"honest_round_total": 151757},
            "evidence_digest": "d" * 64,
        }
        kaizen = {
            "formal_accepted": True,
            "real_benchmark": True,
            "method": "Kaizen",
            "source_commit": "a" * 40,
            "evidence_digest": "e" * 64,
            "metrics": {
                "witness_seconds": 0.5,
                "proof_generation_seconds": 2.0,
                "verification_ms": 5.0,
                "proof_bytes": 192,
            },
        }
        veriblock = {
            "classification": "controlled_veriblockfl_full_verification",
            "real_benchmark": True,
            "formal_accepted": True,
            "source_commit": "a" * 40,
            "evidence_digest": "f" * 64,
            "metrics": {
                "witness_seconds": 6.0,
                "proof_seconds": 18.0,
                "verification_ms": 1.0,
                "proof_bytes": 1145,
                "public_input_bytes": 160,
                "persistent_bytes_per_client": 1305,
                "gas_usd": 0.8,
            },
        }
        seeds[str(seed)] = {
            "vanilla_training": save(f"vanilla-{seed}.json", base),
            "pol_training": save(f"pol-{seed}.json", pol),
            "contract_gas": save(f"gas-{seed}.json", gas),
            "kaizen_proof": save(f"kaizen-{seed}.json", kaizen),
            "veriblock_benchmark": save(f"veriblock-{seed}.json", veriblock),
        }
    results = compose_table7_matrix(
        {"seeds": seeds}, targets, source_lock_digest="1" * 64
    )
    assert len(results) == 12
    assert all(result["formal_accepted"] for result in results)
