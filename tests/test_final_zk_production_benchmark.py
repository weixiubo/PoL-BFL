from scripts.zk_production_benchmark import evaluate_metrics, parse_constraint_count


def test_production_benchmark_parses_real_snarkjs_constraint_output():
    output = "\x1b[32m[INFO]\x1b[0m: # of Wires: 1085296\n[INFO]: # of Constraints: 1090382\n"
    assert parse_constraint_count(output) == 1_090_382


def test_production_benchmark_applies_every_table12_upper_bound():
    metrics = {
        "circuit_constraints": 1_090_382,
        "witness_seconds_median": 1.0,
        "proof_seconds_median": 2.0,
        "prover_memory_gb_max": 2.0,
        "proof_bytes": 192,
        "verification_ms_median": 5.0,
    }
    targets = {
        "table_12_zk": {
            "circuit_constraints": 1_100_000,
            "witness_seconds": 1.8,
            "proof_generation_seconds": 4.2,
            "prover_memory_gb": 2.5,
            "proof_bytes": 192,
            "verification_ms": 8.5,
        }
    }
    report = evaluate_metrics(metrics, targets)
    assert report["passed"]
    metrics["proof_bytes"] = 193
    report = evaluate_metrics(metrics, targets)
    assert not report["passed"]
    assert report["failed"] == ["proof_bytes"]
