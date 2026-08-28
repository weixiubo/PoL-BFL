import json

from experiments.scripts.aggregate_multi_seed_results import (
    aggregate_rq3_results,
    aggregate_rq4_results,
    aggregate_rq5_results,
)
from experiments.scripts.generate_paper_tables import generate_rq5_table


def _write_seed(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rq3_aggregation_covers_all_measured_costs(tmp_path):
    files = [
        _write_seed(
            tmp_path / "rq3_seed_1.json",
            [
                {
                    "method": "PoL_FL_ZKP",
                    "total_training_time": 10.0,
                    "total_communication_mb": 20.0,
                    "total_storage_mb": 3.0,
                    "total_zkp_gen_time": 0.4,
                    "total_zkp_verify_time": 0.1,
                    "total_estimated_fee_eth": 0.02,
                    "avg_round_time": 2.0,
                }
            ],
        ),
        _write_seed(
            tmp_path / "rq3_seed_2.json",
            [
                {
                    "method": "PoL_FL_ZKP",
                    "total_training_time": 14.0,
                    "total_communication_mb": 24.0,
                    "total_storage_mb": 5.0,
                    "total_zkp_gen_time": 0.6,
                    "total_zkp_verify_time": 0.2,
                    "total_estimated_fee_eth": 0.04,
                    "avg_round_time": 3.0,
                }
            ],
        ),
    ]

    aggregated = aggregate_rq3_results(files, tmp_path)

    metrics = aggregated["PoL_FL_ZKP"]
    assert metrics["training_time"]["mean"] == 12.0
    assert metrics["communication_cost"]["mean"] == 22.0
    assert metrics["storage_cost"]["mean"] == 4.0
    assert metrics["zkp_time"]["mean"] == 0.65
    assert metrics["gas_cost"]["num_runs"] == 2
    assert (tmp_path / "rq3_aggregated.json").is_file()


def test_rq4_aggregation_groups_incentive_scenarios(tmp_path):
    files = [
        _write_seed(
            tmp_path / "rq4_seed_1.json",
            [
                {
                    "scenario": "dynamic_reward",
                    "avg_participation_rate": 0.9,
                    "avg_attack_success_rate": 0.1,
                    "total_honest_utility": 8.0,
                    "total_rational_utility": 6.0,
                    "total_malicious_utility": -2.0,
                    "final_accuracy": 0.86,
                }
            ],
        ),
        _write_seed(
            tmp_path / "rq4_seed_2.json",
            [
                {
                    "scenario": "dynamic_reward",
                    "avg_participation_rate": 1.0,
                    "avg_attack_success_rate": 0.0,
                    "total_honest_utility": 10.0,
                    "total_rational_utility": 8.0,
                    "total_malicious_utility": -4.0,
                    "final_accuracy": 0.88,
                }
            ],
        ),
    ]

    metrics = aggregate_rq4_results(files, tmp_path)["dynamic_reward"]

    assert metrics["participation_rate"]["mean"] == 0.95
    assert metrics["attack_success_rate"]["mean"] == 0.05
    assert metrics["honest_utility"]["mean"] == 9.0
    assert metrics["final_accuracy"]["num_runs"] == 2


def test_rq5_aggregation_preserves_attack_and_method_dimensions(tmp_path):
    files = [
        _write_seed(
            tmp_path / "rq5_seed_1.json",
            [
                {
                    "attack_type": "ALIE",
                    "baseline_method": "PoL_Krum",
                    "final_accuracy": 0.87,
                    "convergence_round": 12,
                    "detection_metrics": {"TPR": 0.95, "FPR": 0.02},
                }
            ],
        ),
        _write_seed(
            tmp_path / "rq5_seed_2.json",
            [
                {
                    "attack_type": "ALIE",
                    "baseline_method": "PoL_Krum",
                    "final_accuracy": 0.89,
                    "convergence_round": 10,
                    "detection_metrics": {"TPR": 0.97, "FPR": 0.01},
                }
            ],
        ),
    ]

    metrics = aggregate_rq5_results(files, tmp_path)["ALIE"]["PoL_Krum"]

    assert metrics["final_accuracy"]["mean"] == 0.88
    assert metrics["convergence_round"]["mean"] == 11.0
    assert metrics["TPR"]["mean"] == 0.96
    assert metrics["FPR"]["num_runs"] == 2


def test_rq5_table_renders_each_base_and_pol_pair(tmp_path):
    methods = {}
    for index, name in enumerate(
        [
            "Krum",
            "PoL_Krum",
            "Trimmed_Mean",
            "PoL_Trimmed_Mean",
            "Median",
            "PoL_Median",
            "Bulyan",
            "PoL_Bulyan",
        ]
    ):
        methods[name] = {
            "final_accuracy": {
                "mean": 0.80 + index / 100,
                "std": 0.01,
                "num_runs": 3,
            }
        }
    output = tmp_path / "rq5_table.tex"

    generate_rq5_table({"ALIE": methods}, output)
    rendered = output.read_text(encoding="utf-8")

    assert "Alie" in rendered
    assert rendered.count("\\textbf{") >= 8
    assert rendered.count("&") >= 8
