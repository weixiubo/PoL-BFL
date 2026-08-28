from decimal import Decimal

import pytest

torch = pytest.importorskip("torch")

from experiments.final.run_security_cell import (
    apply_delta,
    client_account_snapshot,
    fake_commitment,
    fake_fingerprint,
    evaluate_cell_acceptance,
    model_delta,
    process_training_policy,
    summarize_security_rates,
)
from polbfl.incentives import ParticipantAccount, ParticipantRole
from polbfl.sybil import screen_trace_fingerprints


def test_security_rates_average_client_decisions_across_rounds():
    summary = summarize_security_rates(
        [
            {"detection_rate": 100.0, "false_positive_rate": 0.0},
            {"detection_rate": 90.0, "false_positive_rate": 2.5},
            {"detection_rate": 95.0, "false_positive_rate": 0.0},
        ]
    )

    assert summary == {
        "DR": 95.0,
        "FPR": pytest.approx(2.5 / 3.0),
        "security_rate_aggregation": "arithmetic_mean_of_per_round_client_rates",
        "security_rate_unit": "client-round",
    }

    with pytest.raises(ValueError, match="at least one round"):
        summarize_security_rates([])
    with pytest.raises(ValueError, match="normalized"):
        summarize_security_rates(
            [
                {
                    "detection_rate": 101.0,
                    "false_positive_rate": 0.0,
                }
            ]
        )

def test_security_runner_delta_roundtrip_and_fake_trace_shape():
    global_state = {"w": torch.tensor([1.0, 2.0]), "counter": torch.tensor(1)}
    local_state = {"w": torch.tensor([1.5, 1.0]), "counter": torch.tensor(2)}
    delta = model_delta(local_state, global_state)
    restored = apply_delta(global_state, delta)
    assert torch.equal(restored["w"], local_state["w"])
    assert torch.equal(restored["counter"], global_state["counter"])

    commitment = fake_commitment("round-1", "client-1", expected_steps=160, seed=7)
    fingerprint = fake_fingerprint(commitment, seed=8, batch_size=32)
    assert commitment.checkpoint_count == 33
    assert len(fingerprint.checkpoint_vectors) == 33
    assert len(fingerprint.checkpoint_vectors[0]) == 14
    other = fake_fingerprint(
        fake_commitment("round-1", "client-2", expected_steps=160, seed=7),
        seed=8,
        batch_size=32,
    )
    assert fingerprint.batch_indices != other.batch_indices

    fake_population = [
        fake_fingerprint(
            fake_commitment("round-1", f"client-{index}", expected_steps=160, seed=7),
            seed=100 + index,
            batch_size=32,
        )
        for index in range(10)
    ]
    assert not screen_trace_fingerprints(fake_population).flagged_clients


def test_process_training_policy_executes_lazy_free_rider_for_one_epoch_without_pol():
    assert process_training_policy(
        attack="FreeRidingLT",
        malicious=True,
        local_epochs=5,
        record_pol=True,
    ) == (1, False)
    assert process_training_policy(
        attack="FreeRidingLT",
        malicious=False,
        local_epochs=5,
        record_pol=True,
    ) == (5, True)
    assert process_training_policy(
        attack="ALIE",
        malicious=True,
        local_epochs=5,
        record_pol=True,
    ) == (5, True)


def test_account_snapshot_preserves_raw_reputation_and_zeroes_inactive_effective_weight():
    accounts = {
        "client-0": ParticipantAccount(
            "client-0", ParticipantRole.CLIENT, Decimal("0.05"), Decimal("0.9")
        ),
        "client-1": ParticipantAccount(
            "client-1",
            ParticipantRole.CLIENT,
            Decimal("0"),
            Decimal("0.45"),
            active=False,
        ),
    }
    snapshot = client_account_snapshot(accounts, {"client-1"})
    assert snapshot["reputation_by_client"]["client-1"] == 0.45
    assert snapshot["effective_reputation_by_client"]["client-1"] == 0.0
    assert snapshot["honest_reputation_mean"] == 0.9
    assert snapshot["malicious_reputation_mean"] == 0.0


def test_no_pol_profile_flag_is_diagnostic_only(monkeypatch, tmp_path):
    from experiments.final import run_security_cell

    base = [
        "run_security_cell.py",
        "--dataset",
        "CIFAR10",
        "--attack",
        "FreeRidingNT",
        "--seed",
        "1337",
        "--run-id",
        "profile",
        "--output",
        str(tmp_path / "profile"),
        "--no-pol",
        "--process-training",
    ]
    monkeypatch.setattr(run_security_cell.sys, "argv", base)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()

    monkeypatch.setattr(run_security_cell.sys, "argv", [*base, "--diagnostic"])
    args = run_security_cell.parse_args()
    assert args.no_pol is True
    assert args.diagnostic is True


def test_spot_check_probability_is_formal_only_in_the_sensitivity_study(monkeypatch, tmp_path):
    from experiments.final import run_security_cell

    base = [
        "run_security_cell.py",
        "--dataset",
        "CIFAR10",
        "--attack",
        "FreeRidingNT",
        "--seed",
        "1337",
        "--run-id",
        "sensitivity",
        "--output",
        str(tmp_path / "sensitivity"),
        "--audit-probability",
        "0.15",
        "--process-training",
        "--train-processes-per-gpu",
        "8",
    ]
    monkeypatch.setattr(run_security_cell.sys, "argv", base)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()
    monkeypatch.setattr(run_security_cell.sys, "argv", [*base, "--study", "sensitivity"])
    assert run_security_cell.parse_args().audit_probability == run_security_cell.Decimal("0.15")
    invalid = [*base[:-1], "0", "--study", "sensitivity"]
    monkeypatch.setattr(run_security_cell.sys, "argv", invalid)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()


def test_security_cell_acceptance_checks_security_and_overhead_directions():
    targets = {
        "table_2_pol_bfl": {
            "CIFAR10": {"FreeRidingNT": {"MA": 86.8, "DR": 96.5, "FPR": 2.1}}
        },
        "table_7_overhead": {
            "runtime_seconds": 78.5,
            "communication_mb": 178.2,
            "storage_mb_per_client": 2.5,
        },
    }
    result = {
        "dataset": "CIFAR10",
        "attack": "FreeRidingNT",
        "MA": 87.0,
        "DR": 97.0,
        "FPR": 2.0,
        "runtime_seconds": 77.0,
        "communication_mb": 110.0,
        "storage_mb_per_client": 1.3,
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]
    result["runtime_seconds"] = 79.0
    report = evaluate_cell_acceptance(result, targets)
    assert not report["passed"]
    assert not report["checks"]["runtime_seconds"]


def test_dirichlet_partition_flag_is_positive_and_study_scoped(monkeypatch, tmp_path):
    from experiments.final import run_security_cell

    base = [
        "run_security_cell.py",
        "--dataset",
        "CIFAR10",
        "--attack",
        "FreeRidingNT",
        "--seed",
        "1337",
        "--run-id",
        "dirichlet",
        "--output",
        str(tmp_path / "dirichlet"),
        "--partition-alpha",
        "0.1",
        "--process-training",
        "--train-processes-per-gpu",
        "8",
    ]
    monkeypatch.setattr(run_security_cell.sys, "argv", base)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()

    noniid_formal = [*base, "--study", "noniid"]
    monkeypatch.setattr(run_security_cell.sys, "argv", noniid_formal)
    args = run_security_cell.parse_args()
    assert args.study == "noniid"
    assert args.partition_alpha == pytest.approx(0.1)
    monkeypatch.setattr(run_security_cell.sys, "argv", [*base, "--diagnostic"])
    assert run_security_cell.parse_args().partition_alpha == pytest.approx(0.1)
    invalid = list(base)
    invalid[invalid.index("--partition-alpha") + 1] = "0"
    invalid.append("--diagnostic")
    monkeypatch.setattr(run_security_cell.sys, "argv", invalid)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()


def test_noniid_cell_acceptance_uses_table_nine_metrics():
    targets = {
        "table_9_noniid": {
            "CIFAR10": {
                "0.1": {
                    "NoAttackMA": 82.5,
                    "FreeRidingDR": 94.2,
                    "ALIEDR": 80.5,
                    "FPR": 4.8,
                }
            }
        }
    }
    base = {
        "study": "noniid",
        "dataset": "CIFAR10",
        "partition_label": "0.1",
        "MA": 83.0,
        "DR": 95.0,
        "FPR": 4.0,
    }
    assert evaluate_cell_acceptance(
        {**base, "attack": "NoAttack"}, targets
    )["passed"]
    assert evaluate_cell_acceptance(
        {**base, "attack": "FreeRidingNT"}, targets
    )["passed"]
    assert evaluate_cell_acceptance(
        {**base, "attack": "ALIE"}, targets
    )["passed"]


def test_composability_acceptance_routes_aggregation_and_attack_labels():
    targets = {
        "table_4_all_modes": {
            "Krum": {
                "FreeRidingNT": {
                    "PoLBFLPrefilter": {"MA": 85.2, "DR": 96.5, "FPR": 2.1}
                }
            }
        }
    }
    result = {
        "study": "composability",
        "dataset": "CIFAR10",
        "attack": "FreeRidingNT",
        "aggregation_method": "krum",
        "composition_mode": "PoLBFLPrefilter",
        "MA": 86.0,
        "DR": 97.0,
        "FPR": 2.0,
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]


def test_scalability_acceptance_and_formal_population(monkeypatch, tmp_path):
    targets = {
        "table_8_scalability": {
            "100": {
                "runtime_seconds": 152.8,
                "communication_mb": 348.5,
                "seconds_per_client": 1.53,
                "MA": 85.5,
                "DR": 95.8,
                "FPR": 2.3,
            }
        }
    }
    result = {
        "study": "scalability",
        "dataset": "CIFAR10",
        "attack": "FreeRidingNT",
        "num_clients": 100,
        "runtime_seconds": 150.0,
        "communication_mb": 340.0,
        "seconds_per_client": 1.5,
        "MA": 86.0,
        "DR": 96.0,
        "FPR": 2.0,
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]

    from experiments.final import run_security_cell

    monkeypatch.setattr(
        run_security_cell.sys,
        "argv",
        [
            "run_security_cell.py",
            "--study",
            "scalability",
            "--dataset",
            "CIFAR10",
            "--attack",
            "FreeRidingNT",
            "--seed",
            "1337",
            "--run-id",
            "scale-100",
            "--output",
            str(tmp_path / "scale-100"),
            "--num-clients",
            "100",
            "--num-malicious",
            "20",
            "--clients-per-round",
            "100",
            "--process-training",
            "--train-processes-per-gpu",
            "8",
        ],
    )
    args = run_security_cell.parse_args()
    assert args.study == "scalability"


def test_layer_acceptance_enforces_target_and_declared_component_profile():
    targets = {
        "table_3_layer_dr": {
            "CIFAR10": {"ALIE": {"L1L2": 85.0}}
        }
    }
    result = {
        "study": "layer",
        "dataset": "CIFAR10",
        "attack": "ALIE",
        "method": "PoLBFL",
        "layer_variant": "L1L2",
        "DR": 86.0,
        "FPR": 3.0,
        "real_groth16": True,
        "real_robust_aggregation": True,
        "real_contract_transition": False,
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]
    result["real_contract_transition"] = True
    assert not evaluate_cell_acceptance(result, targets)["passed"]


def test_layer_profile_is_formal_only_in_the_table3_study(monkeypatch, tmp_path):
    from experiments.final import run_security_cell

    base = [
        "run_security_cell.py",
        "--study",
        "layer",
        "--dataset",
        "CIFAR10",
        "--attack",
        "ALIE",
        "--layer-variant",
        "L1L2",
        "--seed",
        "1337",
        "--run-id",
        "layer",
        "--output",
        str(tmp_path / "layer"),
        "--process-training",
        "--train-processes-per-gpu",
        "8",
    ]
    monkeypatch.setattr(run_security_cell.sys, "argv", base)
    args = run_security_cell.parse_args()
    assert args.study == "layer"
    assert args.layer_variant == "L1L2"

    main_with_ablation = [
        "run_security_cell.py",
        "--dataset",
        "CIFAR10",
        "--attack",
        "ALIE",
        "--layer-variant",
        "L1L2",
        "--seed",
        "1337",
        "--run-id",
        "invalid",
        "--output",
        str(tmp_path / "invalid"),
        "--process-training",
        "--train-processes-per-gpu",
        "8",
    ]
    monkeypatch.setattr(run_security_cell.sys, "argv", main_with_ablation)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()


def test_sybil_scalability_acceptance_uses_dataset_vector_targets_and_stake_floor():
    targets = {
        "figure_6_vector_targets": {
            "CIFAR10": {
                "20": {
                    "MA": 80.5,
                    "DR": 87.5,
                    "FPR": 3.2,
                    "stake_eth": 1.0,
                }
            }
        }
    }
    result = {
        "study": "sybil_scalability",
        "dataset": "CIFAR10",
        "attack": "Sybil",
        "sybil_identity_count": 20,
        "sybil_stake_eth": 1.0,
        "MA": 81.0,
        "DR": 90.0,
        "FPR": 2.0,
        "real_groth16": True,
        "real_contract_transition": True,
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]
    result["sybil_stake_eth"] = 0.95
    assert not evaluate_cell_acceptance(result, targets)["passed"]


def test_sybil_scalability_formal_population_is_identity_bound(monkeypatch, tmp_path):
    from experiments.final import run_security_cell

    command = [
        "run_security_cell.py",
        "--study",
        "sybil_scalability",
        "--dataset",
        "CIFAR10",
        "--attack",
        "Sybil",
        "--sybil-identities",
        "20",
        "--num-clients",
        "60",
        "--num-malicious",
        "20",
        "--clients-per-round",
        "60",
        "--seed",
        "1337",
        "--run-id",
        "figure6",
        "--output",
        str(tmp_path / "figure6"),
        "--process-training",
        "--train-processes-per-gpu",
        "8",
    ]
    monkeypatch.setattr(run_security_cell.sys, "argv", command)
    args = run_security_cell.parse_args()
    assert args.sybil_identities == 20
    assert args.num_clients == 60

    invalid = [
        value
        for index, value in enumerate(command)
        if index
        not in {
            command.index("--num-clients"),
            command.index("--num-clients") + 1,
        }
    ]
    invalid.extend(["--num-clients", "59"])
    monkeypatch.setattr(run_security_cell.sys, "argv", invalid)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()


def test_incentive_acceptance_uses_all_three_directional_metrics():
    targets = {
        "table_5_all_methods": {
            "FedCoin": {
                "ParticipationRate": 75.5,
                "AttackSuccessRate": 20.2,
                "ModelAccuracy": 76.8,
            }
        }
    }
    result = {
        "study": "incentive",
        "dataset": "CIFAR10",
        "attack": "FreeRidingNT",
        "table5_method": "FedCoin",
        "ParticipationRate": 80.0,
        "AttackSuccessRate": 18.0,
        "ModelAccuracy": 78.0,
        "real_training": True,
        "training_rounds": 200,
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]
    result["AttackSuccessRate"] = 21.0
    assert not evaluate_cell_acceptance(result, targets)["passed"]


def test_fedcoin_is_formal_only_in_the_incentive_study(monkeypatch, tmp_path):
    from experiments.final import run_security_cell

    command = [
        "run_security_cell.py",
        "--study",
        "incentive",
        "--dataset",
        "CIFAR10",
        "--attack",
        "FreeRidingNT",
        "--method",
        "FedCoin",
        "--seed",
        "1337",
        "--run-id",
        "fedcoin",
        "--output",
        str(tmp_path / "fedcoin"),
        "--process-training",
        "--train-processes-per-gpu",
        "8",
    ]
    monkeypatch.setattr(run_security_cell.sys, "argv", command)
    assert run_security_cell.parse_args().study == "incentive"
    invalid = list(command)
    invalid[invalid.index("incentive")] = "main"
    monkeypatch.setattr(run_security_cell.sys, "argv", invalid)
    with pytest.raises(SystemExit):
        run_security_cell.parse_args()


def test_worker_seed_domain_accepts_large_paper_derived_seeds():
    import numpy as np
    from experiments.final.client_worker import _seed

    derived = 3_817_739 * 1_000_003 + 199 * 10_007 + 49
    _seed(derived)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)
    _seed(derived)
    assert np.array_equal(first_numpy, np.random.random(4))
    assert torch.equal(first_torch, torch.rand(4))
