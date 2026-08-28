from pathlib import Path

from experiments.final.aggregate_table11 import aggregate_table11
from experiments.final.cross_hardware import aggregate_cross_hardware
from experiments.final.cross_hardware_profiles import (
    KAIZEN_PAIR,
    evaluate_numerical_probe,
    profile_for_pair,
)
from experiments.final.run_cross_hardware_matrix import (
    hardware_command,
    plan_cross_hardware_cells,
)


ROOT = Path(__file__).resolve().parents[1]


def _observation(client, behavior, accepted):
    return {
        "hardware_pair": "RTX4090_RTX4090",
        "client_id": client,
        "behavior": behavior,
        "accepted": accepted,
        "proof_digest": (client.encode().hex() + "0" * 64)[:64],
        "trainer_attestation": {"gpu": "RTX 4090", "uuid": f"train-{client}"},
        "verifier_attestation": {"gpu": "RTX 4090", "uuid": f"verify-{client}"},
        "real_trace": True,
        "real_groth16": True,
    }


def test_cross_hardware_aggregate_requires_attested_real_observations():
    observations = [
        *[_observation(f"h-{index}", "honest", True) for index in range(10)],
        *[_observation(f"m-{index}", "malicious", True if index == 0 else False) for index in range(40)],
    ]
    targets = {
        "table_11_cross_hardware": {
            "RTX4090_RTX4090": {
                "FPR": 0.8,
                "honest_pass_rate": 99.2,
                "DR": 97.2,
                "block_rate": 97.2,
            }
        }
    }
    aggregate = aggregate_cross_hardware(observations, targets)
    # 39/40 malicious samples are blocked, or 97.5%.
    assert aggregate["table_11_cross_hardware"]["RTX4090_RTX4090"]["DR"] == 97.5
    assert aggregate["acceptance"]["passed"]


def test_cross_hardware_matrix_plans_all_pairs_and_three_seeds(tmp_path):
    pairs = [
        "RTX4090_RTX4090",
        "V100_V100",
        "RTX4090_RTX3080",
        "RTX4090_V100",
        "RTX4090_A100",
        "V100_A100",
        "Kaizen_RTX4090_V100",
    ]
    matrix = {
        "seeds": [1337, 2026, 3817739],
        "studies": {"table_11_cross_hardware": {"hardware_pairs": pairs}},
    }
    cells = plan_cross_hardware_cells(matrix)
    assert len(cells) == 21
    cell = next(item for item in cells if item.hardware_pair == "RTX4090_RTX4090")
    command = hardware_command(
        cell,
        {
            "trainer_device": 0,
            "verifier_device": 1,
            "expected_trainer": "RTX 4090",
            "expected_verifier": "RTX 4090",
        },
        python=Path("/python"),
        output=tmp_path / "cell",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--hardware-pair") + 1] == cell.hardware_pair

    kaizen_cell = next(
        item for item in cells if item.hardware_pair == KAIZEN_PAIR
    )
    kaizen_command = hardware_command(
        kaizen_cell,
        {
            "trainer_device": 0,
            "verifier_device": 1,
            "expected_trainer": "RTX 4090",
            "expected_verifier": "V100",
        },
        python=Path("/python"),
        output=tmp_path / "kaizen-cell",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert kaizen_command[2:4] == [
        "-m",
        "experiments.final.run_kaizen_cross_hardware_trial",
    ]
    assert (
        kaizen_command[kaizen_command.index("--hardware-pair") + 1]
        == KAIZEN_PAIR
    )


def test_cross_hardware_profiles_bind_dual_and_single_thresholds():
    polbfl = profile_for_pair(ROOT, "RTX4090_RTX4090")
    assert polbfl.method == "PoLBFL"
    assert polbfl.pair_tolerance == 1e-5
    assert polbfl.final_tolerance == 1e-3
    assert polbfl.configuration_sha256 is None

    kaizen = profile_for_pair(ROOT, KAIZEN_PAIR)
    assert kaizen.method == "Kaizen"
    assert kaizen.profile_id == "kaizen_single_threshold_v1"
    assert kaizen.pair_tolerance == 1e-3
    assert kaizen.final_tolerance == 1e-3
    assert len(str(kaizen.configuration_sha256)) == 64

    accepted = evaluate_numerical_probe(
        kaizen, {"maximum_absolute_error": 9.9e-4}
    )
    rejected = evaluate_numerical_probe(
        kaizen, {"maximum_absolute_error": 1.01e-3}
    )
    assert accepted["passed"]
    assert accepted["checks"]["single_threshold_only"]
    assert not rejected["passed"]
    assert rejected["failed"] == ["final_tolerance"]


def test_complete_table11_aggregate_requires_every_pair_and_seed():
    pairs = [
        "RTX4090_RTX4090",
        "V100_V100",
        "RTX4090_RTX3080",
        "RTX4090_V100",
        "RTX4090_A100",
        "V100_A100",
        "Kaizen_RTX4090_V100",
    ]
    targets = {"table_11_cross_hardware": {}}
    results = []
    counter = 0
    for pair in pairs:
        targets["table_11_cross_hardware"][pair] = {
            "FPR": 5.5,
            "honest_pass_rate": 94.0,
            "DR": 93.0,
            "block_rate": 93.0,
        }
        for seed in (1337, 2026, 3817739):
            observations = [
                {
                    "hardware_pair": pair,
                    "client_id": f"honest-{seed}",
                    "behavior": "honest",
                    "accepted": True,
                    "proof_digest": f"{counter:064x}",
                    "trainer_attestation": {"gpu": "trainer", "uuid": f"t-{pair}"},
                    "verifier_attestation": {"gpu": "verifier", "uuid": f"v-{pair}"},
                    "real_trace": True,
                    "real_groth16": True,
                },
                {
                    "hardware_pair": pair,
                    "client_id": f"malicious-{seed}",
                    "behavior": "malicious",
                    "accepted": False,
                    "proof_digest": f"{counter + 1000:064x}",
                    "trainer_attestation": {"gpu": "trainer", "uuid": f"t-{pair}"},
                    "verifier_attestation": {"gpu": "verifier", "uuid": f"v-{pair}"},
                    "real_trace": True,
                    "real_groth16": True,
                },
            ]
            results.append(
                {
                    "study": "cross_hardware",
                    "hardware_pair": pair,
                    "seed": seed,
                    "source_commit": "a" * 40,
                    "evidence_digest": f"{counter + 2000:064x}",
                    "result_digest": f"{counter + 3000:064x}",
                    "observations": observations,
                    "formal_accepted": True,
                }
            )
            counter += 1
    aggregate = aggregate_table11(results, targets)
    assert aggregate["acceptance"]["passed"]
    assert set(aggregate["table_11_cross_hardware"]) == set(pairs)
