import json
from pathlib import Path

import pytest

from experiments.final.target_provenance import (
    AUTHORITY_TARGET_FILES,
    load_merged_targets,
    validate_all_target_files,
)
from scripts.extract_table3_targets import parse_table3
from scripts.extract_table6_targets import parse_table6
from scripts.extract_table11_targets import parse_table11
from scripts.extract_table13_targets import parse_table13


ROOT = Path(__file__).parents[1]


def test_table3_extractor_reads_every_dataset_attack_and_profile():
    lines = ["Table 3: Layer Contribution Analysis (DR %)."]
    values = {
        "CIFAR10": ((96.0, 96.2, 96.2, 96.5), (78.2, 85.0, 78.5, 85.6), (82.5, 83.0, 92.2, 92.8)),
        "FEMNIST": ((96.5, 96.6, 96.6, 96.8), (80.5, 87.0, 80.8, 87.5), (84.2, 84.5, 93.2, 93.8)),
        "CIFAR100": ((95.2, 95.5, 95.5, 95.8), (76.5, 83.0, 77.0, 83.5), (80.5, 81.0, 90.8, 91.2)),
    }
    for dataset_rows in values.values():
        for label, row in zip(("Free-riding (NT)", "ALIE", "Sybil"), dataset_rows):
            lines.append(label + " " + " ".join(str(value) for value in row))
    lines.append("Table 4: PoL-BFL + Robust Aggregation on CIFAR-10.")
    table = parse_table3("\n".join(lines))["table_3_layer_dr"]
    assert table["CIFAR10"]["ALIE"]["L1L2"] == 85.0
    assert table["FEMNIST"]["Sybil"]["Full"] == 93.8
    assert table["CIFAR100"]["FreeRidingNT"]["L1"] == 95.2


def test_table6_extractor_preserves_money_signs():
    text = "\n".join(
        [
            "Table 6: Client Profit Analysis in PoL-BFL.",
            "Honest $0.172 -$0.022 $0 +$0.150",
            "Rational (LT) $0.098 -$0.016 -$0.008 +$0.074",
            "Malicious (NT) $0.025 -$0.015 -$0.145 -$0.135",
            "Figure 3: Reputation evolution on CIFAR-10.",
        ]
    )
    table = parse_table6(text)["table_6_profit_usd"]
    assert table["Honest"]["profit"] == 0.15
    assert table["RationalLT"]["slash"] == -0.008
    assert table["MaliciousNT"]["profit"] == -0.135


def test_table11_extractor_distinguishes_repeated_hardware_pair():
    text = "\n".join(
        [
            "Table 11: Cross-Hardware Verification Performance on",
            "RTX 4090 -> RTX 4090 0.8 99.2 97.2 97.2",
            "V100 -> V100 1.2 98.8 96.5 96.5",
            "RTX 4090 -> RTX 3080 1.5 98.5 95.8 95.8",
            "RTX 4090 -> V100 1.8 98.2 94.2 94.2",
            "RTX 4090 -> A100 1.3 98.7 96.0 96.0",
            "V100 -> A100 1.6 98.4 95.5 95.5",
            "RTX 4090 -> V100 5.2 94.8 93.5 93.5",
            "A.2    ZK Proof Cost",
        ]
    )
    table = parse_table11(text)["table_11_cross_hardware"]
    assert table["RTX4090_V100"]["FPR"] == 1.8
    assert table["Kaizen_RTX4090_V100"]["FPR"] == 5.2
    assert table["V100_A100"]["honest_pass_rate"] == 98.4


def test_table13_extractor_reads_gas_not_dollar_costs():
    text = "\n".join(
        [
            "Table 13: Gas Cost Breakdown per Round (@1.5 gwei,",
            "Submit Commitment 85,000 $0.32",
            "Submit Proof Receipt 120,000 $0.45",
            "Claim Reward 45,000 $0.17",
            "Slash Penalty 65,000 $0.24",
            "Average/round 225,000 $0.85",
            "A.3    Protocol Pseudocode",
        ]
    )
    assert parse_table13(text)["table_13_gas"] == {
        "commitment": 85000,
        "proof_receipt": 120000,
        "reward_claim": 45000,
        "slash": 65000,
        "honest_round_total": 225000,
    }


def test_remaining_dedicated_targets_match_main_and_all_are_authority_bound():
    main = json.loads((ROOT / "config" / "paper_targets.json").read_text(encoding="utf-8"))
    files = {
        "paper_table3_layer_dr.json": "table_3_layer_dr",
        "paper_table6_profit.json": "table_6_profit_usd",
        "paper_table11_cross_hardware.json": "table_11_cross_hardware",
        "paper_table13_gas.json": "table_13_gas",
    }
    for filename, section in files.items():
        dedicated = json.loads((ROOT / "config" / filename).read_text(encoding="utf-8"))
        assert dedicated[section] == main[section]
        assert load_merged_targets(ROOT, (filename,))[section] == main[section]
    validation = validate_all_target_files(ROOT)
    assert validation["passed"]
    assert set(validation["checks"]) == {"paper_targets.json", *AUTHORITY_TARGET_FILES}


def test_table13_extractor_fails_closed_on_ambiguous_gas():
    text = "\n".join(
        [
            "Table 13: Gas Cost Breakdown per Round (@1.5 gwei,",
            "Submit Commitment 85,000 86,000 $0.32",
            "Submit Proof Receipt 120,000 $0.45",
            "Claim Reward 45,000 $0.17",
            "Slash Penalty 65,000 $0.24",
            "Average/round 225,000 $0.85",
            "A.3    Protocol Pseudocode",
        ]
    )
    with pytest.raises(ValueError, match="missing or ambiguous"):
        parse_table13(text)
