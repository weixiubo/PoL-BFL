from scripts.extract_table2_targets import ATTACK_LABELS, parse_table2


def test_table2_extractor_requires_all_rows_and_preserves_method_columns():
    numeric_row = " 10.0 20.0 21.0 22.0 30.0 31.0 32.0 40.0 41.0 42.0 50.0 51.0 52.0 60.0 61.0 62.0"
    rows = [
        f"  {label}{numeric_row}"
        for _dataset in ("CIFAR10", "FEMNIST", "CIFAR100")
        for label in ATTACK_LABELS
    ]
    text = "\n".join(
        [
            "Table 2: Main Security Results across 8 attacks and 3 datasets.",
            *rows,
            "Table 3: Layer Contribution Analysis",
        ]
    )
    payload = parse_table2(text)
    table = payload["table_2_all_methods"]
    assert len(table) == 3
    assert sum(len(attacks) for attacks in table.values()) == 24
    assert table["CIFAR10"]["FreeRidingNT"]["VanillaFL"] == {"MA": 10.0}
    assert table["CIFAR100"]["Sybil"]["PoLBFL"] == {
        "MA": 60.0,
        "DR": 61.0,
        "FPR": 62.0,
    }
