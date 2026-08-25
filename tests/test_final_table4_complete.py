import json
from pathlib import Path

from experiments.final.aggregate_table4 import aggregate_table4
from experiments.final.run_table4_matrix import plan_table4_cells, table4_command
from scripts.extract_table4_targets import parse_table4


ROOT = Path(__file__).parents[1]


def test_table4_extractor_transcribes_standalone_and_prefilter_columns():
    rows = [
        " ALIE 68.2 45.2 7.5 73.5 85.6 3.5",
        " Free-riding 72.5 15.2 8.5 85.2 96.5 2.1",
        " ALIE 72.5 52.8 5.8 77.8 87.2 3.2",
        " Free-riding 74.5 18.5 6.5 86.0 97.0 2.0 4.3",
        " ALIE 70.2 48.5 6.5 75.5 86.0 3.4",
        " Free-riding 73.2 16.8 7.2 85.5 96.8 2.1",
    ]
    text = "\n".join(
        [
            "Table 4: PoL-BFL + Robust Aggregation on CIFAR-10.",
            *rows,
            "Table 5: Incentive Mechanism Comparison on CIFAR-10.",
        ]
    )
    table = parse_table4(text)["table_4_all_modes"]
    assert table["Krum"]["ALIE"]["Standalone"]["MA"] == 68.2
    assert table["Median"]["FreeRidingNT"]["PoLBFLPrefilter"]["DR"] == 96.8


def test_table4_matrix_plans_both_modes_and_matching_methods(tmp_path):
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    cells = plan_table4_cells(matrix)
    assert len(cells) == 3 * 2 * 2 * 3
    standalone = next(
        cell
        for cell in cells
        if cell.aggregation == "median" and cell.mode == "Standalone"
    )
    assert standalone.method == "Median"
    command = table4_command(
        standalone,
        python=Path("/python"),
        output=tmp_path / "out",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--method") + 1] == "Median"
    assert command[command.index("--composition-mode") + 1] == "Standalone"


def test_table4_aggregate_requires_all_modes_and_three_seeds():
    targets = json.loads(
        (ROOT / "config" / "paper_table4_all_modes.json").read_text(
            encoding="utf-8"
        )
    )
    aggregation_names = {
        "Krum": "krum",
        "TrimmedMean": "trimmed_mean",
        "Median": "median",
    }
    rows = []
    for aggregation, attacks in targets["table_4_all_modes"].items():
        for attack, modes in attacks.items():
            for mode, metrics in modes.items():
                for seed in (1337, 2026, 3817739):
                    rows.append(
                        {
                            "formal_accepted": True,
                            "study": "composability",
                            "aggregation_method": aggregation_names[aggregation],
                            "attack": attack,
                            "composition_mode": mode,
                            "method": (
                                "PoLBFL" if mode == "PoLBFLPrefilter" else aggregation
                            ),
                            "seed": seed,
                            **metrics,
                            "source_commit": "a" * 40,
                        }
                    )
    result = aggregate_table4(rows, targets)
    assert result["acceptance"]["passed"]
    assert len(result["provenance"]) == 12
