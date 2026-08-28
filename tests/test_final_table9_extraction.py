import json
from pathlib import Path

import pytest

from experiments.final.run_security_cell import load_acceptance_targets
from scripts.extract_table9_targets import parse_table9


ROOT = Path(__file__).parents[1]


def _table9_text() -> str:
    return "\n".join(
        [
            "Table 9: Non-IID Sensitivity using Dirichlet distribution alpha.",
            " No Attack MA (%) 82.5 88.2 89.5 91.2",
            " Free-riding DR (%) 94.2 96.5 97.2 98.0",
            " ALIE DR (%) 80.5 85.6 86.8 88.2",
            " FPR (%) 4.8 2.5 2.0 1.5",
            " No Attack MA (%) 85.2 89.5 91.2 92.8",
            " Free-riding DR (%) 95.5 96.8 97.5 98.2",
            " ALIE DR (%) 82.8 87.5 88.5 89.5",
            " FPR (%) 4.2 2.2 1.8 1.2",
            " No Attack MA (%) 52.5 58.2 60.5 62.5",
            " Free-riding DR (%) 92.5 95.8 96.5 97.2",
            " ALIE DR (%) 78.2 83.5 85.2 86.8",
            " FPR (%) 5.5 3.2 2.5 2.0",
            "A.1.2 Gas Price Impact.",
        ]
    )


def test_table9_extractor_reads_all_datasets_metrics_and_partitions():
    table = parse_table9(_table9_text())["table_9_noniid"]
    assert table["CIFAR10"]["0.1"] == {
        "NoAttackMA": 82.5,
        "FreeRidingDR": 94.2,
        "ALIEDR": 80.5,
        "FPR": 4.8,
    }
    assert table["FEMNIST"]["IID"]["NoAttackMA"] == 92.8
    assert table["CIFAR100"]["0.5"]["ALIEDR"] == 83.5


def test_table9_extractor_fails_closed_on_an_incomplete_metric():
    with pytest.raises(ValueError, match="four partition values"):
        parse_table9(_table9_text().replace("5.5 3.2 2.5 2.0", "5.5 3.2 2.5"))


def test_table9_dedicated_targets_match_main_targets_and_runtime_loader():
    dedicated = json.loads(
        (ROOT / "config" / "paper_table9_noniid.json").read_text(encoding="utf-8")
    )
    main = json.loads(
        (ROOT / "config" / "paper_targets.json").read_text(encoding="utf-8")
    )
    assert dedicated["table_9_noniid"] == main["table_9_noniid"]
    assert load_acceptance_targets(ROOT)["table_9_noniid"] == dedicated["table_9_noniid"]
