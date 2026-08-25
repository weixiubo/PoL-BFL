import json
from pathlib import Path

import pytest

from experiments.final import run_matrix
from experiments.final.run_matrix import MatrixCell, cell_command, plan_cells


ROOT = Path(__file__).parents[1]


def test_final_matrix_enumerates_all_paper_security_cells_and_formal_command(tmp_path):
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(encoding="utf-8")
    )
    cells = plan_cells(matrix)
    assert len(cells) == 3 * 8 * 6 * 3
    assert len({cell.run_id for cell in cells}) == len(cells)
    selected = plan_cells(
        matrix,
        datasets=["CIFAR10"],
        attacks=["FreeRidingNT"],
        methods=["PoLBFL"],
        seeds=[1337],
    )
    assert selected == (
        MatrixCell("CIFAR10", "FreeRidingNT", "PoLBFL", 1337, "formal-cifar10-freeridingnt-polbfl-s1337"),
    )
    command = cell_command(
        selected[0],
        python=Path("/python"),
        output=tmp_path / "result",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
        resume=True,
    )
    assert "--process-training" in command
    assert command[command.index("--method") + 1] == "PoLBFL"
    assert command[command.index("--train-processes-per-gpu") + 1] == "8"
    assert command[command.index("--proof-workers") + 1] == "8"
    assert command[-1] == "--resume"


def test_final_matrix_rejects_unknown_filters():
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="unknown values"):
        plan_cells(matrix, datasets=["Unknown"])


def test_final_matrix_catalogs_every_final_paper_study():
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(encoding="utf-8")
    )
    assert set(matrix["studies"]) == {
        "table_2_main_security",
        "table_3_layer_contribution",
        "table_4_composability",
        "table_5_incentive_effectiveness",
        "table_6_client_profit",
        "table_7_system_overhead",
        "table_8_scalability",
        "table_9_noniid",
        "table_10_adaptive",
        "table_11_cross_hardware",
        "table_12_zk_cost",
        "table_13_gas",
        "figure_2_convergence",
        "figure_3_reputation_evolution",
        "figure_4_spot_check_sensitivity",
        "figure_5_gas_price_stress",
        "figure_6_sybil_scalability",
    }


def test_final_matrix_execution_rejects_unidentified_source(monkeypatch):
    matrix = ROOT / "experiments" / "final" / "paper_matrix.json"
    monkeypatch.setattr(
        run_matrix,
        "source_identity",
        lambda _root: {
            "commit": None,
            "dirty": False,
            "deployment_archive": False,
            "status_sha256": "b" * 64,
            "diff_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        run_matrix.sys,
        "argv",
        [
            "run_matrix.py",
            "--matrix",
            str(matrix),
            "--dataset",
            "CIFAR10",
            "--attack",
            "FreeRidingNT",
            "--method",
            "PoLBFL",
            "--seed",
            "1337",
            "--execute",
        ],
    )
    with pytest.raises(RuntimeError, match="clean, identified source tree"):
        run_matrix.main()
