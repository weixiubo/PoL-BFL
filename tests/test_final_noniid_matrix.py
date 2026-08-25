import json
from pathlib import Path

import pytest

from experiments.final import run_noniid_matrix
from experiments.final.run_noniid_matrix import noniid_command, plan_noniid_cells


ROOT = Path(__file__).parents[1]


def test_noniid_matrix_enumerates_paper_cells_and_no_attack_population(tmp_path):
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(encoding="utf-8")
    )
    cells = plan_noniid_cells(matrix)
    assert len(cells) == 3 * 4 * 3 * 3
    no_attack = next(cell for cell in cells if cell.attack == "NoAttack" and cell.partition_label == "0.1")
    command = noniid_command(
        no_attack,
        python=Path("/python"),
        output=tmp_path / "out",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert command[command.index("--num-malicious") + 1] == "0"
    assert command[command.index("--partition-alpha") + 1] == "0.1"
    iid = next(cell for cell in cells if cell.attack == "ALIE" and cell.partition_label == "IID")
    iid_command = noniid_command(
        iid,
        python=Path("/python"),
        output=tmp_path / "iid",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert "--partition-alpha" not in iid_command


def test_noniid_matrix_plan_binds_source_and_execution_rejects_dirty_tree(
    monkeypatch,
    capsys,
):
    matrix = ROOT / "experiments" / "final" / "paper_matrix.json"
    clean = {
        "commit": "a" * 40,
        "dirty": False,
        "deployment_archive": False,
        "status_sha256": "b" * 64,
        "diff_sha256": "c" * 64,
    }
    monkeypatch.setattr(run_noniid_matrix, "source_identity", lambda _root: clean)
    monkeypatch.setattr(
        run_noniid_matrix.sys,
        "argv",
        ["run_noniid_matrix.py", "--matrix", str(matrix), "--seed", "1337"],
    )
    run_noniid_matrix.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["source"] == clean
    assert len(plan["cells"]) == 3 * 4 * 3

    monkeypatch.setattr(
        run_noniid_matrix,
        "source_identity",
        lambda _root: clean | {"dirty": True},
    )
    monkeypatch.setattr(
        run_noniid_matrix.sys,
        "argv",
        [
            "run_noniid_matrix.py",
            "--matrix",
            str(matrix),
            "--seed",
            "1337",
            "--execute",
        ],
    )
    with pytest.raises(RuntimeError, match="clean, identified source"):
        run_noniid_matrix.main()
