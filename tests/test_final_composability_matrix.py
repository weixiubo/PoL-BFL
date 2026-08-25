import json
from pathlib import Path

import pytest

from experiments.final.aggregate_composability import aggregate_composability_cells
from experiments.final import run_composability_matrix
from experiments.final.run_composability_matrix import composability_command, plan_composability_cells


ROOT = Path(__file__).parents[1]


def test_composability_matrix_and_three_seed_aggregate(tmp_path):
    matrix = json.loads((ROOT / "experiments" / "final" / "paper_matrix.json").read_text())
    cells = plan_composability_cells(matrix)
    assert len(cells) == 3 * 2 * 3
    command = composability_command(
        cells[0],
        python=Path("/python"),
        output=tmp_path / "out",
        data_root=tmp_path / "data",
        zk_build=tmp_path / "zk",
    )
    assert "--study" in command and "composability" in command
    rows = [
        {
            "formal_accepted": True,
            "study": "composability",
            "aggregation_method": "krum",
            "attack": "FreeRidingNT",
            "seed": seed,
            "MA": 86.0,
            "DR": 97.0,
            "FPR": 2.0,
            "source_commit": "a" * 40,
        }
        for seed in (1, 2, 3)
    ]
    targets = {
        "table_4_composability_cifar10": {
            "Krum": {"FreeRiding": {"MA": 85.2, "DR": 96.5, "FPR": 2.1}}
        }
    }
    assert aggregate_composability_cells(rows, targets)["acceptance"]["passed"]


def test_composability_plan_binds_source_and_execution_rejects_dirty_tree(
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
    monkeypatch.setattr(run_composability_matrix, "source_identity", lambda _root: clean)
    monkeypatch.setattr(
        run_composability_matrix.sys,
        "argv",
        ["run_composability_matrix.py", "--matrix", str(matrix), "--seed", "1337"],
    )
    run_composability_matrix.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["source"] == clean
    assert len(plan["cells"]) == 3 * 2

    monkeypatch.setattr(
        run_composability_matrix,
        "source_identity",
        lambda _root: clean | {"dirty": True},
    )
    monkeypatch.setattr(
        run_composability_matrix.sys,
        "argv",
        [
            "run_composability_matrix.py",
            "--matrix",
            str(matrix),
            "--seed",
            "1337",
            "--execute",
        ],
    )
    with pytest.raises(RuntimeError, match="clean, identified source"):
        run_composability_matrix.main()
