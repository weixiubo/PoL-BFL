import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts.gpu_idle_supervisor import (
    foreign_gpu_pids,
    gpu_compute_pids,
    gpu_usage,
    training_rounds_complete,
)


def test_gpu_usage_parser_returns_utilization_and_memory_pairs():
    completed = mock.Mock(stdout="0, 12\n9, 1024\n")
    with mock.patch("subprocess.run", return_value=completed):
        assert gpu_usage() == [(0, 12), (9, 1024)]


def test_gpu_compute_pid_parser_rejects_non_numeric_rows():
    completed = mock.Mock(stdout="123\n456\n")
    with mock.patch("subprocess.run", return_value=completed):
        assert gpu_compute_pids() == {123, 456}
    completed = mock.Mock(stdout="N/A\n")
    with mock.patch("subprocess.run", return_value=completed):
        with pytest.raises(RuntimeError, match="invalid GPU compute PID"):
            gpu_compute_pids()


def test_foreign_gpu_pid_detection_uses_the_child_process_group():
    with (
        mock.patch("scripts.gpu_idle_supervisor.gpu_compute_pids", return_value={11, 12, 13}),
        mock.patch("os.getpgid", side_effect=lambda pid: {11: 99, 12: 100, 13: 99}[pid]),
    ):
        assert foreign_gpu_pids(99) == {12}


def test_training_round_completion_requires_contiguous_rows_and_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_parameters": {"rounds": 2}}),
        encoding="utf-8",
    )
    (run_dir / "rounds.jsonl").write_text(
        '{"round":0}\n{"round":1}\n',
        encoding="utf-8",
    )
    assert training_rounds_complete(run_dir) is False
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    assert training_rounds_complete(run_dir) is True
    (run_dir / "rounds.jsonl").write_text(
        '{"round":0}\n{"round":2}\n',
        encoding="utf-8",
    )
    assert training_rounds_complete(run_dir) is False


@pytest.mark.skipif(os.name != "posix", reason="process-group monitoring is POSIX-only")
def test_supervisor_ignores_foreign_gpu_after_all_training_rounds(tmp_path):
    test_bin = tmp_path / "bin"
    test_bin.mkdir()
    test_nvidia = test_bin / "nvidia-smi"
    test_nvidia.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *query-gpu=*) printf '0, 0\\n0, 0\\n' ;;\n"
        "  *query-compute-apps=pid*) printf '%s\\n' \"$FOREIGN_GPU_PID\" ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    test_nvidia.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_parameters": {"rounds": 1}}),
        encoding="utf-8",
    )
    (run_dir / "rounds.jsonl").write_text('{"round":0}\n', encoding="utf-8")
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    child = tmp_path / "child.py"
    child.write_text(
        "import json, os, pathlib, time\n"
        "root = pathlib.Path(os.environ['TEST_RUN_DIR'])\n"
        "time.sleep(0.5)\n"
        "(root / 'result.json').write_text(json.dumps({'passed': True}))\n",
        encoding="utf-8",
    )
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(test_bin) + os.pathsep + environment["PATH"],
            "FOREIGN_GPU_PID": str(os.getpid()),
            "TEST_RUN_DIR": str(run_dir),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "gpu_idle_supervisor.py"),
            "--run-dir",
            str(run_dir),
            "--log",
            str(run_dir / "supervisor.log"),
            "--settle-seconds",
            "1",
            "--poll-seconds",
            "1",
            "--monitor-seconds",
            "0.1",
            "--max-restarts",
            "0",
            "--",
            sys.executable,
            str(child),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0
    events = [
        json.loads(line)
        for line in (run_dir / "supervisor.log").read_text().splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event") == "foreign_gpu_ignored_after_training"
        for event in events
    )
    assert not any(event.get("event") == "foreign_gpu_preemption" for event in events)
    assert (run_dir / "result.json").is_file()


@pytest.mark.skipif(os.name != "posix", reason="process-group preemption is POSIX-only")
def test_supervisor_preempts_foreign_gpu_and_resets_an_uncheckpointed_first_round(
    tmp_path,
):
    test_bin = tmp_path / "bin"
    test_bin.mkdir()
    test_nvidia = test_bin / "nvidia-smi"
    test_nvidia.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *query-gpu=*) printf '0, 0\\n0, 0\\n' ;;\n"
        "  *query-compute-apps=pid*) printf '%s\\n' \"$FOREIGN_GPU_PID\" ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    test_nvidia.chmod(0o755)
    run_dir = tmp_path / "run"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, pathlib, time\n"
        "root = pathlib.Path(os.environ['TEST_RUN_DIR'])\n"
        "(root / 'scratch' / 'round-0').mkdir(parents=True, exist_ok=True)\n"
        "(root / 'rounds.jsonl').write_text('{\\\"round\\\":0}\\n')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(test_bin) + os.pathsep + environment["PATH"],
            "FOREIGN_GPU_PID": str(os.getpid()),
            "TEST_RUN_DIR": str(run_dir),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "gpu_idle_supervisor.py"),
            "--run-dir",
            str(run_dir),
            "--log",
            str(run_dir / "supervisor.log"),
            "--settle-seconds",
            "1",
            "--poll-seconds",
            "1",
            "--monitor-seconds",
            "0.1",
            "--max-restarts",
            "0",
            "--",
            sys.executable,
            str(child),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode != 0
    events = [
        json.loads(line)
        for line in (run_dir / "supervisor.log").read_text().splitlines()
        if line.strip()
    ]
    assert any(event.get("event") == "foreign_gpu_preemption" for event in events)
    assert any(event.get("event") == "preempted_fresh_run_reset" for event in events)
    raw_path = run_dir / "rounds.jsonl"
    assert not raw_path.exists() or raw_path.read_text() == ""
    scratch = run_dir / "scratch"
    assert not scratch.exists() or not any(scratch.iterdir())
