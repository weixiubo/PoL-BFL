from pathlib import Path

from experiments.final.supervision import supervised_gpu_command


def test_supervised_gpu_command_binds_run_directory_and_child_command(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    wrapper = root / "scripts" / "gpu_idle_supervisor.py"
    wrapper.write_text("# wrapper\n", encoding="utf-8")
    run_dir = tmp_path / "result"
    command = supervised_gpu_command(
        ["/child-python", "-m", "formal.cell"],
        python=Path("/python"),
        root=root,
        run_dir=run_dir,
    )
    assert command[:3] == ["/python", "-u", str(wrapper.resolve())]
    assert command[command.index("--run-dir") + 1] == str(run_dir.resolve())
    assert command[command.index("--log") + 1] == str(run_dir.resolve() / "supervisor.log")
    assert command[-4:] == ["--", "/child-python", "-m", "formal.cell"]
