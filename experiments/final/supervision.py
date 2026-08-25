"""Mandatory GPU-exclusivity wrapper for formal experiment cells."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def supervised_gpu_command(
    command: Sequence[str],
    *,
    python: Path,
    root: Path,
    run_dir: Path,
) -> list[str]:
    run_dir = run_dir.resolve()
    return [
        str(python.resolve()),
        "-u",
        str((root / "scripts" / "gpu_idle_supervisor.py").resolve()),
        "--run-dir",
        str(run_dir),
        "--log",
        str(run_dir / "supervisor.log"),
        "--",
        *map(str, command),
    ]
