"""Atomic reconciliation between raw round logs and model checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def align_round_log(path: Path, *, checkpoint_round: int) -> dict[str, Any]:
    path = path.resolve()
    expected_count = int(checkpoint_round) + 1
    if expected_count < 0:
        raise ValueError("checkpoint round cannot be below -1")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    if any(int(row.get("round", -1)) != index for index, row in enumerate(rows)):
        raise ValueError("raw round log is not contiguous from round zero")
    if len(lines) < expected_count:
        raise ValueError("raw round log ends before the checkpoint")
    dropped = lines[expected_count:]
    event = {
        "checkpoint_round": int(checkpoint_round),
        "retained_rounds": expected_count,
        "dropped_rounds": len(dropped),
        "dropped_sha256": [hashlib.sha256(line.encode("utf-8")).hexdigest() for line in dropped],
    }
    if not dropped:
        return event
    payload = ("\n".join(lines[:expected_count]) + ("\n" if expected_count else "")).encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=".round-recovery-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return event


def discard_uncommitted_scratch(
    run_dir: Path,
    *,
    next_round: int,
) -> dict[str, Any]:
    """Remove only scratch rounds at or beyond the next durable round."""

    run_dir = run_dir.resolve()
    scratch_root = (run_dir / "scratch").resolve()
    next_round = int(next_round)
    if next_round < 0:
        raise ValueError("next durable round cannot be negative")
    removed: list[str] = []
    retained: list[str] = []
    if not scratch_root.exists():
        return {
            "next_round": next_round,
            "removed_scratch_rounds": removed,
            "retained_scratch_rounds": retained,
        }
    if not scratch_root.is_dir() or scratch_root.parent != run_dir:
        raise RuntimeError("scratch root escaped the formal run directory")
    for child in sorted(scratch_root.iterdir()):
        resolved = child.resolve()
        if (
            resolved.parent != scratch_root
            or not child.is_dir()
            or not child.name.startswith("round-")
            or not child.name.removeprefix("round-").isdigit()
        ):
            raise RuntimeError(f"unexpected scratch entry: {child}")
        round_number = int(child.name.removeprefix("round-"))
        if round_number < next_round:
            retained.append(child.name)
            continue
        shutil.rmtree(child)
        removed.append(child.name)
    return {
        "next_round": next_round,
        "removed_scratch_rounds": removed,
        "retained_scratch_rounds": retained,
    }


def reset_preempted_fresh_run(run_dir: Path) -> dict[str, Any]:
    """Reset an interrupted round-zero attempt that has no checkpoint."""

    run_dir = run_dir.resolve()
    checkpoint = run_dir / "checkpoint.pt"
    result = run_dir / "result.json"
    if checkpoint.exists() or result.exists():
        raise RuntimeError("fresh-run reset is only valid without a checkpoint or result")
    raw_path = run_dir / "rounds.jsonl"
    round_recovery = (
        align_round_log(raw_path, checkpoint_round=-1)
        if raw_path.is_file()
        else {
            "checkpoint_round": -1,
            "retained_rounds": 0,
            "dropped_rounds": 0,
            "dropped_sha256": [],
        }
    )
    scratch_recovery = discard_uncommitted_scratch(run_dir, next_round=0)
    return {
        "kind": "preempted_fresh_run_reset",
        "round_log": round_recovery,
        "scratch": scratch_recovery,
    }
