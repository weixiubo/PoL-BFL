#!/usr/bin/env python3
"""Resume a formal command only while no external CUDA process is present."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def compute_pids() -> frozenset[int]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    return frozenset(
        int(line.strip())
        for line in process.stdout.splitlines()
        if line.strip().isdigit()
    )


def external_compute_pids(pids: frozenset[int], *, owned_process_group: int) -> frozenset[int]:
    external = set()
    for pid in pids:
        try:
            if os.getpgid(pid) != owned_process_group:
                external.add(pid)
        except ProcessLookupError:
            continue
    return frozenset(external)


def wait_for_exclusive_idle(*, settle_seconds: int, poll_seconds: int) -> None:
    idle_since = None
    while True:
        if not compute_pids():
            idle_since = time.monotonic() if idle_since is None else idle_since
            if time.monotonic() - idle_since >= settle_seconds:
                return
        else:
            idle_since = None
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--completion-file", type=Path)
    parser.add_argument("--manage-child-resume", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--max-external-restarts", type=int, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    completion_file = (
        args.run_dir / "result.json"
        if args.completion_file is None
        else args.completion_file.resolve()
    )
    child: subprocess.Popen[str] | None = None

    def stop(_signum, _frame) -> None:
        nonlocal child
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    external_restarts = 0
    while True:
        if completion_file.is_file():
            return
        wait_for_exclusive_idle(
            settle_seconds=max(1, args.settle_seconds),
            poll_seconds=max(1, args.poll_seconds),
        )
        if completion_file.is_file():
            return
        launch = list(command)
        resumed = bool(
            args.manage_child_resume
            and (args.run_dir / "checkpoint.pt").is_file()
        )
        if resumed and "--resume" not in launch:
            launch.append("--resume")
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "launch",
                        "at_ns": time.time_ns(),
                        "resume": resumed,
                        "command": launch,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            child = subprocess.Popen(
                launch,
                cwd=Path.cwd(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                text=True,
                start_new_session=True,
            )
            interrupted_by = frozenset()
            while child.poll() is None:
                interrupted_by = external_compute_pids(
                    compute_pids(),
                    owned_process_group=child.pid,
                )
                if interrupted_by:
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(max(1, args.poll_seconds))
            returncode = child.wait()
            child = None
            stream.write(
                json.dumps(
                    {
                        "event": "exit",
                        "at_ns": time.time_ns(),
                        "returncode": returncode,
                        "external_pids": sorted(interrupted_by),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if returncode == 0 and completion_file.is_file():
            return
        if not interrupted_by:
            raise SystemExit(returncode or 1)
        external_restarts += 1
        if external_restarts > args.max_external_restarts:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
