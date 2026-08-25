#!/usr/bin/env python3
"""Run/resume a formal cell only while the two reference GPUs are available."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final.recovery import reset_preempted_fresh_run


def gpu_usage() -> list[tuple[int, int]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    ).stdout.strip().splitlines()
    return [tuple(int(part.strip()) for part in line.split(",")) for line in output]


def gpu_compute_pids() -> set[int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    ).stdout.strip().splitlines()
    pids: set[int] = set()
    for line in output:
        value = line.strip()
        if not value:
            continue
        try:
            pids.add(int(value))
        except ValueError as exc:
            raise RuntimeError(f"invalid GPU compute PID: {value!r}") from exc
    return pids


def foreign_gpu_pids(process_group: int) -> set[int]:
    foreign: set[int] = set()
    for pid in gpu_compute_pids():
        try:
            observed_group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            foreign.add(pid)
            continue
        if observed_group != int(process_group):
            foreign.add(pid)
    return foreign


def wait_until_idle(*, settle_seconds: int, poll_seconds: int) -> None:
    idle_since = None
    while True:
        usage = gpu_usage()
        idle = len(usage) == 2 and all(utilization <= 10 and memory <= 1024 for utilization, memory in usage)
        if idle:
            idle_since = time.monotonic() if idle_since is None else idle_since
            if time.monotonic() - idle_since >= settle_seconds:
                return
        else:
            idle_since = None
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--monitor-seconds", type=float, default=1.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    child = None

    def stop(_signum, _frame):
        nonlocal child
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    restarts = 0
    while True:
        if (args.run_dir / "result.json").is_file():
            return
        wait_until_idle(
            settle_seconds=max(1, args.settle_seconds),
            poll_seconds=max(1, args.poll_seconds),
        )
        if (args.run_dir / "result.json").is_file():
            return
        resumed = (args.run_dir / "checkpoint.pt").is_file()
        launch = list(command)
        if resumed and "--resume" not in launch:
            launch.append("--resume")
        with args.log.open("a", encoding="utf-8") as stream:
            launch_offset = stream.tell()
            stream.write(
                json.dumps(
                    {
                        "event": "launch",
                        "at_ns": time.time_ns(),
                        "resume": resumed,
                        "gpu_usage": gpu_usage(),
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
            child_group = child.pid
            preempted = False
            while child.poll() is None:
                try:
                    foreign = foreign_gpu_pids(child_group)
                    monitor_error = None
                except Exception as exc:
                    foreign = set()
                    monitor_error = f"{type(exc).__name__}:{exc}"
                if foreign or monitor_error is not None:
                    event = {
                        "event": "foreign_gpu_preemption",
                        "at_ns": time.time_ns(),
                        "child_group": child_group,
                        "foreign_gpu_pids": sorted(foreign),
                        "monitor_error": monitor_error,
                        "gpu_usage": gpu_usage(),
                    }
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.killpg(child_group, signal.SIGKILL)
                    preempted = True
                    break
                time.sleep(max(0.1, args.monitor_seconds))
            returncode = child.wait()
            try:
                os.killpg(child_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            child = None
            stream.write(
                json.dumps(
                    {
                        "event": "exit",
                        "at_ns": time.time_ns(),
                        "returncode": returncode,
                        "preempted": preempted,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        if returncode == 0 and (args.run_dir / "result.json").is_file():
            return
        if preempted and not (args.run_dir / "checkpoint.pt").is_file():
            reset_event = reset_preempted_fresh_run(args.run_dir)
            with args.log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "preempted_fresh_run_reset",
                            "at_ns": time.time_ns(),
                            "recovery": reset_event,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
        restarts += 1
        if restarts > args.max_restarts:
            raise SystemExit(returncode or 1)
        with args.log.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(launch_offset)
            tail = stream.read()[-20000:]
        retryable = preempted or any(
            marker in tail
            for marker in (
                "CUDA out of memory",
                "CUDA error",
                "ICICLE worker",
                "BrokenPipeError",
                "formal run requires idle reference GPUs",
                "Connection reset",
                "Connection timed out",
                "FOREIGN_GPU_PREEMPTION",
            )
        )
        if not retryable:
            raise SystemExit(returncode or 1)


if __name__ == "__main__":
    main()
