"""Persistent dual-GPU ICICLE-Snark worker pool."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class _IcicleWorker:
    device: int
    process: subprocess.Popen[str]
    output: queue.Queue[str | None]
    history: list[str] = field(default_factory=list)
    commands_completed: int = 0


class IcicleProverPool:
    """Serialize proofs per GPU while sharing ICICLE's parsed-key cache."""

    def __init__(
        self,
        binary: str | Path,
        *,
        backend_directory: str | Path,
        library_directories: Sequence[str | Path],
        devices: Sequence[int],
        timeout_seconds: int = 300,
        max_commands_per_worker: int = 128,
    ):
        self.binary = Path(binary).resolve()
        self.backend_directory = Path(backend_directory).resolve()
        self.library_directories = tuple(
            Path(directory).resolve() for directory in library_directories
        )
        self.devices = tuple(int(device) for device in devices)
        self.timeout_seconds = int(timeout_seconds)
        self.max_commands_per_worker = int(max_commands_per_worker)
        if not self.binary.is_file():
            raise FileNotFoundError(f"missing ICICLE-Snark worker: {self.binary}")
        if not self.backend_directory.is_dir():
            raise FileNotFoundError(
                f"missing ICICLE backend directory: {self.backend_directory}"
            )
        if not self.library_directories or any(
            not directory.is_dir() for directory in self.library_directories
        ):
            raise FileNotFoundError("ICICLE library directories are incomplete")
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("ICICLE devices must be non-empty and unique")
        if self.timeout_seconds <= 0:
            raise ValueError("ICICLE proof timeout must be positive")
        if self.max_commands_per_worker <= 0:
            raise ValueError("ICICLE worker command limit must be positive")
        self._workers: list[_IcicleWorker] = []
        self._available: queue.Queue[_IcicleWorker] = queue.Queue(
            maxsize=len(self.devices)
        )
        self._closed = False
        self._worker_lock = threading.Lock()
        try:
            for device in self.devices:
                worker = self._start_worker(device)
                self._workers.append(worker)
                self._available.put(worker)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_command_path(path: Path) -> None:
        if any(character.isspace() for character in str(path)):
            raise ValueError("ICICLE worker paths cannot contain whitespace")

    def _start_worker(self, device: int) -> _IcicleWorker:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        environment["ICICLE_BACKEND_INSTALL_DIR"] = str(self.backend_directory)
        existing = [
            value
            for value in environment.get("LD_LIBRARY_PATH", "").split(":")
            if value
        ]
        environment["LD_LIBRARY_PATH"] = ":".join(
            [*(str(path) for path in self.library_directories), *existing]
        )
        process = subprocess.Popen(
            [str(self.binary)],
            cwd=self.binary.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        output: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            if process.stdout is None:
                output.put(None)
                return
            for line in process.stdout:
                output.put(line.rstrip("\n"))
            output.put(None)

        threading.Thread(
            target=read_output,
            name=f"icicle-output-gpu-{device}",
            daemon=True,
        ).start()
        return _IcicleWorker(device=device, process=process, output=output)

    @staticmethod
    def _stop_worker(worker: _IcicleWorker) -> None:
        process = worker.process
        if process.poll() is not None:
            return
        try:
            if process.stdin is None:
                raise BrokenPipeError("ICICLE worker stdin is unavailable")
            process.stdin.write("exit\n")
            process.stdin.flush()
            process.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    def _replace_worker(self, worker: _IcicleWorker) -> _IcicleWorker:
        self._stop_worker(worker)
        replacement = self._start_worker(worker.device)
        with self._worker_lock:
            try:
                index = self._workers.index(worker)
            except ValueError:
                self._stop_worker(replacement)
                raise RuntimeError("ICICLE worker disappeared during recycling")
            self._workers[index] = replacement
        return replacement

    @property
    def worker_pids(self) -> tuple[int, ...]:
        if self._closed:
            return ()
        with self._worker_lock:
            return tuple(worker.process.pid for worker in self._workers)

    def prove(
        self,
        *,
        witness: str | Path,
        proving_key: str | Path,
        proof: str | Path,
        public: str | Path,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if self._closed:
            raise RuntimeError("ICICLE prover pool is closed")
        paths = tuple(
            Path(path).resolve()
            for path in (witness, proving_key, proof, public)
        )
        witness_path, proving_key_path, proof_path, public_path = paths
        for path in paths:
            self._validate_command_path(path)
        if not witness_path.is_file() or not proving_key_path.is_file():
            raise FileNotFoundError("ICICLE proof input is missing")
        worker = self._available.get()
        try:
            process = worker.process
            if process.poll() is not None or process.stdin is None:
                raise RuntimeError(
                    f"ICICLE worker on GPU {worker.device} exited before proof"
                )
            command = (
                f"prove --system groth16 --witness {witness_path} "
                f"--zkey {proving_key_path} --proof {proof_path} "
                f"--public {public_path} --device CUDA\n"
            )
            try:
                process.stdin.write(command)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(
                    f"ICICLE worker on GPU {worker.device} rejected the command "
                    f"(exit={process.poll()}): "
                    + "\n".join(worker.history[-20:])
                ) from exc
            command_output: list[str] = []
            while True:
                try:
                    line = worker.output.get(timeout=self.timeout_seconds)
                except queue.Empty as exc:
                    process.terminate()
                    raise TimeoutError(
                        f"ICICLE proof timed out on GPU {worker.device}"
                    ) from exc
                if line is None:
                    raise RuntimeError(
                        "ICICLE worker closed during proof: "
                        + "\n".join(command_output[-20:])
                    )
                command_output.append(line)
                worker.history.append(line)
                del worker.history[:-100]
                if "COMMAND_COMPLETED" in line:
                    break
            if not proof_path.is_file() or not public_path.is_file():
                raise RuntimeError(
                    "ICICLE worker completed without proof outputs: "
                    + "\n".join(command_output[-20:])
                )
            proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
            public_payload = tuple(
                str(value)
                for value in json.loads(public_path.read_text(encoding="utf-8"))
            )
            worker.commands_completed += 1
            return proof_payload, public_payload
        finally:
            if worker.process.poll() is None:
                if worker.commands_completed >= self.max_commands_per_worker:
                    worker = self._replace_worker(worker)
                self._available.put(worker)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._worker_lock:
            workers = tuple(self._workers)
            self._workers.clear()
        for worker in workers:
            self._stop_worker(worker)

    def __del__(self):  # pragma: no cover - interpreter shutdown fallback
        try:
            self.close()
        except Exception:
            pass
