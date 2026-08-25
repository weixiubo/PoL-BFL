"""Small fail-closed bridge to the Circomlib Poseidon implementation."""

from __future__ import annotations

import json
import selectors
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


class PoseidonBridge:
    def __init__(
        self,
        *,
        node_binary: str = "node",
        script: str | Path | None = None,
        native_binary: str | Path | None = None,
        timeout_seconds: int = 120,
        persistent: bool = False,
    ):
        root = Path(__file__).resolve().parents[2]
        self.node_binary = str(node_binary)
        self.script = Path(script) if script is not None else root / "circuits" / "final" / "poseidon_bridge.cjs"
        self.native_binary = (
            None if native_binary is None else Path(native_binary).resolve()
        )
        self.timeout_seconds = int(timeout_seconds)
        self.persistent = bool(persistent)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        if self.native_binary is not None and not self.native_binary.is_file():
            raise FileNotFoundError(f"missing native Poseidon helper: {self.native_binary}")
        if self.native_binary is None and not self.script.is_file():
            raise FileNotFoundError(f"missing Poseidon bridge: {self.script}")
        if self.timeout_seconds <= 0:
            raise ValueError("Poseidon bridge timeout must be positive")
        self._command = (
            [str(self.native_binary)]
            if self.native_binary is not None
            else [self.node_binary, str(self.script)]
        )
        if self.persistent:
            self._process = subprocess.Popen(
                [*self._command, "--stream"],
                cwd=self.script.parents[2],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

    def execute(self, operations: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        if not operations:
            return ()
        request = json.dumps({"operations": list(operations)}, separators=(",", ":"))
        if self._process is not None:
            with self._lock:
                if self._process.poll() is not None or self._process.stdin is None or self._process.stdout is None:
                    raise RuntimeError("persistent Poseidon bridge is not running")
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
                selector = selectors.DefaultSelector()
                selector.register(self._process.stdout, selectors.EVENT_READ)
                ready = selector.select(self.timeout_seconds)
                selector.close()
                if not ready:
                    self._process.terminate()
                    raise TimeoutError("persistent Poseidon bridge timed out")
                output = self._process.stdout.readline()
            if not output:
                error = ""
                if self._process.stderr is not None:
                    error = self._process.stderr.read().strip()
                raise RuntimeError(f"Poseidon commitment failed: {error or 'bridge closed'}")
        else:
            process = subprocess.run(
                self._command,
                cwd=self.script.parents[2],
                input=request,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if process.returncode != 0:
                message = process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}"
                raise RuntimeError(f"Poseidon commitment failed: {message}")
            output = process.stdout
        try:
            result = json.loads(output)
            if "error" in result:
                raise RuntimeError(str(result["error"]))
            values = tuple(str(value) for value in result["results"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Poseidon bridge returned malformed output") from exc
        if len(values) != len(operations) or any(not value.isdigit() for value in values):
            raise RuntimeError("Poseidon bridge returned an invalid result set")
        return values

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)

    def __del__(self):  # pragma: no cover - interpreter shutdown fallback
        try:
            self.close()
        except Exception:
            pass

    def fold2(self, values: Sequence[int | str], *, initial: int | str = 0) -> str:
        return self.execute(({"kind": "fold2", "values": list(values), "initial": str(initial)},))[0]

    def fold3(self, rows: Sequence[Sequence[int | str]], *, initial: int | str = 0) -> str:
        return self.execute(({"kind": "fold3", "rows": [list(row) for row in rows], "initial": str(initial)},))[0]

    def fold_pair_chunks(
        self,
        rows: Sequence[Sequence[int | str]],
        *,
        pairs_per_chunk: int,
        initial: int | str = 0,
    ) -> str:
        return self.execute(
            (
                {
                    "kind": "fold_pair_chunks",
                    "rows": [list(row) for row in rows],
                    "pairs_per_chunk": int(pairs_per_chunk),
                    "initial": str(initial),
                },
            )
        )[0]
