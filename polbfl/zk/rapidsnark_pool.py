"""Thread-safe pool of persistent Rapidsnark prover objects."""

from __future__ import annotations

import ctypes
import json
import mmap
import queue
import threading
from pathlib import Path
from typing import Any


PROVER_OK = 0
PROVER_ERROR_SHORT_BUFFER = 2
_ERROR_BUFFER_SIZE = 8192


class RapidsnarkProverPool:
    """Reuse parsed proving keys through Rapidsnark's supported C API."""

    def __init__(self, library: str | Path, proving_key: str | Path, *, size: int):
        self.library_path = Path(library).resolve()
        self.proving_key = Path(proving_key).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(f"missing Rapidsnark shared library: {self.library_path}")
        if not self.proving_key.is_file():
            raise FileNotFoundError(f"missing proving key: {self.proving_key}")
        if int(size) <= 0:
            raise ValueError("persistent prover pool size must be positive")
        self.size = int(size)
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_abi()
        self._zkey_file = self.proving_key.open("rb")
        self._zkey_mmap = mmap.mmap(
            self._zkey_file.fileno(),
            0,
            access=mmap.ACCESS_COPY,
        )
        self._zkey_pointer = ctypes.c_void_p(
            ctypes.addressof(ctypes.c_char.from_buffer(self._zkey_mmap))
        )
        self._objects: list[ctypes.c_void_p] = []
        self._available: queue.Queue[ctypes.c_void_p] = queue.Queue(maxsize=self.size)
        self._close_lock = threading.Lock()
        self._closed = False
        self._proof_capacity = ctypes.c_ulonglong()
        self._library.groth16_proof_size(ctypes.byref(self._proof_capacity))
        self._public_capacity = ctypes.c_ulonglong()
        self._call_status(
            self._library.groth16_public_size_for_zkey_buf(
                self._zkey_pointer,
                len(self._zkey_mmap),
                ctypes.byref(self._public_capacity),
                self._error_buffer(),
                _ERROR_BUFFER_SIZE,
            ),
            "cannot determine Rapidsnark public-output size",
        )
        try:
            for _ in range(self.size):
                prover = ctypes.c_void_p()
                error = self._error_buffer()
                status = self._library.groth16_prover_create(
                    ctypes.byref(prover),
                    self._zkey_pointer,
                    len(self._zkey_mmap),
                    error,
                    _ERROR_BUFFER_SIZE,
                )
                self._call_status(status, "cannot initialize persistent Rapidsnark prover", error)
                if not prover.value:
                    raise RuntimeError("Rapidsnark returned a null prover object")
                self._objects.append(prover)
                self._available.put(prover)
        except Exception:
            self.close()
            raise

    def _configure_abi(self) -> None:
        unsigned = ctypes.c_ulonglong
        void_pointer = ctypes.c_void_p
        character_pointer = ctypes.c_char_p
        library = self._library
        library.groth16_public_size_for_zkey_buf.argtypes = (
            void_pointer,
            unsigned,
            ctypes.POINTER(unsigned),
            character_pointer,
            unsigned,
        )
        library.groth16_public_size_for_zkey_buf.restype = ctypes.c_int
        library.groth16_public_size_for_zkey_file.argtypes = (
            character_pointer,
            ctypes.POINTER(unsigned),
            character_pointer,
            unsigned,
        )
        library.groth16_public_size_for_zkey_file.restype = ctypes.c_int
        library.groth16_proof_size.argtypes = (ctypes.POINTER(unsigned),)
        library.groth16_proof_size.restype = None
        library.groth16_prover_create_zkey_file.argtypes = (
            ctypes.POINTER(void_pointer),
            character_pointer,
            character_pointer,
            unsigned,
        )
        library.groth16_prover_create_zkey_file.restype = ctypes.c_int
        library.groth16_prover_create.argtypes = (
            ctypes.POINTER(void_pointer),
            void_pointer,
            unsigned,
            character_pointer,
            unsigned,
        )
        library.groth16_prover_create.restype = ctypes.c_int
        library.groth16_prover_prove.argtypes = (
            void_pointer,
            void_pointer,
            unsigned,
            character_pointer,
            ctypes.POINTER(unsigned),
            character_pointer,
            ctypes.POINTER(unsigned),
            character_pointer,
            unsigned,
        )
        library.groth16_prover_prove.restype = ctypes.c_int
        library.groth16_prover_destroy.argtypes = (void_pointer,)
        library.groth16_prover_destroy.restype = None

    @staticmethod
    def _error_buffer():
        return ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)

    @staticmethod
    def _call_status(status: int, prefix: str, error=None) -> None:
        if int(status) == PROVER_OK:
            return
        detail = ""
        if error is not None:
            detail = bytes(error.value).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{prefix}: {detail or f'Rapidsnark status {status}'}")

    def prove(self, witness: str | Path) -> tuple[dict[str, Any], tuple[str, ...]]:
        if self._closed:
            raise RuntimeError("persistent Rapidsnark prover pool is closed")
        witness_path = Path(witness)
        payload = witness_path.read_bytes()
        witness_buffer = ctypes.create_string_buffer(payload)
        prover = self._available.get()
        try:
            proof_size = ctypes.c_ulonglong(self._proof_capacity.value)
            public_size = ctypes.c_ulonglong(self._public_capacity.value)
            proof_buffer = ctypes.create_string_buffer(proof_size.value)
            public_buffer = ctypes.create_string_buffer(public_size.value)
            error = self._error_buffer()
            status = self._library.groth16_prover_prove(
                prover,
                ctypes.cast(witness_buffer, ctypes.c_void_p),
                len(payload),
                proof_buffer,
                ctypes.byref(proof_size),
                public_buffer,
                ctypes.byref(public_size),
                error,
                _ERROR_BUFFER_SIZE,
            )
            if status == PROVER_ERROR_SHORT_BUFFER:
                proof_buffer = ctypes.create_string_buffer(proof_size.value)
                public_buffer = ctypes.create_string_buffer(public_size.value)
                error = self._error_buffer()
                status = self._library.groth16_prover_prove(
                    prover,
                    ctypes.cast(witness_buffer, ctypes.c_void_p),
                    len(payload),
                    proof_buffer,
                    ctypes.byref(proof_size),
                    public_buffer,
                    ctypes.byref(public_size),
                    error,
                    _ERROR_BUFFER_SIZE,
                )
            self._call_status(status, "persistent Rapidsnark proof failed", error)
            proof = json.loads(proof_buffer.raw[: proof_size.value].decode("utf-8"))
            public = tuple(
                str(value)
                for value in json.loads(public_buffer.raw[: public_size.value].decode("utf-8"))
            )
            return proof, public
        finally:
            self._available.put(prover)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            for prover in self._objects:
                if prover.value:
                    self._library.groth16_prover_destroy(prover)
            self._objects.clear()
            self._zkey_pointer = None
            zkey_mmap = getattr(self, "_zkey_mmap", None)
            if zkey_mmap is not None:
                zkey_mmap.close()
                self._zkey_mmap = None
            zkey_file = getattr(self, "_zkey_file", None)
            if zkey_file is not None:
                zkey_file.close()
                self._zkey_file = None

    def __del__(self):  # pragma: no cover - interpreter shutdown fallback
        try:
            self.close()
        except Exception:
            pass
