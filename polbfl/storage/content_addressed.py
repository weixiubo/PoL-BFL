"""Atomic private storage for off-chain PoL evidence blobs."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_PACK_MAGIC = b"POLBFL_EVIDENCE_PACK_V1\x00"
_PACK_REFERENCE_PREFIX = "pack-v1:"


@dataclass(frozen=True)
class BlobRef:
    digest: str
    size: int
    relative_path: str

    def to_dict(self) -> dict[str, object]:
        return {"digest": self.digest, "size": self.size, "relative_path": self.relative_path}


class ContentAddressedStore:
    def __init__(
        self,
        root: str | Path,
        *,
        packed: bool = False,
        durable: bool = True,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.packed = bool(packed)
        self.durable = bool(durable)
        self._pack_stream: BinaryIO | None = None
        self._pack_temporary: Path | None = None
        self._pack_references: dict[str, BlobRef] = {}
        self._pack_finalized = False
        try:
            self.root.chmod(0o700)
        except OSError:  # pragma: no cover - platform-specific permission model
            pass
        if self.packed:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".evidence-pack-",
                suffix=".tmp",
                dir=self.root,
            )
            self._pack_temporary = Path(temporary_name)
            self._pack_stream = os.fdopen(descriptor, "w+b")
            self._pack_stream.write(_PACK_MAGIC)

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid SHA-256 digest")
        return self.root / digest[:2] / digest[2:]

    def put(self, payload: bytes) -> BlobRef:
        digest = hashlib.sha256(payload).hexdigest()
        if self.packed:
            if self._pack_stream is None or self._pack_finalized:
                raise RuntimeError("packed evidence store is already finalized")
            existing = self._pack_references.get(digest)
            if existing is not None:
                if self.get(existing) != payload:
                    raise RuntimeError("content-address collision or storage corruption")
                return existing
            offset = self._pack_stream.tell()
            self._pack_stream.write(payload)
            reference = BlobRef(
                digest=digest,
                size=len(payload),
                relative_path=f"{_PACK_REFERENCE_PREFIX}{offset}",
            )
            self._pack_references[digest] = reference
            return reference
        destination = self._path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists():
            existing = destination.read_bytes()
            if existing != payload:
                raise RuntimeError("content-address collision or storage corruption")
        else:
            fd, temporary_name = tempfile.mkstemp(prefix=".polbfl-", dir=destination.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(0o600)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        relative = destination.relative_to(self.root).as_posix()
        return BlobRef(digest=digest, size=len(payload), relative_path=relative)

    def get(self, reference: BlobRef) -> bytes:
        if reference.relative_path.startswith(_PACK_REFERENCE_PREFIX):
            offset_text = reference.relative_path[len(_PACK_REFERENCE_PREFIX) :]
            if not offset_text.isdigit():
                raise ValueError("invalid packed evidence offset")
            offset = int(offset_text)
            if offset < len(_PACK_MAGIC) or reference.size < 0:
                raise ValueError("invalid packed evidence range")
            if self._pack_stream is not None and not self._pack_finalized:
                current = self._pack_stream.tell()
                self._pack_stream.flush()
                self._pack_stream.seek(0)
                magic = self._pack_stream.read(len(_PACK_MAGIC))
                self._pack_stream.seek(offset)
                payload = self._pack_stream.read(reference.size)
                self._pack_stream.seek(current)
            else:
                pack_path = self.root / "evidence.pack"
                with pack_path.open("rb") as stream:
                    magic = stream.read(len(_PACK_MAGIC))
                    stream.seek(offset)
                    payload = stream.read(reference.size)
            if magic != _PACK_MAGIC:
                raise ValueError("packed evidence header is invalid")
        else:
            path = self._path(reference.digest)
            payload = path.read_bytes()
        if len(payload) != reference.size or hashlib.sha256(payload).hexdigest() != reference.digest:
            raise ValueError("stored evidence does not match its content address")
        return payload

    def has(self, reference: BlobRef) -> bool:
        try:
            self.get(reference)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def finalize(self) -> None:
        """Atomically publish a packed store after one durability barrier."""

        if not self.packed or self._pack_finalized:
            return
        if self._pack_stream is None or self._pack_temporary is None:
            raise RuntimeError("packed evidence store is not writable")
        stream = self._pack_stream
        temporary = self._pack_temporary
        stream.flush()
        if self.durable:
            os.fsync(stream.fileno())
        stream.close()
        temporary.chmod(0o600)
        os.replace(temporary, self.root / "evidence.pack")
        if self.durable and hasattr(os, "O_DIRECTORY"):
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        self._pack_stream = None
        self._pack_temporary = None
        self._pack_finalized = True

    def __del__(self):  # pragma: no cover - interpreter shutdown fallback
        stream = getattr(self, "_pack_stream", None)
        temporary = getattr(self, "_pack_temporary", None)
        try:
            if stream is not None:
                stream.close()
            if temporary is not None and temporary.exists():
                temporary.unlink()
        except Exception:
            pass
