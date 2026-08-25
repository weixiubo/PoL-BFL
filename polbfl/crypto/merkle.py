"""SHA-256 Merkle commitments and inclusion proofs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence


def _decode_digest(value: str) -> bytes:
    if len(value) != 64:
        raise ValueError("SHA-256 digests must contain 64 hexadecimal characters")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid hexadecimal digest") from exc


def _parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


@dataclass(frozen=True)
class MerkleProofStep:
    sibling: str
    position: str

    def __post_init__(self) -> None:
        _decode_digest(self.sibling)
        if self.position not in {"left", "right"}:
            raise ValueError("Merkle sibling position must be 'left' or 'right'")

    def to_dict(self) -> dict[str, str]:
        return {"sibling": self.sibling, "position": self.position}

    def to_bytes(self) -> bytes:
        return (b"\x00" if self.position == "left" else b"\x01") + bytes.fromhex(self.sibling)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "MerkleProofStep":
        if len(payload) != 33 or payload[0] not in (0, 1):
            raise ValueError("binary Merkle step must contain direction and 32-byte sibling")
        return cls(payload[1:].hex(), "left" if payload[0] == 0 else "right")


def encode_merkle_proof(proof: Sequence[MerkleProofStep]) -> bytes:
    if len(proof) >= 2**16:
        raise ValueError("Merkle proof is too deep")
    return len(proof).to_bytes(2, "big") + b"".join(step.to_bytes() for step in proof)


def decode_merkle_proof(payload: bytes) -> tuple[MerkleProofStep, ...]:
    if len(payload) < 2:
        raise ValueError("binary Merkle proof is truncated")
    count = int.from_bytes(payload[:2], "big")
    if len(payload) != 2 + 33 * count:
        raise ValueError("binary Merkle proof length does not match its header")
    return tuple(
        MerkleProofStep.from_bytes(payload[offset : offset + 33])
        for offset in range(2, len(payload), 33)
    )


class MerkleTree:
    """Binary Merkle tree using duplicate-last padding at every odd level."""

    def __init__(self, leaves: Iterable[str]):
        decoded = [_decode_digest(leaf) for leaf in leaves]
        if not decoded:
            raise ValueError("a Merkle tree requires at least one leaf")
        self._levels: list[list[bytes]] = [decoded]
        current = decoded
        while len(current) > 1:
            padded = current if len(current) % 2 == 0 else current + [current[-1]]
            current = [_parent(padded[i], padded[i + 1]) for i in range(0, len(padded), 2)]
            self._levels.append(current)

    @property
    def root(self) -> str:
        return self._levels[-1][0].hex()

    @property
    def leaf_count(self) -> int:
        return len(self._levels[0])

    def proof(self, index: int) -> tuple[MerkleProofStep, ...]:
        if index < 0 or index >= self.leaf_count:
            raise IndexError("Merkle leaf index out of range")
        proof: list[MerkleProofStep] = []
        cursor = index
        for level in self._levels[:-1]:
            sibling_index = cursor - 1 if cursor % 2 else cursor + 1
            if sibling_index >= len(level):
                sibling_index = cursor
            position = "left" if sibling_index < cursor else "right"
            proof.append(MerkleProofStep(level[sibling_index].hex(), position))
            cursor //= 2
        return tuple(proof)

    @staticmethod
    def verify(leaf: str, proof: Sequence[MerkleProofStep | dict[str, str]], root: str) -> bool:
        try:
            current = _decode_digest(leaf)
            expected = _decode_digest(root)
            for raw_step in proof:
                step = raw_step if isinstance(raw_step, MerkleProofStep) else MerkleProofStep(**raw_step)
                sibling = _decode_digest(step.sibling)
                current = _parent(sibling, current) if step.position == "left" else _parent(current, sibling)
            return current == expected
        except (TypeError, ValueError):
            return False
