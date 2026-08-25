"""Canonical 192-byte BN254 Groth16 proof transport encoding."""

from __future__ import annotations

from typing import Any, Mapping


BN254_BASE_FIELD = 21888242871839275222246405745257275088696311157297823662689037894645226208583
PARITY_FLAG = 1 << 255
COORDINATE_MASK = (1 << 255) - 1


def _integer(value: Any) -> int:
    integer = int(value)
    if not 0 <= integer < BN254_BASE_FIELD:
        raise ValueError("Groth16 coordinate is outside the BN254 base field")
    return integer


def _compress_g1(point: Any) -> bytes:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError("Groth16 G1 point is malformed")
    x = _integer(point[0])
    y = _integer(point[1])
    if y * y % BN254_BASE_FIELD != (x * x % BN254_BASE_FIELD * x + 3) % BN254_BASE_FIELD:
        raise ValueError("Groth16 G1 point is not on BN254")
    encoded = x | (PARITY_FLAG if y & 1 else 0)
    return encoded.to_bytes(32, "big")


def _decompress_g1(payload: bytes) -> list[str]:
    if len(payload) != 32:
        raise ValueError("compressed G1 point must contain 32 bytes")
    encoded = int.from_bytes(payload, "big")
    parity = 1 if encoded & PARITY_FLAG else 0
    x = encoded & COORDINATE_MASK
    if x >= BN254_BASE_FIELD:
        raise ValueError("compressed G1 x-coordinate is outside BN254")
    value = (x * x % BN254_BASE_FIELD * x + 3) % BN254_BASE_FIELD
    y = pow(value, (BN254_BASE_FIELD + 1) // 4, BN254_BASE_FIELD)
    if y * y % BN254_BASE_FIELD != value:
        raise ValueError("compressed G1 point has no BN254 square root")
    if y & 1 != parity:
        y = BN254_BASE_FIELD - y
    return [str(x), str(y), "1"]


def encode_groth16_proof(proof: Mapping[str, Any]) -> bytes:
    """Encode compressed G1 A/C and uncompressed G2 B as exactly 192 bytes."""

    try:
        a = _compress_g1(proof["pi_a"])
        c = _compress_g1(proof["pi_c"])
        b = proof["pi_b"]
        if not isinstance(b, (list, tuple)) or len(b) < 2:
            raise ValueError("Groth16 G2 point is malformed")
        limbs = (
            _integer(b[0][0]),
            _integer(b[0][1]),
            _integer(b[1][0]),
            _integer(b[1][1]),
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Groth16 proof is malformed") from exc
    payload = a + b"".join(value.to_bytes(32, "big") for value in limbs) + c
    if len(payload) != 192:  # pragma: no cover - structural invariant
        raise AssertionError("canonical Groth16 proof encoding is not 192 bytes")
    return payload


def decode_groth16_proof(payload: bytes) -> dict[str, Any]:
    if len(payload) != 192:
        raise ValueError("canonical Groth16 proof must contain exactly 192 bytes")
    a = _decompress_g1(payload[:32])
    limbs = [
        str(int.from_bytes(payload[offset : offset + 32], "big"))
        for offset in range(32, 160, 32)
    ]
    if any(int(value) >= BN254_BASE_FIELD for value in limbs):
        raise ValueError("encoded G2 coordinate is outside BN254")
    c = _decompress_g1(payload[160:])
    return {
        "pi_a": a,
        "pi_b": [[limbs[0], limbs[1]], [limbs[2], limbs[3]], ["1", "0"]],
        "pi_c": c,
        "protocol": "groth16",
        "curve": "bn128",
    }
