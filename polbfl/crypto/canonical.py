"""Canonical, cross-process hashing for protocol objects and model states."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _length_prefix(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def domain_hash(domain: str, *parts: bytes | bytearray | memoryview | str | int) -> str:
    """Return a domain-separated SHA-256 digest with unambiguous framing."""

    if not domain or not domain.isascii():
        raise ValueError("hash domain must be a non-empty ASCII string")
    digest = hashlib.sha256()
    digest.update(_length_prefix(domain.encode("ascii")))
    for part in parts:
        if isinstance(part, str):
            encoded = part.encode("utf-8")
        elif isinstance(part, int):
            if part < 0:
                raise ValueError("integer hash parts must be non-negative")
            width = max(1, (part.bit_length() + 7) // 8)
            encoded = part.to_bytes(width, "big")
        else:
            encoded = bytes(part)
        digest.update(_length_prefix(encoded))
    return digest.hexdigest()


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid protocol values")
        return {"$float_hex": value.hex()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$bytes_b64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Path):
        return {"$path": value.as_posix()}
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_json(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Sequence):
        return [_normalize_json(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize_json(value.to_dict())
    raise TypeError(f"unsupported canonical JSON type: {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize protocol metadata without platform-dependent float formatting."""

    return json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _array_payload(value: Any) -> tuple[str, tuple[int, ...], bytes] | None:
    """Return dtype, shape, and C-order bytes for torch/numpy-like arrays."""

    obj = value
    try:
        detach = getattr(obj, "detach", None)
        if callable(detach):
            obj = detach()
        cpu = getattr(obj, "cpu", None)
        if callable(cpu):
            obj = cpu()
        contiguous = getattr(obj, "contiguous", None)
        if callable(contiguous):
            obj = contiguous()
        numpy = getattr(obj, "numpy", None)
        if callable(numpy):
            obj = numpy()
    except Exception as exc:  # pragma: no cover - backend-specific failure
        raise ValueError(f"cannot materialize tensor for hashing: {exc}") from exc

    if not (hasattr(obj, "dtype") and hasattr(obj, "shape") and hasattr(obj, "tobytes")):
        return None
    dtype = str(obj.dtype)
    shape = tuple(int(dim) for dim in obj.shape)
    payload = obj.tobytes(order="C")
    return dtype, shape, payload


def hash_state_dict(state: Mapping[str, Any]) -> str:
    """Hash all state entries including names, dtypes, shapes, and raw bytes."""

    digest = hashlib.sha256()
    digest.update(_length_prefix(b"POLBFL_MODEL_STATE_V1"))
    for key in sorted(state):
        key_bytes = str(key).encode("utf-8")
        array = _array_payload(state[key])
        if array is None:
            type_bytes = b"canonical-json"
            shape_bytes = b"[]"
            payload = canonical_json_bytes(state[key])
        else:
            dtype, shape, payload = array
            type_bytes = dtype.encode("ascii")
            shape_bytes = canonical_json_bytes(list(shape))
        digest.update(_length_prefix(key_bytes))
        digest.update(_length_prefix(type_bytes))
        digest.update(_length_prefix(shape_bytes))
        digest.update(_length_prefix(payload))
    return digest.hexdigest()


def hash_object(value: Any) -> str:
    """Hash nested protocol objects, including tensors inside optimizer state."""

    digest = hashlib.sha256()
    digest.update(_length_prefix(b"POLBFL_NESTED_OBJECT_V1"))

    def visit(item: Any) -> None:
        array = _array_payload(item)
        if array is not None:
            dtype, shape, payload = array
            digest.update(b"A")
            digest.update(_length_prefix(dtype.encode("ascii")))
            digest.update(_length_prefix(canonical_json_bytes(list(shape))))
            digest.update(_length_prefix(payload))
            return
        if item is None:
            digest.update(b"N")
        elif isinstance(item, bool):
            digest.update(b"B1" if item else b"B0")
        elif isinstance(item, int):
            digest.update(b"I")
            digest.update(_length_prefix(str(item).encode("ascii")))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("non-finite floats are not valid protocol values")
            digest.update(b"F")
            digest.update(_length_prefix(item.hex().encode("ascii")))
        elif isinstance(item, str):
            digest.update(b"S")
            digest.update(_length_prefix(item.encode("utf-8")))
        elif isinstance(item, (bytes, bytearray, memoryview)):
            digest.update(b"Y")
            digest.update(_length_prefix(bytes(item)))
        elif isinstance(item, Mapping):
            digest.update(b"M")
            digest.update(struct.pack(">Q", len(item)))
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                visit(str(key))
                visit(child)
        elif isinstance(item, (list, tuple)):
            digest.update(b"L")
            digest.update(struct.pack(">Q", len(item)))
            for child in item:
                visit(child)
        else:
            raise TypeError(f"unsupported nested hash type: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def hash_batch(data: Any, labels: Any) -> str:
    """Commit to the exact ordered private examples and labels in a batch."""

    data_array = _array_payload(data)
    label_array = _array_payload(labels)
    data_payload = (
        canonical_json_bytes(data)
        if data_array is None
        else canonical_json_bytes({"dtype": data_array[0], "shape": data_array[1]})
        + data_array[2]
    )
    label_payload = (
        canonical_json_bytes(labels)
        if label_array is None
        else canonical_json_bytes({"dtype": label_array[0], "shape": label_array[1]})
        + label_array[2]
    )
    return domain_hash("POLBFL_PRIVATE_BATCH_V1", data_payload, label_payload)


def hash_batch_indices(indices: Sequence[int]) -> str:
    ordered = [int(index) for index in indices]
    if any(index < 0 for index in ordered):
        raise ValueError("batch indices must be non-negative")
    return domain_hash("POLBFL_BATCH_INDICES_V1", canonical_json_bytes(ordered))
