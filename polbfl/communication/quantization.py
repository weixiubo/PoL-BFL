"""Deterministic symmetric int8 and packed low-bit update transports."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class QuantizedTensor:
    shape: tuple[int, ...]
    scale: float
    payload: bytes

    @property
    def wire_bytes(self) -> int:
        return 1 + 4 * len(self.shape) + 4 + len(self.payload)


@dataclass(frozen=True)
class CompressedUpdate:
    tensors: Mapping[str, QuantizedTensor]

    @property
    def wire_bytes(self) -> int:
        return 6 + sum(2 + len(name.encode("utf-8")) + tensor.wire_bytes for name, tensor in self.tensors.items())

    def to_bytes(self) -> bytes:
        if len(self.tensors) >= 2**16:
            raise ValueError("too many compressed tensors")
        output = bytearray(b"PBU1" + len(self.tensors).to_bytes(2, "big"))
        for name, tensor in self.tensors.items():
            encoded_name = name.encode("utf-8")
            if len(encoded_name) >= 2**16 or len(tensor.shape) >= 2**8:
                raise ValueError("compressed tensor name or rank is too large")
            output.extend(len(encoded_name).to_bytes(2, "big"))
            output.extend(encoded_name)
            output.append(len(tensor.shape))
            for dimension in tensor.shape:
                if dimension < 0 or dimension >= 2**32:
                    raise ValueError("compressed tensor dimension is invalid")
                output.extend(int(dimension).to_bytes(4, "big"))
            output.extend(struct.pack(">f", float(tensor.scale)))
            output.extend(tensor.payload)
        if len(output) != self.wire_bytes:
            raise AssertionError("compressed update size accounting mismatch")
        return bytes(output)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "CompressedUpdate":
        view = memoryview(payload)
        if len(view) < 6 or bytes(view[:4]) != b"PBU1":
            raise ValueError("compressed update header is invalid")
        count = int.from_bytes(view[4:6], "big")
        cursor = 6
        tensors: dict[str, QuantizedTensor] = {}
        for _ in range(count):
            if cursor + 2 > len(view):
                raise ValueError("compressed update is truncated")
            name_size = int.from_bytes(view[cursor : cursor + 2], "big")
            cursor += 2
            if cursor + name_size + 1 > len(view):
                raise ValueError("compressed tensor name is truncated")
            name = bytes(view[cursor : cursor + name_size]).decode("utf-8")
            cursor += name_size
            rank = int(view[cursor])
            cursor += 1
            if cursor + 4 * rank + 4 > len(view):
                raise ValueError("compressed tensor metadata is truncated")
            shape = tuple(
                int.from_bytes(view[cursor + 4 * index : cursor + 4 * index + 4], "big")
                for index in range(rank)
            )
            cursor += 4 * rank
            scale = struct.unpack(">f", view[cursor : cursor + 4])[0]
            cursor += 4
            size = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if cursor + size > len(view):
                raise ValueError("compressed tensor values are truncated")
            values = bytes(view[cursor : cursor + size])
            cursor += size
            if name in tensors:
                raise ValueError("compressed update contains a duplicate tensor")
            tensors[name] = QuantizedTensor(shape, scale, values)
        if cursor != len(view):
            raise ValueError("compressed update has trailing data")
        return cls(tensors)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compress_update(update: Mapping[str, Any]) -> CompressedUpdate:
    tensors: dict[str, QuantizedTensor] = {}
    for name in sorted(update):
        array = _numpy(update[name])
        if not np.issubdtype(array.dtype, np.floating):
            continue
        values = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("cannot compress a non-finite update")
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        scale = maximum / 127.0 if maximum > 0 else 1.0
        quantized = np.rint(values / scale).clip(-127, 127).astype(np.int8)
        tensors[str(name)] = QuantizedTensor(tuple(values.shape), scale, quantized.tobytes(order="C"))
    if not tensors:
        raise ValueError("compressed update has no floating tensors")
    return CompressedUpdate(tensors)


def decompress_update(compressed: CompressedUpdate, template: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, tensor in compressed.tensors.items():
        if name not in template:
            raise ValueError(f"compressed tensor is absent from template: {name}")
        quantized = np.frombuffer(tensor.payload, dtype=np.int8)
        expected = int(np.prod(tensor.shape, dtype=np.int64)) if tensor.shape else 1
        if quantized.size != expected:
            raise ValueError("compressed tensor payload length is invalid")
        values = (quantized.astype(np.float32) * np.float32(tensor.scale)).reshape(tensor.shape)
        original = template[name]
        if hasattr(original, "detach"):
            import torch

            result[name] = torch.as_tensor(values, dtype=original.dtype, device=original.device)
        else:
            result[name] = values.astype(np.asarray(original).dtype, copy=False)
    return result


def _pack_3bit(values: np.ndarray) -> bytes:
    encoded = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(encoded > 7):
        raise ValueError("3-bit value is out of range")
    padding = (-encoded.size) % 8
    if padding:
        encoded = np.pad(encoded, (0, padding))
    rows = encoded.reshape(-1, 8).astype(np.uint32)
    words = (
        rows[:, 0]
        | (rows[:, 1] << 3)
        | (rows[:, 2] << 6)
        | (rows[:, 3] << 9)
        | (rows[:, 4] << 12)
        | (rows[:, 5] << 15)
        | (rows[:, 6] << 18)
        | (rows[:, 7] << 21)
    )
    output = np.empty((words.size, 3), dtype=np.uint8)
    output[:, 0] = words & 0xFF
    output[:, 1] = (words >> 8) & 0xFF
    output[:, 2] = (words >> 16) & 0xFF
    return output.tobytes(order="C")


def _unpack_3bit(payload: bytes, count: int) -> np.ndarray:
    expected = ((int(count) + 7) // 8) * 3
    if len(payload) != expected:
        raise ValueError("3-bit tensor payload length is invalid")
    octets = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    words = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
    values = np.empty((words.size, 8), dtype=np.uint8)
    for offset in range(8):
        values[:, offset] = (words >> (3 * offset)) & 0x7
    return values.reshape(-1)[:count]


def _pack_4bit(values: np.ndarray) -> bytes:
    encoded = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(encoded > 15):
        raise ValueError("4-bit value is out of range")
    if encoded.size % 2:
        encoded = np.pad(encoded, (0, 1))
    pairs = encoded.reshape(-1, 2)
    return (pairs[:, 0] | (pairs[:, 1] << 4)).tobytes(order="C")


def _unpack_4bit(payload: bytes, count: int) -> np.ndarray:
    expected = (int(count) + 1) // 2
    if len(payload) != expected:
        raise ValueError("4-bit tensor payload length is invalid")
    octets = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty((octets.size, 2), dtype=np.uint8)
    values[:, 0] = octets & 0xF
    values[:, 1] = octets >> 4
    return values.reshape(-1)[:count]


def compress_update_3bit(update: Mapping[str, Any], *, zlib_level: int = 1) -> bytes:
    """Return a real packed 3-bit wire payload, compressed as one frame."""

    entries = []
    for name in sorted(update):
        array = _numpy(update[name])
        if not np.issubdtype(array.dtype, np.floating):
            continue
        values = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("cannot compress a non-finite update")
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        scale = maximum / 3.0 if maximum > 0 else 1.0
        levels = np.rint(values / scale).clip(-3, 3).astype(np.int8)
        entries.append((str(name), tuple(values.shape), scale, _pack_3bit(levels + 3)))
    if not entries or len(entries) >= 2**16:
        raise ValueError("3-bit update has no tensors or too many tensors")
    raw = bytearray(b"PBT3" + len(entries).to_bytes(2, "big"))
    for name, shape, scale, packed in entries:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) >= 2**16 or len(shape) >= 2**8:
            raise ValueError("3-bit tensor name or rank is too large")
        raw.extend(len(name_bytes).to_bytes(2, "big"))
        raw.extend(name_bytes)
        raw.append(len(shape))
        for dimension in shape:
            raw.extend(int(dimension).to_bytes(4, "big"))
        raw.extend(struct.pack(">f", scale))
        raw.extend(packed)
    return b"PBZ3" + zlib.compress(bytes(raw), level=int(zlib_level))


def decompress_update_3bit(payload: bytes, template: Mapping[str, Any]) -> dict[str, Any]:
    if len(payload) < 5 or payload[:4] != b"PBZ3":
        raise ValueError("3-bit update envelope is invalid")
    raw = memoryview(zlib.decompress(payload[4:]))
    if len(raw) < 6 or bytes(raw[:4]) != b"PBT3":
        raise ValueError("3-bit update frame is invalid")
    count = int.from_bytes(raw[4:6], "big")
    cursor = 6
    result = {}
    for _ in range(count):
        if cursor + 2 > len(raw):
            raise ValueError("3-bit update is truncated")
        name_size = int.from_bytes(raw[cursor : cursor + 2], "big")
        cursor += 2
        if cursor + name_size + 1 > len(raw):
            raise ValueError("3-bit tensor name is truncated")
        name = bytes(raw[cursor : cursor + name_size]).decode("utf-8")
        cursor += name_size
        rank = int(raw[cursor])
        cursor += 1
        if cursor + 4 * rank + 4 > len(raw):
            raise ValueError("3-bit tensor metadata is truncated")
        shape = tuple(
            int.from_bytes(raw[cursor + 4 * index : cursor + 4 * index + 4], "big")
            for index in range(rank)
        )
        cursor += 4 * rank
        scale = struct.unpack(">f", raw[cursor : cursor + 4])[0]
        cursor += 4
        value_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        packed_size = ((value_count + 7) // 8) * 3
        if cursor + packed_size > len(raw):
            raise ValueError("3-bit tensor values are truncated")
        levels = _unpack_3bit(bytes(raw[cursor : cursor + packed_size]), value_count)
        cursor += packed_size
        if name in result or name not in template:
            raise ValueError("3-bit tensor is duplicate or absent from template")
        values = ((levels.astype(np.int16) - 3).astype(np.float32) * np.float32(scale)).reshape(shape)
        original = template[name]
        if hasattr(original, "detach"):
            import torch

            result[name] = torch.as_tensor(values, dtype=original.dtype, device=original.device)
        else:
            result[name] = values.astype(np.asarray(original).dtype, copy=False)
    if cursor != len(raw):
        raise ValueError("3-bit update has trailing data")
    return result


def compress_update_4bit(update: Mapping[str, Any], *, zlib_level: int = 1) -> bytes:
    """Return a deterministic packed signed 4-bit update frame."""

    entries = []
    for name in sorted(update):
        array = _numpy(update[name])
        if not np.issubdtype(array.dtype, np.floating):
            continue
        values = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("cannot compress a non-finite update")
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        scale = maximum / 7.0 if maximum > 0 else 1.0
        levels = np.rint(values / scale).clip(-7, 7).astype(np.int8)
        entries.append((str(name), tuple(values.shape), scale, _pack_4bit(levels + 7)))
    if not entries or len(entries) >= 2**16:
        raise ValueError("4-bit update has no tensors or too many tensors")
    raw = bytearray(b"PBT4" + len(entries).to_bytes(2, "big"))
    for name, shape, scale, packed in entries:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) >= 2**16 or len(shape) >= 2**8:
            raise ValueError("4-bit tensor name or rank is too large")
        raw.extend(len(name_bytes).to_bytes(2, "big"))
        raw.extend(name_bytes)
        raw.append(len(shape))
        for dimension in shape:
            raw.extend(int(dimension).to_bytes(4, "big"))
        raw.extend(struct.pack(">f", scale))
        raw.extend(packed)
    return b"PBZ4" + zlib.compress(bytes(raw), level=int(zlib_level))


def decompress_update_4bit(payload: bytes, template: Mapping[str, Any]) -> dict[str, Any]:
    if len(payload) < 5 or payload[:4] != b"PBZ4":
        raise ValueError("4-bit update envelope is invalid")
    raw = memoryview(zlib.decompress(payload[4:]))
    if len(raw) < 6 or bytes(raw[:4]) != b"PBT4":
        raise ValueError("4-bit update frame is invalid")
    count = int.from_bytes(raw[4:6], "big")
    cursor = 6
    result = {}
    for _ in range(count):
        if cursor + 2 > len(raw):
            raise ValueError("4-bit update is truncated")
        name_size = int.from_bytes(raw[cursor : cursor + 2], "big")
        cursor += 2
        if cursor + name_size + 1 > len(raw):
            raise ValueError("4-bit tensor name is truncated")
        name = bytes(raw[cursor : cursor + name_size]).decode("utf-8")
        cursor += name_size
        rank = int(raw[cursor])
        cursor += 1
        if cursor + 4 * rank + 4 > len(raw):
            raise ValueError("4-bit tensor metadata is truncated")
        shape = tuple(
            int.from_bytes(raw[cursor + 4 * index : cursor + 4 * index + 4], "big")
            for index in range(rank)
        )
        cursor += 4 * rank
        scale = struct.unpack(">f", raw[cursor : cursor + 4])[0]
        cursor += 4
        value_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        packed_size = (value_count + 1) // 2
        if cursor + packed_size > len(raw):
            raise ValueError("4-bit tensor values are truncated")
        levels = _unpack_4bit(bytes(raw[cursor : cursor + packed_size]), value_count)
        cursor += packed_size
        if name in result or name not in template:
            raise ValueError("4-bit tensor is duplicate or absent from template")
        values = (
            (levels.astype(np.int16) - 7).astype(np.float32)
            * np.float32(scale)
        ).reshape(shape)
        original = template[name]
        if hasattr(original, "detach"):
            import torch

            result[name] = torch.as_tensor(values, dtype=original.dtype, device=original.device)
        else:
            result[name] = values.astype(np.asarray(original).dtype, copy=False)
    if cursor != len(raw):
        raise ValueError("4-bit update has trailing data")
    return result
