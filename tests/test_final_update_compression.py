import numpy as np
import pytest

from polbfl.communication import (
    CompressedUpdate,
    compress_update,
    compress_update_3bit,
    compress_update_4bit,
    decompress_update,
    decompress_update_3bit,
    decompress_update_4bit,
)


def test_int8_transport_roundtrip_error_and_size():
    rng = np.random.default_rng(41)
    update = {
        "layer.weight": rng.normal(size=(128, 64)).astype(np.float32),
        "layer.bias": rng.normal(size=(128,)).astype(np.float32),
        "counter": np.asarray(1, dtype=np.int64),
    }
    compressed = compress_update(update)
    encoded = compressed.to_bytes()
    assert len(encoded) == compressed.wire_bytes
    compressed = CompressedUpdate.from_bytes(encoded)
    restored = decompress_update(compressed, update)
    assert "counter" not in restored
    for name, values in restored.items():
        tolerance = compressed.tensors[name].scale / 2 + 1e-6
        assert np.max(np.abs(values - update[name])) <= tolerance
    raw_bytes = update["layer.weight"].nbytes + update["layer.bias"].nbytes
    assert compressed.wire_bytes < raw_bytes / 3


def test_compression_rejects_nonfinite_updates():
    with pytest.raises(ValueError, match="non-finite"):
        compress_update({"w": np.asarray([np.nan], dtype=np.float32)})


def test_packed_three_bit_transport_is_real_decodable_and_compact():
    rng = np.random.default_rng(43)
    update = {"w": rng.normal(size=10003).astype(np.float32)}
    payload = compress_update_3bit(update)
    restored = decompress_update_3bit(payload, update)
    maximum = float(np.max(np.abs(update["w"])))
    assert np.max(np.abs(restored["w"] - update["w"])) <= maximum / 6 + 1e-5
    assert len(payload) < update["w"].nbytes / 4


def test_packed_four_bit_transport_improves_error_with_compact_wire_size():
    rng = np.random.default_rng(47)
    update = {"w": rng.normal(size=10003).astype(np.float32)}
    payload = compress_update_4bit(update)
    restored = decompress_update_4bit(payload, update)
    maximum = float(np.max(np.abs(update["w"])))
    assert np.max(np.abs(restored["w"] - update["w"])) <= maximum / 14 + 1e-5
    assert len(payload) < update["w"].nbytes / 3
