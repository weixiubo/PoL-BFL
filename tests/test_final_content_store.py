from dataclasses import replace

import pytest

from polbfl.storage import ContentAddressedStore


def test_packed_content_store_is_atomic_reopenable_and_content_addressed(tmp_path):
    root = tmp_path / "evidence"
    store = ContentAddressedStore(root, packed=True)
    first = store.put(b"first payload")
    second = store.put(b"second payload")
    assert store.put(b"first payload") == first
    assert store.get(first) == b"first payload"
    assert not (root / "evidence.pack").exists()

    store.finalize()
    assert (root / "evidence.pack").is_file()
    reopened = ContentAddressedStore(root)
    assert reopened.get(first) == b"first payload"
    assert reopened.get(second) == b"second payload"
    assert reopened.has(first)

    corrupted = replace(first, digest="0" * 64)
    with pytest.raises(ValueError):
        reopened.get(corrupted)


def test_unfinalized_packed_store_does_not_publish(tmp_path):
    root = tmp_path / "evidence"
    store = ContentAddressedStore(root, packed=True)
    store.put(b"private")
    temporary = store._pack_temporary
    assert temporary is not None and temporary.is_file()
    store.__del__()
    assert not temporary.exists()
    assert not (root / "evidence.pack").exists()
