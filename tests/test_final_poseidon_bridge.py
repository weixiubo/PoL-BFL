from pathlib import Path

import pytest

from polbfl.zk import PoseidonBridge


ROOT = Path(__file__).parents[1]


def test_persistent_poseidon_bridge_matches_one_shot_commitments():
    if not (ROOT / "node_modules" / "circomlibjs").exists():
        pytest.skip("circomlibjs is not installed")
    operations = (
        {"kind": "fold2", "values": [1, 2, 3, 4], "initial": "9"},
        {"kind": "fold3", "rows": [[1, 2], [3, 4]], "initial": "9"},
        {
            "kind": "fold_pair_chunks",
            "rows": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "pairs_per_chunk": 2,
            "initial": "9",
        },
    )
    expected = PoseidonBridge().execute(operations)
    persistent = PoseidonBridge(persistent=True)
    try:
        assert persistent.execute(operations) == expected
        assert persistent.execute(operations) == expected
    finally:
        persistent.close()


def test_native_poseidon_bridge_command_matches_circomlib_when_built():
    native = ROOT / ".tools" / "poseidon-native" / "polbfl-poseidon-native"
    if not native.is_file() or not (ROOT / "node_modules" / "circomlibjs").exists():
        pytest.skip("native Poseidon helper or circomlibjs is not installed")
    operations = (
        {"kind": "fold2", "values": [-3, 0, 2**48 - 1], "initial": "9"},
        {"kind": "fold3", "rows": [[-1, 2], [3, -4]], "initial": "9"},
        {
            "kind": "fold_pair_chunks",
            "rows": [[index - 4, 2 * index + 1] for index in range(8)],
            "pairs_per_chunk": 4,
            "initial": "9",
        },
    )
    expected = PoseidonBridge().execute(operations)
    native_bridge = PoseidonBridge(native_binary=native, persistent=True)
    try:
        assert native_bridge.execute(operations) == expected
        assert native_bridge.execute(operations) == expected
    finally:
        native_bridge.close()
