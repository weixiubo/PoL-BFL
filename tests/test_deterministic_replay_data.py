import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, CODE_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_utils = _load_module("rq_data_utils", "experiments/scripts/utils/data_utils.py")
pol_manager_mod = _load_module("pol_manager_mod", "client/pol/PoLManager.py")
models_mod = _load_module("rq_models", "experiments/scripts/utils/models.py")
verifier_adapter_mod = _load_module("verifier_adapter_mod", "server/pol/verifier_adapter.py")
verifier_node_mod = _load_module("verifier_node_mod", "server/committee/VerifierNode.py")


class CIFAR10(Dataset):
    def __init__(self, n=8):
        rng = np.random.default_rng(7)
        self.data = rng.integers(0, 255, size=(n, 32, 32, 3), dtype=np.uint8)
        self.targets = [int(i % 10) for i in range(n)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).permute(2, 0, 1).float() / 255.0, self.targets[idx]


def test_integrity_mode_enables_deterministic_cifar_wrapper(monkeypatch):
    monkeypatch.delenv("POL_DETERMINISTIC_AUG", raising=False)
    monkeypatch.setenv("POL_INTEGRITY", "1")

    loaders = data_utils.create_dataloaders([Subset(CIFAR10(), [0, 1, 2, 3])], batch_size=2, num_workers=0)

    assert type(loaders[0].dataset).__name__ == "SubsetDeterministicWrapper"


def test_deterministic_cifar_hash_is_stable_in_integrity_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("POL_DETERMINISTIC_AUG", raising=False)
    monkeypatch.setenv("POL_INTEGRITY", "1")

    loader = data_utils.create_dataloaders([Subset(CIFAR10(), [0, 1, 2, 3])], batch_size=2, num_workers=0)[0]
    manager = pol_manager_mod.PoLManager(
        client_id="hash_stability",
        save_dir=str(tmp_path),
        save_freq=1,
        save_to_disk=False,
    )
    manager.record_data_indices([0, 1, 2, 3, 0, 1])

    first = manager.compute_data_hash(loader.dataset)
    second = manager.compute_data_hash(loader.dataset)

    assert first == second


def test_fast_data_hash_does_not_replay_augmentation(monkeypatch, tmp_path):
    monkeypatch.delenv("POL_DETERMINISTIC_AUG", raising=False)
    monkeypatch.setenv("POL_INTEGRITY", "1")

    loader = data_utils.create_dataloaders([Subset(CIFAR10(), [0, 1, 2, 3])], batch_size=2, num_workers=0)[0]
    manager = pol_manager_mod.PoLManager(
        client_id="fast_hash",
        save_dir=str(tmp_path),
        save_freq=1,
        save_to_disk=False,
    )
    manager.record_data_indices([0, 1, 2, 3, 0, 1])

    loader.dataset.set_replay_context(round_num=0, epoch=0)
    first = manager.compute_data_hash(loader.dataset)
    loader.dataset.set_replay_context(round_num=99, epoch=4)
    second = manager.compute_data_hash(loader.dataset)

    assert hasattr(loader.dataset, "fast_data_hash_for_global_indices")
    assert first == second


def test_deterministic_cifar_pickle_uses_compact_partition(monkeypatch):
    monkeypatch.delenv("POL_DETERMINISTIC_AUG", raising=False)
    monkeypatch.setenv("POL_INTEGRITY", "1")

    loader = data_utils.create_dataloaders([Subset(CIFAR10(20), [0, 2, 4, 6])], batch_size=2, num_workers=0)[0]
    ds = loader.dataset
    ds.set_replay_context(round_num=3, epoch=1)
    expected_x, expected_y, expected_gidx = ds[1]
    expected_hash = ds.fast_data_hash_for_global_indices([0, 2, 4, 6, 0])

    buf = io.BytesIO()
    torch.save(ds, buf)
    buf.seek(0)
    try:
        restored = torch.load(buf, map_location="cpu", weights_only=False)
    except TypeError:
        restored = torch.load(buf, map_location="cpu")

    assert restored.subset is None
    assert restored.indices == [0, 2, 4, 6]
    assert getattr(restored, "_compact_data", None) is not None

    restored.set_replay_context(round_num=3, epoch=1)
    actual_x, actual_y, actual_gidx = restored[1]
    actual_hash = restored.fast_data_hash_for_global_indices([0, 2, 4, 6, 0])

    assert actual_gidx == expected_gidx
    assert actual_y == expected_y
    assert torch.equal(actual_x, expected_x)
    assert actual_hash == expected_hash


def test_strict_replay_payload_uses_model_spec_and_compact_dataset(monkeypatch):
    monkeypatch.delenv("POL_STRICT_REPLAY_SERIALIZE_MODEL", raising=False)
    monkeypatch.delenv("POL_DETERMINISTIC_AUG", raising=False)
    monkeypatch.setenv("POL_INTEGRITY", "1")

    loader = data_utils.create_dataloaders([Subset(CIFAR10(20), [0, 2, 4, 6])], batch_size=2, num_workers=0)[0]
    model = models_mod.create_model("ResNet18", num_classes=10, input_channels=3)
    adapter = verifier_adapter_mod.RemoteVerifierAdapter(["http://127.0.0.1:9"], mode="strict_replay")

    fields = adapter._strict_replay_fields(model, loader, torch.nn.CrossEntropyLoss())

    assert fields["model_spec"]["model_name"] == "ResNet18"
    assert fields["model_spec"]["num_classes"] == 10
    assert "ser_model" not in fields

    restored_dataset = verifier_node_mod._decode_torch_object(fields["ser_dataset"])
    assert restored_dataset.subset is None
    assert restored_dataset.indices == [0, 2, 4, 6]

    rebuilt = verifier_node_mod._build_model(fields)
    assert type(rebuilt).__name__ == "ResNet18"


def test_remote_response_compaction_keeps_only_required_optimizer_states(monkeypatch):
    monkeypatch.delenv("POL_COMPACT_REMOTE_RESPONSE", raising=False)
    adapter = verifier_adapter_mod.RemoteVerifierAdapter(["http://127.0.0.1:9"], mode="strict_replay")
    response = {
        "client_id": "client_x",
        "checkpoints": [
            {"data": {"model_state": {"w": torch.tensor([1.0])}, "optimizer_state": {"state": {0: "keep0"}}, "step": 1}},
            {"data": {"model_state": {"w": torch.tensor([2.0])}, "optimizer_state": {"state": {0: "drop1"}}, "step": 2}},
            {"data": {"model_state": {"w": torch.tensor([3.0])}, "optimizer_state": {"state": {0: "keep2"}}, "step": 3}},
            {"data": {"model_state": {"w": torch.tensor([4.0])}, "optimizer_state": {"state": {0: "drop3"}}, "step": 4}},
        ],
    }

    compact = adapter._compact_response_for_pairs(response, pair_indices=[0, 2])

    assert "optimizer_state" in compact["checkpoints"][0]["data"]
    assert "optimizer_state" not in compact["checkpoints"][1]["data"]
    assert "optimizer_state" in compact["checkpoints"][2]["data"]
    assert "optimizer_state" not in compact["checkpoints"][3]["data"]
    assert "optimizer_state" in response["checkpoints"][1]["data"]


def test_deterministic_cifar_aug_changes_by_replay_context(monkeypatch):
    monkeypatch.delenv("POL_DETERMINISTIC_AUG", raising=False)
    monkeypatch.setenv("POL_INTEGRITY", "1")

    loader = data_utils.create_dataloaders([Subset(CIFAR10(), [0, 1, 2, 3])], batch_size=2, num_workers=0)[0]
    ds = loader.dataset
    ds.set_replay_context(round_num=0, epoch=0)
    first, _, _ = ds[0]
    ds.set_replay_context(round_num=0, epoch=0)
    repeat, _, _ = ds[0]
    ds.set_replay_context(round_num=1, epoch=0)
    next_round, _, _ = ds[0]
    ds.set_replay_context(round_num=1, epoch=1)
    next_epoch, _, _ = ds[0]

    assert torch.equal(first, repeat)
    assert not torch.equal(first, next_round)
    assert not torch.equal(next_round, next_epoch)
