import pytest

torch = pytest.importorskip("torch")

from experiments.final.data_attacks import DeterministicLabelPoison


class IndexedToy(torch.utils.data.Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return torch.tensor([float(index)]), index % 5, 100 + index


def test_label_poisoning_is_deterministic_index_bound_and_never_keeps_poisoned_label():
    poisoned = DeterministicLabelPoison(IndexedToy(), num_classes=5, poison_ratio=1.0, seed=7)
    first = [poisoned[index][1] for index in range(len(poisoned))]
    second = [poisoned[index][1] for index in range(len(poisoned))]
    assert first == second
    assert all(label != index % 5 for index, label in enumerate(first))
    assert [poisoned[index][2] for index in range(20)] == list(range(100, 120))
