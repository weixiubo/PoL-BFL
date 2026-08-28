import random

import numpy as np
import torch
from torch.utils.data import TensorDataset

from dataset.DatasetSpliter import DatasetSpliter


def _dataset(size=101):
    features = torch.arange(size, dtype=torch.float32).reshape(-1, 1)
    labels = torch.arange(size) % 7
    return TensorDataset(features, labels)


def _assert_exact_partition(loaders, size):
    assigned = [
        int(index)
        for loader in loaders.values()
        for index in loader.sampler.indices
    ]
    assert len(assigned) == size
    assert len(set(assigned)) == size
    assert sorted(assigned) == list(range(size))


def test_random_split_assigns_every_sample_exactly_once():
    np.random.seed(1337)
    splitter = DatasetSpliter()
    loaders = splitter.random_split(
        _dataset(),
        {f"client-{index}": object() for index in range(9)},
        batch_size=8,
    )

    _assert_exact_partition(loaders, 101)
    sizes = [len(loader.sampler.indices) for loader in loaders.values()]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_split_assigns_every_class_sample_exactly_once():
    random.seed(2026)
    np.random.seed(2026)
    splitter = DatasetSpliter()
    loaders = splitter.dirichlet_split(
        _dataset(),
        {f"client-{index}": object() for index in range(10)},
        batch_size=16,
        alpha=0.1,
    )

    _assert_exact_partition(loaders, 101)


def test_splitter_rejects_invalid_dimensions():
    splitter = DatasetSpliter()
    data = _dataset(10)

    for operation in (
        lambda: splitter.random_split(data, {}, batch_size=2),
        lambda: splitter.dirichlet_split(data, {"client": 1}, alpha=0),
        lambda: splitter.random_split(data, {"client": 1}, batch_size=0),
    ):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid partition dimensions must be rejected")
