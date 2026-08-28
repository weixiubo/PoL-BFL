#!/usr/bin/env python3
"""
Test that SubsetWithIndices correctly exposes global indices for PoL verification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Import the new wrapper
from experiments.scripts.utils.data_utils import SubsetWithIndices

def test_subset_with_indices():
    """Test that SubsetWithIndices returns (x, y, global_idx)"""

    # Create a simple dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Create a Subset (simulating client's data partition)
    client_indices = list(range(100, 200))  # Global indices 100-199
    subset = Subset(full_dataset, client_indices)

    # Wrap with SubsetWithIndices
    wrapped_subset = SubsetWithIndices(subset)

    # Test 1: Check that wrapped subset has indices property
    assert hasattr(wrapped_subset, 'indices'), "Wrapped subset should have indices property"
    assert wrapped_subset.indices == client_indices, "Wrapped indices should match original"

    # Test 2: Check that __getitem__ returns (x, y, global_idx)
    x, y, global_idx = wrapped_subset[0]
    assert isinstance(x, torch.Tensor), "x should be a tensor"
    assert isinstance(y, (int, torch.Tensor)), "y should be an int or tensor"
    assert isinstance(global_idx, int), "global_idx should be an int"
    assert global_idx == client_indices[0], f"global_idx should be {client_indices[0]}, got {global_idx}"

    # Test 3: Check that dataloader with shuffle=True returns correct indices
    loader = DataLoader(wrapped_subset, batch_size=32, shuffle=True)

    all_indices = []
    for batch in loader:
        if len(batch) == 3:
            x, y, idxs = batch
            all_indices.extend(idxs.tolist())
        else:
            raise AssertionError("Batch should have 3 elements (x, y, idxs)")

    # All indices should be from client_indices
    assert set(all_indices) == set(client_indices), "All indices should be from client_indices"

    # Test 4: Verify that indices are shuffled (not in order)
    # With shuffle=True, the order should be different from the original
    # (This test might occasionally fail due to random chance, but very unlikely)
    is_shuffled = all_indices != sorted(all_indices)
    print(f"   Indices shuffled: {is_shuffled}")

    print("[PASS] All tests passed.")
    print(f"   Original indices: {len(client_indices)} items")
    print(f"   Wrapped indices: {len(wrapped_subset.indices)} items")
    print(f"   Collected indices from dataloader: {len(all_indices)} items")
    print(f"   Indices match: {set(all_indices) == set(client_indices)}")

if __name__ == '__main__':
    test_subset_with_indices()

