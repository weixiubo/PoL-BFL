#!/usr/bin/env python3
"""
Test that label flipping wrapper correctly exposes indices for PoL verification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def test_label_flip_wrapper_indices():
    """Test that _LabelFlipDS wrapper exposes indices property"""
    
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
    
    # Create a dataloader
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    # Verify original loader has indices
    assert hasattr(loader.dataset, 'indices'), "Original loader should have indices"
    assert loader.dataset.indices == client_indices, "Original indices should match"
    
    # Create label flipping wrapper (simulating the attack)
    class _LabelFlipDS(torch.utils.data.Dataset):
        def __init__(self, base, p, num_classes):
            self.base = base
            self.p = p
            self.num_classes = num_classes
        
        def __len__(self):
            return len(self.base)
        
        def __getitem__(self, idx):
            item = self.base[idx]
            if isinstance(item, (list, tuple)) and len(item) == 3:
                x, y, orig_idx = item
            else:
                x, y = item
                orig_idx = None
            
            # Flip label
            if self.p >= 1.0:
                new_y = torch.randint(low=0, high=self.num_classes, size=(1,)).item()
                if isinstance(y, torch.Tensor):
                    y_val = int(y.item())
                else:
                    y_val = int(y)
                if new_y == y_val:
                    new_y = (new_y + 1) % self.num_classes
                y_out = new_y
            else:
                y_out = int(y.item()) if isinstance(y, torch.Tensor) else y
            
            if orig_idx is None:
                return x, y_out
            else:
                return x, y_out, orig_idx
        
        @property
        def indices(self):
            """Expose underlying indices for PoL verification compatibility"""
            if hasattr(self.base, 'indices'):
                return self.base.indices
            return None
    
    # Create wrapped dataset
    wrapped_ds = _LabelFlipDS(subset, p=1.0, num_classes=10)
    
    # Test that wrapped dataset exposes indices
    assert hasattr(wrapped_ds, 'indices'), "Wrapped dataset should have indices property"
    assert wrapped_ds.indices == client_indices, "Wrapped indices should match original"
    
    # Create wrapped loader
    wrapped_loader = DataLoader(wrapped_ds, batch_size=32, shuffle=False)
    
    # Test that wrapped loader's dataset exposes indices
    assert hasattr(wrapped_loader.dataset, 'indices'), "Wrapped loader should have indices"
    assert wrapped_loader.dataset.indices == client_indices, "Wrapped loader indices should match"
    
    print("✅ All tests passed!")
    print(f"   Original indices: {len(client_indices)} items")
    print(f"   Wrapped indices: {len(wrapped_loader.dataset.indices)} items")
    print(f"   Indices match: {wrapped_loader.dataset.indices == client_indices}")

if __name__ == '__main__':
    test_label_flip_wrapper_indices()

