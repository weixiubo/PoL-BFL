"""Deterministic indexed datasets for computation-valid poisoning/Sybil cells."""

from __future__ import annotations

import hashlib

import torch


class DeterministicLabelPoison(torch.utils.data.Dataset):
    def __init__(self, base, *, num_classes: int, poison_ratio: float, seed: int):
        if num_classes <= 1 or not 0 <= poison_ratio <= 1:
            raise ValueError("poisoning class count or ratio is invalid")
        self.base = base
        self.num_classes = int(num_classes)
        self.poison_ratio = float(poison_ratio)
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def set_replay_context(self, *, round_num=None, epoch=None):
        setter = getattr(self.base, "set_replay_context", None)
        if callable(setter):
            setter(round_num=round_num, epoch=epoch)

    def __getitem__(self, position: int):
        item = self.base[position]
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError("poison wrapper requires indexed samples")
        data, label, global_index = item
        digest = hashlib.sha256(
            f"POLBFL_POISON_V1:{self.seed}:{int(global_index)}".encode()
        ).digest()
        ticket = int.from_bytes(digest[:8], "big") / 2**64
        if ticket < self.poison_ratio:
            offset = 1 + int.from_bytes(digest[8:12], "big") % (self.num_classes - 1)
            label = (int(label) + offset) % self.num_classes
        return data, label, int(global_index)


def clone_indexed_loader(loader, *, seed: int, dataset=None):
    generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        loader.dataset if dataset is None else dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=loader.drop_last,
    )
