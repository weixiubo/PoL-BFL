"""
Data Utilities for Experiments

Provides data loading and partitioning utilities for FL experiments.
"""

import os
import hashlib
import json
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def _stable_uint64(*parts) -> int:
    """Return a stable 64-bit unsigned int hash for the given parts."""
    h = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def _slice_compact_data(data: Any, indices: List[int]):
    """Copy only the selected partition samples for verifier serialization."""
    if isinstance(data, torch.Tensor):
        return data[list(indices)].detach().cpu().clone()
    try:
        arr = np.asarray(data)
        return np.ascontiguousarray(arr[list(indices)]).copy()
    except Exception:
        return [data[int(i)] for i in indices]


def _slice_compact_targets(targets: Any, indices: List[int]):
    """Copy only the selected partition labels for verifier serialization."""
    if isinstance(targets, torch.Tensor):
        return targets[list(indices)].detach().cpu().clone()
    try:
        arr = np.asarray(targets)
        return np.asarray(arr[list(indices)]).copy()
    except Exception:
        return [targets[int(i)] for i in indices]


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _target_to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _raw_image_to_pil(value: Any) -> Image.Image:
    """Rebuild the PIL image shape expected by torchvision transforms."""
    arr = _to_numpy_array(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = np.rint(arr * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return Image.fromarray(arr[:, :, 0], mode='L')
    return Image.fromarray(arr)


class SubsetWithIndices(Dataset):
    """
    Simple wrapper for Subset that returns (x, y, global_idx) instead of (x, y).
    This enables PoL to record the actual global indices used during training,
    which is critical when dataloader uses shuffle=True.
    """

    def __init__(self, subset: Subset):
        self.subset = subset
        self._sample_digest_cache = {}
        self._compact_indices = None
        self._compact_data = None
        self._compact_targets = None
        self._compact_base_name = None
        self._compact_transform = None
        self._compact_target_transform = None
        self._compact_pos_by_global = None

    def _has_compact_storage(self) -> bool:
        return self._compact_indices is not None and self._compact_data is not None

    def _rebuild_compact_index(self):
        if self._compact_indices is not None:
            self._compact_pos_by_global = {int(g): i for i, g in enumerate(self._compact_indices)}
        else:
            self._compact_pos_by_global = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_sample_digest_cache'] = {}
        state.pop('_compact_pos_by_global', None)
        if self._has_compact_storage():
            state['subset'] = None
            return state
        try:
            base = self.subset.dataset
            if not (hasattr(base, 'data') and hasattr(base, 'targets')):
                return state
            indices = [int(i) for i in list(self.subset.indices)]
            state['subset'] = None
            state['_compact_indices'] = indices
            state['_compact_data'] = _slice_compact_data(base.data, indices)
            state['_compact_targets'] = _slice_compact_targets(base.targets, indices)
            state['_compact_base_name'] = type(base).__name__
            state['_compact_transform'] = getattr(base, 'transform', None)
            state['_compact_target_transform'] = getattr(base, 'target_transform', None)
        except Exception:
            return dict(self.__dict__, _sample_digest_cache={})
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sample_digest_cache = {}
        for name in (
            '_compact_indices', '_compact_data', '_compact_targets',
            '_compact_base_name', '_compact_transform', '_compact_target_transform',
        ):
            if not hasattr(self, name):
                setattr(self, name, None)
        self._rebuild_compact_index()

    @property
    def indices(self):
        """Expose underlying global indices for PoL verification"""
        if self._compact_indices is not None:
            return self._compact_indices
        return self.subset.indices

    def __len__(self):
        if self._has_compact_storage():
            return len(self._compact_indices)
        return len(self.subset)

    def _compact_target_at(self, local_idx: int):
        targets = self._compact_targets
        if isinstance(targets, torch.Tensor):
            target = targets[int(local_idx)]
        else:
            target = targets[int(local_idx)]
        transform = getattr(self, '_compact_target_transform', None)
        if transform is not None:
            target = transform(target)
        return target

    def _compact_sample_at(self, local_idx: int):
        data = self._compact_data[int(local_idx)]
        transform = getattr(self, '_compact_transform', None)
        if transform is not None:
            return transform(_raw_image_to_pil(data))
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().clone()
        arr = np.asarray(data)
        if arr.ndim in (2, 3):
            return _raw_image_to_pil(arr)
        return torch.as_tensor(arr)

    def __getitem__(self, local_idx: int):
        if self._has_compact_storage():
            global_idx = int(self._compact_indices[local_idx])
            x = self._compact_sample_at(local_idx)
            y = self._compact_target_at(local_idx)
            return x, y, global_idx
        # Get global index
        global_idx = int(self.subset.indices[local_idx])
        # Get data from base dataset
        x, y = self.subset[local_idx]
        # Return with global index
        return x, y, global_idx

    def _raw_sample_digest(self, global_idx: int) -> bytes:
        cached = self._sample_digest_cache.get(int(global_idx))
        if cached is not None:
            return cached
        h = hashlib.sha256()
        if self._has_compact_storage():
            pos_map = self._compact_pos_by_global or {}
            if int(global_idx) not in pos_map:
                raise KeyError(f"global index {global_idx} not in compact verifier partition")
            pos = pos_map[int(global_idx)]
            h.update(str(self._compact_base_name or "CompactDataset").encode("utf-8"))
            h.update(str(int(global_idx)).encode("utf-8"))
            h.update(_to_numpy_array(self._compact_data[pos]).tobytes())
            h.update(str(_target_to_int(self._compact_targets[pos])).encode("utf-8"))
            digest = h.digest()
            self._sample_digest_cache[int(global_idx)] = digest
            return digest
        base = self.subset.dataset
        h.update(str(type(base).__name__).encode("utf-8"))
        h.update(str(int(global_idx)).encode("utf-8"))
        if hasattr(base, "data") and hasattr(base, "targets"):
            data = base.data[int(global_idx)]
            if isinstance(data, torch.Tensor):
                arr = data.detach().cpu().numpy()
            else:
                arr = np.asarray(data)
            h.update(arr.tobytes())
            h.update(str(int(base.targets[int(global_idx)])).encode("utf-8"))
        else:
            item = base[int(global_idx)]
            data, label = item[:2] if isinstance(item, (tuple, list)) else (item, 0)
            if isinstance(data, torch.Tensor):
                h.update(data.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(np.asarray(data).tobytes())
            h.update(str(int(label)).encode("utf-8"))
        digest = h.digest()
        self._sample_digest_cache[int(global_idx)] = digest
        return digest

    def fast_data_hash_for_global_indices(self, indices: List[int]) -> str:
        """Hash a recorded global-index sequence using cached raw sample digests."""
        h = hashlib.sha256()
        for idx in indices:
            idx = int(idx)
            h.update(str(idx).encode("utf-8"))
            h.update(b":")
            h.update(self._raw_sample_digest(idx))
            h.update(b";")
        return h.hexdigest()


class SubsetDeterministicWrapper(Dataset):
    """
    Wrap a torch.utils.data.Subset to provide deterministic CIFAR-style augmentation
    and expose global indices. This enables training to record the true global index
    order while verification can reconstruct the same inputs.

    Returns samples as (x, y, global_idx).
    """

    def __init__(self, subset: Subset, dataset_name: str):
        self.subset = subset
        self.dataset_name = dataset_name
        self._sample_digest_cache = {}
        # Normalization per dataset
        if dataset_name == 'CIFAR10':
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
        elif dataset_name == 'CIFAR100':
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
        else:
            self.mean = None
            self.std = None
        # Stable seed (can be overridden)
        self.seed = int(os.getenv('POL_DET_SEED', '1337'))
        self.current_round = 0
        self.current_epoch = 0
        self._compact_indices = None
        self._compact_data = None
        self._compact_targets = None
        self._compact_pos_by_global = None

        # Pre-build simple transform ops
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(self.mean, self.std) if self.mean is not None else None
        if self.mean is not None:
            self._mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
            self._std_tensor = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)
        else:
            self._mean_tensor = None
            self._std_tensor = None

    def set_replay_context(self, *, round_num=None, epoch=None):
        if round_num is not None:
            self.current_round = int(round_num)
        if epoch is not None:
            self.current_epoch = int(epoch)

    def _has_compact_storage(self) -> bool:
        return self._compact_indices is not None and self._compact_data is not None

    def _rebuild_compact_index(self):
        if self._compact_indices is not None:
            self._compact_pos_by_global = {int(g): i for i, g in enumerate(self._compact_indices)}
        else:
            self._compact_pos_by_global = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_sample_digest_cache'] = {}
        state.pop('_compact_pos_by_global', None)
        if self._has_compact_storage():
            state['subset'] = None
            return state
        try:
            base = self.subset.dataset
            if not (hasattr(base, 'data') and hasattr(base, 'targets')):
                return state
            indices = [int(i) for i in list(self.subset.indices)]
            state['subset'] = None
            state['_compact_indices'] = indices
            state['_compact_data'] = _slice_compact_data(base.data, indices)
            state['_compact_targets'] = _slice_compact_targets(base.targets, indices)
        except Exception:
            return dict(self.__dict__, _sample_digest_cache={})
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sample_digest_cache = {}
        for name in ('_compact_indices', '_compact_data', '_compact_targets'):
            if not hasattr(self, name):
                setattr(self, name, None)
        self._rebuild_compact_index()

    @property
    def indices(self):
        # Expose underlying global indices for mapping (global -> local)
        if self._compact_indices is not None:
            return self._compact_indices
        return self.subset.indices

    def __len__(self):
        if self._has_compact_storage():
            return len(self._compact_indices)
        return len(self.subset)

    def _deterministic_cifar_ops(self, img: Image.Image, gidx: int) -> Image.Image:
        # Deterministic pad+crop and horizontal flip following CIFAR recipe
        # padding=4, crop size=32
        padding = 4
        size = 32
        # Zero-pad
        if padding > 0:
            img = ImageOps.expand(img, border=padding, fill=0)
        # Compute deterministic offsets and flip
        seed = _stable_uint64(self.seed, self.dataset_name, self.current_round, self.current_epoch, gidx)
        max_off = padding * 2
        off_x = seed % (max_off + 1)
        off_y = (seed >> 8) % (max_off + 1)
        left, upper = int(off_x), int(off_y)
        img = img.crop((left, upper, left + size, upper + size))
        # Flip decision
        if ((seed >> 16) & 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _deterministic_cifar_array(self, arr: np.ndarray, gidx: int) -> np.ndarray:
        """Fast NumPy equivalent of pad+crop+flip for CIFAR images."""
        padding = 4
        size = 32
        seed = _stable_uint64(self.seed, self.dataset_name, self.current_round, self.current_epoch, gidx)
        max_off = padding * 2
        off_x = int(seed % (max_off + 1))
        off_y = int((seed >> 8) % (max_off + 1))
        if padding > 0:
            arr = np.pad(arr, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
        arr = arr[off_y:off_y + size, off_x:off_x + size, :]
        if ((seed >> 16) & 1) == 1:
            arr = arr[:, ::-1, :]
        return np.ascontiguousarray(arr)

    def _raw_sample_digest(self, global_idx: int) -> bytes:
        cached = self._sample_digest_cache.get(int(global_idx))
        if cached is not None:
            return cached
        h = hashlib.sha256()
        h.update(str(self.dataset_name).encode("utf-8"))
        h.update(str(int(global_idx)).encode("utf-8"))
        if self._has_compact_storage():
            pos_map = self._compact_pos_by_global or {}
            if int(global_idx) not in pos_map:
                raise KeyError(f"global index {global_idx} not in compact verifier partition")
            pos = pos_map[int(global_idx)]
            h.update(_to_numpy_array(self._compact_data[pos]).tobytes())
            h.update(str(_target_to_int(self._compact_targets[pos])).encode("utf-8"))
            digest = h.digest()
            self._sample_digest_cache[int(global_idx)] = digest
            return digest
        base = self.subset.dataset
        if hasattr(base, "data") and hasattr(base, "targets"):
            h.update(np.asarray(base.data[int(global_idx)]).tobytes())
            h.update(str(int(base.targets[int(global_idx)])).encode("utf-8"))
        else:
            item = base[int(global_idx)]
            data, label = item[:2] if isinstance(item, (tuple, list)) else (item, 0)
            if isinstance(data, torch.Tensor):
                h.update(data.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(np.asarray(data).tobytes())
            h.update(str(int(label)).encode("utf-8"))
        digest = h.digest()
        self._sample_digest_cache[int(global_idx)] = digest
        return digest

    def fast_data_hash_for_global_indices(self, indices: List[int]) -> str:
        """Hash a recorded global-index sequence without replaying augmentation."""
        h = hashlib.sha256()
        for idx in indices:
            idx = int(idx)
            h.update(str(idx).encode("utf-8"))
            h.update(b":")
            h.update(self._raw_sample_digest(idx))
            h.update(b";")
        return h.hexdigest()

    def __getitem__(self, local_idx: int):
        compact = self._has_compact_storage()
        base = None if compact else self.subset.dataset
        # Map local to global index
        gidx = int(self._compact_indices[local_idx] if compact else self.subset.indices[local_idx])

        # Fetch raw image/label without triggering base.transform randomness
        # Prefer direct array access where available (CIFAR)
        if compact:
            arr = _to_numpy_array(self._compact_data[local_idx])
            target = _target_to_int(self._compact_targets[local_idx])
            if self.dataset_name in ('CIFAR10', 'CIFAR100') and arr.ndim == 3:
                arr = self._deterministic_cifar_array(np.asarray(arr), gidx)
                x = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
                if self._mean_tensor is not None and self._std_tensor is not None:
                    x = (x - self._mean_tensor) / self._std_tensor
                return x, target, gidx
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode='L')
            else:
                img = Image.fromarray(arr)
        elif hasattr(base, 'data') and hasattr(base, 'targets'):
            arr = base.data[gidx]
            target = int(base.targets[gidx])
            if self.dataset_name in ('CIFAR10', 'CIFAR100') and arr.ndim == 3:
                arr = self._deterministic_cifar_array(np.asarray(arr), gidx)
                x = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
                if self._mean_tensor is not None and self._std_tensor is not None:
                    x = (x - self._mean_tensor) / self._std_tensor
                return x, target, gidx

            # Handle grayscale vs RGB fallback
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode='L')
            else:
                img = Image.fromarray(arr)
        else:
            # Fallback: use base __getitem__ (may already apply transforms)
            sample = base[gidx]
            if isinstance(sample, (tuple, list)):
                img, target = sample[0], int(sample[1])
            else:
                img, target = sample, 0
            # If already tensor, we return as-is with idx
            if isinstance(img, torch.Tensor):
                return img, target, gidx

        # Apply deterministic augment for CIFAR datasets
        if self.dataset_name in ('CIFAR10', 'CIFAR100'):
            img = self._deterministic_cifar_ops(img, gidx)
        # Convert to tensor and normalize
        x = self._to_tensor(img)
        if self._normalize is not None:
            x = self._normalize(x)
        return x, target, gidx


class LEAFFEMNISTDataset(Dataset):
    """
    FEMNIST loader for LEAF-style JSON shards.

    Expected files are the original LEAF JSON format with top-level
    users/user_data fields. FashionMNIST is intentionally not used as a
    replacement because the paper matrix requires natural writer partitions.
    """

    def __init__(self, data_dir: str, train: bool = True):
        self.data_dir = Path(data_dir)
        self.train = bool(train)
        self.samples = []
        self.targets = []
        self.labels = self.targets
        self.users = []
        self._load_leaf_json()

    def _candidate_dirs(self) -> List[Path]:
        split = "train" if self.train else "test"
        return [
            self.data_dir / "leaf" / "data" / split,
            self.data_dir / "data" / split,
            self.data_dir / split,
            self.data_dir,
        ]

    def _json_files(self) -> List[Path]:
        files = []
        for root in self._candidate_dirs():
            if root.is_dir():
                files.extend(sorted(root.glob("*.json")))
        return sorted(set(files))

    def _load_leaf_json(self):
        files = self._json_files()
        if not files:
            raise FileNotFoundError(
                f"FEMNIST LEAF JSON files not found under {self.data_dir}. "
                "Install/download LEAF FEMNIST and point POL_DATA_DIR or data_dir to it."
            )

        for path in files:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            user_data = obj.get("user_data", {})
            users = obj.get("users", list(user_data.keys()))
            for user in users:
                rows = user_data.get(user, {})
                xs = rows.get("x", [])
                ys = rows.get("y", [])
                for x, y in zip(xs, ys):
                    self.samples.append(x)
                    self.targets.append(int(y))
                    self.users.append(str(user))

        if not self.samples:
            raise ValueError(f"No FEMNIST samples found in LEAF JSON files under {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        arr = np.asarray(self.samples[idx], dtype=np.float32)
        if arr.size != 28 * 28:
            raise ValueError(f"FEMNIST sample at index {idx} has {arr.size} values, expected 784")
        arr = arr.reshape(1, 28, 28)
        if float(arr.max()) > 1.0:
            arr = arr / 255.0
        x = torch.from_numpy(arr)
        y = int(self.targets[idx])
        return x, y


def load_dataset(dataset_name: str, data_dir: str = './data', train: bool = True):
    """
    Load dataset

    Args:
        dataset_name: Name of dataset ('MNIST', 'CIFAR10', 'CIFAR100', 'FEMNIST', 'FashionMNIST')
        data_dir: Directory to store data
        train: Load training set or test set

    Returns:
        dataset: PyTorch dataset
    """
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)

    elif dataset_name == 'CIFAR10':
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)

    elif dataset_name == 'CIFAR100':
        # CIFAR-100 uses same normalization as CIFAR-10
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        dataset = datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)

    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)

    elif dataset_name == 'FEMNIST':
        dataset = LEAFFEMNISTDataset(data_dir, train=train)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Loaded {dataset_name} dataset (train={train}, size={len(dataset)})")
    return dataset


def partition_data_iid(dataset, num_clients: int) -> List[Subset]:
    """
    Partition data in IID manner
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
    
    Returns:
        client_datasets: List of client datasets
    """
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    # Split indices evenly
    split_indices = np.array_split(indices, num_clients)
    
    client_datasets = [Subset(dataset, indices.tolist()) for indices in split_indices]
    
    logger.info(f"Partitioned data IID for {num_clients} clients")
    return client_datasets


def partition_data_dirichlet(dataset, num_clients: int, alpha: float = 0.5) -> List[Subset]:
    """
    Partition data using Dirichlet distribution (Non-IID)
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more non-IID)
    
    Returns:
        client_datasets: List of client datasets
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset does not have targets or labels attribute")
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples to clients using Dirichlet
    for k in range(num_classes):
        # Get indices of samples with label k
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Distribute samples according to proportions
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_idx_k = np.split(idx_k, proportions)
        
        # Assign to clients
        for i, idx in enumerate(split_idx_k):
            client_indices[i].extend(idx.tolist())
    
    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    logger.info(f"Partitioned data Non-IID (Dirichlet α={alpha}) for {num_clients} clients")
    return client_datasets


def partition_data_pathological(dataset, num_clients: int, shards_per_client: int = 2) -> List[Subset]:
    """
    Partition data using pathological distribution (Non-IID)
    Each client gets data from only a few classes.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        shards_per_client: Number of shards (classes) per client
    
    Returns:
        client_datasets: List of client datasets
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset does not have targets or labels attribute")
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Sort by label
    sorted_indices = np.argsort(labels)
    
    # Create shards
    num_shards = num_clients * shards_per_client
    shard_size = num_samples // num_shards
    shards = []
    
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else num_samples
        shards.append(sorted_indices[start:end].tolist())
    
    # Shuffle shards
    np.random.shuffle(shards)
    
    # Assign shards to clients
    client_indices = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(shards_per_client):
            shard_idx = i * shards_per_client + j
            client_indices[i].extend(shards[shard_idx])
    
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    logger.info(f"Partitioned data Pathological ({shards_per_client} shards/client) for {num_clients} clients")
    return client_datasets


def partition_data_by_user(dataset, num_clients: int) -> List[Subset]:
    """Partition LEAF-style FEMNIST by natural writer identity."""
    if not hasattr(dataset, 'users'):
        raise ValueError("Natural user partition requires dataset.users")

    groups: Dict[str, List[int]] = {}
    for idx, user in enumerate(dataset.users):
        groups.setdefault(str(user), []).append(int(idx))

    client_indices = [[] for _ in range(num_clients)]
    for pos, user in enumerate(sorted(groups.keys())):
        client_indices[pos % num_clients].extend(groups[user])

    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    logger.info(f"Partitioned FEMNIST by natural writer identity for {num_clients} clients")
    return client_datasets


def create_dataloaders(client_datasets: List[Subset], batch_size: int = 32,
                       num_workers: int = 0) -> List[DataLoader]:
    """
    Create dataloaders for client datasets

    Args:
        client_datasets: List of client datasets
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        dataloaders: List of dataloaders
    """
    det_env = os.getenv('POL_DETERMINISTIC_AUG')
    if det_env is None:
        # Strict PoL replay must see the same inputs during commitment and
        # verification. CIFAR's random crop/flip transforms otherwise make
        # data_hash and replay checks nondeterministic across verifier reads.
        use_det = (
            os.getenv('POL_INTEGRITY', '0') == '1'
            or os.getenv('POL_REMOTE_MODE', '').lower() == 'strict_replay'
            or os.getenv('POL_REQUIRE_REMOTE_VERIFIER', '0') == '1'
        )
    else:
        use_det = str(det_env).lower() in ('1', 'true', 'yes', 'on')
    dataloaders = []
    for subset in client_datasets:
        # Always wrap with SubsetWithIndices to expose global indices
        # This is critical for PoL to record correct indices when shuffle=True
        base = getattr(subset, 'dataset', None)
        base_name = type(base).__name__ if base is not None else ''

        if use_det and base_name in ('CIFAR10', 'CIFAR100'):
            # Use deterministic augmentation wrapper for CIFAR
            ds_to_use = SubsetDeterministicWrapper(subset, base_name)
        else:
            # Use simple indices wrapper for other datasets (MNIST, etc.)
            ds_to_use = SubsetWithIndices(subset)

        loader = DataLoader(
            ds_to_use,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        dataloaders.append(loader)

    return dataloaders


def get_data_statistics(client_datasets: List[Subset]) -> Dict:
    """
    Get statistics of data distribution
    
    Args:
        client_datasets: List of client datasets
    
    Returns:
        stats: Dictionary of statistics
    """
    stats = {
        'num_clients': len(client_datasets),
        'total_samples': sum(len(ds) for ds in client_datasets),
        'samples_per_client': [len(ds) for ds in client_datasets],
        'min_samples': min(len(ds) for ds in client_datasets),
        'max_samples': max(len(ds) for ds in client_datasets),
        'mean_samples': np.mean([len(ds) for ds in client_datasets]),
        'std_samples': np.std([len(ds) for ds in client_datasets])
    }
    
    # Get label distribution for each client
    label_distributions = []
    for dataset in client_datasets:
        if hasattr(dataset.dataset, 'targets'):
            labels = np.array(dataset.dataset.targets)[dataset.indices]
        elif hasattr(dataset.dataset, 'labels'):
            labels = np.array(dataset.dataset.labels)[dataset.indices]
        else:
            continue
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))
        label_distributions.append(label_dist)
    
    stats['label_distributions'] = label_distributions
    
    return stats


def print_data_statistics(stats: Dict):
    """Print data statistics"""
    print("\n" + "="*50)
    print("Data Distribution Statistics")
    print("="*50)
    print(f"Number of clients: {stats['num_clients']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples per client: min={stats['min_samples']}, max={stats['max_samples']}, "
          f"mean={stats['mean_samples']:.1f}, std={stats['std_samples']:.1f}")
    
    if 'label_distributions' in stats:
        print("\nLabel distribution per client:")
        for i, dist in enumerate(stats['label_distributions'][:5]):  # Show first 5 clients
            print(f"  Client {i}: {dist}")
        if len(stats['label_distributions']) > 5:
            print(f"  ... ({len(stats['label_distributions']) - 5} more clients)")
    
    print("="*50 + "\n")
