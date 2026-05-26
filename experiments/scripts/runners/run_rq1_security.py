"""
RQ1: Security Evaluation

Evaluate PoL-FL's ability to defend against Byzantine and free-riding attacks.

Implements the full RQ1 protocol with CLI configurability and academic-complete coverage.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import json
import csv
import gc
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from collections import OrderedDict

# Add parent directory to path
scripts_dir = Path(__file__).parent.parent
experiments_dir = scripts_dir.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / 'utils'))
sys.path.insert(0, str(experiments_dir))

from experiment_config import FL_CONFIG, POL_CONFIG, OUTPUT_CONFIG, DATASETS, NUM_WORKERS, set_random_seed
from data_utils import load_dataset, partition_data_iid, partition_data_dirichlet, partition_data_by_user, create_dataloaders, print_data_statistics, get_data_statistics
from models import create_model, print_model_info
from metrics import MetricsTracker, compute_accuracy, compute_detection_rate, compute_convergence_round
from baselines import create_aggregator
from attacks.byzantine_attacks import create_attack
from attacks.free_riding_attacks import create_free_riding_attack
from attacks.sybil_attacks import SybilAttack
from pol_integration import PoLExperimentHelper

from client.trainer.PoLTrainer import PoLTrainer
from client.PoLClient import PoLClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Academic integrity visibility: log POL_INTEGRITY at start (simulation disabled when =1)
logger.info(f"POL_INTEGRITY={os.getenv('POL_INTEGRITY', '0')} (simulation disabled when =1)")


def _stable_seed32(*parts) -> int:
    import hashlib
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


def _parse_attack_param_value(raw: str):
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." not in text and "e" not in lower:
            return int(text)
        return float(text)
    except ValueError:
        return text


# Ensure deterministic cuBLAS workspace to avoid runtime error when torch.use_deterministic_algorithms(True)
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    logger.info('CUBLAS_WORKSPACE_CONFIG not set; defaulting to :4096:8 for deterministic CUDA')


# RQ1 config (full coverage; CLI can override)
RQ1_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 20,
    'clients_per_round': 10,
    # Per spec (updated): MNIST=20 rounds, CIFAR10/100=100 rounds (defaults set for CIFAR10)
    'num_rounds': 100,
    'data_distribution': 'NonIID_Dirichlet',

    # Full attack scenarios (Original + Blades framework)
    'attacks': {
        'no_attack': {'malicious_ratios': [0.0]},
        # Original Byzantine attacks
        'byzantine_random_noise': {'malicious_ratios': [0.2], 'noise_scale': 1.0, 'scale_mode': 'parameter_scaled'},
        'byzantine_label_flipping': {'malicious_ratios': [0.2]},
        'byzantine_model_replacement': {'malicious_ratios': [0.2], 'replacement_mix': 0.1},
        'byzantine_gradient_inversion': {'malicious_ratios': [0.2]},
        # Blades framework attacks (A-class conferences)
        'byzantine_alie': {'malicious_ratios': [0.2], 'z_max': 2.5},  # NeurIPS 2019
        'byzantine_ipm': {'malicious_ratios': [0.2], 'scale': 1.0},  # UAI 2020
        'byzantine_minmax': {'malicious_ratios': [0.2], 'lambda_init': 1.0},  # NDSS 2021
        # Free-riding attacks
        'free_riding_no_training': {'malicious_ratios': [0.2]},
        'free_riding_lazy_training': {'malicious_ratios': [0.2]},
        'free_riding_minimal_update': {'malicious_ratios': [0.2], 'noise_scale': 1e-5},
        # Paper Table 1 attack axes that must be runnable from RQ1.
        'data_poisoning': {'malicious_ratios': [0.2], 'poison_ratio': 0.1},
        'sybil': {'malicious_ratios': [0.2], 'num_identities': 5, 'shared_data_ratio': 1.0},

    },

    # Baselines to test (Original + SOTA)
    'baselines': ['Vanilla_FL', 'Krum', 'Trimmed_Mean', 'Median', 'ShapleyFL', 'FoolsGold', 'SDEA', 'PoL_FL'],

    # PoL configuration (RQ1 defaults after clearance: delta=5.0, verification_rate=1.0)
    'pol_config': {
        'enable': True,
        'save_freq': 5,
        'verification_rate': 1.0,
        'delta': 5.0,
        'distance_metric': 'l2',
        'use_top_q': False,
        'top_q': 5,
        'enable_zkp': False,
        'zkp_use_simulation': (False if os.getenv('POL_INTEGRITY', '0') == '1' else True),
        # Root-cause improvements
        'min_pair_success_rate': 0.99,
        'always_verify_last_k': 2,
        'random_q': 3,
    }
}


class SecurityExperiment:
    """Security evaluation experiment"""

    def __init__(self, config, output_dir=None):
        """Initialize experiment with configuration

        Args:
            config: Experiment configuration dictionary
            output_dir: Optional custom output directory path (default: experiments/results/rq1_security)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_tracker = MetricsTracker()

        # Set random seed
        set_random_seed()

        # Create output directory
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(OUTPUT_CONFIG['results_dir']) / 'rq1_security'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SecurityExperiment on {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    @staticmethod
    def _num_malicious_from_ratio(num_clients: int, malicious_ratio: float) -> int:
        if malicious_ratio <= 0.0:
            return 0
        return max(1, int(num_clients * malicious_ratio))

    @staticmethod
    def _set_dataset_replay_context(dataset, *, round_num=None, epoch=None):
        seen = set()

        def visit(obj):
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            setter = getattr(obj, 'set_replay_context', None)
            if callable(setter):
                setter(round_num=round_num, epoch=epoch)
            else:
                if round_num is not None and hasattr(obj, 'current_round'):
                    setattr(obj, 'current_round', int(round_num))
                if epoch is not None and hasattr(obj, 'current_epoch'):
                    setattr(obj, 'current_epoch', int(epoch))
            for attr in ('dataset', 'base', 'subset'):
                visit(getattr(obj, attr, None))

        visit(dataset)

    def prepare_data(self):
        """Prepare datasets"""
        logger.info("Preparing datasets...")

        # Load dataset with explicit data root inside repo (avoid cwd-dependent './data')
        ds_name = self.config['dataset']
        ds_root = DATASETS[ds_name]['data_dir']
        train_dataset = load_dataset(ds_name, data_dir=ds_root, train=True)
        test_dataset = load_dataset(ds_name, data_dir=ds_root, train=False)

        # Partition data
        if self.config['data_distribution'] == 'IID':
            client_datasets = partition_data_iid(train_dataset, self.config['num_clients'])
        elif self.config['data_distribution'] == 'NonIID_Dirichlet':
            client_datasets = partition_data_dirichlet(
                train_dataset,
                self.config['num_clients'],
                alpha=float(self.config.get('dirichlet_alpha', 0.5))
            )
        elif self.config['data_distribution'] in ('Natural_Writer', 'FEMNIST_Natural', 'LEAF_Natural'):
            client_datasets = partition_data_by_user(train_dataset, self.config['num_clients'])
        else:
            raise ValueError(f"Unknown data distribution: {self.config['data_distribution']}")

        # Create dataloaders (parallel + pinned for speed)
        self.train_loaders = create_dataloaders(
            client_datasets,
            batch_size=FL_CONFIG['batch_size'],
            num_workers=NUM_WORKERS
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FL_CONFIG['batch_size'],
            shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=torch.cuda.is_available()
        )

        # Print statistics
        stats = get_data_statistics(client_datasets)
        print_data_statistics(stats)

        logger.info(f"Prepared data for {self.config['num_clients']} clients")

    def _num_classes_for_dataset(self) -> int:
        ds = self.config['dataset']
        if ds == 'MNIST':
            return 10
        elif ds == 'CIFAR10':
            return 10
        elif ds == 'CIFAR100':
            return 100
        elif ds == 'FEMNIST':
            return 62
        else:
            raise ValueError(f"Unknown dataset: {ds}")

    def _wrap_loader_with_label_flipping(self, loader: torch.utils.data.DataLoader, flip_prob: float = 1.0) -> torch.utils.data.DataLoader:
        """Return a DataLoader that flips labels on-the-fly for malicious clients.
        This implements the label flipping attack during training (not a post-hoc model edit).
        """
        num_classes = self._num_classes_for_dataset()
        base_ds = loader.dataset

        class _LabelFlipDS(torch.utils.data.Dataset):
            def __init__(self, base, p, num_classes):
                self.base = base
                self.p = p
                self.num_classes = num_classes
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                item = self.base[idx]
                # Support (x, y) or (x, y, idx)
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    x, y, orig_idx = item
                else:
                    x, y = item
                    orig_idx = None
                # Flip label with probability p
                if self.p >= 1.0:
                    flip = True
                elif self.p <= 0.0:
                    flip = False
                else:
                    flip = (torch.rand(1).item() < self.p)
                if flip:
                    new_y = torch.randint(low=0, high=self.num_classes, size=(1,)).item()
                    # ensure change
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

        wrap_ds = _LabelFlipDS(base_ds, flip_prob, num_classes)
        wrapped = torch.utils.data.DataLoader(
            wrap_ds,
            batch_size=loader.batch_size,
            shuffle=True,  # keep randomness similar to honest loaders
            num_workers=loader.num_workers,
            pin_memory=getattr(loader, 'pin_memory', False),
            drop_last=getattr(loader, 'drop_last', False),
        )
        return wrapped

    def _wrap_loader_with_data_poisoning(self, loader: torch.utils.data.DataLoader, poison_ratio: float = 0.1) -> torch.utils.data.DataLoader:
        """Return a DataLoader that flips a deterministic fraction of labels.

        This makes the RQ1 paper Data Poisoning axis runnable in the same
        training path as honest clients and keeps original indices available
        for PoL verification.
        """
        num_classes = self._num_classes_for_dataset()
        base_ds = loader.dataset

        class _PoisonedDS(torch.utils.data.Dataset):
            def __init__(self, base, ratio, num_classes):
                self.base = base
                self.ratio = max(0.0, min(1.0, float(ratio)))
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
                poison_mod = max(1, int(round(1.0 / self.ratio))) if self.ratio > 0.0 else None
                should_poison = poison_mod is not None and (idx % poison_mod == 0)
                y_val = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
                if should_poison:
                    y_val = (y_val + 1) % self.num_classes
                if orig_idx is None:
                    return x, y_val
                return x, y_val, orig_idx

            @property
            def indices(self):
                if hasattr(self.base, 'indices'):
                    return self.base.indices
                return None

        wrap_ds = _PoisonedDS(base_ds, poison_ratio, num_classes)
        return torch.utils.data.DataLoader(
            wrap_ds,
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=loader.num_workers,
            pin_memory=getattr(loader, 'pin_memory', False),
            drop_last=getattr(loader, 'drop_last', False),
        )

    @staticmethod
    def _flatten_model_state(model_state):
        parts = []
        for key in sorted(model_state.keys()):
            tensor = model_state[key]
            if not torch.is_tensor(tensor):
                continue
            vec = tensor.detach().float().cpu().reshape(-1)
            if not torch.isfinite(vec).all():
                vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(vec)
        return torch.cat(parts) if parts else torch.empty(0)

    @classmethod
    def _state_l2_distance(cls, left, right):
        left_vec = cls._flatten_model_state(left)
        right_vec = cls._flatten_model_state(right)
        if left_vec.numel() == 0 or right_vec.numel() == 0 or left_vec.numel() != right_vec.numel():
            return 0.0
        return float(torch.norm(left_vec - right_vec, p=2).item())

    def _baseline_suspects(self, baseline_method, aggregator, client_models, selected_indices, expected_num_suspects):
        """Infer baseline detector decisions from each method's own evidence.

        The older recovery path used one shared median-distance detector and
        the realized malicious count for every baseline.  That made baseline
        DR/FPR look artificially good and decoupled the metric from the method
        being evaluated.  This keeps the clean-room implementation honest by
        deriving suspects from Krum scores, SDEA rejections, ShapleyFL scores,
        or FoolsGold weights.
        """
        if expected_num_suspects <= 0 or not client_models:
            return set()
        k = min(int(expected_num_suspects), len(client_models))
        method = str(baseline_method)
        suspect_positions = []

        if method == "SDEA" and getattr(aggregator, "rejected_indices", None):
            suspect_positions = [int(i) for i in aggregator.rejected_indices]
        elif method == "Krum" and getattr(aggregator, "scores", None):
            scores = np.array(getattr(aggregator, "scores"), dtype=float)
            suspect_positions = np.argsort(scores)[-k:].tolist()
        elif method == "ShapleyFL" and getattr(aggregator, "scores", None):
            scores = np.array(getattr(aggregator, "scores"), dtype=float)
            suspect_positions = np.argsort(scores)[:k].tolist()
        elif method == "FoolsGold" and getattr(aggregator, "client_weights", None):
            weights = np.array(getattr(aggregator, "client_weights"), dtype=float)
            suspect_positions = np.argsort(weights)[:k].tolist()
        elif getattr(aggregator, "rejected_indices", None):
            suspect_positions = [int(i) for i in aggregator.rejected_indices]
        elif getattr(aggregator, "selected_indices", None):
            selected = {int(i) for i in aggregator.selected_indices}
            suspect_positions = [i for i in range(len(client_models)) if i not in selected]

        if not suspect_positions:
            vectors = torch.stack([self._flatten_model_state(model) for model in client_models], dim=0)
            center = torch.median(vectors, dim=0).values
            scores = torch.norm(vectors - center.unsqueeze(0), p=2, dim=1)
            suspect_positions = torch.argsort(scores, descending=True)[:k].tolist()
        suspect_positions = [int(i) for i in suspect_positions[:k]]
        return {f"client_{int(selected_indices[pos])}" for pos in suspect_positions}

    @staticmethod
    def _sum_detection_counts(accum, metrics):
        for key in ["TP_e2e", "FP_e2e", "FN_e2e", "TN_e2e", "total_malicious", "total_honest"]:
            accum[key] = int(accum.get(key, 0)) + int(metrics.get(key, 0))

    @staticmethod
    def _final_detection_metrics_from_counts(accum):
        tp = int(accum.get("TP_e2e", 0))
        fp = int(accum.get("FP_e2e", 0))
        fn = int(accum.get("FN_e2e", 0))
        tn = int(accum.get("TN_e2e", 0))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
        return {
            "TPR": float(tpr),
            "FPR": float(fpr),
            "Precision": float(precision),
            "Recall": float(tpr),
            "F1": float(f1),
            "TP_e2e": tp,
            "FP_e2e": fp,
            "FN_e2e": fn,
            "TN_e2e": tn,
            "total_malicious": int(accum.get("total_malicious", tp + fn)),
            "total_honest": int(accum.get("total_honest", fp + tn)),
        }

    def create_model(self):
        """Create model"""
        ds = self.config['dataset']
        if ds == 'MNIST':
            num_classes, input_channels = 10, 1
        elif ds == 'CIFAR10':
            num_classes, input_channels = 10, 3
        elif ds == 'CIFAR100':
            num_classes, input_channels = 100, 3
        elif ds == 'FEMNIST':
            num_classes, input_channels = 62, 1
        else:
            raise ValueError(f"Unknown dataset: {ds}")

        model = create_model(
            self.config['model'],
            num_classes=num_classes,
            input_channels=input_channels
        )

        if str(os.getenv('POL_SUPPRESS_MODEL_INFO', '0')).strip().lower() not in ('1', 'true', 'yes', 'on'):
            print_model_info(model, self.config['model'])
        return model

    def _client_train_slots(self):
        """Return worker-slot device strings for parallel client training."""
        enabled = str(os.getenv('POL_ENABLE_PARALLEL_CLIENT_TRAINING', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
        if not enabled:
            return []

        raw_devices = str(os.getenv('POL_CLIENT_TRAIN_DEVICES', '')).strip()
        if raw_devices:
            devices = [d.strip() for d in raw_devices.split(',') if d.strip()]
        elif torch.cuda.is_available():
            devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        else:
            devices = ['cpu']

        normalized = []
        for dev in devices:
            low = dev.lower()
            if low == 'cpu':
                normalized.append('cpu')
            elif low.startswith('cuda:'):
                normalized.append(low)
            else:
                normalized.append(f'cuda:{int(dev)}' if str(dev).isdigit() else dev)
        if not normalized:
            return []

        try:
            workers_per_device = max(1, int(os.getenv('POL_CLIENT_TRAIN_WORKERS_PER_DEVICE', '1')))
        except Exception:
            workers_per_device = 1
        slots = []
        for dev in normalized:
            slots.extend([dev] * workers_per_device)

        try:
            max_workers = int(os.getenv('POL_CLIENT_TRAIN_MAX_WORKERS', '0'))
            if max_workers > 0:
                slots = slots[:max_workers]
        except Exception:
            pass
        return slots

    @staticmethod
    def _thread_local_loader(loader, shuffle_seed=None):
        """Avoid spawning DataLoader worker processes from worker threads."""
        try:
            if int(getattr(loader, 'num_workers', 0) or 0) == 0 and shuffle_seed is None:
                return loader
            generator = None
            if shuffle_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(shuffle_seed))
            return torch.utils.data.DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=getattr(loader, 'pin_memory', False),
                drop_last=getattr(loader, 'drop_last', False),
                generator=generator,
            )
        except Exception:
            return loader

    @staticmethod
    def _clone_loader_like(loader, *, dataset=None, shuffle_seed=None):
        """Create a loader with the same surface config and an optional dataset."""
        try:
            generator = None
            if shuffle_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(shuffle_seed))
            return torch.utils.data.DataLoader(
                dataset if dataset is not None else loader.dataset,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=getattr(loader, 'pin_memory', False),
                drop_last=getattr(loader, 'drop_last', False),
                generator=generator,
            )
        except Exception:
            return loader

    def _train_pol_client_for_round(
        self,
        *,
        idx: int,
        round_num: int,
        attack_type: str,
        attack_params: dict,
        malicious_set: set,
        global_state: dict,
        pol_save_dir: str,
        train_device: str,
        sybil_shared_dataset=None,
    ):
        """Build and train one PoL client, returning a client object and benign update."""
        client_id = f"client_{int(idx)}"
        device = torch.device(train_device)
        if device.type == 'cuda':
            try:
                torch.cuda.set_device(device)
            except Exception:
                pass

        client_model = self.create_model().to(device)
        client_model.load_state_dict(global_state)

        trainer_args = {
            'device': str(device),
            'optimizer': FL_CONFIG.get('optimizer', 'SGD'),
            'lr': FL_CONFIG.get('learning_rate', 0.01),
            'momentum': FL_CONFIG.get('momentum', 0.9),
            'weight_decay': FL_CONFIG.get('weight_decay', 1e-4),
            'enable_pol': True,
            'pol_save_freq': self.config.get('pol_config', {}).get('save_freq', 10),
            'pol_save_dir': pol_save_dir,
            'pol_compress': True,
            'client_id': client_id,
            'round_num': int(round_num),
        }

        is_malicious = int(idx) in malicious_set
        base_loader = self.train_loaders[int(idx)]
        if 'sybil' in attack_type and is_malicious and sybil_shared_dataset is not None:
            try:
                base_loader = self._clone_loader_like(
                    base_loader,
                    dataset=sybil_shared_dataset,
                    shuffle_seed=_stable_seed32(int(os.getenv('SEED', '42')), round_num, 'sybil', idx),
                )
                logger.info(
                    "[Sybil] %s using shared data view with %d samples",
                    client_id,
                    len(getattr(base_loader, 'dataset', [])),
                )
            except Exception as e:
                logger.warning("[Sybil] Failed to apply shared data view for %s: %s", client_id, e)
        pol_loader = base_loader
        if ('byzantine_label_flipping' in attack_type) and (idx in malicious_set):
            pol_loader = self._wrap_loader_with_label_flipping(base_loader, flip_prob=attack_params.get('flip_probability', 1.0))
        elif attack_type == 'data_poisoning' and (idx in malicious_set):
            pol_loader = self._wrap_loader_with_data_poisoning(base_loader, poison_ratio=attack_params.get('poison_ratio', 0.1))

        if str(os.getenv('POL_ENABLE_PARALLEL_CLIENT_TRAINING', '0')).strip().lower() in ('1', 'true', 'yes', 'on'):
            try:
                base_seed = int(os.getenv('SEED', '42'))
            except Exception:
                base_seed = 42
            pol_loader = self._thread_local_loader(
                pol_loader,
                shuffle_seed=_stable_seed32(base_seed, round_num, idx),
            )

        trainer = PoLTrainer(
            model=client_model,
            dataloader=pol_loader,
            criterion=nn.CrossEntropyLoss(),
            args=trainer_args
        )
        client = PoLClient(
            client_id=client_id,
            dataloader=base_loader,
            model=client_model,
            trainer=trainer,
            args=trainer_args
        )

        if is_malicious:
            if 'byzantine' in attack_type:
                client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)
            elif 'free_riding' in attack_type:
                attack_name = attack_type.replace('free_riding_', '')
                attack = create_free_riding_attack(attack_name, **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'})
                if attack.should_train():
                    num_epochs = attack.get_training_epochs() if hasattr(attack, 'get_training_epochs') else FL_CONFIG['local_epochs']
                    logger.info(f"[Malicious] {client_id} lazy training ({num_epochs} epochs)")
                    client.train(total_epoch=num_epochs, dataset=None)
            elif attack_type == 'data_poisoning':
                client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)
            elif 'sybil' in attack_type:
                client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)
        else:
            client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)

        try:
            client.model.to('cpu')
        except Exception:
            pass
        benign_update = client.model.state_dict() if not is_malicious else None
        return client, benign_update

    def _train_pol_clients_for_round(
        self,
        *,
        selected_indices,
        round_num: int,
        attack_type: str,
        attack_params: dict,
        malicious_indices,
        global_state: dict,
        pol_save_dir: str,
        sybil_anchor_idx=None,
    ):
        malicious_set = {int(i) for i in malicious_indices}
        sybil_shared_dataset = None
        if 'sybil' in attack_type and malicious_set:
            try:
                anchor_idx = int(sybil_anchor_idx) if sybil_anchor_idx is not None else min(malicious_set)
                sybil_shared_dataset = self.train_loaders[int(anchor_idx)].dataset
                logger.info(
                    "[Sybil] Shared data anchor client_%d for %d malicious identities",
                    int(anchor_idx),
                    len(malicious_set),
                )
            except Exception as e:
                sybil_shared_dataset = None
                logger.warning("[Sybil] Shared data view unavailable: %s", e)
        slots = self._client_train_slots()
        if len(slots) <= 1:
            device = slots[0] if slots else str(self.device)
            clients = []
            benign_updates = []
            for idx in selected_indices:
                client, benign_update = self._train_pol_client_for_round(
                    idx=int(idx),
                    round_num=round_num,
                    attack_type=attack_type,
                    attack_params=attack_params,
                    malicious_set=malicious_set,
                    global_state=global_state,
                    pol_save_dir=pol_save_dir,
                    train_device=device,
                    sybil_shared_dataset=sybil_shared_dataset,
                )
                clients.append(client)
                if benign_update is not None:
                    benign_updates.append(benign_update)
            return clients, benign_updates

        indexed = list(enumerate([int(i) for i in selected_indices]))
        chunks = [[] for _ in slots]
        for pos, item in enumerate(indexed):
            chunks[pos % len(slots)].append(item)

        logger.info(
            "Parallel PoL client training enabled: %d slot(s), devices=%s, clients=%d",
            len(slots),
            slots,
            len(indexed),
        )
        results = [None] * len(indexed)

        def run_chunk(slot_id, device_name, chunk):
            out = []
            for original_pos, client_idx in chunk:
                client, benign_update = self._train_pol_client_for_round(
                    idx=int(client_idx),
                    round_num=round_num,
                    attack_type=attack_type,
                    attack_params=attack_params,
                    malicious_set=malicious_set,
                    global_state=global_state,
                    pol_save_dir=pol_save_dir,
                    train_device=device_name,
                    sybil_shared_dataset=sybil_shared_dataset,
                )
                out.append((original_pos, client, benign_update))
                if str(device_name).startswith('cuda') and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            return out

        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = [
                pool.submit(run_chunk, slot_id, slots[slot_id], chunk)
                for slot_id, chunk in enumerate(chunks)
                if chunk
            ]
            for fut in as_completed(futures):
                for original_pos, client, benign_update in fut.result():
                    results[original_pos] = (client, benign_update)

        clients = [item[0] for item in results if item is not None]
        benign_updates = [item[1] for item in results if item is not None and item[1] is not None]
        return clients, benign_updates

    @staticmethod
    def _cpu_state_dict(model):
        return OrderedDict(
            (k, v.detach().cpu().clone())
            for k, v in model.state_dict().items()
        )

    def _train_baseline_client_for_round(
        self,
        *,
        idx: int,
        round_num: int,
        attack_type: str,
        attack_params: dict,
        malicious_set: set,
        global_state: dict,
        train_device: str,
    ):
        device = torch.device(train_device)
        if device.type == 'cuda':
            try:
                torch.cuda.set_device(device)
            except Exception:
                pass

        client_model = self.create_model().to(device)
        client_model.load_state_dict(global_state)
        is_malicious = int(idx) in malicious_set
        train_loader = self.train_loaders[int(idx)]

        def thread_loader(loader, tag='baseline'):
            if str(os.getenv('POL_ENABLE_PARALLEL_CLIENT_TRAINING', '0')).strip().lower() in ('1', 'true', 'yes', 'on'):
                try:
                    base_seed = int(os.getenv('SEED', '42'))
                except Exception:
                    base_seed = 42
                return self._thread_local_loader(
                    loader,
                    shuffle_seed=_stable_seed32(base_seed, round_num, tag, idx),
                )
            return loader

        if is_malicious:
            if 'byzantine' in attack_type:
                attack_name = attack_type.replace('byzantine_', '')
                if attack_name not in ['random_noise', 'model_replacement', 'alie', 'ipm', 'minmax']:
                    mal_loader = self._wrap_loader_with_label_flipping(
                        train_loader,
                        flip_prob=attack_params.get('flip_probability', 1.0),
                    )
                    self._train_client(client_model, thread_loader(mal_loader, 'label_flip'), round_num=round_num)
                elif attack_name in ['alie', 'ipm', 'minmax']:
                    self._train_client(client_model, thread_loader(train_loader), round_num=round_num)
            elif 'free_riding' in attack_type:
                attack_name = attack_type.replace('free_riding_', '')
                attack = create_free_riding_attack(
                    attack_name,
                    **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'}
                )
                if attack.should_train():
                    num_epochs = attack.get_training_epochs() if hasattr(attack, 'get_training_epochs') else FL_CONFIG['local_epochs']
                    logger.info(f"[Malicious] client_{int(idx)} lazy training ({num_epochs} epochs)")
                    self._train_client(client_model, thread_loader(train_loader), num_epochs=num_epochs, round_num=round_num)
            elif attack_type == 'data_poisoning':
                poisoned_loader = self._wrap_loader_with_data_poisoning(
                    train_loader,
                    poison_ratio=attack_params.get('poison_ratio', 0.1),
                )
                self._train_client(client_model, thread_loader(poisoned_loader, 'data_poison'), round_num=round_num)
            elif 'sybil' in attack_type:
                self._train_client(client_model, thread_loader(train_loader), round_num=round_num)
        else:
            self._train_client(client_model, thread_loader(train_loader), round_num=round_num)

        state = self._cpu_state_dict(client_model)
        benign_update = state if not is_malicious else None
        try:
            client_model.to('cpu')
        except Exception:
            pass
        return state, benign_update

    def _train_baseline_clients_for_round(
        self,
        *,
        selected_indices,
        round_num: int,
        attack_type: str,
        attack_params: dict,
        malicious_indices,
        global_state: dict,
    ):
        malicious_set = {int(i) for i in malicious_indices}
        slots = self._client_train_slots()
        if len(slots) <= 1:
            device = slots[0] if slots else str(self.device)
            client_models = []
            benign_updates = []
            for idx in selected_indices:
                state, benign_update = self._train_baseline_client_for_round(
                    idx=int(idx),
                    round_num=round_num,
                    attack_type=attack_type,
                    attack_params=attack_params,
                    malicious_set=malicious_set,
                    global_state=global_state,
                    train_device=device,
                )
                client_models.append(state)
                if benign_update is not None:
                    benign_updates.append(benign_update)
            return client_models, benign_updates

        indexed = list(enumerate([int(i) for i in selected_indices]))
        chunks = [[] for _ in slots]
        for pos, item in enumerate(indexed):
            chunks[pos % len(slots)].append(item)

        logger.info(
            "Parallel baseline client training enabled: %d slot(s), devices=%s, clients=%d",
            len(slots),
            slots,
            len(indexed),
        )
        results = [None] * len(indexed)

        def run_chunk(slot_id, device_name, chunk):
            out = []
            for original_pos, client_idx in chunk:
                state, benign_update = self._train_baseline_client_for_round(
                    idx=int(client_idx),
                    round_num=round_num,
                    attack_type=attack_type,
                    attack_params=attack_params,
                    malicious_set=malicious_set,
                    global_state=global_state,
                    train_device=device_name,
                )
                out.append((original_pos, state, benign_update))
                if str(device_name).startswith('cuda') and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            return out

        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = [
                pool.submit(run_chunk, slot_id, slots[slot_id], chunk)
                for slot_id, chunk in enumerate(chunks)
                if chunk
            ]
            for fut in as_completed(futures):
                for original_pos, state, benign_update in fut.result():
                    results[original_pos] = (state, benign_update)

        client_models = []
        benign_updates = []
        for item in results:
            if item is None:
                continue
            state, benign_update = item
            client_models.append(state)
            if benign_update is not None:
                benign_updates.append(benign_update)
        return client_models, benign_updates

    @staticmethod
    def _round_csv_fieldnames():
        return [
            'round', 'test_accuracy', 'verification_pass_rate',
            'num_selected_clients', 'num_malicious_in_round',
            'attack_l2_mean', 'attack_l2_max',
            'detection_tpr', 'detection_tpr_e2e', 'detection_tpr_conditional',
            'detection_fpr', 'precision', 'recall', 'f1', 'participation_rate',
            'external_agg_success', 'external_agg_latency_s',
            'remote_majority_responders', 'remote_majority_yes', 'pol_verify_time_s',
            'remote_verify_latency_p50_s', 'remote_verify_latency_p95_s',
            'remote_error_timeout', 'remote_error_network', 'remote_error_invalid',
            'remote_error_business', 'external_agg_error_type',
        ]

    @staticmethod
    def _sanitize_cell_name(value: str) -> str:
        return str(value).replace('/', '_').replace(' ', '_')

    def _write_round_progress(self, attack_type: str, baseline_method: str, rows: list, test_accuracies: list):
        """Flush partial round metrics so interrupted long runs keep usable evidence."""
        try:
            csv_name = (
                f"rq1_rounds_{self._sanitize_cell_name(self.config.get('dataset', 'DATA'))}_"
                f"{self._sanitize_cell_name(attack_type)}_{self._sanitize_cell_name(baseline_method)}.csv"
            )
            csv_path = self.output_dir / csv_name
            fieldnames = self._round_csv_fieldnames()
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: (r.get(k, 0.0) if k != 'round' else r.get(k)) for k in fieldnames})

            partial_path = self.output_dir / (
                f"rq1_partial_{self._sanitize_cell_name(self.config.get('dataset', 'DATA'))}_"
                f"{self._sanitize_cell_name(attack_type)}_{self._sanitize_cell_name(baseline_method)}.json"
            )
            payload = {
                'attack_type': attack_type,
                'baseline_method': baseline_method,
                'completed_rounds': len(rows),
                'test_accuracies': list(test_accuracies),
                'latest_accuracy': float(test_accuracies[-1]) if test_accuracies else None,
                'rounds': rows,
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            }
            with open(partial_path, 'w') as pf:
                json.dump(payload, pf, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write per-round progress: {e}")

    def run_experiment(self, attack_type: str, attack_params: dict, baseline_method: str):
        """
        Run one experiment with specific attack and baseline

        Args:
            attack_type: Type of attack
            attack_params: Attack parameters
            baseline_method: Baseline aggregation method

        Returns:
            results: Experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {baseline_method} vs {attack_type}")
        logger.info(f"Attack params: {attack_params}")
        logger.info(f"{'='*70}\n")

        # Create global model
        global_model = self.create_model().to(self.device)

        # Create aggregator
        # Use the paper threat model: a fixed malicious population of clients,
        # rather than re-labeling the first k sampled clients as malicious each round.
        malicious_ratio = attack_params.get('malicious_ratios', [0.1])[0]
        num_malicious_total = self._num_malicious_from_ratio(self.config['num_clients'], malicious_ratio)
        malicious_population = set(
            int(i) for i in np.random.choice(
                self.config['num_clients'],
                num_malicious_total,
                replace=False,
            )
        ) if num_malicious_total > 0 else set()
        expected_malicious_per_round = self._num_malicious_from_ratio(self.config['clients_per_round'], malicious_ratio)
        blocked_client_ids = set()
        logger.info(
            "Fixed malicious population: %d/%d clients (%s)",
            len(malicious_population),
            self.config['num_clients'],
            sorted(malicious_population),
        )
        sybil_anchor_idx = min(malicious_population) if ('sybil' in attack_type and malicious_population) else None
        if sybil_anchor_idx is not None:
            logger.info(
                "[Sybil] Fixed attack data anchor client_%d for the full run",
                int(sybil_anchor_idx),
            )

        if baseline_method == 'PoL_FL':
            pol_config = dict(self.config.get('pol_config', {}) or {})
            sybil_context = 'sybil' in str(attack_type).lower()
            pol_config['enable_sybil_detector'] = bool(sybil_context)
            os.environ['POL_ATTACK_CONTEXT'] = str(attack_type)
            os.environ['POL_ENABLE_SYBIL_DETECTOR'] = '1' if sybil_context else '0'
            os.environ['POL_SYBIL_TRAJECTORY_ONLY'] = '0'
            aggregator = PoLExperimentHelper.setup_pol_aggregator(
                model=global_model,
                pol_config=pol_config,
                device=str(self.device)
            )
        elif baseline_method == 'Krum':
            aggregator = create_aggregator(
                baseline_method,
                num_byzantine=expected_malicious_per_round,
                multi_krum=True,
            )
        elif baseline_method == 'SDEA':
            aggregator = create_aggregator(baseline_method, num_byzantine=expected_malicious_per_round)
        elif baseline_method == 'Trimmed_Mean':
            aggregator = create_aggregator(baseline_method, trim_ratio=0.1)
        else:
            aggregator = create_aggregator(baseline_method)

        # Training loop
        test_accuracies = []
        verification_results_per_round = []  # Track verification results for PoL-FL
        malicious_ids_per_round = []  # Track malicious client IDs for each round
        malicious_ids_union = set()
        client_ids_union = set()
        per_round_rows = []  # For CSV output
        baseline_detection_accum = {}

        # Initialize Sybil Attack if needed
        sybil_attack = None
        sybil_identities = []
        if 'sybil' in attack_type:
            num_identities = attack_params.get('num_identities', 5)
            shared_data_ratio = attack_params.get('shared_data_ratio', 1.0)
            sybil_attack = SybilAttack(
                num_identities=num_identities,
                shared_data_ratio=shared_data_ratio,
                base_client_id="attacker"
            )
            sybil_identities = sybil_attack.create_identities()
            logger.info(f"Created Sybil attack with {num_identities} fake identities: {sybil_identities}")

        for round_num in range(self.config['num_rounds']):
            logger.info(f"Round {round_num + 1}/{self.config['num_rounds']}")
            det = None
            round_attack_l2 = []

            # Select clients for this round
            num_selected = self.config['clients_per_round']
            if baseline_method == 'PoL_FL' and blocked_client_ids:
                eligible = [
                    idx for idx in range(self.config['num_clients'])
                    if f"client_{int(idx)}" not in blocked_client_ids
                ]
            else:
                eligible = list(range(self.config['num_clients']))
            actual_num_selected = min(num_selected, len(eligible))
            if actual_num_selected < num_selected:
                logger.info(
                    "  Active client pool reduced by PoL exclusions: selected %d/%d",
                    actual_num_selected,
                    num_selected,
                )
            selected_indices = np.random.choice(
                np.array(eligible, dtype=int),
                actual_num_selected,
                replace=False
            )

            # Determine malicious clients
            malicious_indices = np.array(
                [int(idx) for idx in selected_indices if int(idx) in malicious_population],
                dtype=int,
            )
            # Track malicious clients for this round
            malicious_ids_this_round = [f"client_{int(i)}" for i in malicious_indices]
            malicious_ids_per_round.append(malicious_ids_this_round)
            # Track union sets across rounds for academically correct detection metrics
            client_ids_union.update([f"client_{int(i)}" for i in selected_indices])
            malicious_ids_union.update(malicious_ids_this_round)

            # Client training & aggregation (PoL vs Non-PoL)
            if baseline_method == 'PoL_FL':
                # Build PoL clients and use receive_upload() so verification actually runs
                # Two-phase training for Blades attacks to provide benign_updates
                pre_global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
                pol_save_dir = str(self.output_dir / 'pol_data')

                # Phase 1: Train all clients. This helper preserves selected_indices
                # order, so attack application and weighted aggregation keep the same
                # semantics as the original serial implementation.
                clients, benign_updates = self._train_pol_clients_for_round(
                    selected_indices=selected_indices,
                    round_num=round_num,
                    attack_type=attack_type,
                    attack_params=attack_params,
                    malicious_indices=malicious_indices,
                    global_state=pre_global_state,
                    pol_save_dir=pol_save_dir,
                    sybil_anchor_idx=sybil_anchor_idx,
                )

                # Phase 2: Apply Blades attacks with benign_updates
                if 'free_riding' in attack_type:
                    attack_name = attack_type.replace('free_riding_', '')
                    attack = create_free_riding_attack(attack_name, **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'})
                    if not attack.should_train():
                        for i, idx in enumerate(selected_indices):
                            if idx in malicious_indices:
                                client = clients[i]
                                attacked_state = attack.apply(pre_global_state)
                                client.model.load_state_dict(attacked_state)
                                logger.info(f"Applied {attack_name} free-riding attack to {client.client_id}")

                if 'byzantine' in attack_type and len(benign_updates) > 0:
                    attack_name = attack_type.replace('byzantine_', '')
                    # Check if this is a Blades attack that needs benign_updates
                    if attack_name in ['alie', 'ipm', 'minmax']:
                        attack = create_attack(attack_name, **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'})
                        for i, idx in enumerate(selected_indices):
                            if idx in malicious_indices:
                                client = clients[i]
                                # Apply Blades attack with benign_updates
                                original_state = client.model.state_dict()
                                attacked_state = attack.apply(
                                    original_state,
                                    global_model=pre_global_state,
                                    benign_updates=benign_updates
                                )
                                round_attack_l2.append(self._state_l2_distance(attacked_state, original_state))
                                client.model.load_state_dict(attacked_state)
                                logger.info(f"Applied {attack_name} attack to {client.client_id} with {len(benign_updates)} benign updates")
                    else:
                        # FIX: Apply non-Blades Byzantine attacks (random_noise, model_replacement, etc.)
                        attack = create_attack(attack_name, **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'})
                        for i, idx in enumerate(selected_indices):
                            if idx in malicious_indices:
                                client = clients[i]
                                original_state = client.model.state_dict()
                                attacked_state = attack.apply(original_state, global_model=pre_global_state)
                                round_attack_l2.append(self._state_l2_distance(attacked_state, original_state))
                                client.model.load_state_dict(attacked_state)
                                logger.info(f"Applied {attack_name} attack to {client.client_id}")

                # PoL verify + aggregate
                aggregator.receive_upload(clients)
                aggregated_state = aggregator.aggregate()
                global_model.load_state_dict(aggregated_state)
                for cid, passed in getattr(aggregator, 'verification_results', {}).items():
                    if not bool(passed):
                        blocked_client_ids.add(str(cid))
            else:
                # Non-PoL baselines: Two-phase training for Blades attacks
                pre_global_state = self._cpu_state_dict(global_model)
                client_models, benign_updates = self._train_baseline_clients_for_round(
                    selected_indices=selected_indices,
                    round_num=round_num,
                    attack_type=attack_type,
                    attack_params=attack_params,
                    malicious_indices=malicious_indices,
                    global_state=pre_global_state,
                )

                # Phase 2: Apply Blades attacks with benign_updates
                if 'free_riding' in attack_type:
                    attack_name = attack_type.replace('free_riding_', '')
                    attack = create_free_riding_attack(attack_name, **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'})
                    if not attack.should_train():
                        for i, idx in enumerate(selected_indices):
                            if idx in malicious_indices:
                                client_models[i] = attack.apply(pre_global_state)
                                logger.info(f"Applied {attack_name} free-riding attack to client_{int(idx)}")

                if 'byzantine' in attack_type and len(benign_updates) > 0:
                    attack_name = attack_type.replace('byzantine_', '')
                    # Check if this is a Blades attack that needs benign_updates
                    if attack_name in ['alie', 'ipm', 'minmax']:
                        attack = create_attack(attack_name, **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'})
                        for i, idx in enumerate(selected_indices):
                            if idx in malicious_indices:
                                # Apply Blades attack with benign_updates
                                original_state = client_models[i]
                                attacked_state = attack.apply(
                                    original_state,
                                    global_model=pre_global_state,
                                    benign_updates=benign_updates
                                )
                                round_attack_l2.append(self._state_l2_distance(attacked_state, original_state))
                                client_models[i] = attacked_state
                                logger.info(f"Applied {attack_name} attack to client_{int(idx)} with {len(benign_updates)} benign updates")
                    else:
                        # Apply non-Blades Byzantine attacks
                        attack = create_attack(attack_name, **{k: v for k, v in attack_params.items() if k != 'malicious_ratios'})
                        for i, idx in enumerate(selected_indices):
                            if idx in malicious_indices:
                                original_state = client_models[i]
                                attacked_state = attack.apply(original_state, global_model=pre_global_state)
                                round_attack_l2.append(self._state_l2_distance(attacked_state, original_state))
                                client_models[i] = attacked_state
                                logger.info(f"Applied {attack_name} attack to client_{int(idx)}")

                # Use dataset-size weighted FedAvg for fairness with PoL aggregator
                weights = [float(len(self.train_loaders[idx].dataset)) for idx in selected_indices]
                total_w = sum(weights) if weights else 0.0
                if total_w > 0.0:
                    weights = [w / total_w for w in weights]
                else:
                    weights = [1.0 / len(client_models)] * len(client_models)

                aggregated_state = aggregator.aggregate(client_models, weights)
                global_model.load_state_dict(aggregated_state)

                if baseline_method != 'Vanilla_FL':
                    detected_ids = self._baseline_suspects(
                        baseline_method,
                        aggregator,
                        client_models,
                        selected_indices,
                        expected_malicious_per_round,
                    )
                    all_client_ids = [f"client_{int(i)}" for i in selected_indices]
                    malicious_client_ids = [f"client_{int(i)}" for i in malicious_indices]
                    baseline_verification = {cid: (cid not in detected_ids) for cid in all_client_ids}
                    det = PoLExperimentHelper.compute_detection_metrics(
                        baseline_verification,
                        malicious_client_ids,
                        all_client_ids,
                    )
                    self._sum_detection_counts(baseline_detection_accum, det)

            # Extract verification results for PoL-FL and compute per-round metrics
            vpass_rate = 0.0
            if baseline_method == 'PoL_FL' and hasattr(aggregator, 'verification_results'):
                verification_results = PoLExperimentHelper.extract_verification_results(
                    aggregator.verification_results
                )
                verification_results_per_round.append(verification_results)
                # Per-round verification pass rate
                total_verified = len(verification_results)
                pass_count = sum(1 for v in verification_results.values() if v)
                vpass_rate = (pass_count / total_verified) if total_verified > 0 else 0.0
                # Per-round detection metrics (TPR/FPR/Precision/Recall/F1)
                malicious_client_ids = [f"client_{int(i)}" for i in malicious_indices]
                all_client_ids = [f"client_{int(i)}" for i in selected_indices]
                det = PoLExperimentHelper.compute_detection_metrics(
                    verification_results, malicious_client_ids, all_client_ids
                )
                # Conditional TPR among verified malicious clients
                verified_mal = [cid for cid in malicious_client_ids if cid in verification_results]
                if len(verified_mal) > 0:
                    tp_cond = sum((not verification_results[cid]) for cid in verified_mal)
                    tpr_cond = tp_cond / len(verified_mal)
                else:
                    tpr_cond = 0.0

            # Evaluate
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)
            test_accuracies.append(test_acc)

            # Collect per-round row (CSV)
            row = {
                'round': round_num + 1,
                'test_accuracy': float(test_acc),
                'verification_pass_rate': float(vpass_rate),
                'num_selected_clients': int(len(selected_indices)),
                'num_malicious_in_round': int(len(malicious_indices)),
                'attack_l2_mean': float(np.mean(round_attack_l2)) if round_attack_l2 else 0.0,
                'attack_l2_max': float(np.max(round_attack_l2)) if round_attack_l2 else 0.0,
            }
            if isinstance(det, dict):
                row.update({
                    'detection_tpr': float(det.get('TPR', 0.0)),  # e2e for backward compat
                    'detection_tpr_e2e': float(det.get('TPR_e2e', det.get('TPR', 0.0))),
                    'detection_tpr_conditional': float(det.get('TPR_conditional', det.get('TPR', 0.0))),
                    'detection_fpr': float(det.get('FPR', 0.0)),
                    'precision': float(det.get('Precision', 0.0)),
                    'recall': float(det.get('Recall', 0.0)),
                    'f1': float(det.get('F1', 0.0)),
                    'participation_rate': float(det.get('participation_rate', 0.0)),
                })
            # Decentralization metrics
            if baseline_method == 'PoL_FL' and hasattr(aggregator, 'metrics'):
                m = getattr(aggregator, 'metrics', {}) or {}
                try:
                    row.update({
                        'external_agg_success': int(bool(m.get('external_agg_success', False))),
                        'external_agg_latency_s': float(m.get('external_agg_latency_s', 0.0)),
                        'remote_majority_responders': int(m.get('remote_majority_responders', 0)),
                        'remote_majority_yes': int(m.get('remote_majority_yes', 0)),
                        'pol_verify_time_s': float(m.get('pol_verify_time_s', 0.0)),
                        'remote_verify_latency_p50_s': float(m.get('remote_verify_latency_p50_s', 0.0)),
                        'remote_verify_latency_p95_s': float(m.get('remote_verify_latency_p95_s', 0.0)),
                        'remote_error_timeout': int(m.get('remote_error_timeout', 0)),
                        'remote_error_network': int(m.get('remote_error_network', 0)),
                        'remote_error_invalid': int(m.get('remote_error_invalid', 0)),
                        'remote_error_business': int(m.get('remote_error_business', 0)),
                        'external_agg_error_type': str(m.get('external_agg_error_type', '')),
                    })
                except Exception:
                    row.update({
                        'external_agg_success': 0,
                        'external_agg_latency_s': 0.0,
                        'remote_majority_responders': 0,
                        'remote_majority_yes': 0,
                        'pol_verify_time_s': 0.0,
                        'remote_verify_latency_p50_s': 0.0,
                        'remote_verify_latency_p95_s': 0.0,
                        'remote_error_timeout': 0,
                        'remote_error_network': 0,
                        'remote_error_invalid': 0,
                        'remote_error_business': 0,
                        'external_agg_error_type': '',
                    })
            else:
                row.update({
                    'external_agg_success': 0,
                    'external_agg_latency_s': 0.0,
                    'remote_majority_responders': 0,
                    'remote_majority_yes': 0,
                    'pol_verify_time_s': 0.0,
                })
            per_round_rows.append(row)
            self._write_round_progress(attack_type, baseline_method, per_round_rows, test_accuracies)

            logger.info(f"  Test Accuracy: {test_acc:.4f}")
            if isinstance(det, dict):
                logger.info(f"  Detection Metrics: TPR={det.get('TPR', 0.0):.4f}, FPR={det.get('FPR', 0.0):.4f}")
            det = None
            try:
                if baseline_method == 'PoL_FL' and hasattr(aggregator, 'release_round_payloads'):
                    aggregator.release_round_payloads()
                for _round_obj_name in (
                    'clients',
                    'client_models',
                    'benign_updates',
                    'pre_global_state',
                    'aggregated_state',
                ):
                    _round_obj = locals().get(_round_obj_name)
                    if isinstance(_round_obj, (list, dict, OrderedDict)):
                        _round_obj.clear()
                client = None
                trainer = None
                client_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"Round memory cleanup skipped: {e}")

        # Compute convergence round
        convergence_round = compute_convergence_round(test_accuracies, threshold=0.85)

        results = {
            'attack_type': attack_type,
            'attack_params': attack_params,
            'baseline_method': baseline_method,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'convergence_round': convergence_round,
            'rounds': per_round_rows,
        }

        # Add detection metrics for PoL-FL
        if baseline_method == 'PoL_FL' and verification_results_per_round:
            # FIX: Use "blacklist" mechanism to accumulate detections across rounds
            # Once a client is detected as malicious, it should remain marked as malicious
            # This prevents later benign verifications from "whitewashing" previous detections
            detected_malicious_clients = set()
            all_verified_clients = set()

            for round_results in verification_results_per_round:
                for client_id, passed in round_results.items():
                    all_verified_clients.add(client_id)
                    if not passed:  # Verification failed = detected as malicious
                        detected_malicious_clients.add(client_id)

            # Build final verification_results: False for detected malicious, True for others
            all_verification_results = {}
            for client_id in client_ids_union:
                if client_id in detected_malicious_clients:
                    all_verification_results[client_id] = False  # Detected as malicious
                elif client_id in all_verified_clients:
                    all_verification_results[client_id] = True   # Verified as benign
                # Note: Clients never verified are not included in all_verification_results

            # Compute detection metrics over union across rounds (not just last round)
            malicious_client_ids_all = sorted(list(malicious_ids_union))
            client_ids_all = sorted(list(client_ids_union))
            detection_metrics = PoLExperimentHelper.compute_detection_metrics(
                all_verification_results,
                malicious_client_ids_all,
                client_ids_all
            )

            # FIX: Compute TPR_conditional correctly by collecting all verification instances
            # Use per-round malicious client IDs (not union) to avoid counting honest clients as malicious
            tp_cond = 0
            fn_cond = 0
            for round_idx, round_results in enumerate(verification_results_per_round):
                # Get malicious clients for this specific round
                malicious_set_this_round = set(malicious_ids_per_round[round_idx])
                for client_id, is_valid in round_results.items():
                    if client_id in malicious_set_this_round:
                        if not is_valid:
                            tp_cond += 1  # Detected as malicious
                        else:
                            fn_cond += 1  # Missed (passed verification)

            tpr_conditional_corrected = tp_cond / (tp_cond + fn_cond) if (tp_cond + fn_cond) > 0 else 0.0
            detection_metrics['TPR_conditional'] = float(tpr_conditional_corrected)
            detection_metrics['TP_conditional'] = int(tp_cond)
            detection_metrics['FN_conditional'] = int(fn_cond)
            detection_metrics['total_malicious_verifications'] = int(tp_cond + fn_cond)

            results['detection_metrics'] = detection_metrics
            logger.info(f"  Detection Metrics: TPR={detection_metrics['TPR']:.4f}, FPR={detection_metrics['FPR']:.4f}")
            logger.info(f"  TPR_conditional (corrected): {tpr_conditional_corrected:.4f} ({tp_cond}/{tp_cond + fn_cond} malicious verifications)")
        elif baseline_method != 'Vanilla_FL' and baseline_detection_accum:
            detection_metrics = self._final_detection_metrics_from_counts(baseline_detection_accum)
            results['detection_metrics'] = detection_metrics
            logger.info(f"  Baseline Detection Metrics: TPR={detection_metrics['TPR']:.4f}, FPR={detection_metrics['FPR']:.4f}")

        # Persist per-round CSV for plotting
        try:
            ds = str(self.config.get('dataset', 'DATA'))
            def _san(s: str) -> str:
                return s.replace('/', '_').replace(' ', '_')
            csv_name = f"rq1_rounds_{_san(ds)}_{_san(attack_type)}_{_san(baseline_method)}.csv"
            csv_path = self.output_dir / csv_name
            fieldnames = self._round_csv_fieldnames()
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                for r in per_round_rows:
                    writer.writerow({k: (r.get(k, 0.0) if k != 'round' else r.get(k)) for k in fieldnames})
            logger.info(f"Per-round CSV saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to write per-round CSV: {e}")

        return results

    def _train_client(self, model, dataloader, num_epochs=None, round_num=0):
        """Train client model

        Args:
            model: Client model to train
            dataloader: Training data loader
            num_epochs: Number of epochs to train (default: FL_CONFIG['local_epochs'])
        """
        if num_epochs is None:
            num_epochs = FL_CONFIG['local_epochs']

        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=FL_CONFIG['learning_rate'],
            momentum=FL_CONFIG['momentum'],
            weight_decay=FL_CONFIG['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self._set_dataset_replay_context(
                getattr(dataloader, 'dataset', None),
                round_num=round_num,
                epoch=epoch,
            )
            for batch in dataloader:
                # Support both (data, target) and (data, target, idx) formats
                if len(batch) == 3:
                    data, target, _ = batch
                else:
                    data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def run_all_experiments(self):
        """Run all security experiments"""
        logger.info("Starting RQ1: Security Evaluation")

        # Prepare data
        self.prepare_data()

        all_results = []

        # Test each attack
        for attack_type, attack_config in self.config['attacks'].items():
            malicious_ratio = attack_config['malicious_ratios'][0]
            attack_params = {
                'malicious_ratios': [malicious_ratio],
                **{k: v for k, v in attack_config.items() if k != 'malicious_ratios'}
            }

            # Test with each baseline
            for baseline in self.config['baselines']:
                try:
                    results = self.run_experiment(
                        attack_type,
                        attack_params,
                        baseline
                    )
                    all_results.append(results)
                except Exception as e:
                    logger.error(f"Experiment failed: {baseline} vs {attack_type}: {e}")
                    import traceback
                    traceback.print_exc()

        if not all_results:
            raise RuntimeError("RQ1 produced no successful experiments; refusing to write an empty success result")

        # Save results and configuration
        results_file = self.output_dir / 'rq1_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")

        try:
            with open(self.output_dir / 'config.json', 'w') as cf:
                json.dump(self.config, cf, indent=2)
            logger.info(f"Run configuration saved to {self.output_dir / 'config.json'}")
        except Exception as e:
            logger.warning(f"Failed to write config.json: {e}")

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results):
        """Print experiment summary"""
        logger.info("\n" + "="*70)
        logger.info("RQ1: Security Evaluation Summary")
        logger.info("="*70)

        for result in results:
            logger.info(f"\n{result['baseline_method']} vs {result['attack_type']}:")
            logger.info(f"  Final Accuracy: {result['final_accuracy']:.4f}")
            logger.info(f"  Convergence Round: {result['convergence_round']}")

            # Print detection metrics for PoL-FL
            if 'detection_metrics' in result:
                metrics = result['detection_metrics']
                logger.info(f"  Detection Metrics:")
                logger.info(f"    TPR (Detection Rate): {metrics['TPR']:.4f}")
                logger.info(f"    FPR (False Positive Rate): {metrics['FPR']:.4f}")
                logger.info(f"    Precision: {metrics['Precision']:.4f}")
                logger.info(f"    Recall: {metrics['Recall']:.4f}")
                logger.info(f"    F1 Score: {metrics['F1']:.4f}")

        logger.info("="*70)


def main():
    """Main function with CLI configurability"""
    import argparse

    ap = argparse.ArgumentParser(description='RQ1: Security Evaluation')
    ap.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'FEMNIST'])
    ap.add_argument('--model', type=str, default=None, help='Model name (default: auto-select by dataset)')
    ap.add_argument('--num_rounds', type=int, default=None,
                   help='Number of rounds (default: auto-select based on dataset: MNIST=20, CIFAR=100)')
    ap.add_argument('--num_clients', type=int, default=20)
    ap.add_argument('--clients_per_round', type=int, default=10)
    ap.add_argument('--data_distribution', type=str, default=None, choices=['IID', 'NonIID_Dirichlet', 'Natural_Writer', 'FEMNIST_Natural', 'LEAF_Natural'])
    ap.add_argument('--dirichlet_alpha', type=float, default=None)
    ap.add_argument('--local_epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--learning_rate', type=float, default=None)
    ap.add_argument('--momentum', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--attacks', type=str, default='', help='Comma-separated subset of attacks to run')
    ap.add_argument('--baselines', type=str, default='', help='Comma-separated subset of baselines to run')
    ap.add_argument(
        '--attack_param',
        action='append',
        default=[],
        help='Override selected attack parameter as key=value. Can be repeated for calibration runs.',
    )
    # Minimal PoL overrides for quick diagnostics
    ap.add_argument('--pol_delta', type=float, default=None, help='Override PoL L2 distance threshold (delta)')
    ap.add_argument('--verification_rate', type=float, default=None, help='Override PoL verification_rate (0-1)')
    # Output directory override for parallel experiments
    ap.add_argument('--output_dir', type=str, default='experiments/results/rq1_security',
                   help='Output directory for results (default: experiments/results/rq1_security)')
    args = ap.parse_args()

    # Auto-select num_rounds based on dataset if not specified
    if args.num_rounds is None:
        if args.dataset == 'MNIST':
            args.num_rounds = 20
        else:  # CIFAR10, CIFAR100, FEMNIST
            args.num_rounds = 100

    # Build config
    cfg = dict(RQ1_CONFIG)
    cfg['dataset'] = args.dataset
    if args.model is not None:
        cfg['model'] = args.model
    else:
        if args.dataset in ('MNIST', 'FEMNIST'):
            cfg['model'] = 'SimpleCNN'
        elif args.dataset == 'CIFAR100':
            cfg['model'] = 'ResNet34'
        else:
            cfg['model'] = 'ResNet18'
    cfg['num_rounds'] = args.num_rounds
    cfg['num_clients'] = args.num_clients
    cfg['clients_per_round'] = args.clients_per_round
    if args.data_distribution is not None:
        cfg['data_distribution'] = args.data_distribution
    if args.dirichlet_alpha is not None:
        cfg['dirichlet_alpha'] = float(args.dirichlet_alpha)
    if args.local_epochs is not None:
        FL_CONFIG['local_epochs'] = int(args.local_epochs)
    if args.batch_size is not None:
        FL_CONFIG['batch_size'] = int(args.batch_size)
    if args.learning_rate is not None:
        FL_CONFIG['learning_rate'] = float(args.learning_rate)
    if args.momentum is not None:
        FL_CONFIG['momentum'] = float(args.momentum)
    if args.weight_decay is not None:
        FL_CONFIG['weight_decay'] = float(args.weight_decay)
    cfg['local_epochs'] = int(FL_CONFIG.get('local_epochs', 0))
    cfg['batch_size'] = int(FL_CONFIG.get('batch_size', 0))
    cfg['learning_rate'] = float(FL_CONFIG.get('learning_rate', 0.0))
    cfg['momentum'] = float(FL_CONFIG.get('momentum', 0.0))
    cfg['weight_decay'] = float(FL_CONFIG.get('weight_decay', 0.0))
    # Apply PoL overrides if provided
    if args.pol_delta is not None:
        cfg.setdefault('pol_config', {})['delta'] = float(args.pol_delta)
    if args.verification_rate is not None:
        cfg.setdefault('pol_config', {})['verification_rate'] = float(args.verification_rate)
    pol_env_int_overrides = {
        'POL_SAVE_FREQ': ('save_freq', 1),
        'POL_ALWAYS_VERIFY_LAST_K': ('always_verify_last_k', 0),
        'POL_RANDOM_Q': ('random_q', 0),
    }
    for env_name, (config_key, minimum) in pol_env_int_overrides.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        try:
            parsed_value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
        if parsed_value < minimum:
            raise ValueError(f"{env_name} must be >= {minimum}, got {parsed_value}")
        cfg.setdefault('pol_config', {})[config_key] = parsed_value

    # Filter attacks/baselines if provided
    if args.attacks:
        allowed = set(cfg['attacks'].keys())
        chosen = [a.strip() for a in args.attacks.split(',') if a.strip()]
        unknown = [a for a in chosen if a not in allowed]
        if unknown:
            raise ValueError(f"Unknown attacks: {unknown}. Allowed: {sorted(allowed)}")
        cfg['attacks'] = {k: cfg['attacks'][k] for k in chosen}
    if args.attack_param:
        overrides = {}
        for item in args.attack_param:
            if '=' not in item:
                raise ValueError(f"--attack_param must be key=value, got {item!r}")
            key, value = item.split('=', 1)
            key = key.strip()
            if not key:
                raise ValueError(f"--attack_param has empty key: {item!r}")
            overrides[key] = _parse_attack_param_value(value)
        for attack_cfg in cfg['attacks'].values():
            attack_cfg.update(overrides)
    if args.baselines:
        allowed_b = {'Vanilla_FL', 'Krum', 'Trimmed_Mean', 'Median', 'ShapleyFL', 'FoolsGold', 'SDEA', 'PoL_FL'}
        chosen_b = [b.strip() for b in args.baselines.split(',') if b.strip()]
        unknown_b = [b for b in chosen_b if b not in allowed_b]
        if unknown_b:
            raise ValueError(f"Unknown baselines: {unknown_b}. Allowed: {sorted(allowed_b)}")
        cfg['baselines'] = chosen_b

    # Safety guard: warn when using verification_rate < 1.0 in known-bad RQ1 settings
    pol_cfg = cfg.get('pol_config', {})
    vr = pol_cfg.get('verification_rate', None)
    if vr is not None and vr < 1.0:
        has_rn = 'byzantine_random_noise' in cfg.get('attacks', {})
        if args.dataset == 'MNIST' and has_rn:
            logger.warning(
                "RQ1 runner: verification_rate=%.3f < 1.0 with MNIST + byzantine_random_noise. "
                "During clearance we observed systematic model collapse under this setting. "
                "Use this only for ablations, not main RQ1 configs.",
                vr,
            )


    # Phase 1 parameter calibration (low-risk defaults)
    try:
        # Align minimal-steps threshold with this run's local_epochs
        os.environ['POL_MIN_EPOCHS'] = str(FL_CONFIG.get('local_epochs', 5))
        os.environ.setdefault('POL_MIN_STEPS_FRACTION', '0.95')
        # Dataset-specific final consistency threshold
        if cfg['dataset'] == 'MNIST':
            os.environ.setdefault('POL_FINAL_DELTA_OVERRIDE', '0.02')
    except Exception as e:
        logger.debug(f"Parameter calibration env injection skipped: {e}")

    logger.info("Running RQ1 Security Evaluation with configuration:")
    logger.info(json.dumps({k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items() if k != 'attacks'}, indent=2))
    logger.info(f"Attacks: {list(cfg['attacks'].keys())}")

    experiment = SecurityExperiment(cfg, output_dir=args.output_dir)
    results = experiment.run_all_experiments()

    logger.info("\nRQ1: Security Evaluation Completed!")
    logger.info(f"Total experiments: {len(results)}")


if __name__ == '__main__':
    main()
