"""
Experiment Configuration for PoL-BFL Evaluation

This file contains all experimental configurations for the four research questions.
"""

import torch
from pathlib import Path
import os


# ========== General Settings ==========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Project paths (computed relative to this file) and environment overrides
# Path: experiments/scripts/utils/experiment_config.py -> go up 3 levels to PoL-BFL
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # PoL-BFL
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
DEFAULT_RESULTS_DIR = DEFAULT_EXPERIMENTS_DIR / 'results'
DEFAULT_PLOTS_DIR = DEFAULT_EXPERIMENTS_DIR / 'plots'
DEFAULT_CHECKPOINTS_DIR = DEFAULT_EXPERIMENTS_DIR / 'checkpoints'
DEFAULT_LOG_DIR = PROJECT_ROOT / 'log'
# Data root can be overridden by POL_DATA_DIR / VERYFL_DATA_DIR
DATA_ROOT = Path(os.getenv('POL_DATA_DIR') or os.getenv('VERYFL_DATA_DIR') or (PROJECT_ROOT / 'data'))

RANDOM_SEED = 42
DEFAULT_NUM_WORKERS = 4  # For data loading
# Allow env override to adapt concurrency on CPU-only vs GPU runs
# Treat an empty environment variable as an unset value.
_num_workers_env = os.getenv('NUM_WORKERS_OVERRIDE', '').strip()
NUM_WORKERS = int(_num_workers_env) if _num_workers_env else DEFAULT_NUM_WORKERS

# ========== Dataset Settings ==========

DATASETS = {
    'MNIST': {
        'num_classes': 10,
        'input_channels': 1,
        'image_size': 28,
        'data_dir': str(DATA_ROOT / 'MNIST')
    },
    'CIFAR10': {
        'num_classes': 10,
        'input_channels': 3,
        'image_size': 32,
        'data_dir': str(DATA_ROOT / 'CIFAR10')
    },
    'CIFAR100': {
        'num_classes': 100,
        'input_channels': 3,
        'image_size': 32,
        'data_dir': str(DATA_ROOT / 'CIFAR100')
    },
    'FEMNIST': {
        'num_classes': 62,
        'input_channels': 1,
        'image_size': 28,
        'data_dir': str(DATA_ROOT / 'FEMNIST')
    },
    'FashionMNIST': {
        'num_classes': 10,
        'input_channels': 1,
        'image_size': 28,
        'data_dir': str(DATA_ROOT / 'FashionMNIST')
    }
}

# ========== Model Settings ==========

MODELS = {
    'SimpleCNN': {
        'type': 'cnn',
        'description': 'Simple CNN for MNIST/FashionMNIST'
    },
    'ResNet18': {
        'type': 'resnet',
        'description': 'ResNet-18 for CIFAR-10'
    },
    'ResNet34': {
        'type': 'resnet',
        'description': 'ResNet-34 for CIFAR-100'
    },
    'VGG11': {
        'type': 'vgg',
        'description': 'VGG-11 for CIFAR-10'
    }
}

# ========== FL Training Settings ==========

FL_CONFIG = {
    'num_rounds': 50,
    'local_epochs': 2,  # Updated from 5 to 2 (CVPR standard)
    'batch_size': 32,
    'learning_rate': 0.01,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4
}

# ========== PoL Settings ==========

POL_CONFIG = {
    'enable': True,
    # Allow environment overrides for rapid calibration runs
    'save_freq': int(os.getenv('POL_SAVE_FREQ', '10')),               # Save checkpoint every N iterations
    'verification_rate': float(os.getenv('POL_VERIFICATION_RATE', '0.3')),  # Fraction of selected clients to verify
    'delta': 10.0,                # [PARAMETER] Distance threshold (适合SimpleCNN 18万参数)
    'distance_metric': 'l2',      # L1, L2, Linf, or cosine
    'use_top_q': False,           # Use Top-Q optimization
    'top_q': 5,                   # Number of top checkpoints to verify
    # Root-cause improvements (verification acceptance & sampling)
    'min_pair_success_rate': float(os.getenv('POL_MIN_PAIR_SUCCESS_RATE', '0.99')),  # Near-zero tolerance for pairwise replay
    'always_verify_last_k': int(os.getenv('POL_ALWAYS_VERIFY_LAST_K', '2')),         # Always verify tail pairs
    'random_q': int(os.getenv('POL_RANDOM_Q', '3')),                                  # Random pairs to remove sampling bias
}

# ========== Economic Incentive Settings ==========

ECONOMIC_CONFIG = {
    'enable': True,
    'min_stake': 100,
    'base_reward': 50,
    'penalty_rate': 0.5,
    'verification_prob': 0.3,
    'reputation_decay': 0.9,
    'contribution_weight': 0.3,
    'reputation_weight': 0.2,
}

# ========== Data Distribution Settings ==========

DATA_DISTRIBUTION = {
    'IID': {
        'type': 'iid',
        'description': 'Independent and Identically Distributed'
    },
    'NonIID_Dirichlet': {
        'type': 'dirichlet',
        'alpha': 0.5,
        'description': 'Non-IID with Dirichlet distribution'
    },
    'NonIID_Pathological': {
        'type': 'pathological',
        'shards_per_client': 2,
        'description': 'Non-IID with pathological distribution'
    },
    'Natural_Writer': {
        'type': 'natural_writer',
        'description': 'FEMNIST natural writer partition'
    }
}

# ========== RQ1: Security Evaluation ==========

RQ1_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 20,
    'clients_per_round': 10,
    'data_distribution': 'NonIID_Dirichlet',

    # Attack scenarios
    'attacks': {
        'byzantine': {
            'random_noise': {
                'malicious_ratios': [0.1, 0.2, 0.3],
                'noise_scale': 1.0
            },
            'label_flipping': {
                'malicious_ratios': [0.1, 0.2, 0.3],
                'flip_probability': 1.0
            },
            'model_replacement': {
                'malicious_ratios': [0.1, 0.2, 0.3],
                'replacement_type': 'random'
            }
        },
        'free_riding': {
            'no_training': {
                'malicious_ratios': [0.1, 0.2, 0.3]
            },
            'lazy_training': {
                'malicious_ratios': [0.1, 0.2, 0.3],
                'lazy_epochs': 1  # Train only 1 epoch instead of 5
            }
        }
    },

    # Baseline methods
    'baselines': ['Vanilla_FL', 'Krum', 'Trimmed_Mean', 'PoL_FL', 'ZKP_PoL'],

    # Metrics
    'metrics': ['test_accuracy', 'detection_rate', 'rejection_rate', 'convergence_rounds']
}

# ========== RQ2: System Overhead Analysis ==========

RQ2_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 20,
    'clients_per_round': 10,
    'data_distribution': 'NonIID_Dirichlet',

    # Profiling targets
    'profiling': {
        'training_time': True,
        'checkpoint_save_time': True,
        'zkp_generation_time': True,
        'model_upload_size': True,
        'zkp_proof_size': True,
        'gas_cost': True,
        'verification_time': True
    },

    # Methods to compare
    'methods': ['Vanilla_FL', 'PoL_FL', 'ZKP_PoL'],

    # Metrics
    'metrics': [
        'training_time', 'pol_generation_time', 'zkp_generation_time',
        'communication_overhead', 'storage_overhead', 'gas_cost'
    ]
}

# ========== RQ3: Scalability Testing ==========

RQ3_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'data_distribution': 'NonIID_Dirichlet',

    # Scenario 1: Scalability
    'scalability': {
        'num_clients_list': [10, 20, 50, 100],
        'clients_per_round_ratio': 0.5,  # 50% of clients per round
        'verification_rate': 0.3,
        'metrics': ['total_verification_time', 'round_time', 'total_gas_cost']
    },

    # Scenario 2: Parameter Sensitivity
    'parameter_sensitivity': {
        'num_clients': 20,
        'checkpoint_intervals': [1, 5, 10, 20, 50],
        'metrics': ['storage_overhead', 'verification_time', 'detection_rate']
    },

    # Methods to compare
    'methods': ['PoL_FL', 'ZKP_PoL']
}

# ========== RQ4: Economic Incentive Effectiveness ==========

RQ4_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 30,
    'clients_per_round': 15,
    'data_distribution': 'NonIID_Dirichlet',
    'num_rounds': 100,  # Longer simulation for game theory

    # Node types
    'node_types': {
        'honest': 0.6,      # 60% honest nodes
        'rational': 0.3,    # 30% rational nodes
        'malicious': 0.1    # 10% malicious nodes
    },

    # Utility function parameters
    'utility_params': {
        'base_reward': 50,
        'contribution_reward_weight': 0.3,
        'compute_cost': 10,
        'gas_cost': 2,
        'slash_penalty': 100,
        'detection_probability': 0.3
    },

    # Scenarios
    'scenarios': {
        'no_incentive': {
            'enable_staking': False,
            'enable_rewards': False
        },
        'fixed_reward': {
            'enable_staking': True,
            'enable_rewards': True,
            'reward_type': 'fixed'
        },
        'dynamic_reward': {
            'enable_staking': True,
            'enable_rewards': True,
            'reward_type': 'dynamic'
        }
    },

    # Metrics
    'metrics': [
        'honest_utility', 'rational_utility', 'malicious_utility',
        'participation_rate', 'attack_success_rate', 'system_stability'
    ]
}

# ========== Output Settings ==========

OUTPUT_CONFIG = {
    'base_dir': str(Path(os.getenv('POL_EXPERIMENTS_DIR') or DEFAULT_EXPERIMENTS_DIR)),
    'results_dir': str(Path(os.getenv('POL_RESULTS_DIR') or DEFAULT_RESULTS_DIR)),
    'plots_dir': str(Path(os.getenv('POL_PLOTS_DIR') or DEFAULT_PLOTS_DIR)),
    'checkpoints_dir': str(Path(os.getenv('POL_CHECKPOINTS_DIR') or DEFAULT_CHECKPOINTS_DIR)),
    'log_dir': str(Path(os.getenv('POL_LOG_DIR') or os.getenv('VERYFL_LOG_DIR') or DEFAULT_LOG_DIR)),
    'save_checkpoints': True,
    'save_logs': True,
    'log_interval': 10,
    'plot_format': 'pdf',  # or 'png'
    'dpi': 300
}

# ========== Reproducibility ==========

def set_random_seed(seed=None):
    """Set random seed for reproducibility.
    If seed is None, try environment variable SEED, else default to RANDOM_SEED.
    """
    import os, random
    import numpy as np
    if seed is None:
        try:
            seed = int(os.getenv('SEED', RANDOM_SEED))
        except Exception:
            seed = RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
