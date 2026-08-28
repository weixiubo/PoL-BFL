"""
RQ1: Security Evaluation with ResNet-18 on CIFAR-10

This script runs the security evaluation experiments using ResNet-18 on CIFAR-10
to meet the CVPR reviewer expectations for more complex models and datasets.

Based on the original run_rq1_security.py but with ResNet-18 configuration.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import json
from pathlib import Path
from collections import OrderedDict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import FL_CONFIG, POL_CONFIG, OUTPUT_CONFIG, set_random_seed
from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders, print_data_statistics, get_data_statistics
from models import create_model, print_model_info
from metrics import MetricsTracker, compute_accuracy, compute_detection_rate, compute_convergence_round
from baselines import create_aggregator
from attacks.byzantine_attacks import create_attack
from attacks.free_riding_attacks import create_free_riding_attack

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ResNet-18 on CIFAR-10 configuration
# NOTE: Start with a smoke test (5 rounds) to verify the code works
# Then increase to 50 rounds for full experiments
RQ1_RESNET18_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 5,  # Smoke test first, then increase to 50
    'data_distribution': 'NonIID_Dirichlet',
    'dirichlet_alpha': 0.5,  # Same as MNIST experiments

    # Attack scenarios (same as MNIST for comparison)
    'attacks': {
        'no_attack': {'malicious_ratios': [0.0]},
        'byzantine_random_noise': {'malicious_ratios': [0.2], 'noise_scale': 1.0},
        'free_riding_no_training': {'malicious_ratios': [0.2]}
    },

    # Baselines to test
    'baselines': ['Vanilla_FL', 'Krum', 'Trimmed_Mean']
}


class SecurityExperiment:
    """Security evaluation experiment with ResNet-18"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_tracker = MetricsTracker()

        # Set random seed
        set_random_seed()

        # Create output directory
        self.output_dir = Path(OUTPUT_CONFIG['results_dir']) / 'rq1_resnet18'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SecurityExperiment (ResNet-18) on {self.device}")

    def prepare_data(self):
        """Prepare CIFAR-10 datasets"""
        logger.info("Preparing CIFAR-10 datasets...")

        # Load dataset
        train_dataset = load_dataset(
            self.config['dataset'],
            train=True
        )
        test_dataset = load_dataset(
            self.config['dataset'],
            train=False
        )

        # Partition data (Non-IID Dirichlet)
        client_data_indices = partition_data_dirichlet(
            train_dataset,
            self.config['num_clients'],
            alpha=self.config.get('dirichlet_alpha', 0.5)
        )

        # Create dataloaders
        self.train_loaders = create_dataloaders(
            train_dataset,
            client_data_indices,
            batch_size=FL_CONFIG['batch_size']
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FL_CONFIG['batch_size'],
            shuffle=False
        )

        # Print statistics
        print_data_statistics(train_dataset, client_data_indices)

        logger.info(f"Data preparation complete: {len(self.train_loaders)} clients")

    def create_global_model(self):
        """Create ResNet-18 model"""
        model = create_model(
            self.config['model'],
            num_classes=10,
            input_channels=3  # CIFAR-10 has 3 channels
        )
        print_model_info(model)
        return model.to(self.device)

    def train_client(self, model, train_loader, epochs=5):
        """Train a client model"""
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=FL_CONFIG['learning_rate'],
            momentum=FL_CONFIG['momentum'],
            weight_decay=FL_CONFIG['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def evaluate_model(self, model):
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                # Support both (data, target) and (data, target, idx) formats
                if len(batch) == 3:
                    data, target, _ = batch
                else:
                    data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        return accuracy

    def run_experiment(self, baseline_method, attack_type, malicious_ratio):
        """Run a single experiment"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {baseline_method} vs {attack_type} (malicious={malicious_ratio})")
        logger.info(f"{'='*80}")

        # Create global model
        global_model = self.create_global_model()

        # Create aggregator
        aggregator = create_aggregator(
            baseline_method,
            num_byzantine=int(self.config['clients_per_round'] * malicious_ratio)
        )

        # Create attack
        if attack_type == 'no_attack':
            attack = None
        elif 'byzantine' in attack_type:
            attack_config = self.config['attacks'][attack_type]
            attack = create_attack(attack_type, **attack_config)
        elif 'free_riding' in attack_type:
            attack = create_free_riding_attack(attack_type)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Training loop
        test_accuracies = []

        for round_idx in range(self.config['num_rounds']):
            # Select clients
            selected_clients = np.random.choice(
                self.config['num_clients'],
                self.config['clients_per_round'],
                replace=False
            )

            # Determine malicious clients
            num_malicious = int(len(selected_clients) * malicious_ratio)
            malicious_clients = selected_clients[:num_malicious] if num_malicious > 0 else []

            # Collect client updates
            client_models = []
            client_weights = []

            for client_id in selected_clients:
                # Create client model (copy of global model)
                client_model = create_model(
                    self.config['model'],
                    num_classes=10,
                    input_channels=3
                ).to(self.device)
                client_model.load_state_dict(global_model.state_dict())

                # Train or attack
                if client_id in malicious_clients and attack is not None:
                    if 'free_riding' in attack_type:
                        # A free-rider returns the global model without local training.
                        client_update = client_model.state_dict()
                    else:
                        # Byzantine: train then apply attack
                        client_update = self.train_client(
                            client_model,
                            self.train_loaders[client_id],
                            epochs=FL_CONFIG['local_epochs']
                        )
                        client_update = attack.apply(client_update, global_model.state_dict())
                else:
                    # Honest client: normal training
                    client_update = self.train_client(
                        client_model,
                        self.train_loaders[client_id],
                        epochs=FL_CONFIG['local_epochs']
                    )

                client_models.append(client_update)
                client_weights.append(1.0 / len(selected_clients))

            # Aggregate
            aggregated_model = aggregator.aggregate(client_models, client_weights)
            global_model.load_state_dict(aggregated_model)

            # Evaluate
            test_acc = self.evaluate_model(global_model)
            test_accuracies.append(test_acc)

            logger.info(f"Round {round_idx+1}/{self.config['num_rounds']}: Test Accuracy: {test_acc:.4f}")

        # Compute final metrics
        final_accuracy = test_accuracies[-1]
        convergence_round = compute_convergence_round(test_accuracies, threshold=0.7)

        result = {
            'baseline_method': baseline_method,
            'attack_type': attack_type,
            'malicious_ratio': malicious_ratio,
            'test_accuracies': test_accuracies,
            'final_accuracy': final_accuracy,
            'convergence_round': convergence_round,
            'config': {
                'dataset': self.config['dataset'],
                'model': self.config['model'],
                'num_clients': self.config['num_clients'],
                'clients_per_round': self.config['clients_per_round'],
                'num_rounds': self.config['num_rounds'],
                'local_epochs': FL_CONFIG['local_epochs'],
                'batch_size': FL_CONFIG['batch_size'],
                'learning_rate': FL_CONFIG['learning_rate'],
                'dirichlet_alpha': self.config.get('dirichlet_alpha', 0.5),
                'random_seed': 42,
                'device': str(self.device)
            }
        }

        logger.info(f"Final Accuracy: {final_accuracy:.4f}")
        logger.info(f"Convergence Round: {convergence_round}")

        return result

    def run_all_experiments(self):
        """Run all baseline vs attack combinations"""
        all_results = []

        # Prepare data once
        self.prepare_data()

        # Run experiments
        for baseline in self.config['baselines']:
            for attack_type, attack_config in self.config['attacks'].items():
                for malicious_ratio in attack_config['malicious_ratios']:
                    result = self.run_experiment(baseline, attack_type, malicious_ratio)
                    all_results.append(result)

        # Save results
        output_file = self.output_dir / 'rq1_resnet18_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nAll results saved to: {output_file}")

        return all_results


if __name__ == '__main__':
    logger.info("Starting RQ1 Security Evaluation with ResNet-18 on CIFAR-10")

    experiment = SecurityExperiment(RQ1_RESNET18_CONFIG)
    results = experiment.run_all_experiments()

    logger.info("\n" + "="*80)
    logger.info("RQ1 ResNet-18 Experiments Complete.")
    logger.info("="*80)
