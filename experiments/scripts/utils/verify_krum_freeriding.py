"""
验证Krum在Free-riding攻击下的真实表现

目的：重新运行MNIST上的Free-riding实验，验证Krum的真实准确率
论文声称：77.70%
实验结果：9.98%
需要确认哪个是正确的
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

from experiment_config import FL_CONFIG, set_random_seed, OUTPUT_CONFIG
from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders
from models import create_model
from metrics import compute_accuracy
from baselines import create_aggregator
from attacks.free_riding_attacks import create_free_riding_attack

log_dir = Path(OUTPUT_CONFIG.get('log_dir', 'log'))
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'verify_krum_freeriding.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 实验配置
VERIFY_CONFIG = {
    'dataset': 'MNIST',
    'model': 'SimpleCNN',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 10,  # 与原实验一致
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,
    'malicious_ratio': 0.2,  # 20% free-riders
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


class KrumFreeRidingVerification:
    """验证Krum在Free-riding攻击下的表现"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Set random seed
        set_random_seed(config['random_seed'])

        # Create output directory
        self.output_dir = Path('./experiments/results/verification')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized verification on {self.device}")
        logger.info(f"Config: {config}")

    def prepare_data(self):
        """准备数据集"""
        logger.info("Preparing MNIST dataset...")

        # Load dataset
        train_dataset = load_dataset('MNIST', train=True)
        test_dataset = load_dataset('MNIST', train=False)

        # Partition data with Dirichlet distribution (alpha=0.5)
        client_datasets = partition_data_dirichlet(
            train_dataset,
            self.config['num_clients'],
            alpha=0.5
        )

        # Create dataloaders
        self.train_loaders = create_dataloaders(
            client_datasets,
            batch_size=self.config['batch_size']
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        logger.info(f"Prepared data for {self.config['num_clients']} clients")

    def create_model(self):
        """创建模型"""
        model = create_model(
            self.config['model'],
            num_classes=10,
            input_channels=1
        )
        return model

    def train_client(self, model, dataloader):
        """训练客户端模型"""
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config['learning_rate'],
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config['local_epochs']):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def run_experiment(self, baseline_method='Krum'):
        """
        运行实验

        Args:
            baseline_method: 聚合方法 ('Krum', 'Vanilla_FL', 'Trimmed_Mean')

        Returns:
            results: 实验结果
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {baseline_method} vs Free-riding (20%)")
        logger.info(f"{'='*70}\n")

        # Create global model
        global_model = self.create_model().to(self.device)

        # Create aggregator
        num_malicious = int(self.config['clients_per_round'] * self.config['malicious_ratio'])

        if baseline_method == 'Krum':
            aggregator = create_aggregator(baseline_method, num_byzantine=num_malicious)
        elif baseline_method == 'Trimmed_Mean':
            aggregator = create_aggregator(baseline_method, trim_ratio=0.1)
        else:
            aggregator = create_aggregator(baseline_method)

        # Create free-riding attack
        free_riding_attack = create_free_riding_attack('no_training')

        # Training loop
        test_accuracies = []

        for round_num in range(self.config['num_rounds']):
            logger.info(f"Round {round_num + 1}/{self.config['num_rounds']}")

            # Select clients for this round
            num_selected = self.config['clients_per_round']
            selected_indices = np.random.choice(
                self.config['num_clients'],
                num_selected,
                replace=False
            )

            # Determine malicious clients (first 20%)
            num_malicious = int(num_selected * self.config['malicious_ratio'])
            malicious_indices = selected_indices[:num_malicious]

            logger.info(f"  Selected clients: {selected_indices}")
            logger.info(f"  Malicious clients: {malicious_indices}")

            # Client training
            client_models = []

            for idx in selected_indices:
                # Create client model
                client_model = self.create_model().to(self.device)
                client_model.load_state_dict(global_model.state_dict())

                # Check if malicious
                if idx in malicious_indices:
                    # A free-rider returns the global model without local training.
                    logger.info(f"  Client {idx}: Free-rider (no training)")
                    # client_model already has global_model's weights
                else:
                    # Honest client: train normally
                    logger.info(f"  Client {idx}: Honest (training)")
                    self.train_client(client_model, self.train_loaders[idx])

                client_models.append(client_model.state_dict())

            # Aggregate models
            aggregated_state = aggregator.aggregate(client_models)
            global_model.load_state_dict(aggregated_state)

            # Evaluate
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)
            test_accuracies.append(test_acc)

            logger.info(f"  Test Accuracy: {test_acc:.4f}")

        results = {
            'baseline_method': baseline_method,
            'attack_type': 'free_riding_no_training',
            'malicious_ratio': self.config['malicious_ratio'],
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'config': self.config
        }

        return results

    def run_all_baselines(self):
        """运行所有baseline方法"""
        logger.info("Starting Krum Free-riding Verification")

        # Prepare data
        self.prepare_data()

        all_results = []

        # Test all baselines
        for baseline in ['Vanilla_FL', 'Krum', 'Trimmed_Mean']:
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"Testing {baseline}")
                logger.info(f"{'='*70}")

                results = self.run_experiment(baseline)
                all_results.append(results)

                logger.info(f"\n{baseline} Final Accuracy: {results['final_accuracy']:.4f}")

            except Exception as e:
                logger.error(f"Experiment failed for {baseline}: {e}")
                import traceback
                traceback.print_exc()

        # Save results
        output_file = self.output_dir / 'krum_freeriding_verification.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results):
        """打印实验总结"""
        logger.info("\n" + "="*70)
        logger.info("Krum Free-riding Verification Summary")
        logger.info("="*70)

        logger.info("\nFinal Accuracies:")
        for result in results:
            logger.info(f"  {result['baseline_method']}: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)")

        logger.info("\n对比论文数据:")
        logger.info("  论文声称 - Krum: 77.70%")
        logger.info("  论文声称 - Vanilla FL: 98.90%")
        logger.info("  论文声称 - Trimmed Mean: 98.93%")

        logger.info("\n实验结果:")
        for result in results:
            logger.info(f"  {result['baseline_method']}: {result['final_accuracy']*100:.2f}%")

        logger.info("="*70)


def main():
    """主函数"""
    verification = KrumFreeRidingVerification(VERIFY_CONFIG)
    results = verification.run_all_baselines()

    logger.info("\nVerification Completed.")
    logger.info(f"Total experiments: {len(results)}")


if __name__ == '__main__':
    main()
