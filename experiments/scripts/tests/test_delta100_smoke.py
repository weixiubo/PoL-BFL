#!/usr/bin/env python3
"""
Reduced-scale test for delta=100.0
Tests pol_only variant with 5 rounds, 1 repetition
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import torch
import logging
from pathlib import Path
from datetime import datetime

# Now import from the correct paths
from experiment_config import FL_CONFIG, POL_CONFIG, OUTPUT_CONFIG
sys.path.insert(0, os.path.join(parent_dir, 'data'))
sys.path.insert(0, os.path.join(parent_dir, 'models'))
sys.path.insert(0, os.path.join(parent_dir, 'server'))
sys.path.insert(0, os.path.join(parent_dir, 'client'))
sys.path.insert(0, os.path.join(parent_dir, 'utils'))

from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders
from model_factory import create_model
from aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from PoLClient import PoLClient
from trainer.PoLTrainer import PoLTrainer
from seed import set_random_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Smoke test for delta=100.0"""

    logger.info("="*70)
    logger.info("Reduced-scale test: delta=100.0")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset: CIFAR10")
    logger.info(f"  Model: ResNet18")
    logger.info(f"  Rounds: 5")
    logger.info(f"  Clients: 20 (10 per round)")
    logger.info(f"  Malicious ratio: 20%")
    logger.info(f"  PoL Delta: {POL_CONFIG['delta']}")
    logger.info("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set seed
    set_random_seed(42)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(OUTPUT_CONFIG['base_dir']) / 'test_delta100' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Prepare data
    logger.info("\nPreparing datasets...")
    train_dataset = load_dataset('CIFAR10', train=True)
    test_dataset = load_dataset('CIFAR10', train=False)

    client_datasets = partition_data_dirichlet(train_dataset, 20, alpha=0.5)
    train_loaders = create_dataloaders(client_datasets, batch_size=FL_CONFIG['batch_size'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FL_CONFIG['batch_size'],
        shuffle=False
    )
    logger.info("Data prepared")

    # Create model
    logger.info("\nCreating model...")
    global_model = create_model('ResNet18', num_classes=10, input_channels=3)
    logger.info(f"Model created: ResNet18")

    # Create aggregator (pol_only variant)
    logger.info("\nCreating PoL aggregator...")
    agg_args = {
        'device': str(device),
        'enable_pol': True,
        'verification_rate': POL_CONFIG['verification_rate'],
        'pol_delta': POL_CONFIG['delta'],
        'pol_distance_metric': POL_CONFIG.get('distance_metric', 'l2'),
        'use_top_q': POL_CONFIG.get('use_top_q', False),
        'top_q': POL_CONFIG['top_q'],
        'enable_zkp': False,
        'enable_incentives': False,
    }
    aggregator = PoLVerifyAggregator(model=global_model, args=agg_args)
    logger.info(f"Aggregator created with delta={POL_CONFIG['delta']}")

    # Select malicious clients (20%)
    import numpy as np
    np.random.seed(42)
    malicious_indices = np.random.choice(20, size=4, replace=False)
    logger.info(f"\nMalicious clients: {sorted(malicious_indices.tolist())}")

    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70)

    num_rounds = 5
    clients_per_round = 10

    for round_num in range(num_rounds):
        logger.info(f"\n{'='*70}")
        logger.info(f"Round {round_num + 1}/{num_rounds}")
        logger.info(f"{'='*70}")

        # Select clients
        selected_indices = np.random.choice(20, size=clients_per_round, replace=False)
        logger.info(f"Selected clients: {sorted(selected_indices.tolist())}")

        # Build clients
        clients = []
        for idx in selected_indices:
            client_id = f"client_{int(idx)}"
            is_malicious = idx in malicious_indices

            # Create trainer
            trainer_args = {
                'device': device,
                'lr': 0.01,
                'weight_decay': 1e-4,
                'optimizer': 'SGD',
                'enable_pol': True,  # All clients use PoL
                'pol_save_freq': POL_CONFIG['save_freq'],
                'pol_save_dir': str(output_dir / 'pol_data'),
                'pol_compress': True,
                'client_id': client_id,
            }

            trainer = PoLTrainer(
                model=create_model('ResNet18', num_classes=10, input_channels=3),
                args=trainer_args
            )

            client = PoLClient(
                client_id=client_id,
                trainer=trainer,
                train_loader=train_loaders[idx],
                test_loader=test_loader
            )

            clients.append(client)

        # Train clients
        logger.info("\nTraining clients...")
        for i, (client, idx) in enumerate(zip(clients, selected_indices)):
            is_malicious = idx in malicious_indices

            # Train
            client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)

            # Byzantine attack for malicious clients
            if is_malicious:
                with torch.no_grad():
                    for param in client.model.parameters():
                        noise = torch.randn_like(param) * 1.0  # noise_scale=1.0
                        param.add_(noise)
                logger.info(f"  [Malicious] {client.client_id} Byzantine attack")
            else:
                logger.info(f"  [Honest] {client.client_id} training complete")

        # Aggregate
        logger.info("\nAggregating models...")
        aggregator.receive_upload(clients)
        aggregator.aggregate()

        # Get verification results
        verification_results = aggregator.verification_results
        logger.info(f"\nVerification results: {verification_results}")

        # Compute detection metrics
        if verification_results:
            selected_malicious = [idx for idx in selected_indices if idx in malicious_indices]
            selected_honest = [idx for idx in selected_indices if idx not in malicious_indices]

            detected_malicious = sum(1 for idx in selected_malicious
                                    if not verification_results.get(f"client_{idx}", True))
            detected_honest = sum(1 for idx in selected_honest
                                 if not verification_results.get(f"client_{idx}", True))

            tpr = detected_malicious / len(selected_malicious) if selected_malicious else 0
            fpr = detected_honest / len(selected_honest) if selected_honest else 0

            logger.info(f"\n{'='*70}")
            logger.info(f"Detection Metrics:")
            logger.info(f"  Selected malicious: {len(selected_malicious)}")
            logger.info(f"  Detected malicious: {detected_malicious}")
            logger.info(f"  TPR: {tpr:.4f}")
            logger.info(f"  FPR: {fpr:.4f}")
            logger.info(f"{'='*70}")

        # Test accuracy
        global_model.to(device)
        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = correct / total
        logger.info(f"\nTest Accuracy: {accuracy:.4f}")

    logger.info("\n" + "="*70)
    logger.info("Smoke test completed.")
    logger.info("="*70)

    # Check if verification worked
    if verification_results:
        passed_count = sum(1 for v in verification_results.values() if v)
        failed_count = sum(1 for v in verification_results.values() if not v)
        logger.info(f"\nFinal verification summary:")
        logger.info(f"  Passed: {passed_count}")
        logger.info(f"  Failed: {failed_count}")

        if failed_count > 0:
            logger.info(f"\n[PASS] SUCCESS: PoL verification is working.")
            logger.info(f"   Delta=100.0 appears to be effective")
        else:
            logger.warning(f"\n[WARNING] WARNING: All verifications passed")
            logger.warning(f"   This might indicate delta is too large")
    else:
        logger.error(f"\n[FAIL] ERROR: No verification results")
        logger.error(f"   PoL verification may not be enabled")

if __name__ == '__main__':
    main()
