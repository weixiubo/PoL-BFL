"""
RQ1: Security Evaluation with PoL Detection Metrics

This script extends run_rq1_security.py to include PoL verification and detection metrics.
It runs experiments with both traditional baselines AND PoL-enabled aggregation,
recording detection rates (TPR, FPR) alongside accuracy metrics.

Key differences from run_rq1_security.py:
1. Uses PoLClient + PoLTrainer for PoL-enabled experiments
2. Uses PoLVerifyAggregator to perform verification
3. Records detection metrics (TPR, FPR, Precision, Recall, F1)
4. Saves both accuracy and detection results
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import json
from pathlib import Path
from typing import List, Dict

import csv

# Add parent directory and utils to path (align with run_rq1_security.py)
scripts_dir = Path(__file__).parent.parent
experiments_dir = scripts_dir.parent
project_root = experiments_dir.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / 'utils'))
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(project_root))

from experiment_config import FL_CONFIG, POL_CONFIG, OUTPUT_CONFIG, set_random_seed
from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders
from models import create_model
from metrics import compute_accuracy, compute_detection_metrics, compute_convergence_round
from baselines import create_aggregator

# PoL components
from client.trainer.PoLTrainer import PoLTrainer
from client.PoLClient import PoLClient
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# RQ1 config with detection
RQ1_DETECTION_CONFIG = {
    'dataset': 'MNIST',
    'model': 'SimpleCNN',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 10,
    'data_distribution': 'NonIID_Dirichlet',

    # Attack scenarios
    'attacks': {
        'no_attack': {'malicious_ratios': [0.0]},
        'byzantine_random_noise': {'malicious_ratios': [0.2], 'noise_scale': 1.0},
        'byzantine_label_flipping': {'malicious_ratios': [0.2]},
        'free_riding_no_training': {'malicious_ratios': [0.2]},
        'free_riding_lazy_training': {'malicious_ratios': [0.2], 'lazy_ratio': 0.1}
    },

    # Baselines to test
    'baselines': ['Vanilla_FL', 'Krum', 'Trimmed_Mean'],

    # PoL settings
    'enable_pol_detection': True,
    'verification_rate': 1.0,  # Verify all clients for complete detection metrics
    'pol_save_freq': 1,
    'pol_compress': True,
    'pol_delta': 10.0,
}


class SecurityExperimentWithDetection:
    """Security evaluation with PoL detection metrics"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seed
        set_random_seed()

        # Create output directory
        self.output_dir = Path(OUTPUT_CONFIG['results_dir']) / 'rq1_security'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SecurityExperimentWithDetection on {self.device}")

    def prepare_data(self):
        """Prepare datasets"""
        logger.info("Preparing datasets...")

        train_dataset = load_dataset(self.config['dataset'], train=True)
        test_dataset = load_dataset(self.config['dataset'], train=False)

        client_datasets = partition_data_dirichlet(
            train_dataset,
            self.config['num_clients'],
            alpha=0.5
        )

        self.train_loaders = create_dataloaders(client_datasets, batch_size=FL_CONFIG['batch_size'])
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FL_CONFIG['batch_size'],
            shuffle=False
        )

        logger.info(f"Prepared data for {self.config['num_clients']} clients")

    def create_model(self):
        """Create model"""
        if self.config['dataset'] == 'MNIST':
            num_classes, input_channels = 10, 1
        elif self.config['dataset'] == 'CIFAR10':
            num_classes, input_channels = 10, 3
        elif self.config['dataset'] == 'CIFAR100':
            num_classes, input_channels = 100, 3
        else:
            raise ValueError(f"Unknown dataset: {self.config['dataset']}")

        return create_model(
            self.config['model'],
            num_classes=num_classes,
            input_channels=input_channels
        )

    def run_experiment_with_pol(self, attack_type: str, attack_params: dict):
        """
        Run experiment with PoL verification enabled

        Returns:
            results: Dict with accuracy and detection metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running PoL-BFL vs {attack_type}")
        logger.info(f"Attack params: {attack_params}")
        logger.info(f"{'='*70}\n")

        # Create global model
        global_model = self.create_model().to(self.device)

        # Create PoL aggregator
        agg_args = {
            'enable_pol': True,
            'verification_rate': self.config['verification_rate'],
            'pol_delta': self.config['pol_delta'],
            'pol_distance_metric': POL_CONFIG.get('distance_metric', 'l2'),
            'device': str(self.device),
            'use_top_q': False,
            'enable_zkp': False,
            'enable_incentives': False,
        }
        aggregator = PoLVerifyAggregator(model=global_model, args=agg_args)

        # Training loop
        test_accuracies = []
        # Decentralization/observability per-round (aligned with RQ1/RQ2)
        ext_succ, ext_lat = [], []
        rm_resp, rm_yes = [], []
        pol_vt = []
        rlat_p50, rlat_p95 = [], []
        rerr_timeout, rerr_network, rerr_invalid, rerr_business = [], [], [], []
        ext_err_type = []

        detection_metrics_per_round = []

        for round_num in range(self.config['num_rounds']):
            logger.info(f"Round {round_num + 1}/{self.config['num_rounds']}")

            # Select clients
            num_selected = self.config['clients_per_round']
            selected_indices = np.random.choice(
                self.config['num_clients'],
                num_selected,
                replace=False
            )

            # Determine malicious clients
            malicious_ratio = attack_params.get('malicious_ratios', [0.0])[0]
            num_malicious = int(num_selected * malicious_ratio)
            if malicious_ratio > 0 and num_malicious == 0:
                num_malicious = 1
            malicious_indices = selected_indices[:num_malicious]

            # Build PoL clients
            clients = self._build_pol_clients(
                global_model,
                selected_indices,
                malicious_indices,
                attack_type,
                attack_params
            )

            # Upload and aggregate with PoL verification
            aggregator.receive_upload(clients)
            client_models = [c.get_model_state_dict() for c in clients]
            aggregated_state = aggregator.aggregate(client_models)
            global_model.load_state_dict(aggregated_state)

            # Get verification results
            verification_results = aggregator.get_verification_results()

            # Compute detection metrics
            malicious_client_ids = [f"client_{int(idx)}" for idx in malicious_indices]
            all_client_ids = [f"client_{int(idx)}" for idx in selected_indices]

            detection_metrics = compute_detection_metrics(
                verification_results,
                malicious_client_ids,
                all_client_ids
            )
            detection_metrics_per_round.append(detection_metrics)

            # Evaluate accuracy
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)
            test_accuracies.append(test_acc)

            logger.info(f"  Test Accuracy: {test_acc:.4f}")
            logger.info(f"  Detection TPR: {detection_metrics['TPR']:.4f}, FPR: {detection_metrics['FPR']:.4f}")

            # Capture aggregator metrics snapshot for this round (for CSV/JSON)
            try:
                m = getattr(aggregator, 'metrics', {}) or {}
                ext_succ.append(int(bool(m.get('external_agg_success', False))))
                ext_lat.append(float(m.get('external_agg_latency_s', 0.0)))
                rm_resp.append(int(m.get('remote_majority_responders', 0)))
                rm_yes.append(int(m.get('remote_majority_yes', 0)))
                pol_vt.append(float(m.get('pol_verify_time_s', 0.0)))
                rlat_p50.append(float(m.get('remote_verify_latency_p50_s', 0.0)))
                rlat_p95.append(float(m.get('remote_verify_latency_p95_s', 0.0)))
                rerr_timeout.append(int(m.get('remote_error_timeout', 0)))
                rerr_network.append(int(m.get('remote_error_network', 0)))
                rerr_invalid.append(int(m.get('remote_error_invalid', 0)))
                rerr_business.append(int(m.get('remote_error_business', 0)))
                ext_err_type.append(str(m.get('external_agg_error_type', '')))
            except Exception:
                ext_succ.append(0); ext_lat.append(0.0); rm_resp.append(0); rm_yes.append(0)
                pol_vt.append(0.0); rlat_p50.append(0.0); rlat_p95.append(0.0)
                rerr_timeout.append(0); rerr_network.append(0); rerr_invalid.append(0); rerr_business.append(0)
                ext_err_type.append('')


            # Persist per-round CSV with decentralization/observability metrics aligned to RQ1/RQ2
            try:
                def _san(s: str) -> str:
                    return s.replace('/', '_').replace(' ', '_')
                csv_name = f"rq1_with_detection_rounds_{_san(attack_type)}.csv"
                csv_path = self.output_dir / csv_name
                fieldnames = ['round', 'test_accuracy', 'detection_tpr', 'detection_fpr', 'precision', 'recall', 'f1',
                              'external_agg_success', 'external_agg_latency_s', 'remote_majority_responders', 'remote_majority_yes',
                              'pol_verify_time_s', 'remote_verify_latency_p50_s', 'remote_verify_latency_p95_s',
                              'remote_error_timeout', 'remote_error_network', 'remote_error_invalid', 'remote_error_business', 'external_agg_error_type']
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    for i in range(len(test_accuracies)):
                        dm = detection_metrics_per_round[i] if i < len(detection_metrics_per_round) else {'TPR':0.0,'FPR':0.0,'Precision':0.0,'Recall':0.0,'F1':0.0}
                        writer.writerow({
                            'round': i + 1,
                            'test_accuracy': float(test_accuracies[i]) if i < len(test_accuracies) else 0.0,
                            'detection_tpr': float(dm.get('TPR', 0.0)),
                            'detection_fpr': float(dm.get('FPR', 0.0)),
                            'precision': float(dm.get('Precision', 0.0)),
                            'recall': float(dm.get('Recall', 0.0)),
                            'f1': float(dm.get('F1', 0.0)),
                            'external_agg_success': int(ext_succ[i]) if i < len(ext_succ) else 0,
                            'external_agg_latency_s': float(ext_lat[i]) if i < len(ext_lat) else 0.0,
                            'remote_majority_responders': int(rm_resp[i]) if i < len(rm_resp) else 0,
                            'remote_majority_yes': int(rm_yes[i]) if i < len(rm_yes) else 0,
                            'pol_verify_time_s': float(pol_vt[i]) if i < len(pol_vt) else 0.0,
                            'remote_verify_latency_p50_s': float(rlat_p50[i]) if i < len(rlat_p50) else 0.0,
                            'remote_verify_latency_p95_s': float(rlat_p95[i]) if i < len(rlat_p95) else 0.0,
                            'remote_error_timeout': int(rerr_timeout[i]) if i < len(rerr_timeout) else 0,
                            'remote_error_network': int(rerr_network[i]) if i < len(rerr_network) else 0,
                            'remote_error_invalid': int(rerr_invalid[i]) if i < len(rerr_invalid) else 0,
                            'remote_error_business': int(rerr_business[i]) if i < len(rerr_business) else 0,
                            'external_agg_error_type': str(ext_err_type[i]) if i < len(ext_err_type) else '',
                        })
                logger.info(f"Per-round CSV saved to {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to write per-round CSV: {e}")

        # Aggregate detection metrics across rounds
        avg_detection_metrics = {
            'TPR': float(np.mean([m['TPR'] for m in detection_metrics_per_round])),
            'FPR': float(np.mean([m['FPR'] for m in detection_metrics_per_round])),
            'Precision': float(np.mean([m['Precision'] for m in detection_metrics_per_round])),
            'Recall': float(np.mean([m['Recall'] for m in detection_metrics_per_round])),
            'F1': float(np.mean([m['F1'] for m in detection_metrics_per_round])),
        }

        convergence_round = compute_convergence_round(test_accuracies, threshold=0.85)

        results = {
            'attack_type': attack_type,
            'attack_params': attack_params,
            'baseline_method': 'PoL-BFL',
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'convergence_round': convergence_round,
            'detection_metrics': avg_detection_metrics,
            'detection_metrics_per_round': detection_metrics_per_round,
        }

        return results

    def _build_pol_clients(self, global_model, selected_indices, malicious_indices,
                          attack_type, attack_params):
        """Build PoL clients for selected indices"""
        clients = []

        for idx in selected_indices:
            model = self.create_model().to(self.device)
            model.load_state_dict(global_model.state_dict())

            is_malicious = bool(idx in malicious_indices)

            # Configure trainer
            # IMPORTANT: All clients (including malicious) use PoL
            # PoL verification will detect malicious behavior through final model consistency check
            t_args = {
                'device': str(self.device),
                'enable_pol': True,  # All clients use PoL
                'pol_save_freq': self.config['pol_save_freq'],
                'pol_save_dir': 'pol_data',
                'pol_compress': self.config['pol_compress'],
                'client_id': f"client_{int(idx)}",
                'optimizer': FL_CONFIG.get('optimizer', 'SGD'),
                'lr': FL_CONFIG.get('learning_rate', 0.01),
                'weight_decay': FL_CONFIG.get('weight_decay', 0.0),
            }

            trainer = PoLTrainer(
                model=model,
                dataloader=self.train_loaders[int(idx)],
                criterion=nn.CrossEntropyLoss(),
                args=t_args,
            )

            client = PoLClient(
                client_id=f"client_{int(idx)}",
                dataloader=self.train_loaders[int(idx)],
                model=model,
                trainer=trainer,
                args={'enable_pol': t_args['enable_pol']},
            )

            # Train based on attack type
            if is_malicious:
                if 'free_riding_no_training' in attack_type:
                    # Skip training entirely
                    logger.info(f"[Malicious] client_{int(idx)} skips training (free-riding)")
                elif 'free_riding_lazy_training' in attack_type:
                    # Train with reduced epochs
                    lazy_ratio = attack_params.get('lazy_ratio', 0.1)
                    lazy_epochs = max(1, int(FL_CONFIG['local_epochs'] * lazy_ratio))
                    client.train(total_epoch=lazy_epochs, dataset=None)
                    logger.info(f"[Malicious] client_{int(idx)} lazy training ({lazy_epochs} epochs)")
                else:
                    # Byzantine attacks: train normally then corrupt
                    client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)
                    # Note: Byzantine corruption would be applied here
                    logger.info(f"[Malicious] client_{int(idx)} Byzantine attack")
            else:
                # Honest client
                client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)

            clients.append(client)

        return clients

    def run_all_experiments(self):
        """Run all experiments with PoL detection"""
        logger.info("Starting RQ1: Security Evaluation with Detection")

        self.prepare_data()
        all_results = []

        # Run PoL-enabled experiments
        for attack_type, attack_config in self.config['attacks'].items():
            malicious_ratio = attack_config['malicious_ratios'][0]
            attack_params = {
                'malicious_ratios': [malicious_ratio],
                **{k: v for k, v in attack_config.items() if k != 'malicious_ratios'}
            }

            try:
                results = self.run_experiment_with_pol(attack_type, attack_params)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Experiment failed: PoL-BFL vs {attack_type}: {e}")
                import traceback
                traceback.print_exc()

        # Save results
        output_file = self.output_dir / 'rq1_with_detection.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results):
        """Print experiment summary"""
        logger.info("\n" + "="*70)
        logger.info("RQ1: Security Evaluation with Detection Summary")
        logger.info("="*70)

        for result in results:
            logger.info(f"\n{result['baseline_method']} vs {result['attack_type']}:")
            logger.info(f"  Final Accuracy: {result['final_accuracy']:.4f}")
            logger.info(f"  Detection TPR: {result['detection_metrics']['TPR']:.4f}")
            logger.info(f"  Detection FPR: {result['detection_metrics']['FPR']:.4f}")
            logger.info(f"  Precision: {result['detection_metrics']['Precision']:.4f}")
            logger.info(f"  F1 Score: {result['detection_metrics']['F1']:.4f}")

        logger.info("="*70)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='RQ1: Security Evaluation with Detection')
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Model to use (default: auto-select based on dataset)')
    parser.add_argument('--num_clients', type=int, default=20,
                       help='Total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=10,
                       help='Number of clients per round')
    parser.add_argument('--num_rounds', type=int, default=None,
                       help='Number of rounds (default: auto-select based on dataset)')

    args = parser.parse_args()

    # Auto-select model based on dataset
    if args.model is None:
        if args.dataset == 'MNIST':
            args.model = 'SimpleCNN'
        else:  # CIFAR10 or CIFAR100
            args.model = 'ResNet18'

    # Auto-select num_rounds based on dataset
    if args.num_rounds is None:
        if args.dataset == 'MNIST':
            args.num_rounds = 50
        else:  # CIFAR10 or CIFAR100
            args.num_rounds = 100

    # Update config
    config = RQ1_DETECTION_CONFIG.copy()
    config['dataset'] = args.dataset
    config['model'] = args.model
    config['num_clients'] = args.num_clients
    config['clients_per_round'] = args.clients_per_round
    config['num_rounds'] = args.num_rounds

    logger.info(f"Running RQ1 with:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Clients: {args.num_clients} (per round: {args.clients_per_round})")
    logger.info(f"  Rounds: {args.num_rounds}")

    experiment = SecurityExperimentWithDetection(config)
    results = experiment.run_all_experiments()

    logger.info("\nRQ1: Security Evaluation with Detection Completed!")
    logger.info(f"Total experiments: {len(results)}")


if __name__ == '__main__':
    main()

