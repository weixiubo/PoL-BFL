"""
RQ5: Robust Aggregation Composability Verification

Research Question: Can PoL seamlessly integrate with various robust aggregation methods
without degrading their defense capabilities? For attacks that robust aggregation can detect
but PoL cannot, does PoL maintain or even enhance the defense effectiveness?

This experiment tests attacks that:
- Robust aggregation CAN detect (statistical anomalies)
- PoL CANNOT detect (real training with poisoned data/labels)

Attack types:
- Label Flipping: Real training with flipped labels
- Data Poisoning: Real training with poisoned data
- Gradient Inversion: Real training with inverted gradients (if PoL cannot detect)

Baselines:
- Krum only vs PoL + Krum
- Trimmed Mean only vs PoL + Trimmed Mean
- Median only vs PoL + Median
- Bulyan only vs PoL + Bulyan

Expected outcome: PoL should not degrade robust aggregation performance,
and may even improve it by filtering out some attackers.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import json
import csv

from pathlib import Path
from collections import OrderedDict

# Add parent directory to path
scripts_dir = Path(__file__).parent.parent
experiments_dir = scripts_dir.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / 'utils'))
sys.path.insert(0, str(experiments_dir))

from experiment_config import FL_CONFIG, POL_CONFIG, OUTPUT_CONFIG, DATASETS, NUM_WORKERS, set_random_seed
from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders, print_data_statistics, get_data_statistics
from models import create_model, print_model_info
from metrics import MetricsTracker, compute_accuracy, compute_detection_rate, compute_convergence_round
from baselines import create_aggregator
from attacks.byzantine_attacks import create_attack
from attacks.free_riding_attacks import create_free_riding_attack
from pol_integration import PoLExperimentHelper

from client.trainer.PoLTrainer import PoLTrainer
from client.PoLClient import PoLClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure deterministic cuBLAS workspace
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    logger.info('CUBLAS_WORKSPACE_CONFIG not set; defaulting to :4096:8 for deterministic CUDA')


# RQ5 Configuration
RQ5_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 20,
    'clients_per_round': 10,
    'num_rounds': 100,  # MNIST=20, CIFAR=100
    'data_distribution': 'NonIID_Dirichlet',
    'dirichlet_alpha': 0.5,

    # Attacks that robust aggregation CAN detect but PoL CANNOT
    'attacks': {
        'no_attack': {'malicious_ratios': [0.0]},
        'byzantine_alie': {'malicious_ratios': [0.2], 'z_max': 2.5},
        'free_riding_no_training': {'malicious_ratios': [0.2]},
        'label_flipping': {'malicious_ratios': [0.2]},
        'data_poisoning': {'malicious_ratios': [0.2], 'poison_ratio': 0.1},
        # 'byzantine_gradient_inversion': {'malicious_ratios': [0.2]},  # Enable if PoL cannot detect
    },

    # Test each robust aggregation method with and without PoL
    'baselines': [
        # Krum
        'Krum',
        'PoL_Krum',
        # Trimmed Mean
        'Trimmed_Mean',
        'PoL_Trimmed_Mean',
        # Median
        'Median',
        'PoL_Median',
        # Bulyan
        'Bulyan',
        'PoL_Bulyan',
    ],

    # PoL configuration (aligned with RQ1 clearance defaults)
    'pol_config': {
        'enable': True,
        'save_freq': 5,
        # RQ1 clearance defaults: delta=5.0, verification_rate=1.0
        'verification_rate': 1.0,
        'delta': 5.0,
        'distance_metric': 'l2',
        'use_top_q': False,
        'top_q': 5,
        'enable_zkp': False,
        'zkp_use_simulation': True,
        'min_pair_success_rate': 0.99,
        'always_verify_last_k': 2,
        'random_q': 3,
    }
}


class ComposabilityExperiment:
    """RQ5: Composability experiment"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_tracker = MetricsTracker()

        # Set random seed
        set_random_seed()

        # Create output directory
        self.output_dir = Path(OUTPUT_CONFIG['results_dir']) / 'rq5_composability'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ComposabilityExperiment on {self.device}")

    def prepare_data(self):
        """Prepare datasets"""
        logger.info("Preparing datasets...")

        ds_name = self.config['dataset']
        ds_root = DATASETS[ds_name]['data_dir']
        train_dataset = load_dataset(ds_name, data_dir=ds_root, train=True)
        test_dataset = load_dataset(ds_name, data_dir=ds_root, train=False)

        # Partition data
        if self.config['data_distribution'] == 'NonIID_Dirichlet':
            client_datasets = partition_data_dirichlet(
                train_dataset,
                self.config['num_clients'],
                alpha=float(self.config.get('dirichlet_alpha', 0.5))
            )
        elif self.config['data_distribution'] == 'IID':
            client_datasets = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) // self.config['num_clients']] * (self.config['num_clients'] - 1)
                + [len(train_dataset) - (len(train_dataset) // self.config['num_clients']) * (self.config['num_clients'] - 1)]
            )
        else:
            raise ValueError(f"Unknown data distribution: {self.config['data_distribution']}")

        # Create dataloaders
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

        logger.info(f"Prepared {len(self.train_loaders)} client dataloaders")

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

    def _robust_suspects(self, client_models, selected_indices, num_malicious):
        """Estimate robust aggregation rejections from update-distance outliers."""
        if num_malicious <= 0 or not client_models:
            return set()
        vectors = torch.stack([self._flatten_model_state(model) for model in client_models], dim=0)
        center = torch.median(vectors, dim=0).values
        scores = torch.norm(vectors - center.unsqueeze(0), p=2, dim=1)
        k = min(int(num_malicious), len(client_models))
        suspect_positions = torch.argsort(scores, descending=True)[:k].tolist()
        return {f"client_{int(selected_indices[pos])}" for pos in suspect_positions}

    @staticmethod
    def _sum_detection_counts(accum, metrics):
        for key in ["TP_e2e", "FP_e2e", "FN_e2e", "TN_e2e", "total_malicious", "total_honest"]:
            accum[key] = int(accum.get(key, 0)) + int(metrics.get(key, 0))

    @staticmethod
    def _final_detection_metrics(accum):
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
        """Create model based on dataset and model name"""
        model_name = self.config['model']
        ds = self.config['dataset']
        if ds == 'MNIST':
            num_classes = 10
            input_channels = 1
        elif ds == 'CIFAR10':
            num_classes = 10
            input_channels = 3
        elif ds == 'CIFAR100':
            num_classes = 100
            input_channels = 3
        else:
            raise ValueError(f"Unknown dataset: {ds}")
        model = create_model(model_name, num_classes=num_classes, input_channels=input_channels)
        print_model_info(model, model_name)
        return model

    def run_single_experiment(self, attack_type, attack_params, baseline_method):
        """
        Run a single experiment with given attack and baseline
        
        This is similar to RQ1 but focuses on composability testing
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Attack: {attack_type}, Baseline: {baseline_method}")
        logger.info(f"{'='*70}")

        # Create global model
        global_model = self.create_model().to(self.device)

        # Determine malicious clients
        num_malicious = int(self.config['num_clients'] * attack_params.get('malicious_ratios', [0.2])[0])
        malicious_indices = set(range(num_malicious))

        logger.info(f"Malicious clients: {num_malicious}/{self.config['num_clients']}")
        logger.info(f"Malicious indices: {malicious_indices}")

        # Create attack
        attack = None
        if attack_type != 'no_attack':
            if attack_type.startswith('byzantine_'):
                attack_name = attack_type.replace('byzantine_', '')
                clean_params = {k: v for k, v in attack_params.items() if k != 'malicious_ratios'}
                attack = create_attack(attack_name, **clean_params)
            elif attack_type == 'label_flipping':
                clean_params = {k: v for k, v in attack_params.items() if k != 'malicious_ratios'}
                attack = create_attack('label_flipping', **clean_params)
            elif attack_type.startswith('free_riding_'):
                fr_name = attack_type.replace('free_riding_', '')
                clean_params = {k: v for k, v in attack_params.items() if k != 'malicious_ratios'}
                attack = create_free_riding_attack(fr_name, **clean_params)
            elif attack_type.startswith('data_poisoning'):
                from attacks.free_riding_attacks import DataPoisoningAttack
                attack = DataPoisoningAttack(poison_ratio=attack_params.get('poison_ratio', 0.1))
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

        # Create aggregator
        # Parse baseline method to determine if PoL is enabled
        use_pol = baseline_method.startswith('PoL_')
        base_method = baseline_method.replace('PoL_', '') if use_pol else baseline_method

        if use_pol:
            # PoL + Robust Aggregation
            aggregator = PoLExperimentHelper.setup_pol_aggregator(
                model=global_model,
                pol_config=self.config.get('pol_config', {}),
                device=str(self.device),
                robust_aggregation=base_method  # Use robust aggregation as inner aggregator
            )
        else:
            # Robust Aggregation only
            if base_method == 'Krum':
                aggregator = create_aggregator(base_method, num_byzantine=num_malicious)
            elif base_method == 'Trimmed_Mean':
                aggregator = create_aggregator(base_method, trim_ratio=0.1)
            elif base_method == 'Bulyan':
                aggregator = create_aggregator(base_method, num_byzantine=num_malicious)
            else:
                aggregator = create_aggregator(base_method)

        # Training loop
        test_accuracies = []
        verification_results_per_round = []
        per_round_rows = []
        detection_accum = {}

        for round_num in range(self.config['num_rounds']):
            logger.info(f"\n--- Round {round_num + 1}/{self.config['num_rounds']} ---")

            # Select clients
            selected_indices = np.random.choice(
                self.config['num_clients'],
                self.config['clients_per_round'],
                replace=False
            )

            # Track malicious clients in this round
            malicious_in_round = [idx for idx in selected_indices if idx in malicious_indices]
            logger.info(f"Selected clients: {selected_indices.tolist()}")
            logger.info(f"Malicious in round: {malicious_in_round}")

            # Client training
            client_models = []
            client_pool = []

            for idx in selected_indices:
                is_malicious = idx in malicious_indices

                # Create client model
                client_model = self.create_model().to(self.device)
                client_model.load_state_dict(global_model.state_dict())

                # Train client
                if use_pol:
                    # PoL-enabled client
                    client = PoLClient(
                        client_id=f"client_{idx}",
                        model=client_model,
                        dataloader=self.train_loaders[idx],
                        device=self.device,
                        enable_pol=True
                    )
                    client.trainer.pol_manager.save_freq = self.config['pol_config']['save_freq']
                    if is_malicious and attack_type.startswith('free_riding_') and attack is not None:
                        if hasattr(attack, 'should_train') and not attack.should_train():
                            pass
                        elif hasattr(attack, 'get_training_epochs'):
                            client.train(total_epoch=attack.get_training_epochs())
                        else:
                            client.train(total_epoch=FL_CONFIG['local_epochs'])
                    else:
                        client.train(total_epoch=FL_CONFIG['local_epochs'])
                    client_pool.append(client)
                else:
                    # Regular training
                    optimizer = torch.optim.SGD(
                        client_model.parameters(),
                        lr=FL_CONFIG['learning_rate'],
                        momentum=FL_CONFIG['momentum']
                    )
                    criterion = nn.CrossEntropyLoss()
                    client_model.train()

                    train_epochs = FL_CONFIG['local_epochs']
                    should_train = True
                    if is_malicious and attack_type.startswith('free_riding_') and attack is not None:
                        should_train = bool(attack.should_train()) if hasattr(attack, 'should_train') else True
                        if hasattr(attack, 'get_training_epochs'):
                            train_epochs = int(attack.get_training_epochs())

                    if should_train:
                        for epoch in range(train_epochs):
                            for batch in self.train_loaders[idx]:
                                # Support both (data, target) and (data, target, idx) formats
                                if len(batch) == 3:
                                    data, target, _ = batch
                                else:
                                    data, target = batch
                                data, target = data.to(self.device), target.to(self.device)

                                # Apply data poisoning if needed
                                if is_malicious and attack_type == 'data_poisoning':
                                    target = attack.poison_data(data, target)[1]
                                elif is_malicious and attack_type == 'label_flipping':
                                    target = attack.flip_labels(target, DATASETS[self.config['dataset']]['num_classes'])

                                optimizer.zero_grad()
                                output = client_model(data)
                                loss = criterion(output, target)
                                loss.backward()
                                optimizer.step()

                # Apply attack if malicious
                if is_malicious and attack is not None and attack_type != 'data_poisoning':
                    attacked_state = attack.apply(client_model.state_dict())
                    client_model.load_state_dict(attacked_state)

                client_models.append(client_model.state_dict())

            # Aggregation
            if use_pol:
                # PoL aggregation with verification
                aggregated_state = aggregator.aggregate(
                    raw_client_model_or_grad_list=client_models,
                    client_pool=client_pool
                )
            else:
                # Regular aggregation
                weights = [float(len(self.train_loaders[idx].dataset)) for idx in selected_indices]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                aggregated_state = aggregator.aggregate(client_models, weights)

            robust_detected = self._robust_suspects(client_models, selected_indices, num_malicious)
            if use_pol:
                pol_results = aggregator.get_verification_results() if hasattr(aggregator, 'get_verification_results') else {}
                detected_ids = {cid for cid, is_valid in pol_results.items() if not is_valid}
                detected_ids.update(robust_detected)
            else:
                detected_ids = set(robust_detected)

            all_round_clients = [f"client_{int(idx)}" for idx in selected_indices]
            malicious_round_clients = [f"client_{int(idx)}" for idx in selected_indices if int(idx) in malicious_indices]
            round_verification = {cid: (cid not in detected_ids) for cid in all_round_clients}
            det_metrics = PoLExperimentHelper.compute_detection_metrics(
                round_verification,
                malicious_round_clients,
                all_round_clients,
            )
            self._sum_detection_counts(detection_accum, det_metrics)
            verification_results_per_round.append(det_metrics)

            global_model.load_state_dict(aggregated_state)

            # Evaluation
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)
            test_accuracies.append(test_acc)

            logger.info(f"Test Accuracy: {test_acc:.4f}")

            # Record per-round data
            row = {
                'round': round_num + 1,
                'test_accuracy': test_acc,
                'num_malicious_in_round': len(malicious_in_round),
                'detection_TPR': det_metrics.get('TPR', 0.0),
                'detection_FPR': det_metrics.get('FPR', 0.0),
            }
            per_round_rows.append(row)

        # Compute final metrics
        convergence_round = compute_convergence_round(test_accuracies, threshold=0.85)

        results = {
            'attack_type': attack_type,
            'attack_params': attack_params,
            'baseline_method': baseline_method,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'detection_metrics': self._final_detection_metrics(detection_accum),
            'convergence_round': convergence_round,
            'rounds': per_round_rows,
        }

        return results

    def run_all_experiments(self):
        """Run all RQ5 experiments"""
        logger.info("\n" + "="*70)
        logger.info("Starting RQ5: Robust Aggregation Composability Verification")
        logger.info("="*70)

        # Prepare data once
        self.prepare_data()

        all_results = []

        # Run experiments for each attack and baseline combination
        for attack_type, attack_params in self.config['attacks'].items():
            for malicious_ratio in attack_params.get('malicious_ratios', [0.0]):
                params = attack_params.copy()
                params['malicious_ratios'] = [malicious_ratio]

                for baseline_method in self.config['baselines']:
                    result = self.run_single_experiment(
                        attack_type=attack_type,
                        attack_params=params,
                        baseline_method=baseline_method
                    )
                    all_results.append(result)

        # Save results
        self.save_results(all_results)

        # Generate summary
        self.generate_summary(all_results)

        logger.info("\n" + "="*70)
        logger.info("RQ5 Experiments Completed!")
        logger.info("="*70)

        return all_results

    def save_results(self, results):
        """Save experiment results"""
        # Save JSON
        json_path = self.output_dir / 'rq5_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {json_path}")

        # Save CSV summary
        csv_path = self.output_dir / 'rq5_summary.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'attack_type', 'baseline_method', 'final_accuracy',
                'convergence_round'
            ])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'attack_type': result['attack_type'],
                    'baseline_method': result['baseline_method'],
                    'final_accuracy': result['final_accuracy'],
                    'convergence_round': result['convergence_round'],
                })
        logger.info(f"Saved summary to {csv_path}")

    def generate_summary(self, results):
        """Generate and print summary statistics"""
        logger.info("\n" + "="*70)
        logger.info("RQ5 Summary: Composability Analysis")
        logger.info("="*70)

        # Group results by attack type
        from collections import defaultdict
        by_attack = defaultdict(list)
        for result in results:
            by_attack[result['attack_type']].append(result)

        # For each attack, compare robust aggregation vs PoL + robust aggregation
        for attack_type, attack_results in by_attack.items():
            logger.info(f"\n--- Attack: {attack_type} ---")

            # Group by base method
            by_base_method = defaultdict(list)
            for result in attack_results:
                baseline = result['baseline_method']
                base_method = baseline.replace('PoL_', '')
                by_base_method[base_method].append(result)

            # Compare each base method
            for base_method, method_results in by_base_method.items():
                # Find results with and without PoL
                without_pol = [r for r in method_results if not r['baseline_method'].startswith('PoL_')]
                with_pol = [r for r in method_results if r['baseline_method'].startswith('PoL_')]

                if without_pol and with_pol:
                    acc_without = without_pol[0]['final_accuracy']
                    acc_with = with_pol[0]['final_accuracy']
                    improvement = acc_with - acc_without

                    logger.info(f"  {base_method}:")
                    logger.info(f"    Without PoL: {acc_without:.4f}")
                    logger.info(f"    With PoL:    {acc_with:.4f}")
                    logger.info(f"    Improvement: {improvement:+.4f}")

        logger.info("\n" + "="*70)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='RQ5: Robust Aggregation Composability')
    parser.add_argument('--dataset', type=str, default=None, choices=['MNIST', 'CIFAR10', 'CIFAR100'], help='Dataset name')
    parser.add_argument('--model', type=str, default=None, help='Model name (default: auto-select by dataset)')
    parser.add_argument('--num_rounds', type=int, default=None, help='Number of rounds')
    parser.add_argument('--num_clients', type=int, default=None, help='Total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=None, help='Number of clients per round')
    parser.add_argument('--local_epochs', type=int, default=None, help='Local training epochs')
    parser.add_argument('--data_distribution', type=str, default=None, choices=['IID', 'NonIID_Dirichlet'], help='Data partition')
    parser.add_argument('--dirichlet_alpha', type=float, default=None, help='Dirichlet alpha for Non-IID partition')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--attacks', type=str, default='', help='Comma-separated subset of attacks to run')
    parser.add_argument('--baselines', type=str, default='', help='Comma-separated subset of baselines to run')
    parser.add_argument('--pol_delta', type=float, default=None, help='Override PoL L2 distance threshold (delta)')
    parser.add_argument('--verification_rate', type=float, default=None, help='Override PoL verification_rate (0-1)')
    args = parser.parse_args()

    # Update config with CLI args
    config = dict(RQ5_CONFIG)
    if args.dataset:
        config['dataset'] = args.dataset
    if args.num_rounds:
        config['num_rounds'] = args.num_rounds
    if args.num_clients:
        config['num_clients'] = args.num_clients
    if args.clients_per_round:
        config['clients_per_round'] = args.clients_per_round
    if args.local_epochs is not None:
        FL_CONFIG['local_epochs'] = int(args.local_epochs)
    if args.data_distribution:
        config['data_distribution'] = args.data_distribution
    if args.dirichlet_alpha is not None:
        config['dirichlet_alpha'] = float(args.dirichlet_alpha)
    if args.output_dir:
        OUTPUT_CONFIG['results_dir'] = args.output_dir
    config['local_epochs'] = int(FL_CONFIG.get('local_epochs', 0))
    config['batch_size'] = int(FL_CONFIG.get('batch_size', 0))
    # Model selection: explicit argument wins; otherwise auto-select for MNIST
    if args.model is not None:
        config['model'] = args.model
    else:
        if config['dataset'] == 'MNIST':
            config['model'] = 'SimpleCNN'

    # Apply PoL overrides if provided
    pol_cfg = config.get('pol_config', {})
    if args.pol_delta is not None:
        pol_cfg['delta'] = float(args.pol_delta)
    if args.verification_rate is not None:
        pol_cfg['verification_rate'] = float(args.verification_rate)
    config['pol_config'] = pol_cfg

    # Filter attacks/baselines if provided
    if args.attacks:
        allowed_attacks = set(config['attacks'].keys())
        chosen_attacks = [a.strip() for a in args.attacks.split(',') if a.strip()]
        unknown = [a for a in chosen_attacks if a not in allowed_attacks]
        if unknown:
            raise ValueError(f"Unknown attacks: {unknown}. Allowed: {sorted(allowed_attacks)}")
        config['attacks'] = {k: config['attacks'][k] for k in chosen_attacks}
    if args.baselines:
        allowed_b = {
            'Krum', 'PoL_Krum',
            'Trimmed_Mean', 'PoL_Trimmed_Mean',
            'Median', 'PoL_Median',
            'Bulyan', 'PoL_Bulyan',
        }
        chosen_b = [b.strip() for b in args.baselines.split(',') if b.strip()]
        unknown_b = [b for b in chosen_b if b not in allowed_b]
        if unknown_b:
            raise ValueError(f"Unknown baselines: {unknown_b}. Allowed: {sorted(allowed_b)}")
        config['baselines'] = chosen_b

    # Run experiments
    experiment = ComposabilityExperiment(config)
    results = experiment.run_all_experiments()

    return results


if __name__ == '__main__':
    main()
