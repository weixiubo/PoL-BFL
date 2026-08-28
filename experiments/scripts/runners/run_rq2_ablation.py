"""
RQ2: Ablation Study - Component Importance Analysis

This script evaluates the contribution of each component (PoL, ZKP, Incentive)
to the overall system performance.

Variants tested:
1. Vanilla FL (no defense)
2. PoL only (PoL verification only)
3. PoL + ZKP (PoL + zero-knowledge proofs)
4. PoL + Incentive (PoL + incentive mechanism)
5. PoL + ZKP + Incentive (full system)

Metrics: MA, DR, FPR, Participation Rate, Verify Pass Rate
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import csv

# Set CUBLAS_WORKSPACE_CONFIG for deterministic behavior
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    logging.info("CUBLAS_WORKSPACE_CONFIG not set; defaulting to :4096:8 for deterministic CUDA")

# Add parent directory to path
scripts_dir = Path(__file__).parent.parent
utils_dir = scripts_dir / 'utils'
project_root = scripts_dir.parent.parent  # PoL-BFL
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(utils_dir))

from experiment_config import FL_CONFIG, POL_CONFIG, OUTPUT_CONFIG, DATASETS, NUM_WORKERS, set_random_seed
from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders
from models import create_model
from metrics import compute_accuracy, compute_detection_metrics

# PoL components
from client.trainer.PoLTrainer import PoLTrainer
from client.PoLClient import PoLClient
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from baselines import create_aggregator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# RQ2 Ablation Study Config
# [WARNING] IMPORTANT: Original config is 100 rounds (see RQ2_EXPERIMENT_CONFIGS.md)
# Current config: 20 rounds for faster validation of improved design
RQ2_ABLATION_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 20,
    'clients_per_round': 10,
    'num_rounds': 20,  # Reduced-scale compatibility default.
    'data_distribution': 'NonIID_Dirichlet',

    # Attack scenario (fixed for ablation)
    'attack_type': 'byzantine_random_noise',
    'malicious_ratio': 0.2,
    'noise_scale': 1.0,

    # Variants to test
    'variants': [
        'vanilla_fl',
        'pol_only',
        'pol_zkp',
        'pol_incentive',
        'pol_zkp_incentive'
    ],

    # Number of repetitions for statistical significance
    'num_repetitions': 3,
}


class AblationStudyExperiment:
    """Ablation Study Experiment Runner"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seed
        set_random_seed(42)

        # Create base output directory (microsecond precision) to avoid collisions
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.base_dir = Path(OUTPUT_CONFIG['results_dir']) / 'rq2_ablation' / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: self.output_dir and self.pol_data_dir will be set per-variant in run_all_experiments()

        logger.info(f"Initialized AblationStudyExperiment on {self.device}")

    def prepare_data(self):
        """Prepare datasets"""
        logger.info("Preparing datasets...")

        ds_name = self.config['dataset']
        ds_root = DATASETS[ds_name]['data_dir']
        train_dataset = load_dataset(ds_name, data_dir=ds_root, train=True)
        test_dataset = load_dataset(ds_name, data_dir=ds_root, train=False)

        client_datasets = partition_data_dirichlet(
            train_dataset,
            self.config['num_clients'],
            alpha=0.5
        )

        self.train_loaders = create_dataloaders(
            client_datasets,
            batch_size=FL_CONFIG['batch_size'],
            num_workers=NUM_WORKERS
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FL_CONFIG['batch_size'],
            shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2)
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

        model = create_model(
            self.config['model'],
            num_classes=num_classes,
            input_channels=input_channels
        )
        return model.to(self.device)

    def run_all_experiments(self):
        """Run all ablation experiments"""
        self.prepare_data()

        all_results = []

        for variant in self.config['variants']:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running variant: {variant}")
            logger.info(f"{'='*70}")

            # Set per-variant output directories to avoid collisions across variants
            self.output_dir = self.base_dir / variant
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.pol_data_dir = self.output_dir / 'pol_data'
            self.pol_data_dir.mkdir(parents=True, exist_ok=True)

            variant_results = []

            for rep in range(self.config['num_repetitions']):
                logger.info(f"\nRepetition {rep + 1}/{self.config['num_repetitions']}")

                # Set different seed for each repetition
                set_random_seed(42 + rep)

                result = self._run_single_experiment(variant, rep)
                variant_results.append(result)

                # Persist the result for this configuration.
                self._save_results(all_results + [{'variant': variant, 'results': variant_results}])

            # Aggregate results across repetitions
            aggregated = self._aggregate_results(variant, variant_results)
            all_results.append(aggregated)

        # Save final results at base_dir to provide a single aggregated summary for this run
        _prev_output_dir = getattr(self, 'output_dir', None)
        self.output_dir = self.base_dir
        self._save_results(all_results)
        self.output_dir = _prev_output_dir
        self._print_summary(all_results)

        return all_results

    def _run_single_experiment(self, variant, repetition):
        """Run a single experiment with specified variant"""
        # Create model
        global_model = self.create_model()

        # Configure variant
        enable_pol = variant != 'vanilla_fl'
        enable_zkp = 'zkp' in variant
        enable_incentive = 'incentive' in variant

        # Create aggregator
        if enable_pol:
            # PoLVerifyAggregator consumes a flat configuration structure.
            # Override distance metric/delta per model to reduce FPR and stabilize pass rate
            # Allow env override for smoke probe: POL_DELTA_OVERRIDE
            _model_name = self.config.get('model', '')
            _default_delta = 100.0 if _model_name == 'ResNet18' else 10.0
            delta_override = float(os.getenv('POL_DELTA_OVERRIDE', str(_default_delta)))
            agg_args = {
                'device': str(self.device),
                'enable_pol': True,  # Must be at top level.
                'verification_rate': POL_CONFIG['verification_rate'],
                'pol_delta': delta_override,
                'pol_distance_metric': 'l2',
                'use_top_q': POL_CONFIG.get('use_top_q', False),
                'top_q': POL_CONFIG['top_q'],
                # Root-cause improvements
                'min_pair_success_rate': POL_CONFIG.get('min_pair_success_rate', 0.99),
                'always_verify_last_k': POL_CONFIG.get('always_verify_last_k', 2),
                'random_q': POL_CONFIG.get('random_q', 3),
                'enable_zkp': enable_zkp,
                'enable_incentives': enable_incentive,
            }
            aggregator = PoLVerifyAggregator(model=global_model, args=agg_args)
        else:
            aggregator = create_aggregator('Vanilla_FL')

        # Training loop
        test_accuracies = []
        detection_metrics_per_round = []
        conditional_tprs = []
        participation_rates = []
        verification_pass_rates = []

        # Decentralization/observability per-round (aligned with RQ1)
        ext_succ, ext_lat = [], []
        rm_resp, rm_yes = [], []
        pol_vt = []
        rlat_p50, rlat_p95 = [], []
        rerr_timeout, rerr_network, rerr_invalid, rerr_business = [], [], [], []
        ext_err_type = []

        for round_num in range(self.config['num_rounds']):
            logger.info(f"Round {round_num + 1}/{self.config['num_rounds']}")

            # Select clients
            selected_indices = np.random.choice(
                self.config['num_clients'],
                self.config['clients_per_round'],
                replace=False
            )

            # Determine malicious clients
            num_malicious = int(self.config['malicious_ratio'] * self.config['clients_per_round'])
            malicious_indices = selected_indices[:num_malicious]

            # Persist per-round selection for offline analysis (preflight instrumentation)
            try:
                sel_rec = {
                    'round': int(round_num + 1),
                    'selected_indices': [int(x) for x in selected_indices.tolist()],
                    'malicious_indices': [int(x) for x in malicious_indices.tolist()],
                }
                sel_path = self.output_dir / f"round_{round_num+1:03d}_selection.json"
                with open(sel_path, 'w') as sf:
                    json.dump(sel_rec, sf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to persist round selection info: {e}")

            # Build clients
            if enable_pol:
                clients = self._build_pol_clients(
                    global_model, selected_indices, malicious_indices,
                    enable_zkp=enable_zkp
                )
            else:
                clients = self._build_vanilla_clients(
                    global_model, selected_indices, malicious_indices
                )

            # Aggregate models
            if enable_pol:
                # PoL-enabled: use receive_upload() which handles verification
                aggregator.receive_upload(clients)

                # Aggregate through the parameter-free compatibility method.
                # NOTE: Verification happens inside aggregate() via _on_before_aggregation()
                global_model_dict = aggregator.aggregate()

                # Get verification results (AFTER aggregate(), not before!)
                verification_results = aggregator.verification_results

                # Compute detection metrics
                malicious_client_ids = [f"client_{int(idx)}" for idx in malicious_indices]
                all_client_ids = [f"client_{int(idx)}" for idx in selected_indices]

                detection_metrics = compute_detection_metrics(
                    verification_results,
                    malicious_client_ids,
                    all_client_ids
                )
                detection_metrics_per_round.append(detection_metrics)
                # Conditional TPR among verified malicious clients
                verified_mal = [cid for cid in malicious_client_ids if cid in verification_results]
                if len(verified_mal) > 0:
                    tp_cond = sum((not verification_results[cid]) for cid in verified_mal)
                    tpr_cond = tp_cond / len(verified_mal)
                else:
                    tpr_cond = 0.0
                conditional_tprs.append(float(tpr_cond))

                # Compute participation/verification metrics
                selected_count = len(selected_indices)

                # Use verification_results dict which contains all verified clients
                # Key: client_id, Value: bool (True=passed, False=failed)
                verification_results_dict = getattr(aggregator, 'verification_results', {})
                total_verified = len(verification_results_dict)
                pass_count = sum(1 for v in verification_results_dict.values() if v)

                # Participation Rate = fraction of selected clients that were actually verified
                # (i.e., clients that responded to challenge, regardless of pass/fail)
                participation_rate = (total_verified / selected_count) if selected_count > 0 else 0.0
                participation_rates.append(participation_rate)

                # Verification Pass Rate = passes among verified clients
                # (i.e., what fraction of verified clients passed verification)
                verification_pass_rate = (pass_count / total_verified) if total_verified > 0 else 0.0
                verification_pass_rates.append(verification_pass_rate)
            else:
                # Vanilla FL: no detection, 100% participation
                detection_metrics_per_round.append({'TPR': 0.0, 'FPR': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0})
                conditional_tprs.append(0.0)
                participation_rates.append(1.0)
                # For Vanilla FL, fill decentralization/observability with zeros for this round
                ext_succ.append(0); ext_lat.append(0.0); rm_resp.append(0); rm_yes.append(0)
                pol_vt.append(0.0); rlat_p50.append(0.0); rlat_p95.append(0.0)
                rerr_timeout.append(0); rerr_network.append(0); rerr_invalid.append(0); rerr_business.append(0)
                ext_err_type.append('')

                # Aggregate (VanillaFLAggregator.aggregate() needs models parameter)
                client_models = [client.get_model_state_dict() for client in clients]
                global_model_dict = aggregator.aggregate(client_models)

            global_model.load_state_dict(global_model_dict)

            # Evaluate accuracy
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)
            test_accuracies.append(test_acc)

            logger.info(f"  Test Accuracy: {test_acc:.4f}")
            if enable_pol:
                logger.info(f"  Detection TPR: {detection_metrics['TPR']:.4f}, FPR: {detection_metrics['FPR']:.4f}")
                logger.info(f"  Participation Rate: {participation_rate:.4f}, Verify Pass Rate: {verification_pass_rate:.4f}")

                # Capture aggregator metrics snapshot for CSV (PoL variants)
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


        # Aggregate metrics
        avg_detection_metrics = {
            'TPR': float(np.mean([m['TPR'] for m in detection_metrics_per_round])),
            'FPR': float(np.mean([m['FPR'] for m in detection_metrics_per_round])),
            'Precision': float(np.mean([m['Precision'] for m in detection_metrics_per_round])),
            'Recall': float(np.mean([m['Recall'] for m in detection_metrics_per_round])),
            'F1': float(np.mean([m['F1'] for m in detection_metrics_per_round])),
        }

        avg_participation_rate = float(np.mean(participation_rates)) if participation_rates else 0.0
        avg_verification_pass_rate = float(np.mean(verification_pass_rates)) if verification_pass_rates else 0.0

        # Build per-round CSV rows
        per_round_rows = []
        num_rounds = self.config['num_rounds']
        for i in range(num_rounds):
            tacc = float(test_accuracies[i]) if i < len(test_accuracies) else 0.0
            pr = float(participation_rates[i]) if i < len(participation_rates) else 0.0
            vpr = float(verification_pass_rates[i]) if i < len(verification_pass_rates) else 0.0
            tpr = float(detection_metrics_per_round[i]['TPR']) if i < len(detection_metrics_per_round) else 0.0
            tpr_cond = float(conditional_tprs[i]) if i < len(conditional_tprs) else 0.0
            fpr = float(detection_metrics_per_round[i]['FPR']) if i < len(detection_metrics_per_round) else 0.0
            prec = float(detection_metrics_per_round[i]['Precision']) if i < len(detection_metrics_per_round) else 0.0
            rec = float(detection_metrics_per_round[i]['Recall']) if i < len(detection_metrics_per_round) else 0.0
            f1 = float(detection_metrics_per_round[i]['F1']) if i < len(detection_metrics_per_round) else 0.0
            per_round_rows.append({
                'round': i + 1,
                'test_accuracy': tacc,
                'detection_tpr_conditional': tpr_cond,
                'participation_rate': pr,
                'verification_pass_rate': vpr,
                'detection_tpr': tpr,
                'detection_fpr': fpr,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                # Decentralization / Observability (if present)
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

        # Persist per-round CSV for plotting
        try:
            def _san(s: str) -> str:
                return s.replace('/', '_').replace(' ', '_')
            csv_name = f"rq2_rounds_{_san(variant)}_rep{int(repetition)}.csv"
            csv_path = self.output_dir / csv_name
            fieldnames = ['round', 'test_accuracy', 'participation_rate', 'verification_pass_rate', 'detection_tpr', 'detection_tpr_conditional', 'detection_fpr', 'precision', 'recall', 'f1',
                           'external_agg_success', 'external_agg_latency_s', 'remote_majority_responders', 'remote_majority_yes', 'pol_verify_time_s',
                           'remote_verify_latency_p50_s', 'remote_verify_latency_p95_s', 'remote_error_timeout', 'remote_error_network', 'remote_error_invalid', 'remote_error_business', 'external_agg_error_type']
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                for r in per_round_rows:
                    writer.writerow({k: (r.get(k, 0.0) if k != 'round' else r.get(k)) for k in fieldnames})
            logger.info(f"Per-round CSV saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to write per-round CSV: {e}")

        results = {
            'variant': variant,
            'repetition': repetition,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'detection_metrics': avg_detection_metrics,
            'participation_rate': avg_participation_rate,
            'verification_pass_rate': avg_verification_pass_rate,
            'rounds': per_round_rows,
        }

        return results

    def _build_pol_clients(self, global_model, selected_indices, malicious_indices, enable_zkp=False):
        """Build PoL clients"""
        clients = []

        for idx in selected_indices:
            client_id = f"client_{int(idx)}"
            is_malicious = idx in malicious_indices

            # Create trainer
            # IMPORTANT: All clients (including malicious) use PoL
            # PoL verification will detect malicious behavior (noise injection, insufficient training)
            trainer_args = {
                'device': self.device,
                'lr': 0.01,
                'weight_decay': 1e-4,
                'optimizer': 'SGD',
                'enable_pol': True,  # All clients use PoL
                'pol_save_freq': POL_CONFIG['save_freq'],
                'pol_save_dir': str(self.pol_data_dir),  # Use consistent path: experiments/results/rq2_ablation/{run_id}/pol_data
                'pol_compress': True,
                'client_id': client_id,
            }

            trainer = PoLTrainer(
                model=global_model,
                dataloader=self.train_loaders[idx],
                criterion=torch.nn.CrossEntropyLoss(),
                args=trainer_args
            )

            # Create client
            client = PoLClient(
                client_id=client_id,
                dataloader=self.train_loaders[idx],
                model=global_model,
                trainer=trainer,
                args=trainer_args
            )

            # Train
            if idx in malicious_indices:
                # Malicious: Byzantine random noise attack
                client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)
                # Add noise to model
                with torch.no_grad():
                    for param in client.model.parameters():
                        noise = torch.randn_like(param) * self.config['noise_scale']
                        param.add_(noise)
                logger.info(f"[Malicious] {client_id} Byzantine attack")
            else:
                # Honest client
                client.train(total_epoch=FL_CONFIG['local_epochs'], dataset=None)

            clients.append(client)

        return clients

    def _build_vanilla_clients(self, global_model, selected_indices, malicious_indices):
        """Build vanilla FL clients (no PoL) - simplified version without using BaseClient"""
        from client.trainer.normalTrainer import normalTrainer
        from copy import deepcopy

        clients = []

        for idx in selected_indices:
            client_id = f"client_{int(idx)}"

            # Create a copy of the global model for this client
            client_model = deepcopy(global_model)

            # Prepare trainer args
            trainer_args = {
                'device': self.device,
                'lr': 0.01,
                'weight_decay': 1e-4,
                'optimizer': 'SGD'
            }

            # Create trainer instance
            trainer = normalTrainer(
                model=client_model,
                dataloader=self.train_loaders[idx],
                criterion=torch.nn.CrossEntropyLoss(),
                args=trainer_args
            )

            # Train directly using trainer (no need for Client wrapper for vanilla FL)
            if idx in malicious_indices:
                # Malicious: Byzantine random noise attack
                trainer.train(total_epoch=FL_CONFIG['local_epochs'])
                with torch.no_grad():
                    for param in client_model.parameters():
                        noise = torch.randn_like(param) * self.config['noise_scale']
                        param.add_(noise)
                logger.info(f"[Malicious] {client_id} Byzantine attack")
            else:
                # Honest client
                trainer.train(total_epoch=FL_CONFIG['local_epochs'])

            # Store the trained model for subsequent state extraction.
            # Create a simple object to hold client_id and model
            class SimpleClient:
                def __init__(self, client_id, model):
                    self.client_id = client_id
                    self.model = model

                def get_model_state_dict(self):
                    return self.model.state_dict()

            client = SimpleClient(client_id, client_model)
            clients.append(client)

        return clients

    def _aggregate_results(self, variant, results_list):
        """Aggregate results across repetitions"""
        final_accs = [r['final_accuracy'] for r in results_list]
        tprs = [r['detection_metrics']['TPR'] for r in results_list]
        fprs = [r['detection_metrics']['FPR'] for r in results_list]
        participation_rates = [r['participation_rate'] for r in results_list]
        verify_pass_rates = [r.get('verification_pass_rate', 0.0) for r in results_list]

        return {
            'variant': variant,
            'final_accuracy_mean': float(np.mean(final_accs)),
            'final_accuracy_std': float(np.std(final_accs)),
            'tpr_mean': float(np.mean(tprs)),
            'tpr_std': float(np.std(tprs)),
            'fpr_mean': float(np.mean(fprs)),
            'fpr_std': float(np.std(fprs)),
            'participation_rate_mean': float(np.mean(participation_rates)),
            'participation_rate_std': float(np.std(participation_rates)),
            'verify_pass_rate_mean': float(np.mean(verify_pass_rates)),
            'verify_pass_rate_std': float(np.std(verify_pass_rates)),
        }

    def _save_results(self, results):
        """Save results and run configuration to JSON"""
        # Results file aligned with spec (primary)
        results_file = self.output_dir / 'rq2_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

        # Backward-compatible alias for downstream scripts
        ablation_file = self.output_dir / 'ablation_results.json'
        try:
            with open(ablation_file, 'w') as af:
                json.dump(results, af, indent=2)
            logger.info(f"Alias saved to {ablation_file}")
        except Exception as e:
            logger.warning(f"Failed to write ablation_results.json: {e}")

        # Persist run configuration for reproducibility
        config_file = self.output_dir / 'config.json'
        try:
            with open(config_file, 'w') as cf:
                json.dump(self.config, cf, indent=2)
            logger.info(f"Run configuration saved to {config_file}")
        except Exception as e:
            logger.warning(f"Failed to write config.json: {e}")

    def _print_summary(self, results):
        """Print summary table"""
        logger.info("\n" + "="*70)
        logger.info("RQ2: Ablation Study Results")
        logger.info("="*70)

        logger.info(f"\n{'Variant':<25} {'MA (%)':<15} {'DR (%)':<15} {'FPR (%)':<15} {'VerifyPass (%)':<17} {'Participation (%)':<18}")
        logger.info("-"*110)

        for result in results:
            variant = result['variant'].replace('_', ' ').title()
            ma = f"{result['final_accuracy_mean']*100:.1f}±{result['final_accuracy_std']*100:.1f}"
            dr = f"{result['tpr_mean']*100:.1f}±{result['tpr_std']*100:.1f}"
            fpr = f"{result['fpr_mean']*100:.1f}±{result['fpr_std']*100:.1f}"
            vpass = f"{result.get('verify_pass_rate_mean', 0.0)*100:.1f}±{result.get('verify_pass_rate_std', 0.0)*100:.1f}"
            part = f"{result['participation_rate_mean']*100:.1f}±{result['participation_rate_std']*100:.1f}"

            logger.info(f"{variant:<25} {ma:<15} {dr:<15} {fpr:<15} {vpass:<17} {part:<18}")

        logger.info("="*70)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='RQ2: Ablation Study')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='Dataset to use')
    parser.add_argument('--num_rounds', type=int, default=None,
                       help='Number of rounds (default: auto-select based on dataset: MNIST=20, CIFAR=100)')
    parser.add_argument('--num_repetitions', type=int, default=3,
                       help='Number of repetitions for statistical significance')
    parser.add_argument('--variants', type=str, default='',
                       help='Comma-separated subset of variants to run (allowed: vanilla_fl,pol_only,pol_zkp,pol_incentive,pol_zkp_incentive)')

    args = parser.parse_args()

    # Auto-select num_rounds based on dataset if not specified
    if args.num_rounds is None:
        if args.dataset == 'MNIST':
            args.num_rounds = 20
        else:  # CIFAR10, CIFAR100
            args.num_rounds = 100

    # Update config
    config = RQ2_ABLATION_CONFIG.copy()
    config['dataset'] = args.dataset
    config['num_rounds'] = args.num_rounds
    config['num_repetitions'] = args.num_repetitions

    # Override variants if provided
    if args.variants:
        vlist = [v.strip() for v in args.variants.split(',') if v.strip()]
        allowed = set(RQ2_ABLATION_CONFIG['variants'])
        unknown = [v for v in vlist if v not in allowed]
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}. Allowed: {sorted(allowed)}")
        config['variants'] = vlist

    # Auto-select model
    if args.dataset == 'MNIST':
        config['model'] = 'SimpleCNN'
    else:
        config['model'] = 'ResNet18'

    logger.info(f"Running RQ2 Ablation Study with:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model: {config['model']}")
    logger.info(f"  Rounds: {args.num_rounds}")
    logger.info(f"  Repetitions: {args.num_repetitions}")
    logger.info(f"  Variants: {', '.join(config['variants'])}")

    experiment = AblationStudyExperiment(config)
    results = experiment.run_all_experiments()

    logger.info("\nRQ2: Ablation Study Completed.")


if __name__ == '__main__':
    main()
