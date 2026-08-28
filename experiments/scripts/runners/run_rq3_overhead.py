"""
RQ3: System Overhead Analysis

Measure storage, computation, and communication overheads of PoL-FL.

Metrics:
- Training Time
- Checkpoint Save Time
- PoL replay and Groth16 verification time
- Storage (checkpoint size)
- Communication (model size)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import json
import time
from pathlib import Path
import argparse
import shutil


import csv

from collections import OrderedDict

# Set CUBLAS_WORKSPACE_CONFIG for deterministic behavior
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add parent directory to path
scripts_dir = Path(__file__).parent.parent
experiments_dir = scripts_dir.parent
project_root = experiments_dir.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / 'utils'))
sys.path.insert(0, str(project_root))

from experiment_config import FL_CONFIG, POL_CONFIG, OUTPUT_CONFIG, DATASETS, NUM_WORKERS, set_random_seed
from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders
from models import create_model, get_model_size
from metrics import Profiler, compute_accuracy

# PoL/ZKP components for realistic measurements
from client.pol.PoLManager import PoLManager
from client.zkp.ZKPProver import ZKPProver
try:
    from server.zkp.ZKPVerifier import ZKPVerifier
except Exception:
    ZKPVerifier = None
# Optional blockchain proxy (for gas estimation)
try:
    from chainfl.interact import chainProxy
except Exception:
    chainProxy = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# RQ3 Configuration
RQ3_CONFIG = {
    'dataset': 'MNIST',
    'model': 'SimpleCNN',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 3,  # Testing: 3 rounds (Production: 5 rounds)
    'data_distribution': 'NonIID_Dirichlet',

    # Methods to compare
    'methods': ['Vanilla_FL', 'PoL_FL', 'PoL_FL_ZKP'],

    # PoL settings
    'pol_save_freq': 10,
    'checkpoint_dir': OUTPUT_CONFIG['checkpoints_dir'],
    'zkp_use_simulation': True,
    'measure_zkp': False,

    # Chain/Gas estimation (Plan A)
    'chain': {
        'gas_estimation': True,
        'gas_price_gwei': 10,
    },
}


class OverheadExperiment:
    """System overhead analysis experiment"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = Profiler()

        # Set random seed
        set_random_seed()

        # Create output directory
        self.output_dir = Path(OUTPUT_CONFIG['results_dir']) / 'rq3_overhead'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized OverheadExperiment on {self.device}")
        # Track measurement checkpoints for deterministic cleanup.
        self.created_checkpoint_files = []

    def prepare_data(self):
        """Prepare datasets"""
        logger.info("Preparing datasets...")

        # Load dataset with explicit repo data root
        ds_name = self.config['dataset']
        ds_root = DATASETS[ds_name]['data_dir']
        train_dataset = load_dataset(ds_name, data_dir=ds_root, train=True)
        test_dataset = load_dataset(ds_name, data_dir=ds_root, train=False)

        # Partition data
        client_datasets = partition_data_dirichlet(
            train_dataset,
            self.config['num_clients'],
            alpha=0.5
        )

        # Create dataloaders (parallel workers + pinned)
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

        logger.info(f"Prepared data for {self.config['num_clients']} clients")

    def create_model(self):
        """Create model"""
        if self.config['dataset'] == 'MNIST':
            num_classes = 10
            input_channels = 1
        elif self.config['dataset'] == 'CIFAR10':
            num_classes = 10
            input_channels = 3
        else:
            raise ValueError(f"Unknown dataset: {self.config['dataset']}")

        model = create_model(
            self.config['model'],
            num_classes=num_classes,
            input_channels=input_channels
        )

        return model

    def _estimate_gas_for_zkp(self, proof: dict, public_signals: dict):
        """Estimate gas for on-chain integrated ZKP verification (Plan A).
        Returns integer gas units or None if unavailable.
        """
        try:
            if chainProxy is None:
                return None
            proxy = chainProxy()
            pol = getattr(proxy, 'pol_contract', None)
            if pol is None:
                return None
            # Build a,b,c,inputs as in chainfl.interact helpers
            pi_a = proof.get('pi_a') or proof.get('A')
            pi_b = proof.get('pi_b') or proof.get('B')
            pi_c = proof.get('pi_c') or proof.get('C')
            a = [int(pi_a[0][0]), int(pi_a[0][1])] if isinstance(pi_a[0], (list, tuple)) else [int(pi_a[0]), int(pi_a[1])]
            b = [
                [int(pi_b[0][0]), int(pi_b[0][1])],
                [int(pi_b[1][0]), int(pi_b[1][1])],
            ]
            c = [int(pi_c[0][0]), int(pi_c[0][1])] if isinstance(pi_c[0], (list, tuple)) else [int(pi_c[0]), int(pi_c[1])]
            inputs = [
                int(public_signals.get('W_t_hash', 0)),
                int(public_signals.get('W_t1_hash', 0)),
                int(public_signals.get('data_hash', 0)),
                int(public_signals.get('max_distance', 0)),
            ]
            method = getattr(pol, 'challengeProofOnchainVerify', None) or pol.get_method('challengeProofOnchainVerify')
            cid_bytes = bytes(32)  # Zero-valued challenge identifier.
            tx_params = {'from': getattr(proxy, 'server_accounts', None)}
            gas = method.estimate_gas(cid_bytes, a, b, c, inputs, "", tx_params)
            return int(gas)
        except Exception as e:
            logger.debug(f"Gas estimation failed: {e}")
            return None

    def train_client(self, model, dataloader, save_checkpoints=False, client_id=0):
        """
        Train client model with optional checkpoint saving

        Returns:
            metrics: Dictionary of timing and size metrics
        """
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=FL_CONFIG['learning_rate'],
            momentum=FL_CONFIG['momentum']
        )
        criterion = nn.CrossEntropyLoss()

        metrics = {
            'training_time': 0.0,
            'checkpoint_save_time': 0.0,  # per-client total seconds (no cross-client accumulation)
            'num_checkpoints': 0,
            'checkpoint_size_mb': 0.0,
            'merkle_proof_size_mb': 0.0,
            'zkp_proof_gen_time_s': 0.0,
            'zkp_verify_time_s': 0.0,
            'zkp_estimated_gas': 0,
            'zkp_estimated_fee_eth': 0.0,
        }
        # local accumulator to avoid cross-client double counting
        _ckpt_time_total = 0.0

        # Optional PoL manager for proof size measurement
        pol_manager = None
        last_ckpt = None
        prev_ckpt = None
        last_saved_step = None
        prev_saved_step = None
        if save_checkpoints:
            try:
                pol_manager = PoLManager(
                    client_id=str(client_id),
                    save_dir=str(self.checkpoint_dir),
                    save_freq=self.config['pol_save_freq'],
                    save_to_disk=False
                )
            except Exception as e:
                logger.warning(f"Failed to init PoLManager for client {client_id}: {e}")
                pol_manager = None

        # Training
        self.profiler.start('training')

        iteration = 0
        for epoch in range(FL_CONFIG['local_epochs']):
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

                iteration += 1

                # Save checkpoint if PoL enabled
                if save_checkpoints and iteration % self.config['pol_save_freq'] == 0:
                    self.profiler.start('checkpoint_save')

                    checkpoint = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'iteration': iteration
                    }

                    checkpoint_path = self.checkpoint_dir / f'client_{client_id}_iter_{iteration}.pt'
                    torch.save(checkpoint, checkpoint_path)
                    # Record created measurement checkpoint for later cleanup
                    try:
                        self.created_checkpoint_files.append(str(checkpoint_path))
                    except Exception:
                        pass

                    # Track last/prev checkpoints for ZKP binding
                    prev_ckpt = last_ckpt
                    prev_saved_step = last_saved_step
                    last_ckpt = checkpoint
                    last_saved_step = iteration

                    # Also save to PoL manager (memory) to enable Merkle proof
                    if pol_manager is not None:
                        try:
                            pol_manager.save_checkpoint(step=iteration, checkpoint_data=checkpoint)
                        except Exception as e:
                            logger.warning(f"PoLManager.save_checkpoint failed: {e}")

                    # Measure checkpoint size
                    checkpoint_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
                    metrics['checkpoint_size_mb'] += checkpoint_size
                    metrics['num_checkpoints'] += 1

                    # accumulate local checkpoint time (seconds)
                    _ckpt_time_total += float(self.profiler.stop('checkpoint_save'))

        metrics['training_time'] = self.profiler.stop('training')
        # use local accumulator to avoid cross-client cumulative totals
        metrics['checkpoint_save_time'] = _ckpt_time_total

        # After training, compute Merkle proof size and ZKP metrics if enabled
        if save_checkpoints and pol_manager is not None and last_saved_step is not None:
            try:
                pol_manager.generate_commitment()
                proof = pol_manager.get_merkle_proof_by_step(last_saved_step)
                import json as _json
                proof_bytes = _json.dumps(proof).encode('utf-8')
                metrics['merkle_proof_size_mb'] = len(proof_bytes) / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Failed to compute Merkle proof size: {e}")

            if self.config.get('measure_zkp', False) and prev_ckpt is not None:
                # ZKP proof generation
                try:
                    prover = ZKPProver(use_simulation=self.config.get('zkp_use_simulation', True))
                    t0 = time.perf_counter()
                    # Signature may vary; attempt common interface
                    try:
                        proof_obj, public_signals = prover.generate_proof(prev_ckpt['model_state'], last_ckpt['model_state'], data_indices=[], max_distance=None)
                    except TypeError:
                        result = prover.generate_proof(prev_ckpt['model_state'], last_ckpt['model_state'])
                        if isinstance(result, tuple) and len(result) == 2:
                            proof_obj, public_signals = result
                        else:
                            proof_obj, public_signals = result, None
                    metrics['zkp_proof_gen_time_s'] = time.perf_counter() - t0
                except Exception as e:
                    logger.warning(f"ZKP proof generation failed: {e}")

                # Optional gas estimation (Plan A)
                try:
                    if self.config.get('chain', {}).get('gas_estimation', True) and public_signals is not None:
                        gas_est = self._estimate_gas_for_zkp(proof_obj, public_signals)
                        if gas_est is not None:
                            metrics['zkp_estimated_gas'] = int(gas_est)
                            gp = float(self.config.get('chain', {}).get('gas_price_gwei', 10))
                            metrics['zkp_estimated_fee_eth'] = float(gas_est) * gp / 1e9
                except Exception as ge:
                    logger.debug(f"Gas estimation skipped: {ge}")

                # ZKP verification (simulation) if verifier available
                if ZKPVerifier is not None and 'proof_obj' in locals():
                    try:
                        verifier = ZKPVerifier(use_simulation=self.config.get('zkp_use_simulation', True))
                        t1 = time.perf_counter()
                        # Try a permissive verify call
                        try:
                            ok = verifier.verify_proof_with_binding(
                                current_ckpt=prev_ckpt,

                                next_ckpt=last_ckpt,
                                data_indices=[],
                                proof=proof_obj,
                                public_signals=public_signals
                            )
                        except TypeError:
                            ok = verifier.verify_proof(proof_obj, public_signals)
                        metrics['zkp_verify_time_s'] = time.perf_counter() - t1
                        if not ok:
                            logger.warning("ZKP verification reported failure in simulation mode.")
                    except Exception as e:
                        logger.warning(f"ZKP verification failed: {e}")

        return metrics

    def measure_communication_overhead(self, model):
        """Measure communication overhead components that are model-dependent"""
        # Model size only; proof size is measured empirically per-client
        model_size_mb = get_model_size(model)
        return {
            'model_size_mb': model_size_mb
        }

    def run_vanilla_fl(self):
        """Run Vanilla FL and measure overhead"""
        logger.info("\n=== Running Vanilla FL ===")

        global_model = self.create_model().to(self.device)

        results = {
            'method': 'Vanilla_FL',
            'rounds': [],
            'total_training_time': 0.0,
            'total_communication_mb': 0.0,
            'avg_round_time': 0.0
        }

        for round_num in range(self.config['num_rounds']):
            logger.info(f"Round {round_num + 1}/{self.config['num_rounds']}")

            round_start = time.time()

            # Select clients
            selected_indices = np.random.choice(
                self.config['num_clients'],
                self.config['clients_per_round'],
                replace=False
            )

            # Train clients
            client_models = []
            round_training_time = 0.0

            for idx in selected_indices:
                client_model = self.create_model().to(self.device)
                client_model.load_state_dict(global_model.state_dict())

                metrics = self.train_client(client_model, self.train_loaders[idx], save_checkpoints=False)
                round_training_time += metrics['training_time']

                client_models.append(client_model.state_dict())

            # Aggregate (simple average)
            aggregated_state = OrderedDict()
            for key in client_models[0].keys():
                aggregated_state[key] = sum(model[key] for model in client_models) / len(client_models)

            global_model.load_state_dict(aggregated_state)

            # Measure communication
            comm_metrics = self.measure_communication_overhead(global_model)
            round_comm = comm_metrics['model_size_mb'] * self.config['clients_per_round'] * 2  # Upload + Download

            # Evaluate
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)

            round_time = time.time() - round_start

            round_results = {
                'round': round_num + 1,
                'training_time': round_training_time,
                'communication_mb': round_comm,
                'round_time': round_time,
                'test_accuracy': test_acc,
                # Decentralization/observability (uniform columns; Vanilla=0)
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
            }

            results['rounds'].append(round_results)
            results['total_training_time'] += round_training_time
            results['total_communication_mb'] += round_comm

            logger.info(f"  Training Time: {round_training_time:.2f}s")
            logger.info(f"  Communication: {round_comm:.2f}MB")
            logger.info(f"  Test Accuracy: {test_acc:.4f}")

        results['avg_round_time'] = np.mean([r['round_time'] for r in results['rounds']])

        return results

    def run_pol_fl(self, with_zkp: bool = False):
        """Run PoL-FL and measure overhead. If with_zkp=True, also measure ZKP costs."""
        title = "PoL-FL+ZKP" if with_zkp else "PoL-FL"
        logger.info(f"\n=== Running {title} ===")

        # Toggle ZKP measurement according to variant
        prev_measure = self.config.get('measure_zkp', False)
        self.config['measure_zkp'] = bool(with_zkp)

        global_model = self.create_model().to(self.device)

        results = {
            'method': 'PoL_FL_ZKP' if with_zkp else 'PoL_FL',
            'rounds': [],
            'total_training_time': 0.0,
            'total_checkpoint_time': 0.0,
            'total_storage_mb': 0.0,
            'total_communication_mb': 0.0,
            'total_estimated_gas': 0,
            'total_estimated_fee_eth': 0.0,
            'total_zkp_gen_time': 0.0,
            'total_zkp_verify_time': 0.0,
            'avg_round_time': 0.0
        }

        for round_num in range(self.config['num_rounds']):
            logger.info(f"Round {round_num + 1}/{self.config['num_rounds']}")

            round_start = time.time()

            # Select clients
            selected_indices = np.random.choice(
                self.config['num_clients'],
                self.config['clients_per_round'],
                replace=False
            )

            # Train clients with PoL
            client_models = []
            client_metrics = []
            round_training_time = 0.0
            round_checkpoint_time = 0.0
            round_storage_mb = 0.0

            for idx in selected_indices:
                client_model = self.create_model().to(self.device)
                client_model.load_state_dict(global_model.state_dict())

                metrics = self.train_client(
                    client_model,
                    self.train_loaders[idx],
                    save_checkpoints=True,
                    client_id=idx
                )

                round_training_time += metrics['training_time']
                round_checkpoint_time += metrics['checkpoint_save_time']
                round_storage_mb += metrics['checkpoint_size_mb']

                client_models.append(client_model.state_dict())
                client_metrics.append(metrics)

            # Aggregate
            aggregated_state = OrderedDict()
            for key in client_models[0].keys():
                aggregated_state[key] = sum(model[key] for model in client_models) / len(client_models)

            global_model.load_state_dict(aggregated_state)

            # Measure communication (model + empirically measured average Merkle proof)
            comm_metrics = self.measure_communication_overhead(global_model)
            model_size_mb = comm_metrics['model_size_mb']
            avg_proof_mb = float(np.mean([m.get('merkle_proof_size_mb', 0.0) for m in client_metrics])) if client_metrics else 0.0
            round_comm = (model_size_mb + avg_proof_mb) * self.config['clients_per_round'] * 2

            # Evaluate
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)
            # Gas estimation sums (if any)
            est_gas_round = int(np.sum([m.get('zkp_estimated_gas', 0) for m in client_metrics])) if client_metrics else 0
            est_fee_round = float(np.sum([m.get('zkp_estimated_fee_eth', 0.0) for m in client_metrics])) if client_metrics else 0.0


            round_time = time.time() - round_start

            # ZKP times (if collected)
            zkp_gen_time = float(np.sum([m.get('zkp_proof_gen_time_s', 0.0) for m in client_metrics])) if client_metrics else 0.0
            zkp_verify_time = float(np.sum([m.get('zkp_verify_time_s', 0.0) for m in client_metrics])) if client_metrics else 0.0

            round_results = {
                'round': round_num + 1,
                'training_time': round_training_time,
                'checkpoint_time': round_checkpoint_time,
                'storage_mb': round_storage_mb,
                'communication_mb': round_comm,
                'zkp_proof_gen_time_s': zkp_gen_time,
                'zkp_verify_time_s': zkp_verify_time,
                'estimated_gas': est_gas_round,
                'estimated_fee_eth': est_fee_round,
                'round_time': round_time,
                'test_accuracy': test_acc,
                # Decentralization/observability (uniform columns)
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
            }

            results['rounds'].append(round_results)
            results['total_training_time'] += round_training_time
            results['total_checkpoint_time'] += round_checkpoint_time
            results['total_storage_mb'] += round_storage_mb
            results['total_communication_mb'] += round_comm
            results['total_estimated_gas'] += int(est_gas_round)
            results['total_estimated_fee_eth'] += float(est_fee_round)
            results['total_zkp_gen_time'] += zkp_gen_time
            results['total_zkp_verify_time'] += zkp_verify_time

            logger.info(f"  Training Time: {round_training_time:.2f}s")
            logger.info(f"  Checkpoint Time: {round_checkpoint_time:.2f}s")
            logger.info(f"  Storage: {round_storage_mb:.2f}MB")
            logger.info(f"  Communication: {round_comm:.2f}MB")
            logger.info(f"  Test Accuracy: {test_acc:.4f}")

        results['avg_round_time'] = np.mean([r['round_time'] for r in results['rounds']])

        # Restore previous config flag
        self.config['measure_zkp'] = prev_measure
        return results

    def run_all_experiments(self):
        """Run all overhead experiments"""
        logger.info("Starting RQ3: System Overhead Analysis")

        # Prepare data
        self.prepare_data()

        all_results = []

        for method in self.config.get('methods', ['Vanilla_FL', 'PoL_FL', 'PoL_FL_ZKP']):
            if method == 'Vanilla_FL':
                res = self.run_vanilla_fl()
            elif method == 'PoL_FL':
                res = self.run_pol_fl(with_zkp=False)
            elif method == 'PoL_FL_ZKP':
                res = self.run_pol_fl(with_zkp=True)
            else:
                logger.warning(f"Unknown method '{method}', skipping.")
                continue
            all_results.append(res)

            # Persist per-round CSV for each method with superset columns
            try:
                def _san(s: str) -> str:
                    return s.replace('/', '_').replace(' ', '_')
                csv_name = f"rq3_rounds_{_san(res['method'])}.csv"
                csv_path = self.output_dir / csv_name
                rounds = res.get('rounds', [])
                if rounds:
                    fieldnames = ['round', 'training_time', 'checkpoint_time', 'storage_mb', 'communication_mb',
                                  'zkp_proof_gen_time_s', 'zkp_verify_time_s', 'estimated_gas', 'estimated_fee_eth',
                                  'round_time', 'test_accuracy',
                                  'external_agg_success', 'external_agg_latency_s', 'remote_majority_responders', 'remote_majority_yes',
                                  'pol_verify_time_s', 'remote_verify_latency_p50_s', 'remote_verify_latency_p95_s',
                                  'remote_error_timeout', 'remote_error_network', 'remote_error_invalid', 'remote_error_business', 'external_agg_error_type']
                    with open(csv_path, 'w', newline='') as cf:
                        writer = csv.DictWriter(cf, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in rounds:
                            writer.writerow({
                                'round': r.get('round'),
                                'training_time': float(r.get('training_time', 0.0)),
                                'checkpoint_time': float(r.get('checkpoint_time', 0.0)),
                                'storage_mb': float(r.get('storage_mb', 0.0)),
                                'communication_mb': float(r.get('communication_mb', 0.0)),
                                'zkp_proof_gen_time_s': float(r.get('zkp_proof_gen_time_s', 0.0)),
                                'zkp_verify_time_s': float(r.get('zkp_verify_time_s', 0.0)),
                                'estimated_gas': int(r.get('estimated_gas', 0)),
                                'estimated_fee_eth': float(r.get('estimated_fee_eth', 0.0)),
                                'round_time': float(r.get('round_time', 0.0)),
                                'test_accuracy': float(r.get('test_accuracy', 0.0)),
                                'external_agg_success': int(r.get('external_agg_success', 0)),
                                'external_agg_latency_s': float(r.get('external_agg_latency_s', 0.0)),
                                'remote_majority_responders': int(r.get('remote_majority_responders', 0)),
                                'remote_majority_yes': int(r.get('remote_majority_yes', 0)),
                                'pol_verify_time_s': float(r.get('pol_verify_time_s', 0.0)),
                                'remote_verify_latency_p50_s': float(r.get('remote_verify_latency_p50_s', 0.0)),
                                'remote_verify_latency_p95_s': float(r.get('remote_verify_latency_p95_s', 0.0)),
                                'remote_error_timeout': int(r.get('remote_error_timeout', 0)),
                                'remote_error_network': int(r.get('remote_error_network', 0)),
                                'remote_error_invalid': int(r.get('remote_error_invalid', 0)),
                                'remote_error_business': int(r.get('remote_error_business', 0)),
                                'external_agg_error_type': str(r.get('external_agg_error_type', '')),
                            })
                    logger.info(f"Per-round CSV saved to {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to write per-round CSV: {e}")

        # Save results and configuration
        output_file = self.output_dir / 'rq3_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        try:
            with open(self.output_dir / 'config.json', 'w') as cf:
                json.dump(self.config, cf, indent=2)
            logger.info(f"Run configuration saved to {self.output_dir / 'config.json'}")
        except Exception as e:
            logger.warning(f"Failed to write config.json: {e}")

        logger.info(f"\nResults saved to {output_file}")

        # Optional archiving to timestamped compare/{label}_TS directory
        try:
            label_env = os.getenv('RQ3_ARCHIVE_LABEL','').strip().lower()
            if label_env not in ('remote','local'):
                label_env = 'remote' if os.getenv('POL_DECENT_MODE','0') in ('1','true','True') else 'local'
            ts = time.strftime('%Y%m%d_%H%M%S')
            dest = self.output_dir / 'compare' / f"{label_env}_{ts}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, dest / output_file.name)
            cfg_path = self.output_dir / 'config.json'
            if cfg_path.exists():
                shutil.copy2(cfg_path, dest / cfg_path.name)
            logger.info(f"Archived RQ3 results to {dest}")
        except Exception as e:
            logger.warning(f"Archiving skipped: {e}")

        # Print summary
        self._print_summary(all_results)

        # Auto-clean measurement checkpoints unless user opts to keep them
        try:
            keep = bool(self.config.get('keep_checkpoints', False))
        except Exception:
            keep = False
        if not keep:
            try:
                removed = 0
                freed_mb = 0.0
                for fp in list(set(self.created_checkpoint_files)):
                    p = Path(fp)
                    if p.exists() and p.name.startswith('client_') and '_iter_' in p.name:
                        try:
                            size_mb = p.stat().st_size / (1024 * 1024)
                        except Exception:
                            size_mb = 0.0
                        try:
                            p.unlink()
                            removed += 1
                            freed_mb += size_mb
                        except Exception as e:
                            logger.warning(f"Failed to delete measurement checkpoint {p}: {e}")
                if removed > 0:
                    logger.info(f"Cleaned {removed} measurement checkpoints (~{freed_mb:.2f} MB) from {self.checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Checkpoint cleanup skipped due to error: {e}")

        return all_results

    def _print_summary(self, results):
        """Print experiment summary"""
        logger.info("\n" + "="*70)
        logger.info("RQ3: System Overhead Analysis Summary")
        logger.info("="*70)

        for result in results:
            logger.info(f"\n{result['method']}:")
            logger.info(f"  Total Training Time: {result['total_training_time']:.2f}s")
            logger.info(f"  Total Communication: {result['total_communication_mb']:.2f}MB")
            logger.info(f"  Avg Round Time: {result['avg_round_time']:.2f}s")

            if 'total_checkpoint_time' in result:
                logger.info(f"  Total Checkpoint Time: {result['total_checkpoint_time']:.2f}s")
                logger.info(f"  Total Storage: {result['total_storage_mb']:.2f}MB")

            if 'total_estimated_gas' in result:
                logger.info(f"  Total Estimated Gas (verify): {result['total_estimated_gas']}")
                logger.info(f"  Total Estimated Fee (@{self.config.get('chain', {}).get('gas_price_gwei', 10)} gwei): {result['total_estimated_fee_eth']:.6f} ETH")

        # Compute overhead
        if len(results) == 2:
            vanilla = results[0]
            pol = results[1]

            training_overhead = (pol['total_training_time'] - vanilla['total_training_time']) / vanilla['total_training_time'] * 100
            comm_overhead = (pol['total_communication_mb'] - vanilla['total_communication_mb']) / vanilla['total_communication_mb'] * 100

            logger.info(f"\nPoL-FL Overhead:")
            logger.info(f"  Training Time Overhead: +{training_overhead:.1f}%")
            logger.info(f"  Communication Overhead: +{comm_overhead:.1f}%")
            logger.info(f"  Storage Overhead: {pol['total_storage_mb']:.2f}MB")

        logger.info("="*70)


def main():
    """Main function"""
    ap = argparse.ArgumentParser(description='RQ3: System Overhead Analysis')
    ap.add_argument('--dataset', type=str, default='CIFAR10',
                   choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                   help='Dataset for profiling')
    ap.add_argument('--model', type=str, default=None,
                   help='Model name (default: auto-select based on dataset)')
    ap.add_argument('--rounds', type=int, default=20,
                   help='Number of rounds to profile (default: 20 for all datasets)')
    ap.add_argument('--num_clients', type=int, default=20,
                   help='Total number of clients')
    ap.add_argument('--clients_per_round', type=int, default=10,
                   help='Number of clients per round')
    ap.add_argument('--output_dir', type=str, default=None,
                   help='Base output directory; rq3_overhead will be created below it')
    ap.add_argument('--keep_checkpoints', action='store_true',
                   help='Do not delete measurement checkpoints created during the run')
    args = ap.parse_args()

    # Auto-select model based on dataset
    if args.model is None:
        if args.dataset == 'MNIST':
            args.model = 'SimpleCNN'
        else:
            args.model = 'ResNet18'

    # Apply overrides
    cfg = dict(RQ3_CONFIG)
    cfg['dataset'] = args.dataset
    cfg['model'] = args.model
    cfg['num_rounds'] = args.rounds
    cfg['num_clients'] = args.num_clients
    cfg['clients_per_round'] = args.clients_per_round
    cfg['keep_checkpoints'] = bool(getattr(args, 'keep_checkpoints', False))
    if args.output_dir:
        OUTPUT_CONFIG['results_dir'] = args.output_dir

    logger.info(f"Running RQ3 with:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Rounds: {args.rounds}")
    logger.info(f"  Clients: {args.num_clients} (per round: {args.clients_per_round})")

    experiment = OverheadExperiment(cfg)
    results = experiment.run_all_experiments()

    logger.info("\nRQ3: System Overhead Analysis Completed.")


if __name__ == '__main__':
    main()
