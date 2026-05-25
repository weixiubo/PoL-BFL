"""
RQ4: Economic Incentive Effectiveness

Evaluate the effectiveness of economic incentive system through game theory simulation.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import json
import random
from pathlib import Path
from collections import OrderedDict

import time

# Set CUBLAS_WORKSPACE_CONFIG for deterministic behavior
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Configure import paths (scripts, utils, project root)
scripts_dir = Path(__file__).parent.parent
experiments_dir = scripts_dir.parent
project_root = experiments_dir.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / 'utils'))
sys.path.insert(0, str(project_root))

from experiment_config import FL_CONFIG, OUTPUT_CONFIG, POL_CONFIG, DATASETS, NUM_WORKERS, set_random_seed
from data_utils import load_dataset, partition_data_dirichlet, create_dataloaders
from models import create_model
from metrics import compute_accuracy

# Import economic incentive components
# sys.path already configured above
from server.incentive.StakingManager import StakingManager
from server.incentive.RewardCalculator import RewardCalculator
from server.incentive.ReputationSystem import ReputationSystem

# PoL imports for real verification
from client.pol.PoLManager import PoLManager
from server.pol.PoLVerifier import PoLVerifier
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# RQ4 Configuration
RQ4_CONFIG = {
    'dataset': 'MNIST',
    'model': 'SimpleCNN',
    'num_clients': 20,
    'clients_per_round': 10,
    'num_rounds': 50,  # default per spec
    'data_distribution': 'NonIID_Dirichlet',

    # Node types
    'node_types': {
        'honest': 0.6,      # 60% honest
        'rational': 0.3,    # 30% rational
        'malicious': 0.1    # 10% malicious
    },

    # Utility parameters
    'utility_params': {
        'base_reward': 500,  # per-round reward pool (aligned with RewardCalculator default)
        'compute_cost': 10,
        'gas_cost': 2,
        'slash_penalty': 100,
        'detection_probability': 0.3
    },

    # Scenarios
    'scenarios': ['no_incentive', 'fixed_reward', 'dynamic_reward', 'sybil_attack'],

    # Sybil Attack configuration (for sybil_attack scenario)
    'sybil_config': {
        'base_scenario': 'dynamic_reward',  # Sybil attack uses dynamic_reward as base
        'attacker_count': 1,
        'identities_per_attacker': 5,
        'shared_resources': True
    }
}


class IncentiveExperiment:
    """Economic incentive effectiveness experiment"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seed
        set_random_seed()

        # Create output directory
        self.output_dir = Path(OUTPUT_CONFIG['results_dir']) / 'rq4_incentive'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Optional progress/status file for foreground runs
        self.status_path = self.config.get('status_path', None)


        # Initialize node types
        self.node_types = self._initialize_node_types()

        logger.info(f"Initialized IncentiveExperiment on {self.device}")
        logger.info(f"Node distribution: {sum(1 for t in self.node_types.values() if t == 'honest')} honest, "
                   f"{sum(1 for t in self.node_types.values() if t == 'rational')} rational, "
                   f"{sum(1 for t in self.node_types.values() if t == 'malicious')} malicious")

    def _initialize_node_types(self):
        """Initialize node types based on configuration"""
        node_types = {}
        num_clients = self.config['num_clients']

        num_honest = int(num_clients * self.config['node_types']['honest'])
        num_rational = int(num_clients * self.config['node_types']['rational'])
        num_malicious = num_clients - num_honest - num_rational

        client_ids = [f'client{i}' for i in range(num_clients)]

        for i, client_id in enumerate(client_ids):
            if i < num_honest:
                node_types[client_id] = 'honest'
            elif i < num_honest + num_rational:
                node_types[client_id] = 'rational'
            else:
                node_types[client_id] = 'malicious'

        return node_types
    def _write_status(self, payload: dict):
        """Write lightweight status JSON for foreground progress monitoring."""
        try:
            if not getattr(self, 'status_path', None):
                return
            data = {
                'ts': time.time(),
                'pid': os.getpid(),
                'device': str(self.device),
            }
            if isinstance(payload, dict):
                data.update(payload)
            with open(self.status_path, 'w') as sf:
                json.dump(data, sf, ensure_ascii=False, indent=2)
        except Exception:
            # Best-effort only; never interrupt training due to status I/O
            pass


    def prepare_data(self):
        """Prepare datasets"""
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
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=torch.cuda.is_available()
        )

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


    def train_client_with_pol(self, model, dataloader, client_id):
        """Train client and capture real PoL artifacts (commitment + checkpoints).
        Returns (model_state_dict, commitment, response_dict)
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=FL_CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # Setup PoL manager (memory mode; minimal overhead)
        pol_dir = self.output_dir / 'pol'
        pol_manager = PoLManager(client_id=str(client_id), save_dir=str(pol_dir), save_freq=1, save_to_disk=False)

        # Save pre-step checkpoint (step=0)
        step = 0
        ckpt0 = {
            'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
            'epoch': 0,
            'step': step,
        }
        pol_manager.save_checkpoint(step=step, checkpoint_data=ckpt0)

        # Build a reproducible one-step batch from the client's dataset indices
        ds = dataloader.dataset
        # Get the actual subset indices (global indices in the base dataset)
        idx_list = getattr(ds, 'indices', list(range(len(ds))))
        batch_size = min(FL_CONFIG['batch_size'], len(idx_list))
        # Use local indices (0, 1, 2, ...) to access the wrapped dataset
        local_indices_used = list(range(batch_size))
        # Get global indices for PoL recording
        global_indices_used = idx_list[:batch_size]

        xs, ys = [], []
        for local_idx in local_indices_used:
            # Access dataset using local index
            item = ds[local_idx]
            # Support both (x, y) and (x, y, idx) formats
            if len(item) == 3:
                x, y, _ = item
            else:
                x, y = item
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(self.device)
        y = torch.tensor(ys, dtype=torch.long).to(self.device)

        # Record indices for PoL (use global indices)
        try:
            pol_manager.record_data_indices(global_indices_used)
        except Exception:
            pass

        # Perform one training step on the recorded batch
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Save post-step checkpoint (step=1)
        step = 1
        ckpt1 = {
            'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
            'epoch': 0,
            'step': step,
        }
        pol_manager.save_checkpoint(step=step, checkpoint_data=ckpt1)

        # Build commitment and proofs
        commitment = pol_manager.generate_commitment()
        proof0 = pol_manager.get_merkle_proof_by_step(0)
        proof1 = pol_manager.get_merkle_proof_by_step(1)
        response = {
            'checkpoints': [
                {'data': ckpt0, 'merkle_proof': proof0},
                {'data': ckpt1, 'merkle_proof': proof1},
            ],
            'data_indices': global_indices_used,
        }

        # Continue normal local training for remaining epochs
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

        return model.state_dict(), commitment, response

    def train_client(self, model, dataloader):
        """Train client model"""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=FL_CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()

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

    def decide_behavior(self, client_id, scenario, expected_reward, reputation):
        """
        Decide client behavior based on node type and scenario

        Returns:
            behavior: 'train' or 'cheat'
        """
        node_type = self.node_types[client_id]

        if node_type == 'honest':
            return 'train'

        elif node_type == 'malicious':
            return 'cheat'

        elif node_type == 'rational':
            # Rational nodes compute utility
            if scenario == 'no_incentive':
                # No incentive, likely to cheat
                return 'cheat' if random.random() < 0.7 else 'train'

            elif scenario == 'fixed_reward':
                # Fixed reward, compute utility
                compute_cost = self.config['utility_params']['compute_cost']
                gas_cost = self.config['utility_params']['gas_cost']

                # Utility of training
                utility_train = expected_reward - compute_cost - gas_cost

                # Utility of cheating (might get caught)
                detection_prob = self.config['utility_params']['detection_probability']
                penalty = self.config['utility_params']['slash_penalty']
                utility_cheat = expected_reward * (1 - detection_prob) - penalty * detection_prob - gas_cost

                return 'train' if utility_train > utility_cheat else 'cheat'

            elif scenario == 'dynamic_reward':
                # Dynamic reward with reputation
                # Higher reputation -> higher reward, lower detection probability
                compute_cost = self.config['utility_params']['compute_cost']
                gas_cost = self.config['utility_params']['gas_cost']

                # Adjusted reward based on reputation
                adjusted_reward = expected_reward * (1 + reputation * 0.5)

                # Adjusted detection probability (inverse to reputation)
                base_detection = self.config['utility_params']['detection_probability']
                detection_prob = base_detection * (1 + (1 - reputation))

                penalty = self.config['utility_params']['slash_penalty']

                utility_train = adjusted_reward - compute_cost - gas_cost
                utility_cheat = adjusted_reward * (1 - detection_prob) - penalty * detection_prob - gas_cost

                return 'train' if utility_train > utility_cheat else 'cheat'

            elif scenario == 'sybil_attack':
                # Sybil attack scenario: use dynamic_reward as base
                # Rational nodes try to create multiple identities
                compute_cost = self.config['utility_params']['compute_cost']
                gas_cost = self.config['utility_params']['gas_cost']

                # Adjusted reward based on reputation
                adjusted_reward = expected_reward * (1 + reputation * 0.5)

                # Adjusted detection probability (inverse to reputation)
                base_detection = self.config['utility_params']['detection_probability']
                detection_prob = base_detection * (1 + (1 - reputation))

                penalty = self.config['utility_params']['slash_penalty']

                utility_train = adjusted_reward - compute_cost - gas_cost
                utility_cheat = adjusted_reward * (1 - detection_prob) - penalty * detection_prob - gas_cost

                return 'train' if utility_train > utility_cheat else 'cheat'

        return 'train'

    def run_scenario(self, scenario):
        """Run one incentive scenario"""
        logger.info(f"\n=== Running Scenario: {scenario} ===")
        # Status: scenario start
        self._write_status({'state': 'SCENARIO_START', 'scenario': scenario})


        # Prepare data
        self.prepare_data()

        # Create global model
        global_model = self.create_model().to(self.device)

        # Initialize incentive components
        if scenario in ['fixed_reward', 'dynamic_reward', 'sybil_attack']:
            staking = StakingManager(min_stake=100.0)
            rewards = RewardCalculator(base_reward_per_round=self.config['utility_params']['base_reward'])
            reputation = ReputationSystem(initial_reputation=0.5)

            # Initial staking
            for client_id in self.node_types.keys():
                staking.stake(client_id, 150.0)

        # Initialize Sybil Attack if needed
        sybil_attack = None
        sybil_identities = []
        if scenario == 'sybil_attack':
            from experiments.attacks.sybil_attacks import SybilAttack
            sybil_config = self.config.get('sybil_config', {})
            sybil_attack = SybilAttack(
                num_identities=sybil_config.get('identities_per_attacker', 5),
                shared_data_ratio=1.0,
                base_client_id="attacker"
            )
            sybil_identities = sybil_attack.create_identities()
            logger.info(f"Created Sybil attack with {len(sybil_identities)} fake identities: {sybil_identities}")

        # Metrics
        results = {
            'scenario': scenario,
            'rounds': [],
            'total_honest_utility': 0.0,
            'total_rational_utility': 0.0,
            'total_malicious_utility': 0.0,
            'participation_rate': [],
            'attack_success_rate': []
        }

        for round_num in range(self.config['num_rounds']):
            logger.info(f"Round {round_num + 1}/{self.config['num_rounds']}")

            # Select clients
            all_clients = list(self.node_types.keys())
            selected_clients = random.sample(all_clients, self.config['clients_per_round'])

            # Decide behaviors
            behaviors = {}
            for client_id in selected_clients:
                if scenario in ['fixed_reward', 'dynamic_reward', 'sybil_attack']:
                    rep = reputation.get_reputation(client_id)
                    expected_reward = self.config['utility_params']['base_reward']
                else:
                    rep = 0.5
                    expected_reward = 0

                behavior = self.decide_behavior(client_id, scenario, expected_reward, rep)
                behaviors[client_id] = behavior

            # Train clients with real PoL capture
            client_models = []
            trained_clients = []
            pol_artifacts = {}

            for i, client_id in enumerate(selected_clients):
                client_model = self.create_model().to(self.device)
                client_model.load_state_dict(global_model.state_dict())

                if behaviors[client_id] == 'train':
                    state_dict, commitment, response = self.train_client_with_pol(client_model, self.train_loaders[i], client_id)
                    trained_clients.append(client_id)
                    pol_artifacts[client_id] = {'commitment': commitment, 'response': response, 'loader_idx': i}
                    client_models.append(state_dict)
                else:
                    # cheating: no training, provide empty PoL response
                    pol_artifacts[client_id] = {'commitment': '', 'response': {'checkpoints': [], 'data_indices': []}, 'loader_idx': i}
                    client_models.append(client_model.state_dict())

            # Aggregate
            aggregated_state = OrderedDict()
            for key in client_models[0].keys():
                aggregated_state[key] = sum(model[key] for model in client_models) / len(client_models)

            global_model.load_state_dict(aggregated_state)

            # Evaluate
            test_acc = compute_accuracy(global_model, self.test_loader, self.device)

            # Verification using real PoL
            verification_results = {}
            try:
                pol_delta = POL_CONFIG.get('delta', 100.0)
            except Exception:
                pol_delta = 100.0
            distance_metric = POL_CONFIG.get('distance_metric', 'l2') if isinstance(POL_CONFIG, dict) else 'l2'
            verifier = PoLVerifier({'delta': pol_delta, 'distance_metric': distance_metric, 'device': str(self.device)})
            criterion = nn.CrossEntropyLoss()

            for client_id in selected_clients:
                art = pol_artifacts.get(client_id, {})
                resp = art.get('response', {})
                if isinstance(resp, dict) and resp.get('checkpoints'):
                    loader_idx = art.get('loader_idx', 0)
                    try:
                        is_valid = verifier.verify_with_top_q(
                            challenge={},
                            response=resp,
                            commitment=art.get('commitment', ''),
                            model=self.create_model().to(self.device),
                            dataloader=self.train_loaders[loader_idx],
                            criterion=criterion,
                            optimizer_class=torch.optim.SGD,
                            lr=FL_CONFIG['learning_rate'],
                            q=1
                        )
                        verification_results[client_id] = bool(is_valid)
                    except Exception as ve:
                        logger.debug(f"PoL verification failed for {client_id}: {ve}")
                        verification_results[client_id] = False
                else:
                    verification_results[client_id] = False

            # Update incentives and per-round utilities
            honest_util_round = 0.0
            rational_util_round = 0.0
            malicious_util_round = 0.0
            if scenario in ['fixed_reward', 'dynamic_reward']:
                # Update reputation
                for client_id in selected_clients:
                    performance = 1.0 if verification_results[client_id] else 0.0
                    reputation.update_reputation(client_id, performance)

                # Calculate rewards
                data_sizes = {client_id: 1000 for client_id in selected_clients}
                reputations_dict = {client_id: reputation.get_reputation(client_id) for client_id in selected_clients}

                reward_amounts = rewards.calculate_rewards(
                    selected_clients,
                    verification_results,
                    data_sizes,
                    reputations_dict
                )

                # Update utilities (both per-round and totals)
                for client_id in selected_clients:
                    reward = reward_amounts.get(client_id, 0)
                    cost = self.config['utility_params']['compute_cost'] if behaviors[client_id] == 'train' else 0
                    gas = self.config['utility_params']['gas_cost']
                    penalty = 0

                    # Apply slashing only when a cheating client is detected
                    if behaviors.get(client_id) == 'cheat' and not verification_results[client_id]:
                        penalty = self.config['utility_params']['slash_penalty']
                        staking.penalize(client_id, 'moderate')

                    utility = reward - cost - gas - penalty
                    node_type = self.node_types[client_id]
                    if node_type == 'honest':
                        results['total_honest_utility'] += utility
                        honest_util_round += utility
                    elif node_type == 'rational':
                        results['total_rational_utility'] += utility
                        rational_util_round += utility
                    else:
                        results['total_malicious_utility'] += utility
                        malicious_util_round += utility

            # Metrics
            participation_rate = len(trained_clients) / len(selected_clients)
            verification_pass_rate = sum(1 for c in selected_clients if verification_results.get(c, False)) / len(selected_clients)
            # Attack success = cheat AND passed verification (i.e., undetected)
            success = sum(1 for c in selected_clients if behaviors[c] == 'cheat' and verification_results.get(c, False))
            attack_success_rate = success / len(selected_clients)

            results['participation_rate'].append(participation_rate)
            results['attack_success_rate'].append(attack_success_rate)

            round_results = {
                'round': round_num + 1,
                'test_accuracy': test_acc,
                'participation_rate': participation_rate,
                'verification_pass_rate': verification_pass_rate,
                'attack_success_rate': attack_success_rate,
                'honest_utility': float(honest_util_round),
                'rational_utility': float(rational_util_round),
                'malicious_utility': float(malicious_util_round)
            }
            results['rounds'].append(round_results)

            logger.info(f"  Acc: {test_acc:.4f}, Part: {participation_rate:.2f}, VerPass: {verification_pass_rate:.2f}, Attacks: {attack_success_rate:.2f}")

            # Status: per-round progress
            self._write_status({
                'state': 'RUNNING',
                'scenario': scenario,
                'round': round_num + 1,
                'participation_rate': participation_rate,
                'attack_success_rate': attack_success_rate,
                'test_accuracy': float(test_acc),
            })

        # Compute averages
        results['avg_participation_rate'] = np.mean(results['participation_rate'])
        results['avg_attack_success_rate'] = np.mean(results['attack_success_rate'])
        results['final_accuracy'] = results['rounds'][-1]['test_accuracy']

        return results

    def run_all_experiments(self):
        """Run all incentive experiments"""
        logger.info("Starting RQ4: Economic Incentive Effectiveness")
        # Status: overall start
        self._write_status({'state': 'STARTED', 'scenarios': list(self.config.get('scenarios', []))})


        all_results = []

        for scenario in self.config['scenarios']:
            results = self.run_scenario(scenario)
            all_results.append(results)
            # Persist per-round CSV for plotting (per scenario)
            try:
                csv_path = self.output_dir / f"rq4_rounds_{scenario}.csv"
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=['round', 'test_accuracy', 'participation_rate', 'verification_pass_rate', 'attack_success_rate', 'honest_utility', 'rational_utility', 'malicious_utility'])
                    writer.writeheader()
                    for r in results['rounds']:
                        writer.writerow({
                            'round': r['round'],
                            'test_accuracy': r['test_accuracy'],
                            'participation_rate': r['participation_rate'],
                            'verification_pass_rate': r.get('verification_pass_rate', 0.0),
                            'attack_success_rate': r['attack_success_rate'],
                            'honest_utility': r.get('honest_utility', 0.0),
                            'rational_utility': r.get('rational_utility', 0.0),
                            'malicious_utility': r.get('malicious_utility', 0.0)
                        })
                logger.info(f"Per-round CSV saved to {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to write per-round CSV: {e}")


        # Save results and configuration
        output_file = self.output_dir / 'rq4_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        try:
            with open(self.output_dir / 'config.json', 'w') as cf:
                json.dump(self.config, cf, indent=2)
            logger.info(f"Run configuration saved to {self.output_dir / 'config.json'}")
        except Exception as e:
            logger.warning(f"Failed to write config.json: {e}")

        logger.info(f"\nResults saved to {output_file}")

        # Print summary
        # Status: done
        self._write_status({'state': 'DONE'})

        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results):
        """Print experiment summary"""
        logger.info("\n" + "="*70)
        logger.info("RQ4: Economic Incentive Effectiveness Summary")
        logger.info("="*70)

        logger.info(f"\n{'Scenario':<20} {'Participation':<15} {'Attack Rate':<15} {'Accuracy':<10}")
        logger.info("-"*70)

        for r in results:
            logger.info(f"{r['scenario']:<20} {r['avg_participation_rate']:<15.2f} "
                       f"{r['avg_attack_success_rate']:<15.2f} {r['final_accuracy']:<10.4f}")

        logger.info("\nUtility Analysis:")
        for r in results:
            if 'total_honest_utility' in r:
                logger.info(f"\n{r['scenario']}:")
                logger.info(f"  Honest Utility: {r['total_honest_utility']:.2f}")
                logger.info(f"  Rational Utility: {r['total_rational_utility']:.2f}")
                logger.info(f"  Malicious Utility: {r['total_malicious_utility']:.2f}")

        logger.info("="*70)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='RQ4: Economic Incentive Effectiveness')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Model to use (default: auto-select based on dataset)')
    parser.add_argument('--num_clients', type=int, default=15,
                       help='Total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=10,
                       help='Number of clients per round (default: 10)')
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of rounds (default: 50 for all datasets)')
    # Utility parameter overrides (optional)
    parser.add_argument('--base_reward', type=float, default=None,
                       help='Per-round reward pool (default: 500)')
    parser.add_argument('--compute_cost', type=float, default=None,
                       help='Per-round compute cost (default: 10)')
    parser.add_argument('--gas_cost', type=float, default=None,
                       help='Per-round gas cost (default: 2)')
    parser.add_argument('--slash_penalty', type=float, default=None,
                       help='Penalty applied when cheater is detected (default: 100)')
    parser.add_argument('--detection_probability', type=float, default=None,
                       help='Probability of detecting a cheating client (default: 0.3)')
    parser.add_argument('--scenario', type=str, default=None,
                       choices=['no_incentive', 'fixed_reward', 'dynamic_reward', 'sybil_attack'],
                       help='Run only the specified scenario (default: run all)')
    parser.add_argument('--status_path', type=str, default=None,
                       help='Write progress status JSON to this path (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory; rq4_incentive will be created below it')



    args = parser.parse_args()

    # Auto-select model based on dataset
    if args.model is None:
        if args.dataset == 'MNIST':
            args.model = 'SimpleCNN'
        else:
            args.model = 'ResNet18'

    # Update config
    config = RQ4_CONFIG.copy()
    config['dataset'] = args.dataset
    config['model'] = args.model
    config['num_clients'] = args.num_clients
    config['num_rounds'] = args.num_rounds
    config['clients_per_round'] = args.clients_per_round
    # Apply utility overrides if provided
    up = config['utility_params']
    if args.base_reward is not None:
        up['base_reward'] = args.base_reward
    if args.compute_cost is not None:
        up['compute_cost'] = args.compute_cost
    if args.gas_cost is not None:
        up['gas_cost'] = args.gas_cost
    if args.slash_penalty is not None:
        up['slash_penalty'] = args.slash_penalty
    if args.detection_probability is not None:
        up['detection_probability'] = args.detection_probability

    # If a specific scenario is requested, limit to that scenario only
    if args.scenario:
        config['scenarios'] = [args.scenario]
    if args.output_dir:
        OUTPUT_CONFIG['results_dir'] = args.output_dir

    # Optional progress status path
    if args.status_path:
        config['status_path'] = args.status_path

    logger.info(f"Running RQ4 with:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Clients: {args.num_clients}")
    logger.info(f"  Rounds: {args.num_rounds}")
    logger.info(f"  Utility: base_reward={up['base_reward']}, compute_cost={up['compute_cost']}, gas_cost={up['gas_cost']}, slash_penalty={up['slash_penalty']}, det_prob={up['detection_probability']}")

    experiment = IncentiveExperiment(config)
    results = experiment.run_all_experiments()

    logger.info("\nRQ4: Economic Incentive Effectiveness Completed!")


if __name__ == '__main__':
    main()
