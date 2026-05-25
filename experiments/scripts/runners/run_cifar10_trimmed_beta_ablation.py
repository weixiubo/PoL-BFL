#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_cifar10_paper import (
    SimpleCNN, load_cifar10, dirichlet_split, train_one_epoch, evaluate,
)
from experiment_config import OUTPUT_CONFIG

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 10,
    'local_epochs': 2,
    'batch_size': 64,
    'learning_rate': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'alpha': 0.5,
    'seed': 123,
}

BETAS = [0.05, 0.1, 0.2]


def trimmed_mean_aggregate_beta(updates, beta: float):
    aggregated = {}
    n = len(updates)
    num_trim = int(n * beta)
    for key in updates[0].keys():
        stacked = torch.stack([u[key].float() for u in updates])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[num_trim:-num_trim] if num_trim > 0 else sorted_vals
        aggregated[key] = trimmed.mean(0)
    return aggregated


def run(beta: float):
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])

    train_dataset, test_dataset = load_cifar10()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    client_loaders = dirichlet_split(train_dataset, CONFIG['num_clients'], CONFIG['alpha'])

    global_model = SimpleCNN().to(device)

    # short run with Byzantine attack to show robustness trend
    attack_ratio = 0.2
    num_malicious = int(CONFIG['clients_per_round'] * attack_ratio)

    accuracies = []
    for round_num in range(CONFIG['num_rounds']):
        selected = np.random.choice(list(client_loaders.keys()), CONFIG['clients_per_round'], replace=False)
        updates = []
        for i, cid in enumerate(selected):
            client = SimpleCNN().to(device)
            client.load_state_dict(global_model.state_dict())
            for _ in range(CONFIG['local_epochs']):
                _ = train_one_epoch(client, client_loaders[cid], device, CONFIG['learning_rate'])
            upd = client.state_dict()
            # Byzantine: add noise to first K malicious
            if i < num_malicious:
                noisy = {}
                for k, v in upd.items():
                    noisy[k] = v + torch.randn_like(v) * 0.1
                upd = noisy
            updates.append(upd)
        aggregated = trimmed_mean_aggregate_beta(updates, beta)
        global_model.load_state_dict(aggregated)
        if (round_num + 1) % 5 == 0 or round_num == CONFIG['num_rounds'] - 1:
            acc = evaluate(global_model, test_loader, device)
            accuracies.append(acc)
            logger.info(f"beta={beta} round {round_num+1}: acc={acc:.4f}")
    return accuracies


if __name__ == '__main__':
    out_dir = Path(OUTPUT_CONFIG['results_dir']) / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for b in BETAS:
        accs = run(b)
        results.append({'beta': b, 'accuracies': accs})
    with open(out_dir / 'beta_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('saved', out_dir / 'beta_results.json')

