#!/usr/bin/env python3
"""
CIFAR-10 ablation runner (baseline + Byzantine noise-scale variants)
- Baseline (no attack) for Vanilla_FL and Trimmed_Mean
- Byzantine attacks at noise_scale in a list, 20% malicious by default
- Short runs to quickly sanity-check trends and prepare paper paragraphs

Outputs JSON to experiments/results/cifar10_ablation/results.json
"""

import sys
import json
import logging
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num-rounds', type=int, default=10)
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-clients', type=int, default=10)
    p.add_argument('--clients-per-round', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--malicious-ratio', type=float, default=0.2)
    p.add_argument('--noise-scales', type=str, default='0.3,0.5', help='Comma-separated noise scales for Byzantine ablation')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--skip-baseline', action='store_true', help='Skip running the no-attack baseline runs')
    return p.parse_args()


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    return train_dataset, test_dataset


def dirichlet_split(dataset, num_clients, alpha=0.5, batch_size=64):
    num_classes = 10
    num_samples = len(dataset)
    labels = np.array([dataset[i][1] for i in range(num_samples)])
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split)
    client_loaders = {}
    for i in range(num_clients):
        subset = Subset(dataset, client_indices[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders[f'client_{i}'] = loader
    return client_loaders


def train_one_epoch(model, dataloader, device, lr=0.01):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        # Support both (data, target) and (data, target, idx) formats
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model.state_dict()


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            # Support both (data, target) and (data, target, idx) formats
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


def fedavg_aggregate(updates):
    aggregated = {}
    for key in updates[0].keys():
        aggregated[key] = torch.stack([u[key].float() for u in updates]).mean(0)
    return aggregated


def trimmed_mean_aggregate(updates, beta=0.1):
    aggregated = {}
    num_trim = int(len(updates) * beta)
    for key in updates[0].keys():
        stacked = torch.stack([u[key].float() for u in updates])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[num_trim:-num_trim] if num_trim > 0 else sorted_vals
        aggregated[key] = trimmed.mean(0)
    return aggregated


def add_byzantine_noise(update, noise_scale=0.1):
    noisy_update = {}
    for key, value in update.items():
        noisy_update[key] = value + torch.randn_like(value) * noise_scale
    return noisy_update


def run(method_name, aggregator_fn, attack_type=None, attack_ratio=0.0, noise_scale=0.1,
        num_clients=10, clients_per_round=5, num_rounds=10, local_epochs=2, batch_size=64,
        lr=0.01, alpha=0.5, device='cuda'):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(device)
    train_dataset, test_dataset = load_cifar10()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    client_loaders = dirichlet_split(train_dataset, num_clients, alpha=alpha, batch_size=batch_size)
    global_model = SimpleCNN().to(device)
    accuracies = []
    num_malicious = int(clients_per_round * attack_ratio)
    for round_num in range(num_rounds):
        selected_clients = np.random.choice(list(client_loaders.keys()), clients_per_round, replace=False)
        client_updates = []
        for i, client_id in enumerate(selected_clients):
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            if attack_type == 'free_riding' and i < num_malicious:
                update = global_model.state_dict()
            else:
                for _ in range(local_epochs):
                    update = train_one_epoch(client_model, client_loaders[client_id], device, lr)
                if attack_type == 'byzantine' and i < num_malicious:
                    update = add_byzantine_noise(update, noise_scale=noise_scale)
            client_updates.append(update)
        aggregated = aggregator_fn(client_updates)
        global_model.load_state_dict(aggregated)
        if (round_num + 1) % max(1, num_rounds // 5) == 0 or round_num == num_rounds - 1:
            acc = evaluate(global_model, test_loader, device)
            accuracies.append(acc)
            logger.info(f"[{method_name} | {attack_type or 'no_attack'} | noise={noise_scale}] Round {round_num+1}/{num_rounds}: Acc={acc:.4f}")
    final_acc = accuracies[-1]
    return {
        'method': method_name,
        'attack_type': attack_type or 'no_attack',
        'attack_ratio': attack_ratio,
        'noise_scale': noise_scale if attack_type == 'byzantine' else 0.0,
        'accuracies': accuracies,
        'final_accuracy': final_acc,
        'config': {
            'num_rounds': num_rounds, 'local_epochs': local_epochs, 'batch_size': batch_size,
            'num_clients': num_clients, 'clients_per_round': clients_per_round,
            'alpha': alpha, 'lr': lr
        }
    }


def main():
    args = parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    logger.info(f"Device: {device}")

    # Baseline (no attack)
    results = []
    if not args.skip_baseline:
        results.append(run('Vanilla_FL', fedavg_aggregate, attack_type=None, attack_ratio=0.0,
                           num_rounds=args.num_rounds, local_epochs=args.local_epochs,
                           batch_size=args.batch_size, num_clients=args.num_clients,
                           clients_per_round=args.clients_per_round, lr=args.lr,
                           alpha=args.alpha, device=device))
        results.append(run('Trimmed_Mean', trimmed_mean_aggregate, attack_type=None, attack_ratio=0.0,
                           num_rounds=args.num_rounds, local_epochs=args.local_epochs,
                           batch_size=args.batch_size, num_clients=args.num_clients,
                           clients_per_round=args.clients_per_round, lr=args.lr,
                           alpha=args.alpha, device=device))

    # Byzantine ablations at different noise scales
    scales = [float(s.strip()) for s in args.noise_scales.split(',') if s.strip()]
    for s in scales:
        results.append(run('Vanilla_FL', fedavg_aggregate, attack_type='byzantine', attack_ratio=args.malicious_ratio,
                           noise_scale=s, num_rounds=args.num_rounds, local_epochs=args.local_epochs,
                           batch_size=args.batch_size, num_clients=args.num_clients,
                           clients_per_round=args.clients_per_round, lr=args.lr,
                           alpha=args.alpha, device=device))
        results.append(run('Trimmed_Mean', trimmed_mean_aggregate, attack_type='byzantine', attack_ratio=args.malicious_ratio,
                           noise_scale=s, num_rounds=args.num_rounds, local_epochs=args.local_epochs,
                           batch_size=args.batch_size, num_clients=args.num_clients,
                           clients_per_round=args.clients_per_round, lr=args.lr,
                           alpha=args.alpha, device=device))

    outdir = Path('experiments/results/cifar10_ablation')
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / 'results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved ablation results to {outfile}")

    # brief stdout summary
    for r in results:
        logger.info(f"Summary | {r['method']} | {r['attack_type']} | noise={r['noise_scale']}: final_acc={r['final_accuracy']:.4f}")


if __name__ == '__main__':
    main()

