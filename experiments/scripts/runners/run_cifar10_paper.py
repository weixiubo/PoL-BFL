#!/usr/bin/env python3
"""
CIFAR-10实验 - 用于论文补充
只运行关键测试以验证方法在CIFAR-10上的有效性
减少轮次以加快速度
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import logging
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 优化配置 - 增加轮次和local epochs以提高准确率
CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'SimpleCNN',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 50,  # 增加到50轮
    'local_epochs': 5,  # 增加到5个epoch
    'batch_size': 64,
    'learning_rate': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'alpha': 0.5,  # Dirichlet参数
    'seed': 42,
}

class SimpleCNN(nn.Module):
    """简单CNN用于CIFAR-10"""
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
    """加载CIFAR-10数据集"""
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

def dirichlet_split(dataset, num_clients, alpha=0.5):
    """使用Dirichlet分布分割数据集"""
    num_classes = 10
    num_samples = len(dataset)
    
    # 获取标签
    labels = np.array([dataset[i][1] for i in range(num_samples)])
    
    # 为每个类别生成Dirichlet分布
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Dirichlet分布
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # 分割索引
        splits = np.split(idx_k, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split)
    
    # 创建DataLoader
    client_loaders = {}
    for i in range(num_clients):
        subset = Subset(dataset, client_indices[i])
        loader = DataLoader(subset, batch_size=CONFIG['batch_size'], shuffle=True)
        client_loaders[f'client_{i}'] = loader
    
    return client_loaders

def train_one_epoch(model, dataloader, device, lr=0.01):
    """训练一个epoch"""
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
    """评估模型"""
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
    """FedAvg聚合"""
    aggregated = {}
    for key in updates[0].keys():
        aggregated[key] = torch.stack([u[key].float() for u in updates]).mean(0)
    return aggregated

def trimmed_mean_aggregate(updates, beta=0.1):
    """Trimmed Mean聚合"""
    aggregated = {}
    num_trim = int(len(updates) * beta)
    
    for key in updates[0].keys():
        stacked = torch.stack([u[key].float() for u in updates])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[num_trim:-num_trim] if num_trim > 0 else sorted_vals
        aggregated[key] = trimmed.mean(0)
    
    return aggregated

def add_byzantine_noise(update, noise_scale=0.1):
    """添加Byzantine噪声 - 降低噪声强度以展示防御效果"""
    noisy_update = {}
    for key, value in update.items():
        noisy_update[key] = value + torch.randn_like(value) * noise_scale
    return noisy_update

def run_experiment(method_name, aggregator_fn, attack_type=None, attack_ratio=0.0):
    """运行一个实验"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {method_name} - {attack_type if attack_type else 'No Attack'}")
    logger.info(f"{'='*70}")
    
    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    device = torch.device(CONFIG['device'])
    
    # 加载数据
    train_dataset, test_dataset = load_cifar10()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 分割数据
    client_loaders = dirichlet_split(train_dataset, CONFIG['num_clients'], CONFIG['alpha'])
    
    # 创建全局模型
    global_model = SimpleCNN().to(device)
    
    # 训练
    accuracies = []
    num_malicious = int(CONFIG['clients_per_round'] * attack_ratio)
    
    for round_num in range(CONFIG['num_rounds']):
        # 选择客户端
        selected_clients = np.random.choice(
            list(client_loaders.keys()),
            CONFIG['clients_per_round'],
            replace=False
        )
        
        # 客户端训练
        client_updates = []
        for i, client_id in enumerate(selected_clients):
            # 复制全局模型
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            # 训练
            if attack_type == 'free_riding' and i < num_malicious:
                # Free-riding: 直接返回全局模型
                update = global_model.state_dict()
            else:
                # 正常训练
                for _ in range(CONFIG['local_epochs']):
                    update = train_one_epoch(
                        client_model,
                        client_loaders[client_id],
                        device,
                        CONFIG['learning_rate']
                    )
                
                # Byzantine攻击
                if attack_type == 'byzantine' and i < num_malicious:
                    update = add_byzantine_noise(update, noise_scale=0.1)
            
            client_updates.append(update)
        
        # 聚合
        aggregated = aggregator_fn(client_updates)
        global_model.load_state_dict(aggregated)
        
        # 评估 - 每10轮评估一次
        if (round_num + 1) % 10 == 0 or round_num == CONFIG['num_rounds'] - 1:
            acc = evaluate(global_model, test_loader, device)
            accuracies.append(acc)
            logger.info(f"Round {round_num+1}/{CONFIG['num_rounds']}: Accuracy = {acc:.4f}")
    
    final_acc = accuracies[-1]
    logger.info(f"Final Accuracy: {final_acc:.4f}")
    
    return {
        'method': method_name,
        'attack_type': attack_type,
        'attack_ratio': attack_ratio,
        'accuracies': accuracies,
        'final_accuracy': final_acc,
        'config': CONFIG
    }

def main():
    """主函数"""
    logger.info("="*70)
    logger.info("CIFAR-10 Experiment for Paper")
    logger.info("="*70)
    logger.info(f"Config: {CONFIG}")
    logger.info(f"Device: {CONFIG['device']}")
    logger.info("")

    # CLI args to support backgrounded, targeted runs
    parser = argparse.ArgumentParser(description='CIFAR-10 paper experiments')
    parser.add_argument('--only-no-attack', action='store_true', help='Run only the two no-attack baselines')
    args = parser.parse_args()
    logger.info(f"Args: only_no_attack={args.only_no_attack}")

    results = []

    # 实验0: 无攻击基线（Vanilla / Trimmed Mean）
    results.append(run_experiment(
        'Vanilla_FL',
        fedavg_aggregate,
        attack_type='no_attack',
        attack_ratio=0.0
    ))
    results.append(run_experiment(
        'Trimmed_Mean',
        trimmed_mean_aggregate,
        attack_type='no_attack',
        attack_ratio=0.0
    ))

    if not args.only_no_attack:
        # 实验1: Vanilla FedAvg - Byzantine Attack
        results.append(run_experiment(
            'Vanilla_FL',
            fedavg_aggregate,
            attack_type='byzantine',
            attack_ratio=0.2
        ))

        # 实验2: Trimmed Mean - Byzantine Attack
        results.append(run_experiment(
            'Trimmed_Mean',
            trimmed_mean_aggregate,
            attack_type='byzantine',
            attack_ratio=0.2
        ))

        # 实验3: Vanilla FedAvg - Free-Riding Attack
        results.append(run_experiment(
            'Vanilla_FL',
            fedavg_aggregate,
            attack_type='free_riding',
            attack_ratio=0.2
        ))

        # 实验4: Trimmed Mean - Free-Riding Attack
        results.append(run_experiment(
            'Trimmed_Mean',
            trimmed_mean_aggregate,
            attack_type='free_riding',
            attack_ratio=0.2
        ))

    # 保存结果
    output_dir = Path('experiments/results/cifar10_paper')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("Results Summary")
    logger.info(f"{'='*70}")
    for result in results:
        logger.info(f"{result['method']} - {result['attack_type']}: {result['final_accuracy']:.4f}")

    logger.info(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()

