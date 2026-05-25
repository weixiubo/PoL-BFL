"""
MNIST上的PoL-FL完整示例
演示如何使用PoL机制进行联邦学习
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.pol.PoLManager import PoLManager
from client.trainer.PoLTrainer import PoLTrainer
from client.PoLClient import PoLClient
from server.pol.PoLVerifier import PoLVerifier
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from config.pol_config import get_pol_config, get_experiment_config, merge_configs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """简单的CNN模型（用于MNIST）"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_mock_mnist_data(num_samples=1000):
    """创建模拟MNIST数据"""
    logger.info(f"Creating mock MNIST data with {num_samples} samples")
    
    # 生成随机数据（模拟MNIST）
    data = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    
    return data, labels


def split_data_for_clients(data, labels, num_clients):
    """将数据分割给多个客户端"""
    logger.info(f"Splitting data for {num_clients} clients")
    
    dataset = TensorDataset(data, labels)
    
    # 计算每个客户端的数据量
    samples_per_client = len(dataset) // num_clients
    split_sizes = [samples_per_client] * num_clients
    
    # 处理余数
    remainder = len(dataset) - sum(split_sizes)
    split_sizes[-1] += remainder
    
    # 分割数据集
    client_datasets = random_split(dataset, split_sizes)
    
    return client_datasets


def run_pol_fl_experiment(config):
    """运行PoL-FL实验"""
    logger.info("=" * 80)
    logger.info("Starting PoL-FL Experiment on MNIST")
    logger.info("=" * 80)
    
    # 1. 准备数据
    logger.info("\n[Step 1] Preparing data...")
    train_data, train_labels = create_mock_mnist_data(num_samples=config['num_samples'])
    test_data, test_labels = create_mock_mnist_data(num_samples=200)
    
    # 分割数据给客户端
    client_datasets = split_data_for_clients(
        train_data, train_labels, config['num_clients']
    )
    
    # 创建测试数据加载器
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 2. 初始化全局模型
    logger.info("\n[Step 2] Initializing global model...")
    global_model = SimpleCNN()
    logger.info(f"  Model: {global_model.__class__.__name__}")
    logger.info(f"  Parameters: {sum(p.numel() for p in global_model.parameters())}")
    
    # 3. 创建客户端
    logger.info(f"\n[Step 3] Creating {config['num_clients']} clients...")
    clients = []
    
    for i in range(config['num_clients']):
        # 创建数据加载器
        dataloader = DataLoader(
            client_datasets[i],
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        # 创建客户端模型（复制全局模型）
        client_model = SimpleCNN()
        client_model.load_state_dict(global_model.state_dict())
        
        # 创建trainer配置
        trainer_args = {
            'enable_pol': config['enable_pol'],
            'pol_save_freq': config['pol_save_freq'],
            'pol_save_dir': config['pol_save_dir'],
            'pol_compress': config['pol_compress'],
            'client_id': f'client_{i}',
            'device': config['device'],
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'optimizer': config['optimizer']
        }
        
        # 创建trainer
        trainer = PoLTrainer(
            model=client_model,
            dataloader=dataloader,
            criterion=nn.CrossEntropyLoss(),
            args=trainer_args
        )
        
        # 创建客户端
        client = PoLClient(
            client_id=f'client_{i}',
            dataloader=dataloader,
            model=client_model,
            trainer=trainer,
            args=trainer_args
        )
        
        clients.append(client)
        logger.info(f"  Created client_{i} with {len(client_datasets[i])} samples")
    
    # 4. 创建聚合器
    logger.info("\n[Step 4] Creating aggregator...")
    aggregator_args = {
        'enable_pol': config['enable_pol'],
        'verification_rate': config['verification_rate'],
        'pol_delta': config['pol_delta'],
        'pol_distance_metric': config['pol_distance_metric'],
        'device': config['device'],
        'use_top_q': config['use_top_q'],
        'top_q': config['top_q']
    }
    
    aggregator = PoLVerifyAggregator(model=global_model, args=aggregator_args)
    logger.info(f"  Verification rate: {config['verification_rate']}")
    logger.info(f"  Distance metric: {config['pol_distance_metric']}")
    logger.info(f"  Delta threshold: {config['pol_delta']}")
    
    # 5. 联邦学习训练循环
    logger.info(f"\n[Step 5] Starting FL training for {config['num_rounds']} rounds...")
    
    for round_idx in range(config['num_rounds']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx + 1}/{config['num_rounds']}")
        logger.info(f"{'='*60}")
        
        # 5.1 客户端本地训练
        logger.info("\n  [5.1] Local training...")
        client_models = []
        pol_commitments = []
        
        for i, client in enumerate(clients):
            logger.info(f"    Training client_{i}...")
            
            # 下载全局模型
            client.load_state_dict(global_model.state_dict())
            
            # 本地训练
            results = client.train(
                total_epoch=config['local_epochs'],
                dataset=client.dataloader.dataset
            )
            
            # 获取训练后的模型
            client_models.append(client.get_model_state_dict())
            
            # 获取PoL承诺
            if config['enable_pol']:
                commitment = client.get_pol_commitment()
                pol_commitments.append(commitment)
                logger.info(f"      PoL commitment: {commitment['commitment'][:16]}...")
                logger.info(f"      Checkpoints: {commitment['num_checkpoints']}")
        
        # 5.2 服务器聚合
        logger.info("\n  [5.2] Server aggregation...")
        aggregated_model = aggregator.aggregate(client_models)
        
        # 更新全局模型
        global_model.load_state_dict(aggregated_model)
        
        # 5.3 验证结果
        if config['enable_pol']:
            logger.info("\n  [5.3] PoL verification results...")
            verification_results = aggregator.get_verification_results()
            logger.info(f"    Verified clients: {len(verification_results)}")
            for client_id, is_valid in verification_results.items():
                status = "✓ PASS" if is_valid else "✗ FAIL"
                logger.info(f"      {client_id}: {status}")
        
        # 5.4 测试全局模型
        logger.info("\n  [5.4] Testing global model...")
        test_results = aggregator.test(test_loader, config['device'], config)
        logger.info(f"    Test loss: {test_results['loss']:.4f}")
        logger.info(f"    Test accuracy: {test_results['accuracy']:.2f}%")
    
    # 6. 实验总结
    logger.info("\n" + "=" * 80)
    logger.info("Experiment Completed!")
    logger.info("=" * 80)
    logger.info(f"Total rounds: {config['num_rounds']}")
    logger.info(f"Total clients: {config['num_clients']}")
    logger.info(f"PoL enabled: {config['enable_pol']}")
    logger.info(f"Final test accuracy: {test_results['accuracy']:.2f}%")
    
    return {
        'global_model': global_model,
        'test_results': test_results,
        'clients': clients,
        'aggregator': aggregator
    }


def main():
    """主函数"""
    # 合并配置
    pol_config = get_pol_config()
    exp_config = get_experiment_config()
    
    # 自定义配置
    custom_config = {
        'num_samples': 600,  # 总样本数
        'num_clients': 5,    # 客户端数量
        'num_rounds': 3,     # 训练轮数
        'local_epochs': 2,   # 本地训练epoch数
        'batch_size': 32,
        'learning_rate': 0.01,
        'pol_save_freq': 5,  # 每5个batch保存一次checkpoint
        'verification_rate': 0.6,  # 验证60%的客户端
        'pol_delta': 10.0,   # 距离阈值
        'enable_pol': True,  # 启用PoL
        'pol_compress': True,  # 压缩checkpoint
        'pol_save_dir': 'pol_data',  # PoL数据目录
        'pol_distance_metric': 'l2',  # 距离度量
        'use_top_q': False,  # 不使用Top-Q
        'top_q': 5,  # Top-Q值
        'optimizer': 'SGD',  # 优化器
        'weight_decay': 1e-4,  # 权重衰减
    }
    
    config = merge_configs(pol_config, exp_config, custom_config)
    
    # 运行实验
    results = run_pol_fl_experiment(config)
    
    logger.info("\n✓ All done!")


if __name__ == "__main__":
    main()

