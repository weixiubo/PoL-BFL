"""
端到端集成测试
在MNIST数据集上测试完整的PoL-FL流程
"""

import unittest
import os
import shutil
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock chainProxy以避免Brownie冲突
from tests.mock_chain_proxy import mock_chain_proxy
import chainfl.interact as interact_module
interact_module.chain_proxy = mock_chain_proxy

from client.pol.PoLManager import PoLManager
from client.trainer.PoLTrainer import PoLTrainer
from server.pol.PoLVerifier import PoLVerifier
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from config.pol_config import POL_CONFIG, EXPERIMENT_CONFIG


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


class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 创建临时目录
        cls.test_dir = tempfile.mkdtemp()
        
        # 创建模拟MNIST数据
        cls.num_samples = 100
        cls.batch_size = 10
        
        # 生成随机数据（模拟MNIST）
        cls.train_data = torch.randn(cls.num_samples, 1, 28, 28)
        cls.train_labels = torch.randint(0, 10, (cls.num_samples,))
        
        cls.test_data = torch.randn(20, 1, 28, 28)
        cls.test_labels = torch.randint(0, 10, (20,))
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_pol_trainer_basic(self):
        """测试PoLTrainer基本功能"""
        # 创建数据加载器
        dataset = TensorDataset(self.train_data[:50], self.train_labels[:50])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # 创建模型
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        
        # 创建PoLTrainer
        args = {
            'enable_pol': True,
            'pol_save_freq': 5,
            'pol_save_dir': os.path.join(self.test_dir, 'pol_data'),
            'pol_compress': True,
            'client_id': 'test_client_1',
            'device': 'cpu',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'optimizer': 'SGD'
        }
        
        trainer = PoLTrainer(model, dataloader, criterion, args)
        
        # 训练1个epoch
        results = trainer.train(total_epoch=1)
        
        # 验证训练结果
        self.assertEqual(len(results), 1)
        self.assertIn('loss', results[0])
        
        # 生成PoL承诺
        pol_commitment = trainer.finalize_pol(epoch=0, dataset=dataset)
        
        # 验证PoL承诺
        self.assertIsNotNone(pol_commitment)
        self.assertIn('commitment', pol_commitment)
        self.assertIn('num_checkpoints', pol_commitment)
        self.assertGreater(pol_commitment['num_checkpoints'], 0)
        
        print(f"\n✓ PoLTrainer test passed")
        print(f"  Checkpoints: {pol_commitment['num_checkpoints']}")
        print(f"  Commitment: {pol_commitment['commitment'][:16]}...")
    
    def test_pol_trainer_workflow(self):
        """测试PoLTrainer完整工作流（不使用PoLClient避免chainProxy冲突）"""
        # 创建数据加载器
        dataset = TensorDataset(self.train_data[:50], self.train_labels[:50])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # 创建模型和trainer
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()

        args = {
            'enable_pol': True,
            'pol_save_freq': 5,
            'pol_save_dir': os.path.join(self.test_dir, 'pol_data_trainer'),
            'pol_compress': True,
            'client_id': 'test_trainer_2',
            'device': 'cpu',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'optimizer': 'SGD'
        }

        trainer = PoLTrainer(model, dataloader, criterion, args)

        # 训练
        results = trainer.train(total_epoch=1)

        # 获取PoL承诺
        commitment = trainer.finalize_pol(epoch=0, dataset=dataset)

        # 验证
        self.assertIsNotNone(commitment)
        self.assertIn('commitment', commitment)

        # 测试挑战响应
        challenge = {
            'checkpoint_indices': [0, 1]
        }
        response = trainer.respond_to_challenge(challenge)

        # 验证响应
        self.assertIsNotNone(response)
        self.assertIn('checkpoints', response)

        print(f"\n✓ PoLTrainer workflow test passed")
        print(f"  Client ID: {args['client_id']}")
        print(f"  Commitment: {commitment['commitment'][:16]}...")
    
    def test_aggregator_with_pol(self):
        """测试带PoL验证的聚合器"""
        # 创建全局模型
        global_model = SimpleCNN()
        
        # 创建聚合器
        args = {
            'enable_pol': True,
            'verification_rate': 0.5,
            'pol_delta': 1.0,
            'pol_distance_metric': 'l2',
            'device': 'cpu',
            'use_top_q': False
        }
        
        aggregator = PoLVerifyAggregator(model=global_model, args=args)
        
        # 创建模拟客户端模型
        client_models = []
        for i in range(3):
            model = SimpleCNN()
            # 添加一些随机扰动
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            client_models.append(model.state_dict())
        
        # 执行聚合
        aggregated_model = aggregator.aggregate(client_models)
        
        # 验证聚合结果
        self.assertIsNotNone(aggregated_model)
        self.assertEqual(len(aggregated_model), len(global_model.state_dict()))
        
        # 获取验证结果
        verification_results = aggregator.get_verification_results()
        
        print(f"\n✓ Aggregator test passed")
        print(f"  Aggregated {len(client_models)} client models")
        print(f"  Verification results: {len(verification_results)} clients verified")
    
    def test_full_fl_round(self):
        """测试完整的FL训练轮次（不使用PoLClient避免chainProxy冲突）"""
        print(f"\n{'='*60}")
        print("Testing Full FL Round with PoL")
        print(f"{'='*60}")

        # 1. 创建全局模型
        global_model = SimpleCNN()

        # 2. 创建多个客户端trainer
        num_clients = 3
        trainers = []
        datasets = []

        for i in range(num_clients):
            # 为每个客户端创建数据
            start_idx = i * 30
            end_idx = start_idx + 30
            dataset = TensorDataset(
                self.train_data[start_idx:end_idx],
                self.train_labels[start_idx:end_idx]
            )
            dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

            # 创建客户端模型（复制全局模型）
            client_model = SimpleCNN()
            client_model.load_state_dict(global_model.state_dict())

            # 创建trainer
            args = {
                'enable_pol': True,
                'pol_save_freq': 3,
                'pol_save_dir': os.path.join(self.test_dir, f'pol_data_fl_trainer_{i}'),
                'pol_compress': True,
                'client_id': f'fl_trainer_{i}',
                'device': 'cpu',
                'lr': 0.01,
                'weight_decay': 1e-4,
                'optimizer': 'SGD'
            }

            trainer = PoLTrainer(client_model, dataloader, nn.CrossEntropyLoss(), args)
            trainers.append(trainer)
            datasets.append(dataset)

        # 3. 客户端本地训练
        print(f"\nStep 1: Local training on {num_clients} clients")
        client_models = []
        for i, trainer in enumerate(trainers):
            print(f"  Training client {i}...")
            trainer.train(total_epoch=1)
            client_models.append(trainer.model.state_dict())

            # 获取PoL承诺
            commitment = trainer.finalize_pol(epoch=0, dataset=datasets[i])
            print(f"    PoL commitment: {commitment['commitment'][:16]}...")

        # 4. 服务器聚合
        print(f"\nStep 2: Server aggregation with PoL verification")
        aggregator_args = {
            'enable_pol': True,
            'verification_rate': 1.0,  # 验证所有客户端
            'pol_delta': 10.0,
            'pol_distance_metric': 'l2',
            'device': 'cpu',
            'use_top_q': False
        }

        aggregator = PoLVerifyAggregator(model=global_model, args=aggregator_args)
        aggregated_model = aggregator.aggregate(client_models)

        # 5. 更新全局模型
        global_model.load_state_dict(aggregated_model)

        # 6. 验证
        print(f"\nStep 3: Verification")
        verification_results = aggregator.get_verification_results()
        print(f"  Verified clients: {len(verification_results)}")

        print(f"\n{'='*60}")
        print("✓ Full FL round completed successfully!")
        print(f"{'='*60}")

        # 断言
        self.assertEqual(len(client_models), num_clients)
        self.assertIsNotNone(aggregated_model)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

