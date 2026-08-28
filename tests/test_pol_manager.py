"""
PoLManager单元测试
"""

import unittest
import os
import shutil
import tempfile
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.pol.PoLManager import PoLManager
from client.pol.MerkleTree import MerkleTree


class TestPoLManager(unittest.TestCase):
    """PoLManager测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        self.client_id = "test_client_1"
        self.save_freq = 5

        # 创建PoLManager实例
        self.pol_manager = PoLManager(
            client_id=self.client_id,
            save_dir=self.test_dir,
            save_freq=self.save_freq,
            compress=True
        )

    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.pol_manager.client_id, self.client_id)
        self.assertEqual(self.pol_manager.save_freq, self.save_freq)
        self.assertTrue(self.pol_manager.compress)
        self.assertEqual(self.pol_manager.checkpoint_count, 0)

        # 检查目录是否创建
        expected_dir = os.path.join(self.test_dir, f"client_{self.client_id}")
        self.assertTrue(os.path.exists(expected_dir))

    def test_save_and_load_checkpoint(self):
        """测试checkpoint保存和加载"""
        # 创建测试checkpoint
        checkpoint_data = {
            'model_state': {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10),
            },
            'optimizer_state': {},
            'epoch': 1,
            'step': 10,
            'loss': 0.5
        }

        # 保存checkpoint
        step = 10
        ckpt_hash = self.pol_manager.save_checkpoint(step, checkpoint_data)

        # 验证哈希不为空
        self.assertIsNotNone(ckpt_hash)
        self.assertTrue(len(ckpt_hash) > 0)

        # 验证checkpoint计数
        self.assertEqual(self.pol_manager.checkpoint_count, 1)

        # 加载checkpoint
        loaded_ckpt = self.pol_manager.load_checkpoint(step)

        # 验证加载的数据
        self.assertIsNotNone(loaded_ckpt)
        self.assertEqual(loaded_ckpt['epoch'], checkpoint_data['epoch'])
        self.assertEqual(loaded_ckpt['step'], checkpoint_data['step'])
        self.assertEqual(loaded_ckpt['loss'], checkpoint_data['loss'])

        # 验证模型参数
        for key in checkpoint_data['model_state']:
            self.assertTrue(torch.allclose(
                loaded_ckpt['model_state'][key],
                checkpoint_data['model_state'][key]
            ))

    def test_memory_checkpoint_is_immutable_snapshot(self):
        """Memory mode should keep real snapshots and still persist metadata."""
        pm = PoLManager(
            client_id="memory_client",
            save_dir=self.test_dir,
            save_freq=1,
            compress=False,
            save_to_disk=False,
            memory_limit=4,
        )
        weight = torch.tensor([1.0, 2.0])
        checkpoint_data = {
            'model_state': {'w': weight},
            'optimizer_state': {},
            'epoch': 0,
            'step': 1,
            'loss': 0.1,
        }

        pm.save_checkpoint(1, checkpoint_data)
        weight.add_(10.0)
        checkpoint_data['model_state']['w'].add_(10.0)
        loaded = pm.load_checkpoint(1)

        self.assertIsNotNone(loaded)
        self.assertTrue(torch.allclose(loaded['model_state']['w'], torch.tensor([1.0, 2.0])))
        pm.save_data_indices([1, 2, 3])
        pm.save_metadata()
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "client_memory_client", "metadata.json")))

    def test_save_and_load_data_indices(self):
        """测试数据索引保存和加载"""
        # 创建测试索引
        indices = [1, 5, 3, 8, 2, 9, 4, 7, 6, 0]

        # 保存索引
        self.pol_manager.save_data_indices(indices)

        # 加载索引
        loaded_indices = self.pol_manager.load_data_indices()

        # 验证
        self.assertIsNotNone(loaded_indices)
        np.testing.assert_array_equal(loaded_indices, np.array(indices))

    def test_generate_commitment(self):
        """测试生成PoL承诺"""
        # 保存多个checkpoint
        for i in range(3):
            checkpoint_data = {
                'model_state': {
                    'layer1.weight': torch.randn(10, 5),
                },
                'optimizer_state': {},
                'epoch': 0,
                'step': i * self.save_freq,
                'loss': 0.5
            }
            self.pol_manager.save_checkpoint(i * self.save_freq, checkpoint_data)

        # 生成承诺
        commitment = self.pol_manager.generate_commitment()

        # 验证承诺不为空
        self.assertIsNotNone(commitment)
        self.assertTrue(len(commitment) > 0)

        # 验证是有效的十六进制字符串
        try:
            int(commitment, 16)
            is_hex = True
        except ValueError:
            is_hex = False
        self.assertTrue(is_hex)

    def test_metadata(self):
        """测试元数据"""
        # 保存一个checkpoint
        checkpoint_data = {
            'model_state': {'layer1.weight': torch.randn(10, 5)},
            'optimizer_state': {},
            'epoch': 0,
            'step': 0,
            'loss': 0.5
        }
        self.pol_manager.save_checkpoint(0, checkpoint_data)

        # 获取元数据
        metadata = self.pol_manager.get_metadata()

        # 验证元数据
        self.assertEqual(metadata['client_id'], self.client_id)
        self.assertEqual(metadata['save_freq'], self.save_freq)
        self.assertEqual(len(metadata['checkpoints']), 1)

        # 保存元数据到文件
        self.pol_manager.save_metadata()

        # 验证文件存在
        metadata_path = os.path.join(
            self.test_dir, f"client_{self.client_id}", "metadata.json"
        )
        self.assertTrue(os.path.exists(metadata_path))

    def test_cleanup_old_checkpoints(self):
        """测试清理旧checkpoint"""
        # 保存多个checkpoint
        num_checkpoints = 20
        for i in range(num_checkpoints):
            checkpoint_data = {
                'model_state': {'layer1.weight': torch.randn(10, 5)},
                'optimizer_state': {},
                'epoch': 0,
                'step': i,
                'loss': 0.5
            }
            self.pol_manager.save_checkpoint(i, checkpoint_data)

        # 清理，保留每10个中的一个
        self.pol_manager.cleanup_old_checkpoints(keep_every_n=10)

        # 验证剩余的checkpoint数量
        checkpoint_dir = os.path.join(
            self.test_dir, f"client_{self.client_id}", "checkpoints"
        )
        remaining_files = os.listdir(checkpoint_dir)

        # 应该保留2个checkpoint（索引0和10）
        self.assertEqual(len(remaining_files), 2)


class TestMerkleTree(unittest.TestCase):
    """MerkleTree测试类"""

    def test_single_leaf(self):
        """测试单个叶子节点"""
        leaves = ["abc123"]
        tree = MerkleTree(leaves)

        self.assertEqual(tree.get_root(), leaves[0])

        # 验证proof
        proof = tree.get_proof(0)
        is_valid = MerkleTree.verify_proof(leaves[0], proof, tree.get_root())
        self.assertTrue(is_valid)

    def test_multiple_leaves(self):
        """测试多个叶子节点"""
        import hashlib
        leaves = [
            hashlib.sha256(f"data_{i}".encode()).hexdigest()
            for i in range(5)
        ]

        tree = MerkleTree(leaves)

        # 验证所有叶子的proof
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            is_valid = MerkleTree.verify_proof(leaf, proof, tree.get_root())
            self.assertTrue(is_valid, f"Proof for leaf {i} should be valid")

    def test_invalid_proof(self):
        """测试无效proof"""
        import hashlib
        leaves = [
            hashlib.sha256(f"data_{i}".encode()).hexdigest()
            for i in range(5)
        ]

        tree = MerkleTree(leaves)

        # 使用错误的叶子
        invalid_leaf = hashlib.sha256(b"invalid_data").hexdigest()
        proof = tree.get_proof(0)
        is_valid = MerkleTree.verify_proof(invalid_leaf, proof, tree.get_root())
        self.assertFalse(is_valid)

    def test_tree_info(self):
        """测试树信息"""
        import hashlib
        leaves = [
            hashlib.sha256(f"data_{i}".encode()).hexdigest()
            for i in range(7)
        ]

        tree = MerkleTree(leaves)
        info = tree.get_tree_info()

        self.assertEqual(info['num_leaves'], 7)
        self.assertGreater(info['num_levels'], 0)
        self.assertIsNotNone(info['root'])


if __name__ == '__main__':
    unittest.main()
