"""
区块链集成测试
测试PoL与区块链的完整集成
"""

import unittest
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from brownie import project, network, accounts
from chainfl.interact import chainProxy


class TestBlockchainIntegration(unittest.TestCase):
    """区块链集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 连接到开发网络
        if not network.is_connected():
            network.connect('development')
        
        # 创建chainProxy实例
        cls.chain_proxy = chainProxy()
        
        # 获取测试账户
        cls.server_account = accounts[0]
        cls.client_accounts = accounts[1:6]  # 5个客户端
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        # 断开网络连接
        if network.is_connected():
            network.disconnect()
    
    def test_pol_contract_deployment(self):
        """测试PoL合约部署"""
        # 验证合约已部署
        self.assertIsNotNone(self.chain_proxy.pol_contract)
        
        # 验证合约owner
        owner = self.chain_proxy.pol_contract.owner()
        self.assertEqual(owner, self.server_account.address)
        
        print(f"\n✓ PoL contract deployed at: {self.chain_proxy.pol_contract.address}")
        print(f"  Owner: {owner}")
    
    def test_client_registration(self):
        """测试客户端注册"""
        # 检查是否已注册
        is_registered = self.chain_proxy.pol_contract.isClientRegistered(
            self.client_accounts[0].address
        )

        if not is_registered:
            # 注册第一个客户端
            success = self.chain_proxy.pol_register_client('1')
            self.assertTrue(success)

        # 验证注册状态
        is_registered = self.chain_proxy.pol_contract.isClientRegistered(
            self.client_accounts[0].address
        )
        self.assertTrue(is_registered)

        print(f"\n✓ Client registration verified")
        print(f"  Address: {self.client_accounts[0].address}")
    
    def test_submit_pol_proof(self):
        """测试提交PoL证明"""
        # 先注册客户端
        self.chain_proxy.pol_register_client('2')
        
        # 准备测试数据
        commitment = "a" * 64  # 64个十六进制字符 = 32字节
        data_hash = "b" * 64
        num_checkpoints = 5
        total_steps = 50
        
        # 提交证明
        tx_hash = self.chain_proxy.submit_pol_proof(
            client_id='2',
            commitment=commitment,
            data_hash=data_hash,
            num_checkpoints=num_checkpoints,
            total_steps=total_steps
        )
        
        # 验证交易成功
        self.assertNotEqual(tx_hash, "")
        
        # 获取证明
        proof = self.chain_proxy.get_pol_proof('2')
        
        # 验证证明内容
        self.assertEqual(proof['num_checkpoints'], num_checkpoints)
        self.assertEqual(proof['total_steps'], total_steps)
        self.assertFalse(proof['verified'])  # 尚未验证
        
        print(f"\n✓ PoL proof submitted successfully")
        print(f"  Commitment: {proof['commitment'][:16]}...")
        print(f"  Checkpoints: {proof['num_checkpoints']}")
        print(f"  Total steps: {proof['total_steps']}")
    
    def test_record_verification(self):
        """测试记录验证结果"""
        # 先注册并提交证明
        self.chain_proxy.pol_register_client('3')
        self.chain_proxy.submit_pol_proof(
            client_id='3',
            commitment="c" * 64,
            data_hash="d" * 64,
            num_checkpoints=3,
            total_steps=30
        )
        
        # 记录验证结果
        success = self.chain_proxy.record_pol_verification(
            client_id='3',
            is_valid=True
        )
        self.assertTrue(success)
        
        # 获取证明并验证
        proof = self.chain_proxy.get_pol_proof('3')
        self.assertTrue(proof['verified'])
        self.assertTrue(proof['is_valid'])
        
        print(f"\n✓ Verification recorded successfully")
        print(f"  Verified: {proof['verified']}")
        print(f"  Is valid: {proof['is_valid']}")
    
    def test_batch_record_verification(self):
        """测试批量记录验证结果"""
        # 注册并提交多个客户端的证明
        client_ids = ['4', '5']
        for cid in client_ids:
            self.chain_proxy.pol_register_client(cid)
            self.chain_proxy.submit_pol_proof(
                client_id=cid,
                commitment="e" * 64,
                data_hash="f" * 64,
                num_checkpoints=2,
                total_steps=20
            )
        
        # 批量记录验证结果
        results = [True, False]  # 第一个通过，第二个失败
        success = self.chain_proxy.batch_record_pol_verification(
            client_ids=client_ids,
            results=results
        )
        self.assertTrue(success)
        
        # 验证结果
        for i, cid in enumerate(client_ids):
            proof = self.chain_proxy.get_pol_proof(cid)
            self.assertTrue(proof['verified'])
            self.assertEqual(proof['is_valid'], results[i])
        
        print(f"\n✓ Batch verification recorded successfully")
        print(f"  Clients: {len(client_ids)}")
        print(f"  Results: {results}")
    
    def test_get_pol_stats(self):
        """测试获取统计信息"""
        stats = self.chain_proxy.get_pol_stats()
        
        # 验证统计信息
        self.assertIn('total_proofs', stats)
        self.assertIn('total_verifications', stats)
        self.assertIn('total_clients', stats)
        
        # 统计应该大于0（因为前面的测试已经提交了证明）
        self.assertGreater(stats['total_clients'], 0)
        
        print(f"\n✓ PoL stats retrieved successfully")
        print(f"  Total clients: {stats['total_clients']}")
        print(f"  Total proofs: {stats['total_proofs']}")
        print(f"  Total verifications: {stats['total_verifications']}")
    
    def test_full_workflow(self):
        """测试完整的工作流程"""
        print(f"\n{'='*60}")
        print("Testing Full Blockchain Workflow")
        print(f"{'='*60}")
        
        client_id = '1'  # 使用已注册的客户端
        
        # 1. 提交证明
        print(f"\nStep 1: Submit PoL proof")
        commitment = "1234567890abcdef" * 4  # 64字符
        data_hash = "fedcba0987654321" * 4
        
        tx_hash = self.chain_proxy.submit_pol_proof(
            client_id=client_id,
            commitment=commitment,
            data_hash=data_hash,
            num_checkpoints=10,
            total_steps=100
        )
        self.assertNotEqual(tx_hash, "")
        print(f"  ✓ Proof submitted: {tx_hash[:16]}...")
        
        # 2. 获取证明
        print(f"\nStep 2: Retrieve proof")
        proof = self.chain_proxy.get_pol_proof(client_id)
        self.assertEqual(proof['num_checkpoints'], 10)
        print(f"  ✓ Proof retrieved")
        print(f"    Checkpoints: {proof['num_checkpoints']}")
        print(f"    Verified: {proof['verified']}")
        
        # 3. 记录验证
        print(f"\nStep 3: Record verification")
        success = self.chain_proxy.record_pol_verification(
            client_id=client_id,
            is_valid=True
        )
        self.assertTrue(success)
        print(f"  ✓ Verification recorded")
        
        # 4. 验证更新
        print(f"\nStep 4: Verify update")
        proof = self.chain_proxy.get_pol_proof(client_id)
        self.assertTrue(proof['verified'])
        self.assertTrue(proof['is_valid'])
        print(f"  ✓ Proof verified: {proof['is_valid']}")
        
        # 5. 获取统计
        print(f"\nStep 5: Get statistics")
        stats = self.chain_proxy.get_pol_stats()
        print(f"  ✓ Stats retrieved")
        print(f"    Total clients: {stats['total_clients']}")
        print(f"    Total proofs: {stats['total_proofs']}")
        
        print(f"\n{'='*60}")
        print("✓ Full workflow completed successfully!")
        print(f"{'='*60}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

