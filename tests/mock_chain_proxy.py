"""
Mock chainProxy for testing
避免在单元测试中初始化真实的区块链连接
"""


class MockChainProxy:
    """Mock版本的chainProxy，用于单元测试"""
    
    def __init__(self):
        """初始化mock proxy"""
        self.pol_contract = None
        self.watermark_proxy = None
        self.client_manager = None
        self.network_manager = None
        self.server_accounts = None
    
    def pol_register_client(self, client_id: str) -> bool:
        """Mock注册客户端"""
        return True
    
    def submit_pol_proof(self, client_id: str, commitment: str,
                        data_hash: str, num_checkpoints: int,
                        total_steps: int) -> str:
        """Mock提交PoL证明"""
        return "0x" + "a" * 64
    
    def record_pol_verification(self, client_id: str, is_valid: bool) -> bool:
        """Mock记录验证结果"""
        return True
    
    def batch_record_pol_verification(self, client_ids: list, results: list) -> bool:
        """Mock批量记录验证结果"""
        return True
    
    def get_pol_proof(self, client_id: str) -> dict:
        """Mock获取PoL证明"""
        return {
            'commitment': "a" * 64,
            'data_hash': "b" * 64,
            'num_checkpoints': 5,
            'total_steps': 50,
            'timestamp': 0,
            'verified': False,
            'is_valid': False
        }
    
    def get_pol_stats(self) -> dict:
        """Mock获取统计信息"""
        return {
            'total_proofs': 0,
            'total_verifications': 0,
            'total_clients': 0
        }


# 创建全局mock实例
mock_chain_proxy = MockChainProxy()

