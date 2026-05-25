"""
PoL Client
扩展BaseClient，支持PoL承诺生成和挑战响应
"""

import logging
from typing import Dict, Optional
from torch.utils.data import DataLoader
import torch.nn as nn

from client.clients import Client
from client.trainer.PoLTrainer import PoLTrainer

logger = logging.getLogger(__name__)


class PoLClient(Client):
    """
    支持PoL的客户端
    
    扩展功能:
    1. 使用PoLTrainer进行训练
    2. 生成PoL承诺
    3. 响应服务器的验证挑战
    4. 将PoL承诺提交到区块链
    """
    
    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        model: nn.Module,
        trainer: PoLTrainer,
        args: dict = {},
        test_dataloader: DataLoader = None,
        watermarks: dict = {},
    ) -> None:
        """
        初始化PoL客户端
        
        Args:
            client_id: 客户端ID
            dataloader: 训练数据加载器
            model: 模型
            trainer: PoLTrainer实例
            args: 参数字典
            test_dataloader: 测试数据加载器
            watermarks: 水印参数
        """
        super().__init__(
            client_id=client_id,
            dataloader=dataloader,
            model=model,
            trainer=trainer,
            args=args,
            test_dataloader=test_dataloader,
            watermarks=watermarks
        )
        
        # 确保trainer是PoLTrainer
        if not isinstance(trainer, PoLTrainer):
            logger.warning(f"Trainer is not PoLTrainer, PoL features may not work")
        
        self.pol_commitment = None
        self.enable_pol = args.get('enable_pol', False)
        self.enable_auto_register = args.get('enable_auto_register', False)
        self.enable_auto_submit_pol = args.get('enable_auto_submit_pol', False)
        
        logger.info(f"PoLClient {client_id} initialized")
        logger.info(f"  PoL enabled: {self.enable_pol}")
    
    def train(self, total_epoch: int, dataset=None) -> list:
        """
        训练模型并生成PoL承诺
        
        Args:
            total_epoch: 训练轮数
            dataset: 完整数据集（用于计算数据哈希）
        
        Returns:
            ret_list: 训练结果列表
        """
        # 调用trainer的train方法
        ret_list = self.trainer.train(total_epoch)
        
        # 如果启用PoL，生成承诺，并根据开关自动上链
        if self.enable_pol and isinstance(self.trainer, PoLTrainer):
            hash_dataset = dataset
            if hash_dataset is None:
                try:
                    hash_dataset = self.trainer.dataloader.dataset
                except Exception:
                    hash_dataset = None
            self.pol_commitment = self.trainer.finalize_pol(
                epoch=total_epoch - 1,
                dataset=hash_dataset
            )
            
            if self.pol_commitment:
                logger.info(f"Client {self.client_id} generated PoL commitment")
                logger.info(f"  Commitment: {self.pol_commitment['commitment'][:16]}...")
        
        return ret_list
    
    def get_pol_commitment(self) -> Optional[Dict]:
        """
        获取PoL承诺
        
        Returns:
            pol_commitment: PoL承诺数据
        """
        return self.pol_commitment
    
    def respond_to_challenge(self, challenge_data: Dict) -> Optional[Dict]:
        """
        响应验证挑战
        
        Args:
            challenge_data: 挑战数据
        
        Returns:
            response: 响应数据
        """
        if not self.enable_pol:
            logger.warning(f"Client {self.client_id}: PoL not enabled")
            return None
        
        if not isinstance(self.trainer, PoLTrainer):
            logger.error(f"Client {self.client_id}: Trainer is not PoLTrainer")
            return None
        
        response = self.trainer.respond_to_challenge(challenge_data)
        
        if response:
            logger.info(f"Client {self.client_id} responded to challenge")
        
        return response
    
    def submit_pol_to_blockchain(self, chain_proxy) -> bool:
        """
        将PoL承诺提交到区块链
        
        Args:
            chain_proxy: 区块链代理对象
        
        Returns:
            success: 是否成功提交
        """
        if not self.enable_pol or self.pol_commitment is None:
            logger.warning(f"Client {self.client_id}: No PoL commitment to submit")
            return False
        
        try:
            # 调用chainProxy的方法提交PoL
            # 这个方法将在后续实现chainProxy扩展时添加
            if hasattr(chain_proxy, 'submit_pol_proof'):
                tx_hash = chain_proxy.submit_pol_proof(
                    client_id=self.client_id,
                    commitment=self.pol_commitment['commitment'],
                    data_hash=self.pol_commitment['data_hash'],
                    num_checkpoints=self.pol_commitment['num_checkpoints'],
                    total_steps=self.pol_commitment['total_steps']
                )
                logger.info(f"Client {self.client_id} submitted PoL to blockchain")
                logger.info(f"  Transaction hash: {tx_hash}")
                return True
            else:
                logger.error("chainProxy does not support submit_pol_proof")
                return False
        except Exception as e:
            logger.error(f"Failed to submit PoL to blockchain: {e}")
            return False
    
    def get_pol_metadata(self) -> Dict:
        """
        获取PoL元数据
        
        Returns:
            metadata: 元数据字典
        """
        if not self.enable_pol or not isinstance(self.trainer, PoLTrainer):
            return {}
        
        if self.trainer.pol_manager is None:
            return {}
        
        return self.trainer.pol_manager.get_metadata()
    
    def __repr__(self) -> str:
        pol_status = "enabled" if self.enable_pol else "disabled"
        return f"PoLClient(id={self.client_id}, PoL={pol_status})"
