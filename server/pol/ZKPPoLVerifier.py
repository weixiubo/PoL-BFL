"""
ZKP-PoL验证器
将ZKP集成到PoL验证流程中，实现隐私保护的可验证计算
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from server.zkp.ZKPProver import ZKPProver, ZKPVerifier
from server.pol.PoLVerifier import PoLVerifier

logger = logging.getLogger(__name__)


class ZKPPoLVerifier:
    """
    ZKP-PoL验证器

    功能:
    1. 生成SGD更新步骤的ZKP证明
    2. 验证ZKP证明
    3. 集成到PoL验证流程
    4. 支持隐私保护的可验证计算
    """

    def __init__(self, args: Dict = None):
        """
        初始化ZKPPoLVerifier

        Args:
            args: 配置参数
        """
        if args is None:
            args = {}

        self.model_dim = args.get('model_dim', 256)
        self.data_size = args.get('data_size', 32)
        self.use_zkp = args.get('use_zkp', True)
        self.zkp_tolerance = args.get('zkp_tolerance', 0.01)

        # 初始化ZKP组件
        self.zkp_prover = ZKPProver(
            model_dim=self.model_dim,
            data_size=self.data_size
        )
        self.zkp_verifier = ZKPVerifier()

        # 初始化PoL验证器
        self.pol_verifier = PoLVerifier(args)

        logger.info(f"ZKPPoLVerifier initialized")
        logger.info(f"  Model dimension: {self.model_dim}")
        logger.info(f"  Data size: {self.data_size}")
        logger.info(f"  Use ZKP: {self.use_zkp}")
        logger.info(f"  ZKP tolerance: {self.zkp_tolerance}")

    def verify_with_zkp(self,
                       W_t: np.ndarray,
                       W_t_plus_1: np.ndarray,
                       data: np.ndarray,
                       learning_rate: float,
                       batch_size: int,
                       step_number: int) -> Tuple[bool, Dict]:
        """
        使用ZKP验证SGD更新步骤

        Args:
            W_t: 初始模型权重
            W_t_plus_1: 更新后的模型权重
            data: 数据批次
            learning_rate: 学习率
            batch_size: 批次大小
            step_number: 训练步数

        Returns:
            (验证结果, 详细信息)
        """
        try:
            # 1. 生成ZKP证明
            proof = self.zkp_prover.generate_proof(
                W_t=W_t,
                W_t_plus_1=W_t_plus_1,
                data=data,
                learning_rate=learning_rate,
                batch_size=batch_size,
                step_number=step_number
            )

            if proof is None:
                return False, {
                    'error': 'Failed to generate ZKP proof',
                    'step': step_number
                }

            # 2. 验证ZKP证明
            is_valid, error_msg = self.zkp_verifier.verify_proof(proof)

            if not is_valid:
                return False, {
                    'error': f'ZKP verification failed: {error_msg}',
                    'step': step_number,
                    'l2_error': proof['verification_result']['l2_error']
                }

            # 3. 返回验证结果
            return True, {
                'step': step_number,
                'l2_error': proof['verification_result']['l2_error'],
                'proof_size': self.zkp_prover.get_proof_size(proof),
                'proof_id': hash(str(proof))
            }

        except Exception as e:
            logger.error(f"Error in ZKP verification: {e}")
            return False, {'error': str(e), 'step': step_number}

    def verify_checkpoint_with_zkp(self,
                                   checkpoint: Dict,
                                   learning_rate: float) -> Tuple[bool, Dict]:
        """
        使用ZKP验证checkpoint

        Args:
            checkpoint: Checkpoint数据
            learning_rate: 学习率

        Returns:
            (验证结果, 详细信息)
        """
        try:
            # 提取checkpoint数据
            W_t = np.array(checkpoint.get('W_t', []))
            W_t_plus_1 = np.array(checkpoint.get('W_t_plus_1', []))
            data = np.array(checkpoint.get('data', []))
            batch_size = checkpoint.get('batch_size', 32)
            step_number = checkpoint.get('step_number', 0)

            # 验证数据有效性
            if W_t.size == 0 or W_t_plus_1.size == 0:
                return False, {'error': 'Invalid checkpoint data'}

            # 使用ZKP验证
            return self.verify_with_zkp(
                W_t=W_t,
                W_t_plus_1=W_t_plus_1,
                data=data,
                learning_rate=learning_rate,
                batch_size=batch_size,
                step_number=step_number
            )

        except Exception as e:
            logger.error(f"Error verifying checkpoint with ZKP: {e}")
            return False, {'error': str(e)}

    def verify_training_trajectory_with_zkp(self,
                                           checkpoints: list,
                                           learning_rate: float) -> Tuple[bool, Dict]:
        """
        使用ZKP验证完整的训练轨迹

        Args:
            checkpoints: Checkpoint列表
            learning_rate: 学习率

        Returns:
            (验证结果, 详细信息)
        """
        try:
            results = {
                'total_checkpoints': len(checkpoints),
                'verified_checkpoints': 0,
                'failed_checkpoints': 0,
                'total_proof_size': 0,
                'details': []
            }

            for i, checkpoint in enumerate(checkpoints):
                is_valid, detail = self.verify_checkpoint_with_zkp(
                    checkpoint,
                    learning_rate
                )

                results['details'].append(detail)

                if is_valid:
                    results['verified_checkpoints'] += 1
                    results['total_proof_size'] += detail.get('proof_size', 0)
                else:
                    results['failed_checkpoints'] += 1
                    logger.warning(f"Checkpoint {i} verification failed: {detail}")

            # 判断整体验证结果
            is_all_valid = results['failed_checkpoints'] == 0

            logger.info(f"Training trajectory verification: {results['verified_checkpoints']}/{results['total_checkpoints']} passed")

            return is_all_valid, results

        except Exception as e:
            logger.error(f"Error verifying training trajectory: {e}")
            return False, {'error': str(e)}

    def get_verification_stats(self) -> Dict:
        """
        获取验证统计信息

        Returns:
            统计信息字典
        """
        return {
            'model_dim': self.model_dim,
            'data_size': self.data_size,
            'use_zkp': self.use_zkp,
            'zkp_tolerance': self.zkp_tolerance,
            'zkp_prover_initialized': self.zkp_prover is not None,
            'zkp_verifier_initialized': self.zkp_verifier is not None,
            'pol_verifier_initialized': self.pol_verifier is not None
        }


class ZKPPoLAggregator:
    """
    ZKP-PoL聚合器

    功能:
    1. 聚合通过ZKP验证的模型更新
    2. 记录验证结果
    3. 支持激励机制
    """

    def __init__(self, args: Dict = None):
        """
        初始化ZKPPoLAggregator

        Args:
            args: 配置参数
        """
        if args is None:
            args = {}

        self.zkp_pol_verifier = ZKPPoLVerifier(args)
        self.verified_updates = []
        self.failed_updates = []

        logger.info("ZKPPoLAggregator initialized")

    def aggregate_with_zkp_verification(self,
                                       client_updates: Dict,
                                       learning_rate: float) -> Tuple[bool, Dict]:
        """
        使用ZKP验证聚合客户端更新

        Args:
            client_updates: 客户端更新字典 {client_id: update_data}
            learning_rate: 学习率

        Returns:
            (聚合结果, 详细信息)
        """
        try:
            aggregation_result = {
                'total_clients': len(client_updates),
                'verified_clients': 0,
                'failed_clients': 0,
                'verified_updates': [],
                'failed_updates': []
            }

            for client_id, update_data in client_updates.items():
                # 验证客户端更新
                is_valid, detail = self.zkp_pol_verifier.verify_checkpoint_with_zkp(
                    update_data,
                    learning_rate
                )

                if is_valid:
                    aggregation_result['verified_clients'] += 1
                    aggregation_result['verified_updates'].append({
                        'client_id': client_id,
                        'detail': detail
                    })
                    self.verified_updates.append((client_id, update_data))
                else:
                    aggregation_result['failed_clients'] += 1
                    aggregation_result['failed_updates'].append({
                        'client_id': client_id,
                        'error': detail.get('error', 'Unknown error')
                    })
                    self.failed_updates.append((client_id, update_data))

            logger.info(f"Aggregation: {aggregation_result['verified_clients']}/{aggregation_result['total_clients']} clients verified")

            return aggregation_result['failed_clients'] == 0, aggregation_result

        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            return False, {'error': str(e)}

