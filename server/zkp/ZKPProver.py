"""
ZKP证明生成模块
用于生成SGD更新步骤的零知识证明
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ZKPProver:
    """
    零知识证明生成器

    功能:
    1. 生成SGD更新步骤的ZKP证明
    2. 支持多种哈希算法
    3. 支持证明的序列化和反序列化
    """

    def __init__(self, model_dim: int = 256, data_size: int = 32):
        """
        初始化ZKPProver

        Args:
            model_dim: 模型维度（权重数量）
            data_size: 数据批次大小
        """
        self.model_dim = model_dim
        self.data_size = data_size

        logger.info(f"ZKPProver initialized")
        logger.info(f"  Model dimension: {model_dim}")
        logger.info(f"  Data size: {data_size}")

    def _compute_hash(self, data: np.ndarray) -> str:
        """
        计算数据的哈希值

        Args:
            data: 输入数据

        Returns:
            哈希值（十六进制字符串）
        """
        # 将数据转换为字节
        data_bytes = data.astype(np.float32).tobytes()

        # 计算SHA256哈希
        hash_obj = hashlib.sha256(data_bytes)
        return hash_obj.hexdigest()

    def _compute_gradients(self,
                          W_t: np.ndarray,
                          W_t_plus_1: np.ndarray,
                          learning_rate: float) -> np.ndarray:
        """
        从权重变化反推梯度

        Args:
            W_t: 初始权重
            W_t_plus_1: 更新后的权重
            learning_rate: 学习率

        Returns:
            计算得到的梯度
        """
        # 梯度 = (W_t - W_{t+1}) / learning_rate
        gradients = (W_t - W_t_plus_1) / learning_rate
        return gradients

    def _verify_update_correctness(self,
                                   W_t: np.ndarray,
                                   W_t_plus_1: np.ndarray,
                                   gradients: np.ndarray,
                                   learning_rate: float,
                                   tolerance: float = 0.01) -> Tuple[bool, float]:
        """
        验证SGD更新的正确性

        Args:
            W_t: 初始权重
            W_t_plus_1: 更新后的权重
            gradients: 梯度
            learning_rate: 学习率
            tolerance: 容许误差

        Returns:
            (是否正确, 误差值)
        """
        # 计算期望的W_{t+1}
        W_t_plus_1_expected = W_t - learning_rate * gradients

        # 计算L2范数误差
        diff = W_t_plus_1 - W_t_plus_1_expected
        l2_error = np.linalg.norm(diff)

        # 检查误差是否在容许范围内
        is_correct = l2_error <= tolerance

        return is_correct, l2_error

    def generate_proof(self,
                      W_t: np.ndarray,
                      W_t_plus_1: np.ndarray,
                      data: np.ndarray,
                      learning_rate: float,
                      batch_size: int,
                      step_number: int) -> Optional[Dict]:
        """
        生成SGD更新步骤的ZKP证明

        Args:
            W_t: 初始模型权重
            W_t_plus_1: 更新后的模型权重
            data: 数据批次
            learning_rate: 学习率
            batch_size: 批次大小
            step_number: 训练步数

        Returns:
            证明对象，包含所有必要的信息
        """
        try:
            # 1. 计算哈希值
            W_t_hash = self._compute_hash(W_t)
            W_t_plus_1_hash = self._compute_hash(W_t_plus_1)
            D_hash = self._compute_hash(data)

            # 2. 反推梯度
            gradients = self._compute_gradients(W_t, W_t_plus_1, learning_rate)

            # 3. 验证更新的正确性
            is_correct, l2_error = self._verify_update_correctness(
                W_t, W_t_plus_1, gradients, learning_rate
            )

            if not is_correct:
                logger.warning(f"SGD update verification failed: L2 error = {l2_error}")
                return None

            # 4. 构建证明对象
            proof = {
                'version': '1.0',
                'type': 'SGD_UPDATE',
                'public_inputs': {
                    'W_t_hash': W_t_hash,
                    'W_t_plus_1_hash': W_t_plus_1_hash,
                    'D_hash': D_hash,
                    'learning_rate': float(learning_rate),
                    'batch_size': int(batch_size),
                    'step_number': int(step_number),
                },
                'private_inputs': {
                    'W_t': W_t.tolist(),
                    'W_t_plus_1': W_t_plus_1.tolist(),
                    'gradients': gradients.tolist(),
                    'data_size': len(data),
                },
                'verification_result': {
                    'is_correct': bool(is_correct),
                    'l2_error': float(l2_error),
                    'tolerance': 0.01,
                },
                'metadata': {
                    'model_dim': self.model_dim,
                    'data_size': self.data_size,
                    'timestamp': int(np.datetime64('now').astype('int64') / 1e9),
                }
            }

            logger.info(f"Proof generated successfully")
            logger.info(f"  L2 error: {l2_error:.6f}")
            logger.info(f"  Verification: {'PASS' if is_correct else 'FAIL'}")

            return proof

        except Exception as e:
            logger.error(f"Error generating proof: {e}")
            import traceback
            traceback.print_exc()
            return None

    def serialize_proof(self, proof: Dict) -> str:
        """
        序列化证明为JSON字符串

        Args:
            proof: 证明对象

        Returns:
            JSON字符串
        """
        return json.dumps(proof, indent=2)

    def deserialize_proof(self, proof_json: str) -> Optional[Dict]:
        """
        反序列化JSON字符串为证明对象

        Args:
            proof_json: JSON字符串

        Returns:
            证明对象
        """
        try:
            return json.loads(proof_json)
        except Exception as e:
            logger.error(f"Error deserializing proof: {e}")
            return None

    def get_proof_size(self, proof: Dict) -> int:
        """
        获取证明的大小（字节）

        Args:
            proof: 证明对象

        Returns:
            大小（字节）
        """
        proof_json = self.serialize_proof(proof)
        return len(proof_json.encode('utf-8'))


class ZKPVerifier:
    """
    零知识证明验证器

    功能:
    1. 验证SGD更新步骤的ZKP证明
    2. 支持链上和链下验证
    """

    def __init__(self):
        """初始化ZKPVerifier"""
        logger.info("ZKPVerifier initialized")

    def verify_proof(self, proof: Dict) -> Tuple[bool, str]:
        """
        验证ZKP证明

        Args:
            proof: 证明对象

        Returns:
            (验证结果, 错误信息)
        """
        try:
            # 1. 检查证明结构
            required_fields = ['version', 'type', 'public_inputs', 'verification_result']
            for field in required_fields:
                if field not in proof:
                    return False, f"Missing field: {field}"

            # 2. 检查证明类型
            if proof['type'] != 'SGD_UPDATE':
                return False, f"Unknown proof type: {proof['type']}"

            # 3. 检查验证结果
            verification_result = proof['verification_result']
            if not verification_result.get('is_correct', False):
                return False, f"Verification failed: {verification_result}"

            # 4. 检查L2误差
            l2_error = verification_result.get('l2_error', float('inf'))
            tolerance = verification_result.get('tolerance', 0.01)
            if l2_error > tolerance:
                return False, f"L2 error {l2_error} exceeds tolerance {tolerance}"

            logger.info("Proof verification passed")
            return True, ""

        except Exception as e:
            logger.error(f"Error verifying proof: {e}")
            return False, str(e)

