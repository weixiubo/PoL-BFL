"""
经济激励系统
集成质押、奖励、声誉等机制
"""

import logging
from typing import Dict, Optional, Tuple
from server.incentive.StakingManager import StakingManager
from server.incentive.RewardCalculator import RewardCalculator
from server.incentive.ReputationSystem import ReputationSystem

logger = logging.getLogger(__name__)


class EconomicIncentiveSystem:
    """
    经济激励系统
    
    功能:
    1. 集成质押、奖励、声誉机制
    2. 支持女巫攻击防御
    3. 支持动态激励
    4. 支持完整的经济模型
    """
    
    def __init__(self, args: Dict = None):
        """
        初始化EconomicIncentiveSystem
        
        Args:
            args: 配置参数
        """
        if args is None:
            args = {}
        
        # 初始化各个组件
        self.staking_manager = StakingManager(
            min_stake=args.get('min_stake', 100.0)
        )
        
        self.reward_calculator = RewardCalculator(
            base_reward_per_round=args.get('base_reward', 500.0),
            reputation_weight=args.get('reputation_weight', 0.2)
        )
        
        self.reputation_system = ReputationSystem(
            initial_reputation=args.get('initial_reputation', 0.5),
            decay_factor=args.get('decay_factor', 0.9)
        )
        
        # 女巫攻击防御参数
        self.min_stake_for_participation = args.get('min_stake_for_participation', 100.0)
        self.reputation_threshold = args.get('reputation_threshold', 0.3)
        self.max_clients_per_address = args.get('max_clients_per_address', 1)
        
        # 统计数据
        self.total_rounds = 0
        self.total_rewards_distributed = 0.0
        self.total_stakes_slashed = 0.0
        
        logger.info(f"EconomicIncentiveSystem initialized")
        logger.info(f"  Min stake: {self.min_stake_for_participation}")
        logger.info(f"  Reputation threshold: {self.reputation_threshold}")
    
    def register_client(self, client_id: str, initial_stake: float) -> Tuple[bool, str]:
        """
        注册客户端

        Args:
            client_id: 客户端ID
            initial_stake: 初始质押

        Returns:
            (成功标志, 消息)
        """
        try:
            # 检查最小质押
            if initial_stake < self.min_stake_for_participation:
                return False, f"Initial stake {initial_stake} is less than minimum {self.min_stake_for_participation}"

            # 质押
            self.staking_manager.stake(client_id, initial_stake)

            logger.info(f"Registered client {client_id} with stake {initial_stake}")

            return True, f"Successfully registered client {client_id}"

        except Exception as e:
            logger.error(f"Error registering client: {e}")
            return False, str(e)
    
    def verify_client_eligibility(self, client_id: str) -> Tuple[bool, str]:
        """
        验证客户端是否有资格参与

        Args:
            client_id: 客户端ID

        Returns:
            (是否有资格, 原因)
        """
        try:
            # 检查质押
            stake = self.staking_manager.stakes.get(client_id, 0.0)
            if stake < self.min_stake_for_participation:
                return False, f"Insufficient stake: {stake} < {self.min_stake_for_participation}"

            # 检查声誉
            reputation = self.reputation_system.reputations.get(client_id, self.reputation_system.initial_reputation)
            if reputation < self.reputation_threshold:
                return False, f"Reputation too low: {reputation} < {self.reputation_threshold}"

            return True, "Client is eligible"

        except Exception as e:
            logger.error(f"Error verifying eligibility: {e}")
            return False, str(e)
    
    def process_verification_result(self,
                                   client_id: str,
                                   is_verified: bool,
                                   training_steps: int = 0,
                                   total_steps: int = 0) -> Dict:
        """
        处理验证结果

        Args:
            client_id: 客户端ID
            is_verified: 验证是否成功
            training_steps: 训练步数
            total_steps: 总步数

        Returns:
            处理结果字典
        """
        try:
            result = {
                'client_id': client_id,
                'verified': is_verified,
                'reputation_change': 0.0,
                'reward': 0.0,
                'slash': 0.0
            }

            # 更新声誉
            old_reputation = self.reputation_system.reputations.get(client_id, self.reputation_system.initial_reputation)
            if is_verified:
                self.reputation_system.update_reputation(client_id, 0.05)  # 增加声誉
            else:
                self.reputation_system.update_reputation(client_id, -0.1)  # 降低声誉

            new_reputation = self.reputation_system.reputations.get(client_id, self.reputation_system.initial_reputation)
            result['reputation_change'] = new_reputation - old_reputation

            # 如果验证失败，罚没质押
            if not is_verified:
                stake = self.staking_manager.stakes.get(client_id, 0.0)
                slash_amount = stake * 0.1  # 罚没10%

                # 直接修改质押
                self.staking_manager.stakes[client_id] -= slash_amount
                self.staking_manager.penalty_pool += slash_amount

                result['slash'] = slash_amount
                self.total_stakes_slashed += slash_amount

            # 如果验证成功，计算奖励
            if is_verified and training_steps > 0:
                # 简单的奖励计算
                reward = self.reward_calculator.base_reward_per_round * (training_steps / max(total_steps, 1))

                result['reward'] = reward
                self.total_rewards_distributed += reward

            logger.info(f"Processed verification for {client_id}: verified={is_verified}, reward={result['reward']:.2f}, slash={result['slash']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error processing verification result: {e}")
            return {'error': str(e)}
    
    def end_round(self) -> Dict:
        """
        结束一轮（应用衰减等）

        Returns:
            轮次统计
        """
        try:
            self.total_rounds += 1

            # 应用声誉衰减
            for client_id in list(self.reputation_system.reputations.keys()):
                current_rep = self.reputation_system.reputations[client_id]
                decayed_rep = current_rep * (1 - 0.01)  # 1%衰减
                self.reputation_system.reputations[client_id] = max(decayed_rep, 0.0)

            logger.info(f"Ended round {self.total_rounds}, applied reputation decay")

            return {
                'round': self.total_rounds,
                'total_rewards_distributed': self.total_rewards_distributed,
                'total_stakes_slashed': self.total_stakes_slashed
            }

        except Exception as e:
            logger.error(f"Error ending round: {e}")
            return {'error': str(e)}

    def get_system_statistics(self) -> Dict:
        """获取系统统计"""
        return {
            'total_rounds': self.total_rounds,
            'total_rewards_distributed': self.total_rewards_distributed,
            'total_stakes_slashed': self.total_stakes_slashed,
            'total_clients': len(self.staking_manager.stakes),
            'total_staked': sum(self.staking_manager.stakes.values())
        }

    def get_client_status(self, client_id: str) -> Dict:
        """获取客户端状态"""
        return {
            'client_id': client_id,
            'stake': self.staking_manager.stakes.get(client_id, 0.0),
            'reputation': self.reputation_system.reputations.get(client_id, self.reputation_system.initial_reputation),
            'eligible': self.verify_client_eligibility(client_id)[0]
        }

