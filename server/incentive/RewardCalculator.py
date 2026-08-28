"""
Reward Calculator for PoL-FL Economic Incentive System

Calculates and distributes rewards based on contribution, reputation, and verification results.
"""

import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Calculates rewards for clients based on multiple factors

    Reward components:
    - Base reward: Equal distribution with quality factor
    - Contribution reward: Based on data size
    - Reputation reward: Based on reputation score
    """

    def __init__(self, base_reward_per_round: float = 500.0,
                 contribution_weight: float = 0.3,
                 reputation_weight: float = 0.2,
                 chain_proxy=None):
        """
        Initialize Reward Calculator

        Args:
            base_reward_per_round: Base reward pool per round
            contribution_weight: Weight for contribution-based reward
            reputation_weight: Weight for reputation-based reward
            chain_proxy: Blockchain proxy for on-chain reward distribution
        """
        self.base_reward_per_round = base_reward_per_round
        self.contribution_weight = contribution_weight
        self.reputation_weight = reputation_weight
        self.chain_proxy = chain_proxy

        # Ensure weights sum to reasonable value
        total_weight = contribution_weight + reputation_weight
        if total_weight > 1.0:
            logger.warning(f"Total weight {total_weight} > 1.0, normalizing...")
            self.contribution_weight /= total_weight
            self.reputation_weight /= total_weight

        # Track rewards
        self.total_rewards_distributed = 0.0
        self.rewards_history: Dict[str, List[float]] = defaultdict(list)

        logger.info(f"RewardCalculator initialized (base={base_reward_per_round}, "
                   f"contrib_w={contribution_weight}, rep_w={reputation_weight})")

    def calculate_rewards(self,
                         clients: List[str],
                         verification_results: Dict[str, bool],
                         data_sizes: Dict[str, int],
                         reputations: Dict[str, float],
                         penalty_pool: float = 0.0) -> Dict[str, float]:
        """
        Calculate rewards for all clients

        Args:
            clients: List of client IDs
            verification_results: Verification results {client_id: passed}
            data_sizes: Data sizes {client_id: size}
            reputations: Reputation scores {client_id: score}
            penalty_pool: Additional reward from penalty redistribution

        Returns:
            rewards: Dictionary {client_id: reward_amount}
        """
        if not clients:
            logger.warning("No clients to calculate rewards for")
            return {}

        # Total reward pool
        total_pool = self.base_reward_per_round + penalty_pool

        # Calculate each component
        base_rewards = self._calculate_base_rewards(clients, verification_results, total_pool)
        contribution_rewards = self._calculate_contribution_rewards(clients, data_sizes, total_pool)
        reputation_rewards = self._calculate_reputation_rewards(clients, reputations, total_pool)

        # Combine rewards
        total_rewards = {}
        for client_id in clients:
            total_rewards[client_id] = (
                base_rewards.get(client_id, 0.0) +
                contribution_rewards.get(client_id, 0.0) +
                reputation_rewards.get(client_id, 0.0)
            )

        # Log rewards
        logger.info(f"Calculated rewards for {len(clients)} clients:")
        for client_id, reward in total_rewards.items():
            logger.info(f"  {client_id}: {reward:.2f} "
                       f"(base={base_rewards.get(client_id, 0):.2f}, "
                       f"contrib={contribution_rewards.get(client_id, 0):.2f}, "
                       f"rep={reputation_rewards.get(client_id, 0):.2f})")

        return total_rewards

    def _calculate_base_rewards(self, clients: List[str],
                                verification_results: Dict[str, bool],
                                total_pool: float) -> Dict[str, float]:
        """
        Calculate base rewards with quality factor

        Quality factor:
        - 1.0 if verification passed
        - 0.5 if not verified (random sampling)
        - 0.0 if verification failed
        """
        base_pool = total_pool * (1.0 - self.contribution_weight - self.reputation_weight)

        # Calculate quality factors
        quality_factors = {}
        for client_id in clients:
            if client_id in verification_results:
                # Was verified
                quality_factors[client_id] = 1.0 if verification_results[client_id] else 0.0
            else:
                # Not selected by probabilistic verification.
                quality_factors[client_id] = 0.5

        # Calculate rewards
        total_quality = sum(quality_factors.values())
        if total_quality == 0:
            logger.warning("Total quality is 0, no base rewards")
            return {client_id: 0.0 for client_id in clients}

        base_rewards = {}
        for client_id in clients:
            base_rewards[client_id] = base_pool * (quality_factors[client_id] / total_quality)

        return base_rewards

    def _calculate_contribution_rewards(self, clients: List[str],
                                       data_sizes: Dict[str, int],
                                       total_pool: float) -> Dict[str, float]:
        """
        Calculate contribution-based rewards

        Reward proportional to data size
        """
        contrib_pool = total_pool * self.contribution_weight

        # Calculate total data size
        total_data = sum(data_sizes.get(client_id, 0) for client_id in clients)

        if total_data == 0:
            logger.warning("Total data size is 0, no contribution rewards")
            return {client_id: 0.0 for client_id in clients}

        # Calculate rewards
        contrib_rewards = {}
        for client_id in clients:
            data_size = data_sizes.get(client_id, 0)
            contrib_rewards[client_id] = contrib_pool * (data_size / total_data)

        return contrib_rewards

    def _calculate_reputation_rewards(self, clients: List[str],
                                     reputations: Dict[str, float],
                                     total_pool: float) -> Dict[str, float]:
        """
        Calculate reputation-based rewards

        Reward proportional to reputation score
        """
        rep_pool = total_pool * self.reputation_weight

        # Calculate total reputation
        total_rep = sum(reputations.get(client_id, 0.5) for client_id in clients)

        if total_rep == 0:
            logger.warning("Total reputation is 0, no reputation rewards")
            return {client_id: 0.0 for client_id in clients}

        # Calculate rewards
        rep_rewards = {}
        for client_id in clients:
            reputation = reputations.get(client_id, 0.5)
            rep_rewards[client_id] = rep_pool * (reputation / total_rep)

        return rep_rewards

    def distribute_rewards(self, rewards: Dict[str, float]) -> bool:
        """
        Distribute rewards to clients

        Args:
            rewards: Dictionary {client_id: reward_amount}

        Returns:
            success: True if distribution successful
        """
        if not rewards:
            logger.warning("No rewards to distribute")
            return True

        total_reward = sum(rewards.values())

        if self.chain_proxy:
            # On-chain distribution
            try:
                client_ids = list(rewards.keys())
                amounts = list(rewards.values())
                tx_hash = self.chain_proxy.distribute_rewards(client_ids, amounts)
                logger.info(f"Distributed {total_reward:.2f} tokens on-chain (tx: {tx_hash})")

                # Update history
                for client_id, amount in rewards.items():
                    self.rewards_history[client_id].append(amount)
                self.total_rewards_distributed += total_reward

                return True
            except Exception as e:
                logger.error(f"On-chain reward distribution failed: {e}")
                return False
        else:
            # Off-chain distribution (for testing)
            logger.info(f"Distributed {total_reward:.2f} tokens off-chain:")
            for client_id, amount in rewards.items():
                logger.info(f"  {client_id}: {amount:.2f}")
                self.rewards_history[client_id].append(amount)

            self.total_rewards_distributed += total_reward
            return True

    def get_reward_history(self, client_id: str) -> List[float]:
        """Get reward history for a client"""
        return self.rewards_history.get(client_id, [])

    def get_total_rewards(self, client_id: str) -> float:
        """Get total rewards earned by a client"""
        return sum(self.rewards_history.get(client_id, []))

    def get_statistics(self) -> Dict:
        """Get reward statistics"""
        return {
            'total_distributed': self.total_rewards_distributed,
            'num_clients': len(self.rewards_history),
            'avg_reward_per_client': (
                self.total_rewards_distributed / len(self.rewards_history)
                if self.rewards_history else 0.0
            )
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create reward calculator
    calculator = RewardCalculator(base_reward_per_round=500.0)

    # Example data
    clients = ['client1', 'client2', 'client3', 'client4']
    verification_results = {
        'client1': True,   # Verified and passed
        'client2': False,  # Verified and failed
        # client3 not verified (random sampling)
        'client4': True    # Verified and passed
    }
    data_sizes = {
        'client1': 1000,
        'client2': 1500,
        'client3': 800,
        'client4': 1200
    }
    reputations = {
        'client1': 0.8,
        'client2': 0.3,
        'client3': 0.6,
        'client4': 0.9
    }

    # Calculate rewards
    rewards = calculator.calculate_rewards(
        clients, verification_results, data_sizes, reputations, penalty_pool=50.0
    )

    # Distribute rewards
    calculator.distribute_rewards(rewards)

    # Get statistics
    print(f"\nReward Statistics:")
    stats = calculator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
