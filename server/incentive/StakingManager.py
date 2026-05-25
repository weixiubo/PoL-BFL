"""
Staking Manager for PoL-FL Economic Incentive System

Manages client staking, penalties, and stake requirements.
"""

import logging
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class StakingManager:
    """
    Manages staking for economic incentive system
    
    Features:
    - Stake requirement based on reputation
    - Penalty for misbehavior
    - Stake locking and unlocking
    - Penalty redistribution
    """
    
    def __init__(self, min_stake: float = 100.0, 
                 penalty_rates: Dict[str, float] = None,
                 chain_proxy=None):
        """
        Initialize Staking Manager
        
        Args:
            min_stake: Minimum stake required
            penalty_rates: Penalty rates for different violations
            chain_proxy: Blockchain proxy for on-chain staking
        """
        self.min_stake = min_stake
        self.chain_proxy = chain_proxy
        
        # Default penalty rates
        self.penalty_rates = penalty_rates or {
            'minor': 0.1,      # First-time verification failure
            'moderate': 0.3,   # Multiple verification failures
            'severe': 0.5,     # Malicious attack
            'critical': 1.0    # Persistent attack
        }
        
        # Local state (if not using blockchain)
        self.stakes: Dict[str, float] = defaultdict(float)
        self.locked_stakes: Dict[str, float] = defaultdict(float)
        self.penalty_pool: float = 0.0
        
        logger.info(f"StakingManager initialized (min_stake={min_stake})")
    
    def calculate_required_stake(self, client_id: str, reputation: float) -> float:
        """
        Calculate required stake based on reputation
        
        Args:
            client_id: Client identifier
            reputation: Client reputation (0-1)
        
        Returns:
            required_stake: Required stake amount
        """
        # Reputation factor
        if reputation < 0.3:
            reputation_factor = -0.5  # Low reputation needs more stake
        elif reputation > 0.7:
            reputation_factor = -0.3  # High reputation can reduce stake
        else:
            reputation_factor = 0.0
        
        required_stake = self.min_stake * (1 + reputation_factor)
        
        logger.debug(f"Client {client_id}: reputation={reputation:.2f}, "
                    f"required_stake={required_stake:.2f}")
        
        return required_stake
    
    def stake(self, client_id: str, amount: float) -> bool:
        """
        Stake tokens
        
        Args:
            client_id: Client identifier
            amount: Amount to stake
        
        Returns:
            success: True if staking successful
        """
        if amount <= 0:
            logger.error(f"Invalid stake amount: {amount}")
            return False
        
        if self.chain_proxy:
            # On-chain staking
            try:
                tx_hash = self.chain_proxy.stake(client_id, amount)
                logger.info(f"Client {client_id} staked {amount} tokens (tx: {tx_hash})")
                return True
            except Exception as e:
                logger.error(f"On-chain staking failed: {e}")
                return False
        else:
            # Off-chain staking (for testing)
            self.stakes[client_id] += amount
            logger.info(f"Client {client_id} staked {amount} tokens "
                       f"(total: {self.stakes[client_id]})")
            return True
    
    def lock_stake(self, client_id: str, amount: float) -> bool:
        """
        Lock stake for participation
        
        Args:
            client_id: Client identifier
            amount: Amount to lock
        
        Returns:
            success: True if locking successful
        """
        available = self.stakes[client_id] - self.locked_stakes[client_id]
        
        if available < amount:
            logger.error(f"Insufficient stake for {client_id}: "
                        f"available={available}, required={amount}")
            return False
        
        self.locked_stakes[client_id] += amount
        logger.info(f"Locked {amount} tokens for {client_id}")
        return True
    
    def unlock_stake(self, client_id: str, amount: float) -> bool:
        """
        Unlock stake after participation
        
        Args:
            client_id: Client identifier
            amount: Amount to unlock
        
        Returns:
            success: True if unlocking successful
        """
        if self.locked_stakes[client_id] < amount:
            logger.error(f"Cannot unlock {amount} for {client_id}: "
                        f"locked={self.locked_stakes[client_id]}")
            return False
        
        self.locked_stakes[client_id] -= amount
        logger.info(f"Unlocked {amount} tokens for {client_id}")
        return True
    
    def penalize(self, client_id: str, violation_type: str = 'minor') -> float:
        """
        Penalize client for misbehavior
        
        Args:
            client_id: Client identifier
            violation_type: Type of violation ('minor', 'moderate', 'severe', 'critical')
        
        Returns:
            penalty_amount: Amount penalized
        """
        if violation_type not in self.penalty_rates:
            logger.error(f"Unknown violation type: {violation_type}")
            violation_type = 'minor'
        
        penalty_rate = self.penalty_rates[violation_type]
        penalty_amount = self.stakes[client_id] * penalty_rate
        
        if penalty_amount > self.stakes[client_id]:
            penalty_amount = self.stakes[client_id]
        
        # Deduct from stake
        self.stakes[client_id] -= penalty_amount
        
        # 50% to penalty pool (for redistribution)
        # 50% burned (removed from system)
        redistribution = penalty_amount * 0.5
        burned = penalty_amount * 0.5
        
        self.penalty_pool += redistribution
        
        logger.warning(f"Penalized {client_id}: {penalty_amount:.2f} tokens "
                      f"({violation_type}, rate={penalty_rate})")
        logger.info(f"  Redistributed: {redistribution:.2f}, Burned: {burned:.2f}")
        
        return penalty_amount
    
    def unstake(self, client_id: str, amount: float) -> bool:
        """
        Unstake tokens
        
        Args:
            client_id: Client identifier
            amount: Amount to unstake
        
        Returns:
            success: True if unstaking successful
        """
        available = self.stakes[client_id] - self.locked_stakes[client_id]
        
        if available < amount:
            logger.error(f"Cannot unstake {amount} for {client_id}: "
                        f"available={available}")
            return False
        
        if self.chain_proxy:
            # On-chain unstaking
            try:
                tx_hash = self.chain_proxy.unstake(client_id, amount)
                logger.info(f"Client {client_id} unstaked {amount} tokens (tx: {tx_hash})")
                return True
            except Exception as e:
                logger.error(f"On-chain unstaking failed: {e}")
                return False
        else:
            # Off-chain unstaking
            self.stakes[client_id] -= amount
            logger.info(f"Client {client_id} unstaked {amount} tokens "
                       f"(remaining: {self.stakes[client_id]})")
            return True
    
    def get_stake(self, client_id: str) -> Dict[str, float]:
        """
        Get stake information for client
        
        Args:
            client_id: Client identifier
        
        Returns:
            stake_info: Dictionary with stake information
        """
        total = self.stakes[client_id]
        locked = self.locked_stakes[client_id]
        available = total - locked
        
        return {
            'total': total,
            'locked': locked,
            'available': available
        }
    
    def check_stake_requirement(self, client_id: str, reputation: float) -> bool:
        """
        Check if client meets stake requirement
        
        Args:
            client_id: Client identifier
            reputation: Client reputation
        
        Returns:
            meets_requirement: True if stake is sufficient
        """
        required = self.calculate_required_stake(client_id, reputation)
        available = self.stakes[client_id] - self.locked_stakes[client_id]
        
        meets = available >= required
        
        if not meets:
            logger.warning(f"Client {client_id} does not meet stake requirement: "
                          f"available={available:.2f}, required={required:.2f}")
        
        return meets
    
    def get_penalty_pool(self) -> float:
        """Get current penalty pool for redistribution"""
        return self.penalty_pool
    
    def distribute_penalty_pool(self, clients: list, amounts: list) -> bool:
        """
        Distribute penalty pool to honest clients
        
        Args:
            clients: List of client IDs
            amounts: List of amounts to distribute
        
        Returns:
            success: True if distribution successful
        """
        total_distribution = sum(amounts)
        
        if total_distribution > self.penalty_pool:
            logger.error(f"Cannot distribute {total_distribution}: "
                        f"pool={self.penalty_pool}")
            return False
        
        for client_id, amount in zip(clients, amounts):
            self.stakes[client_id] += amount
            logger.info(f"Distributed {amount:.2f} to {client_id} from penalty pool")
        
        self.penalty_pool -= total_distribution
        logger.info(f"Distributed {total_distribution:.2f} from penalty pool "
                   f"(remaining: {self.penalty_pool:.2f})")
        
        return True


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create staking manager
    manager = StakingManager(min_stake=100.0)
    
    # Client stakes
    manager.stake('client1', 150.0)
    manager.stake('client2', 200.0)
    
    # Lock stake for participation
    manager.lock_stake('client1', 100.0)
    
    # Check stake info
    print(f"\nClient1 stake: {manager.get_stake('client1')}")
    
    # Penalize misbehavior
    penalty = manager.penalize('client2', 'moderate')
    print(f"\nPenalty pool: {manager.get_penalty_pool():.2f}")
    
    # Unlock and unstake
    manager.unlock_stake('client1', 100.0)
    manager.unstake('client1', 50.0)
    
    print(f"\nFinal stakes:")
    print(f"  Client1: {manager.get_stake('client1')}")
    print(f"  Client2: {manager.get_stake('client2')}")

