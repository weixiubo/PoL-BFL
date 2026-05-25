"""
Reputation System for PoL-FL Economic Incentive System

Tracks client behavior and maintains reputation scores.
"""

import logging
from typing import Dict, List, Optional
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class ReputationSystem:
    """
    Manages client reputation scores
    
    Features:
    - Reputation tracking based on verification results
    - Dynamic verification probability based on reputation
    - Reputation recovery mechanism
    - History tracking
    """
    
    def __init__(self, initial_reputation: float = 0.5,
                 decay_factor: float = 0.9,
                 base_verification_prob: float = 0.3,
                 chain_proxy=None):
        """
        Initialize Reputation System
        
        Args:
            initial_reputation: Initial reputation for new clients (0-1)
            decay_factor: Weight for historical reputation (0-1)
            base_verification_prob: Base verification probability
            chain_proxy: Blockchain proxy for on-chain reputation
        """
        self.initial_reputation = initial_reputation
        self.decay_factor = decay_factor
        self.base_verification_prob = base_verification_prob
        self.chain_proxy = chain_proxy
        
        # Local state
        self.reputations: Dict[str, float] = defaultdict(lambda: initial_reputation)
        self.history: Dict[str, List[Dict]] = defaultdict(list)
        
        logger.info(f"ReputationSystem initialized (initial={initial_reputation}, "
                   f"decay={decay_factor})")
    
    def get_reputation(self, client_id: str) -> float:
        """
        Get reputation score for client
        
        Args:
            client_id: Client identifier
        
        Returns:
            reputation: Reputation score (0-1)
        """
        if self.chain_proxy:
            # Get from blockchain
            try:
                reputation = self.chain_proxy.get_reputation(client_id)
                return reputation / 1000.0  # Convert from 0-1000 to 0-1
            except Exception as e:
                logger.warning(f"Failed to get on-chain reputation: {e}")
                return self.reputations[client_id]
        else:
            return self.reputations[client_id]
    
    def update_reputation(self, client_id: str, performance: float) -> float:
        """
        Update reputation based on performance
        
        Args:
            client_id: Client identifier
            performance: Performance score (0-1)
                - 1.0: Verification passed
                - 0.5: Not verified
                - 0.0: Verification failed
        
        Returns:
            new_reputation: Updated reputation score
        """
        old_reputation = self.get_reputation(client_id)
        
        # Exponential moving average
        new_reputation = (
            old_reputation * self.decay_factor +
            performance * (1 - self.decay_factor)
        )
        
        # Clamp to [0, 1]
        new_reputation = max(0.0, min(1.0, new_reputation))
        
        # Update
        if self.chain_proxy:
            # Update on blockchain
            try:
                reputation_int = int(new_reputation * 1000)  # Convert to 0-1000
                self.chain_proxy.update_reputation(client_id, reputation_int)
            except Exception as e:
                logger.warning(f"Failed to update on-chain reputation: {e}")
        
        self.reputations[client_id] = new_reputation
        
        # Record history
        self.history[client_id].append({
            'timestamp': time.time(),
            'old_reputation': old_reputation,
            'new_reputation': new_reputation,
            'performance': performance
        })
        
        logger.info(f"Updated reputation for {client_id}: "
                   f"{old_reputation:.3f} -> {new_reputation:.3f} "
                   f"(performance={performance:.1f})")
        
        return new_reputation
    
    def batch_update_reputations(self, performances: Dict[str, float]) -> Dict[str, float]:
        """
        Batch update reputations
        
        Args:
            performances: Dictionary {client_id: performance}
        
        Returns:
            new_reputations: Dictionary {client_id: new_reputation}
        """
        new_reputations = {}
        for client_id, performance in performances.items():
            new_reputations[client_id] = self.update_reputation(client_id, performance)
        
        return new_reputations
    
    def get_verification_probability(self, client_id: str) -> float:
        """
        Calculate verification probability based on reputation
        
        Lower reputation -> Higher verification probability
        
        Args:
            client_id: Client identifier
        
        Returns:
            prob: Verification probability (0-1)
        """
        reputation = self.get_reputation(client_id)
        
        # Inverse relationship: low reputation -> high verification
        prob = self.base_verification_prob * (1 + (1 - reputation))
        
        # Clamp to [0, 1]
        prob = max(0.0, min(1.0, prob))
        
        logger.debug(f"Verification probability for {client_id}: {prob:.3f} "
                    f"(reputation={reputation:.3f})")
        
        return prob
    
    def get_trust_level(self, client_id: str) -> str:
        """
        Get trust level based on reputation
        
        Args:
            client_id: Client identifier
        
        Returns:
            trust_level: 'low', 'medium', or 'high'
        """
        reputation = self.get_reputation(client_id)
        
        if reputation < 0.3:
            return 'low'
        elif reputation < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def get_recovery_rate(self, client_id: str) -> float:
        """
        Get reputation recovery rate
        
        Lower reputation -> Slower recovery
        
        Args:
            client_id: Client identifier
        
        Returns:
            recovery_rate: Recovery rate per round
        """
        reputation = self.get_reputation(client_id)
        
        if reputation < 0.3:
            return 0.01  # Slow recovery
        elif reputation < 0.5:
            return 0.02  # Medium recovery
        else:
            return 0.03  # Fast recovery
    
    def apply_recovery(self, client_id: str) -> float:
        """
        Apply reputation recovery (for consecutive honest behavior)
        
        Args:
            client_id: Client identifier
        
        Returns:
            new_reputation: Updated reputation
        """
        recovery_rate = self.get_recovery_rate(client_id)
        old_reputation = self.get_reputation(client_id)
        new_reputation = min(1.0, old_reputation + recovery_rate)
        
        self.reputations[client_id] = new_reputation
        
        logger.info(f"Applied recovery for {client_id}: "
                   f"{old_reputation:.3f} -> {new_reputation:.3f} "
                   f"(rate={recovery_rate:.3f})")
        
        return new_reputation
    
    def get_history(self, client_id: str, limit: int = 10) -> List[Dict]:
        """
        Get reputation history for client
        
        Args:
            client_id: Client identifier
            limit: Maximum number of records to return
        
        Returns:
            history: List of history records
        """
        return self.history[client_id][-limit:]
    
    def get_statistics(self) -> Dict:
        """Get reputation statistics"""
        if not self.reputations:
            return {
                'num_clients': 0,
                'avg_reputation': 0.0,
                'min_reputation': 0.0,
                'max_reputation': 0.0
            }
        
        reputations = list(self.reputations.values())
        
        return {
            'num_clients': len(reputations),
            'avg_reputation': sum(reputations) / len(reputations),
            'min_reputation': min(reputations),
            'max_reputation': max(reputations),
            'low_trust_count': sum(1 for r in reputations if r < 0.3),
            'medium_trust_count': sum(1 for r in reputations if 0.3 <= r < 0.7),
            'high_trust_count': sum(1 for r in reputations if r >= 0.7)
        }
    
    def reset_reputation(self, client_id: str) -> None:
        """Reset reputation to initial value (for testing)"""
        self.reputations[client_id] = self.initial_reputation
        logger.warning(f"Reset reputation for {client_id} to {self.initial_reputation}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create reputation system
    rep_system = ReputationSystem(initial_reputation=0.5, decay_factor=0.9)
    
    # Simulate client behavior
    client_id = 'client1'
    
    print(f"Initial reputation: {rep_system.get_reputation(client_id):.3f}")
    print(f"Trust level: {rep_system.get_trust_level(client_id)}")
    print(f"Verification probability: {rep_system.get_verification_probability(client_id):.3f}")
    
    # Simulate verification results
    print("\n--- Simulating behavior ---")
    
    # Pass verification
    rep_system.update_reputation(client_id, 1.0)
    rep_system.update_reputation(client_id, 1.0)
    print(f"After 2 passes: {rep_system.get_reputation(client_id):.3f}")
    
    # Fail verification
    rep_system.update_reputation(client_id, 0.0)
    print(f"After 1 fail: {rep_system.get_reputation(client_id):.3f}")
    
    # Recovery
    rep_system.apply_recovery(client_id)
    print(f"After recovery: {rep_system.get_reputation(client_id):.3f}")
    
    # Statistics
    print(f"\n--- Statistics ---")
    stats = rep_system.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

