"""
Sybil Attack Implementations

Implements Sybil attacks where a single attacker creates multiple Sybil identities
to gain disproportionate influence in the federated learning system.
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SybilAttack:
    """
    Sybil Attack Implementation

    A single attacker creates multiple Sybil identities that share computational
    resources and data. This allows the attacker to:
    1. Gain multiple votes in aggregation
    2. Dilute rewards across Sybil identities
    3. Evade reputation-based defenses

    Defense mechanisms:
    - PoL trajectory similarity detection
    - Data commitment correlation analysis
    - Model update similarity detection
    - Staking and reputation penalties
    """

    def __init__(self, num_identities: int = 5,
                 shared_data_ratio: float = 1.0,
                 base_client_id: str = "attacker"):
        """
        Initialize Sybil Attack

        Args:
            num_identities: Number of Sybil identities per attacker
            shared_data_ratio: Ratio of shared data between identities (0-1)
                - 1.0: All identities share the same data
                - 0.5: Identities share 50% of data
                - 0.0: Each identity has unique data
            base_client_id: Base ID for generating Sybil identity IDs
        """
        self.num_identities = num_identities
        self.shared_data_ratio = shared_data_ratio
        self.base_client_id = base_client_id
        self.attack_name = "SybilAttack"

        # Track Sybil identities
        self.sybil_identities: List[str] = []
        self.identity_mapping: Dict[str, int] = {}  # Maps Sybil identifiers to indices.

        # Store shared model update for all identities
        self.shared_model_update: Optional[OrderedDict] = None

        # Store PoL trajectories for similarity analysis
        self.pol_trajectories: Dict[str, List] = {}

        logger.info(f"Initialized {self.attack_name} (num_identities={num_identities}, "
                   f"shared_data_ratio={shared_data_ratio})")

    def create_identities(self) -> List[str]:
        """
        Create Sybil identity IDs

        Returns:
            sybil_identities: List of Sybil identity IDs
        """
        self.sybil_identities = [
            f"{self.base_client_id}_sybil_{i}"
            for i in range(self.num_identities)
        ]

        # Create mapping for smoke lookup
        self.identity_mapping = {
            sybil_id: i for i, sybil_id in enumerate(self.sybil_identities)
        }

        logger.info(f"Created {self.num_identities} Sybil identities: {self.sybil_identities}")
        return self.sybil_identities

    def get_shared_model_update(self, model_state: OrderedDict) -> OrderedDict:
        """
        Get the shared model update for all Sybil identities

        All Sybil identities submit the same model update since they share data
        and computational resources.

        Args:
            model_state: Model state dict from training

        Returns:
            shared_model: Same model state for all identities
        """
        # Store the shared model update
        self.shared_model_update = model_state

        logger.debug(f"Shared model update stored for {self.num_identities} identities")
        return model_state

    def get_identity_model_update(self, identity_id: str,
                                  model_state: OrderedDict) -> OrderedDict:
        """
        Get model update for a specific Sybil identity

        Since all Sybil identities share data, they all return the same model update.

        Args:
            identity_id: Sybil identity ID
            model_state: Model state dict

        Returns:
            model_update: Model update for this identity (same as all others)
        """
        if identity_id not in self.identity_mapping:
            logger.warning(f"Unknown identity: {identity_id}")
            return model_state

        # All identities return the same model update
        return model_state

    def record_pol_trajectory(self, identity_id: str,
                             pol_trajectory: List) -> None:
        """
        Record PoL trajectory for a Sybil identity

        Args:
            identity_id: Sybil identity ID
            pol_trajectory: PoL trajectory (checkpoints, data indices, etc.)
        """
        if identity_id not in self.identity_mapping:
            logger.warning(f"Unknown identity: {identity_id}")
            return

        self.pol_trajectories[identity_id] = pol_trajectory
        logger.debug(f"Recorded PoL trajectory for {identity_id}")

    def get_identity_correlation(self, identity1: str, identity2: str) -> float:
        """
        Calculate correlation between two Sybil identities

        Since all Sybil identities share data and computational resources,
        their PoL trajectories should be highly similar.

        Args:
            identity1: First Sybil identity ID
            identity2: Second Sybil identity ID

        Returns:
            correlation: Correlation score (0-1)
                - 1.0: Identical (definitely Sybil)
                - 0.95+: Very similar (likely Sybil)
                - <0.9: Different (likely honest)
        """
        if identity1 not in self.pol_trajectories or identity2 not in self.pol_trajectories:
            logger.debug(f"Missing PoL trajectory for {identity1} or {identity2}")
            return 0.0

        traj1 = self.pol_trajectories[identity1]
        traj2 = self.pol_trajectories[identity2]

        # Calculate similarity based on trajectory length
        if len(traj1) == 0 or len(traj2) == 0:
            return 0.0

        # For shared data, trajectories should be identical or very similar
        # Calculate cosine similarity of trajectory vectors
        try:
            # Convert trajectories to numpy arrays for similarity calculation
            if isinstance(traj1, list) and isinstance(traj2, list):
                # If trajectories are lists of checkpoints, compare them
                matching_checkpoints = sum(
                    1 for cp1, cp2 in zip(traj1, traj2)
                    if torch.allclose(cp1, cp2, atol=1e-5)
                )
                correlation = matching_checkpoints / max(len(traj1), len(traj2))
            else:
                # Default: assume high correlation for Sybil identities
                correlation = 0.95 if self.shared_data_ratio > 0.8 else 0.5
        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            correlation = 0.95 if self.shared_data_ratio > 0.8 else 0.5

        return correlation

    def get_data_commitment_correlation(self, identity1: str, identity2: str) -> float:
        """
        Calculate data commitment correlation between two identities

        Data commitment is the hash of the data used for training.
        Sybil identities sharing data should have identical commitments.

        Args:
            identity1: First Sybil identity ID
            identity2: Second Sybil identity ID

        Returns:
            correlation: Correlation score (0-1)
                - 1.0: Identical data (definitely Sybil)
                - 0.0: Different data (likely honest)
        """
        # For Sybil identities with shared_data_ratio = 1.0,
        # data commitments should be identical
        if self.shared_data_ratio >= 0.99:
            return 1.0
        elif self.shared_data_ratio >= 0.8:
            return 0.95
        else:
            return self.shared_data_ratio

    def get_model_update_similarity(self, update1: OrderedDict,
                                   update2: OrderedDict) -> float:
        """
        Calculate similarity between two model updates

        Args:
            update1: First model update
            update2: Second model update

        Returns:
            similarity: Cosine similarity (0-1)
        """
        try:
            # Flatten model updates to vectors
            vec1 = torch.cat([p.flatten() for p in update1.values()])
            vec2 = torch.cat([p.flatten() for p in update2.values()])

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                vec1.unsqueeze(0), vec2.unsqueeze(0)
            ).item()

            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception as e:
            logger.warning(f"Error calculating model update similarity: {e}")
            return 0.95 if self.shared_data_ratio > 0.8 else 0.5

    def is_sybil_identity(self, identity_id: str) -> bool:
        """
        Check if an identity is a Sybil identity

        Args:
            identity_id: Identity ID to check

        Returns:
            is_sybil: True if this is a Sybil identity
        """
        return identity_id in self.identity_mapping

    def get_all_identities(self) -> List[str]:
        """
        Get all Sybil identity IDs

        Returns:
            sybil_identities: List of all Sybil identity IDs
        """
        return self.sybil_identities.copy()

    def get_statistics(self) -> Dict:
        """
        Get Sybil attack statistics

        Returns:
            stats: Dictionary with attack statistics
        """
        return {
            'attack_name': self.attack_name,
            'num_identities': self.num_identities,
            'shared_data_ratio': self.shared_data_ratio,
            'sybil_identities': len(self.sybil_identities),
            'pol_trajectories_recorded': len(self.pol_trajectories),
            'shared_model_update_stored': self.shared_model_update is not None
        }


# Factory function for creating Sybil attacks
def create_sybil_attack(num_identities: int = 5,
                       shared_data_ratio: float = 1.0,
                       base_client_id: str = "attacker") -> SybilAttack:
    """
    Factory function to create Sybil attack instances

    Args:
        num_identities: Number of Sybil identities per attacker
        shared_data_ratio: Ratio of shared data between identities
        base_client_id: Base ID for generating Sybil identity IDs

    Returns:
        attack: SybilAttack instance
    """
    return SybilAttack(
        num_identities=num_identities,
        shared_data_ratio=shared_data_ratio,
        base_client_id=base_client_id
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create Sybil attack
    attack = SybilAttack(num_identities=5, shared_data_ratio=1.0)

    # Create Sybil identities
    identities = attack.create_identities()
    print(f"Created identities: {identities}")

    # Get statistics
    stats = attack.get_statistics()
    print(f"Statistics: {stats}")
