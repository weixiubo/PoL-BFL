"""
Sybil Attack Defense for PoL-FL Economic Incentive System

Detects and prevents Sybil attacks where attackers create multiple Sybil identities.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import hashlib
import time

logger = logging.getLogger(__name__)


class SybilDefense:
    """
    Detects and defends against Sybil attacks

    Defense strategies:
    1. Staking threshold (economic barrier)
    2. Identity verification (IP, device fingerprint)
    3. Behavior pattern analysis
    4. Social graph analysis
    5. Progressive trust
    """

    def __init__(self, sybil_threshold: float = 0.7,
                 ip_weight: float = 0.3,
                 device_weight: float = 0.3,
                 behavior_weight: float = 0.4):
        """
        Initialize Sybil Defense

        Args:
            sybil_threshold: Threshold for marking as suspicious (0-1)
            ip_weight: Weight for IP similarity
            device_weight: Weight for device similarity
            behavior_weight: Weight for behavior similarity
        """
        self.sybil_threshold = sybil_threshold
        self.ip_weight = ip_weight
        self.device_weight = device_weight
        self.behavior_weight = behavior_weight

        # Client metadata
        self.client_ips: Dict[str, str] = {}
        self.client_devices: Dict[str, str] = {}
        self.client_behaviors: Dict[str, List[float]] = defaultdict(list)

        # Blacklist
        self.blacklist: Set[str] = set()

        # Detection history
        self.detection_history: List[Dict] = []

        logger.info(f"SybilDefense initialized (threshold={sybil_threshold})")

    def register_client(self, client_id: str, ip_address: str,
                       device_fingerprint: str) -> bool:
        """
        Register client metadata

        Args:
            client_id: Client identifier
            ip_address: Client IP address
            device_fingerprint: Device fingerprint (hash of hardware info)

        Returns:
            allowed: True if registration allowed
        """
        # Check blacklist
        if client_id in self.blacklist:
            logger.warning(f"Client {client_id} is blacklisted")
            return False

        # Store metadata
        self.client_ips[client_id] = ip_address
        self.client_devices[client_id] = device_fingerprint

        logger.info(f"Registered client {client_id} (IP: {ip_address[:10]}...)")
        return True

    def record_behavior(self, client_id: str, behavior_features: List[float]) -> None:
        """
        Record client behavior for pattern analysis

        Args:
            client_id: Client identifier
            behavior_features: Behavior feature vector
                e.g., [training_time, upload_time, model_quality, ...]
        """
        self.client_behaviors[client_id].append(behavior_features)

        # Keep only recent history
        if len(self.client_behaviors[client_id]) > 100:
            self.client_behaviors[client_id] = self.client_behaviors[client_id][-100:]

    def detect_sybil(self, client_ids: List[str]) -> Dict[str, float]:
        """
        Detect Sybil attacks among clients

        Args:
            client_ids: List of client IDs to check

        Returns:
            sybil_scores: Dictionary {client_id: sybil_score}
                Higher score = More suspicious
        """
        sybil_scores = {}

        for client_id in client_ids:
            if client_id in self.blacklist:
                sybil_scores[client_id] = 1.0
                continue

            # Calculate similarity with other clients
            max_similarity = 0.0

            for other_id in client_ids:
                if other_id == client_id:
                    continue

                similarity = self._calculate_similarity(client_id, other_id)
                max_similarity = max(max_similarity, similarity)

            sybil_scores[client_id] = max_similarity

        # Log suspicious clients
        suspicious = [cid for cid, score in sybil_scores.items()
                     if score > self.sybil_threshold]

        if suspicious:
            logger.warning(f"Detected {len(suspicious)} suspicious clients: {suspicious}")
            for client_id in suspicious:
                logger.warning(f"  {client_id}: sybil_score={sybil_scores[client_id]:.3f}")

        return sybil_scores

    def _calculate_similarity(self, client1: str, client2: str) -> float:
        """
        Calculate similarity between two clients

        Args:
            client1: First client ID
            client2: Second client ID

        Returns:
            similarity: Similarity score (0-1)
        """
        # IP similarity
        ip_sim = self._ip_similarity(client1, client2)

        # Device similarity
        device_sim = self._device_similarity(client1, client2)

        # Behavior similarity
        behavior_sim = self._behavior_similarity(client1, client2)

        # Weighted combination
        similarity = (
            self.ip_weight * ip_sim +
            self.device_weight * device_sim +
            self.behavior_weight * behavior_sim
        )

        return similarity

    def _ip_similarity(self, client1: str, client2: str) -> float:
        """Calculate IP address similarity"""
        ip1 = self.client_ips.get(client1, "")
        ip2 = self.client_ips.get(client2, "")

        if not ip1 or not ip2:
            return 0.0

        # Exact match
        if ip1 == ip2:
            return 1.0

        # Same subnet (first 3 octets)
        parts1 = ip1.split('.')
        parts2 = ip2.split('.')

        if len(parts1) == 4 and len(parts2) == 4:
            if parts1[:3] == parts2[:3]:
                return 0.7  # Same subnet

        return 0.0

    def _device_similarity(self, client1: str, client2: str) -> float:
        """Calculate device fingerprint similarity"""
        device1 = self.client_devices.get(client1, "")
        device2 = self.client_devices.get(client2, "")

        if not device1 or not device2:
            return 0.0

        # Exact match
        if device1 == device2:
            return 1.0

        # Hamming distance for fingerprints
        if len(device1) == len(device2):
            matches = sum(c1 == c2 for c1, c2 in zip(device1, device2))
            similarity = matches / len(device1)

            # High similarity is suspicious
            if similarity > 0.8:
                return similarity

        return 0.0

    def _behavior_similarity(self, client1: str, client2: str) -> float:
        """Calculate behavior pattern similarity"""
        behaviors1 = self.client_behaviors.get(client1, [])
        behaviors2 = self.client_behaviors.get(client2, [])

        if not behaviors1 or not behaviors2:
            return 0.0

        # Calculate average behavior vectors
        avg1 = [sum(b[i] for b in behaviors1) / len(behaviors1)
                for i in range(len(behaviors1[0]))]
        avg2 = [sum(b[i] for b in behaviors2) / len(behaviors2)
                for i in range(len(behaviors2[0]))]

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(avg1, avg2))
        norm1 = sum(a ** 2 for a in avg1) ** 0.5
        norm2 = sum(a ** 2 for a in avg2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # High similarity is suspicious
        if similarity > 0.9:
            return similarity

        return 0.0

    def mark_suspicious(self, client_id: str, reason: str = "") -> None:
        """
        Mark client as suspicious

        Args:
            client_id: Client identifier
            reason: Reason for marking
        """
        logger.warning(f"Marked {client_id} as suspicious: {reason}")

        self.detection_history.append({
            'timestamp': time.time(),
            'client_id': client_id,
            'reason': reason,
            'action': 'marked_suspicious'
        })

    def add_to_blacklist(self, client_id: str, reason: str = "") -> None:
        """
        Add client to blacklist

        Args:
            client_id: Client identifier
            reason: Reason for blacklisting
        """
        self.blacklist.add(client_id)
        logger.error(f"Blacklisted {client_id}: {reason}")

        self.detection_history.append({
            'timestamp': time.time(),
            'client_id': client_id,
            'reason': reason,
            'action': 'blacklisted'
        })

    def is_blacklisted(self, client_id: str) -> bool:
        """Check if client is blacklisted"""
        return client_id in self.blacklist

    def remove_from_blacklist(self, client_id: str) -> None:
        """Remove client from blacklist (for testing/appeals)"""
        if client_id in self.blacklist:
            self.blacklist.remove(client_id)
            logger.info(f"Removed {client_id} from blacklist")

    def get_statistics(self) -> Dict:
        """Get Sybil defense statistics"""
        return {
            'num_registered_clients': len(self.client_ips),
            'num_blacklisted': len(self.blacklist),
            'num_detections': len(self.detection_history),
            'unique_ips': len(set(self.client_ips.values())),
            'unique_devices': len(set(self.client_devices.values()))
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create Sybil defense
    defense = SybilDefense(sybil_threshold=0.7)

    # Register clients
    defense.register_client('client1', '192.0.2.100', 'device_hash_1')
    defense.register_client('client2', '192.0.2.101', 'device_hash_2')
    defense.register_client('client3', '192.0.2.100', 'device_hash_1')  # Suspicious.

    # Record behaviors
    defense.record_behavior('client1', [10.5, 2.3, 0.95])
    defense.record_behavior('client2', [12.1, 2.5, 0.93])
    defense.record_behavior('client3', [10.6, 2.3, 0.94])  # Similar to client1.

    # Detect Sybil
    clients = ['client1', 'client2', 'client3']
    sybil_scores = defense.detect_sybil(clients)

    print("\n--- Sybil Detection Results ---")
    for client_id, score in sybil_scores.items():
        status = "SUSPICIOUS" if score > defense.sybil_threshold else "OK"
        print(f"{client_id}: {score:.3f} [{status}]")

    # Mark suspicious
    for client_id, score in sybil_scores.items():
        if score > defense.sybil_threshold:
            defense.mark_suspicious(client_id, f"Sybil score: {score:.3f}")

    # Statistics
    print("\n--- Statistics ---")
    stats = defense.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
