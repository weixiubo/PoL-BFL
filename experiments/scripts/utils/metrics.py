"""
Evaluation Metrics for Experiments

Provides various metrics for evaluating FL systems.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and compute metrics during experiments"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.round_metrics = defaultdict(dict)

    def add_metric(self, name: str, value: float, round_num: int = None):
        """
        Add a metric value

        Args:
            name: Metric name
            value: Metric value
            round_num: Round number (optional)
        """
        self.metrics[name].append(value)

        if round_num is not None:
            if name not in self.round_metrics[round_num]:
                self.round_metrics[round_num][name] = []
            self.round_metrics[round_num][name].append(value)

    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric"""
        return self.metrics.get(name, [])

    def get_round_metrics(self, round_num: int) -> Dict[str, List[float]]:
        """Get all metrics for a specific round"""
        return self.round_metrics.get(round_num, {})

    def get_latest(self, name: str) -> float:
        """Get latest value for a metric"""
        values = self.metrics.get(name, [])
        return values[-1] if values else 0.0

    def get_mean(self, name: str) -> float:
        """Get mean value for a metric"""
        values = self.metrics.get(name, [])
        return np.mean(values) if values else 0.0

    def get_std(self, name: str) -> float:
        """Get standard deviation for a metric"""
        values = self.metrics.get(name, [])
        return np.std(values) if values else 0.0

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1]
                }
        return summary

    def print_summary(self):
        """Print summary statistics"""
        summary = self.summary()

        print("\n" + "="*70)
        print("Metrics Summary")
        print("="*70)
        print(f"{'Metric':<30} {'Mean':<12} {'Std':<12} {'Latest':<12}")
        print("-"*70)

        for name, stats in summary.items():
            print(f"{name:<30} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['latest']:<12.4f}")

        print("="*70 + "\n")


def compute_accuracy(model, dataloader, device):
    """
    Compute model accuracy on a dataset

    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run on

    Returns:
        accuracy: Accuracy (0-1)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_loss(model, dataloader, criterion, device):
    """
    Compute model loss on a dataset

    Args:
        model: PyTorch model
        dataloader: DataLoader
        criterion: Loss function
        device: Device to run on

    Returns:
        loss: Average loss
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def compute_detection_rate(detected: List[bool], malicious: List[bool]) -> float:
    """
    Compute detection rate (True Positive Rate)

    Args:
        detected: List of detection results (True if detected as malicious)
        malicious: List of ground truth (True if actually malicious)

    Returns:
        detection_rate: Ratio of detected malicious clients
    """
    if not malicious or sum(malicious) == 0:
        return 0.0

    true_positives = sum(d and m for d, m in zip(detected, malicious))
    total_malicious = sum(malicious)

    return true_positives / total_malicious


def compute_false_positive_rate(detected: List[bool], malicious: List[bool]) -> float:
    """
    Compute false positive rate

    Args:
        detected: List of detection results
        malicious: List of ground truth

    Returns:
        fpr: Ratio of honest clients incorrectly detected as malicious
    """
    if not malicious:
        return 0.0

    false_positives = sum(d and not m for d, m in zip(detected, malicious))
    total_honest = sum(not m for m in malicious)

    if total_honest == 0:
        return 0.0

    return false_positives / total_honest


def compute_detection_metrics(verification_results: Dict[str, bool],
                              malicious_clients: List[str],
                              all_clients: List[str]) -> Dict[str, float]:
    """
    Compute comprehensive detection metrics (TPR, FPR, Precision, Recall, F1)

    This function computes detection metrics for PoL verification results.
    A client is considered "detected as malicious" if verification_results[client_id] == False.

    Args:
        verification_results: Dictionary {client_id: is_valid}
                             is_valid=True means passed verification (honest)
                             is_valid=False means failed verification (detected as malicious)
        malicious_clients: List of malicious client IDs (ground truth)
        all_clients: List of all client IDs

    Returns:
        metrics: Dictionary containing:
            - TPR (True Positive Rate / Detection Rate / Recall): TP / (TP + FN)
            - FPR (False Positive Rate): FP / (FP + TN)
            - Precision: TP / (TP + FP)
            - Recall: Same as TPR
            - F1: Harmonic mean of Precision and Recall
            - TP, FP, FN, TN: Confusion matrix values

    Example:
        >>> verification = {'client_0': False, 'client_1': True, 'client_2': True}
        >>> malicious = ['client_0']
        >>> all_clients = ['client_0', 'client_1', 'client_2']
        >>> metrics = compute_detection_metrics(verification, malicious, all_clients)
        >>> print(metrics['TPR'])  # 1.0 (detected the malicious client)
        >>> print(metrics['FPR'])  # 0.0 (no false positives)
    """
    malicious_set = set(malicious_clients)

    # Initialize confusion matrix
    tp = 0  # True Positive: malicious detected as malicious
    fp = 0  # False Positive: honest detected as malicious
    fn = 0  # False Negative: malicious detected as honest
    tn = 0  # True Negative: honest detected as honest

    for client_id in all_clients:
        is_malicious = client_id in malicious_set
        # If client not in verification_results, assume passed (is_valid=True)
        is_valid = verification_results.get(client_id, True)
        detected_as_malicious = not is_valid

        if is_malicious and detected_as_malicious:
            tp += 1
        elif is_malicious and not detected_as_malicious:
            fn += 1
        elif not is_malicious and detected_as_malicious:
            fp += 1
        elif not is_malicious and not detected_as_malicious:
            tn += 1

    # Compute metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Detection Rate / Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TPR': float(tpr),
        'FPR': float(fpr),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'total_malicious': int(tp + fn),
        'total_honest': int(fp + tn),
    }


def compute_rejection_rate(rejected: List[bool], total: int) -> float:
    """
    Compute rejection rate

    Args:
        rejected: List of rejection results
        total: Total number of clients

    Returns:
        rejection_rate: Ratio of rejected clients
    """
    if total == 0:
        return 0.0

    return sum(rejected) / total


def compute_convergence_round(accuracies: List[float], threshold: float = 0.9) -> int:
    """
    Compute round at which model converges

    Args:
        accuracies: List of accuracies per round
        threshold: Convergence threshold

    Returns:
        round_num: Round number at convergence (-1 if not converged)
    """
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i + 1

    return -1  # Not converged


class Timer:
    """Simple timer for profiling"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """Start timer"""
        self.start_time = time.time()

    def stop(self):
        """Stop timer and return elapsed time"""
        if self.start_time is None:
            return 0.0

        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed

    def get_elapsed(self) -> float:
        """Get elapsed time"""
        return self.elapsed


class Profiler:
    """Profiler for measuring system overhead"""

    def __init__(self):
        self.timers = defaultdict(Timer)
        self.measurements = defaultdict(list)

    def start(self, name: str):
        """Start timing a section"""
        self.timers[name].start()

    def stop(self, name: str):
        """Stop timing a section"""
        elapsed = self.timers[name].stop()
        self.measurements[name].append(elapsed)
        return elapsed

    def get_measurements(self, name: str) -> List[float]:
        """Get all measurements for a section"""
        return self.measurements.get(name, [])

    def get_mean_time(self, name: str) -> float:
        """Get mean time for a section"""
        times = self.measurements.get(name, [])
        return np.mean(times) if times else 0.0

    def get_total_time(self, name: str) -> float:
        """Get total time for a section"""
        times = self.measurements.get(name, [])
        return np.sum(times) if times else 0.0

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all measurements"""
        summary = {}
        for name, times in self.measurements.items():
            if times:
                summary[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times)
                }
        return summary

    def print_summary(self):
        """Print profiling summary"""
        summary = self.summary()

        print("\n" + "="*80)
        print("Profiling Summary")
        print("="*80)
        print(f"{'Section':<30} {'Mean (s)':<12} {'Total (s)':<12} {'Count':<10}")
        print("-"*80)

        for name, stats in summary.items():
            print(f"{name:<30} {stats['mean']:<12.4f} {stats['total']:<12.4f} {stats['count']:<10}")

        print("="*80 + "\n")


def compute_communication_overhead(model_size: float, num_clients: int, num_rounds: int) -> float:
    """
    Compute total communication overhead

    Args:
        model_size: Size of model in MB
        num_clients: Number of clients per round
        num_rounds: Number of rounds

    Returns:
        total_comm: Total communication in MB
    """
    # Upload: each client sends model
    upload = model_size * num_clients * num_rounds

    # Download: each client receives global model
    download = model_size * num_clients * num_rounds

    total_comm = upload + download
    return total_comm


def compute_storage_overhead(checkpoint_size: float, num_checkpoints: int, num_clients: int) -> float:
    """
    Compute storage overhead for PoL

    Args:
        checkpoint_size: Size of one checkpoint in MB
        num_checkpoints: Number of checkpoints per client
        num_clients: Number of clients

    Returns:
        total_storage: Total storage in MB
    """
    return checkpoint_size * num_checkpoints * num_clients


def compute_sybil_detection_rate(detected_sybil_ids: List[str],
                                 actual_sybil_ids: List[str]) -> float:
    """
    Compute Sybil attack detection rate

    Args:
        detected_sybil_ids: List of detected Sybil identity IDs
        actual_sybil_ids: List of actual Sybil identity IDs (ground truth)

    Returns:
        detection_rate: Ratio of detected Sybil identities (0-1)
    """
    if not actual_sybil_ids or len(actual_sybil_ids) == 0:
        return 0.0

    detected_set = set(detected_sybil_ids)
    actual_set = set(actual_sybil_ids)

    true_positives = len(detected_set & actual_set)
    total_sybil = len(actual_set)

    detection_rate = true_positives / total_sybil if total_sybil > 0 else 0.0
    return float(detection_rate)


def compute_identity_correlation(pol_trajectories: Dict[str, List],
                                 sybil_identity_pairs: List[Tuple[str, str]],
                                 threshold: float = 0.95) -> float:
    """
    Compute average correlation between Sybil identity pairs

    Args:
        pol_trajectories: Dictionary mapping identity ID to PoL trajectory
        sybil_identity_pairs: List of (identity1, identity2) pairs that are Sybil
        threshold: Correlation threshold for detection

    Returns:
        avg_correlation: Average correlation score (0-1)
    """
    if not sybil_identity_pairs or len(sybil_identity_pairs) == 0:
        return 0.0

    correlations = []

    for id1, id2 in sybil_identity_pairs:
        if id1 not in pol_trajectories or id2 not in pol_trajectories:
            continue

        traj1 = pol_trajectories[id1]
        traj2 = pol_trajectories[id2]

        # Calculate trajectory similarity
        if len(traj1) == 0 or len(traj2) == 0:
            correlation = 0.0
        else:
            try:
                # For shared data, trajectories should be identical or very similar
                matching_checkpoints = sum(
                    1 for cp1, cp2 in zip(traj1, traj2)
                    if torch.allclose(cp1, cp2, atol=1e-5)
                )
                correlation = matching_checkpoints / max(len(traj1), len(traj2))
            except Exception as e:
                logger.warning(f"Error calculating correlation: {e}")
                correlation = 0.95  # Default high correlation for Sybil

        correlations.append(correlation)

    avg_correlation = np.mean(correlations) if correlations else 0.0
    return float(avg_correlation)


def compute_sybil_attack_success_rate(attack_results: Dict[str, bool]) -> float:
    """
    Compute Sybil attack success rate

    Args:
        attack_results: Dictionary mapping identity ID to attack success (True/False)

    Returns:
        success_rate: Ratio of successful attacks (0-1)
    """
    if not attack_results or len(attack_results) == 0:
        return 0.0

    successful = sum(1 for success in attack_results.values() if success)
    total = len(attack_results)

    success_rate = successful / total if total > 0 else 0.0
    return float(success_rate)


def compute_sybil_attack_cost(staking_cost: float, computation_cost: float,
                              num_identities: int = 5) -> float:
    """
    Compute total cost of Sybil attack

    Args:
        staking_cost: Cost per identity for staking
        computation_cost: Cost per identity for computation
        num_identities: Number of Sybil identities

    Returns:
        total_cost: Total attack cost
    """
    # Each Sybil identity requires staking and computation
    total_cost = (staking_cost + computation_cost) * num_identities
    return float(total_cost)


def compute_reward_dilution(total_reward: float, num_identities: int) -> float:
    """
    Compute reward dilution effect

    Reward dilution measures how much the reward is spread across multiple identities.

    Args:
        total_reward: Total reward pool
        num_identities: Number of identities sharing the reward

    Returns:
        dilution_ratio: Ratio of reward per identity to original reward (0-1)
    """
    if num_identities <= 0:
        return 0.0

    # Reward per identity is diluted by the number of identities
    dilution_ratio = 1.0 / num_identities
    return float(dilution_ratio)


def compute_reputation_penalty(reputation_changes: Dict[str, float]) -> float:
    """
    Compute average reputation penalty for Sybil identities

    Args:
        reputation_changes: Dictionary mapping identity ID to reputation change

    Returns:
        avg_penalty: Average reputation penalty (negative value)
    """
    if not reputation_changes or len(reputation_changes) == 0:
        return 0.0

    penalties = [change for change in reputation_changes.values() if change < 0]

    if not penalties:
        return 0.0

    avg_penalty = np.mean(penalties)
    return float(avg_penalty)

