"""
Baseline Methods for Comparison

Implements various baseline aggregation methods for comparison with PoL-FL.
"""

import torch
import logging
import numpy as np
import os
from typing import List, Dict, OrderedDict
from collections import OrderedDict as ODict

logger = logging.getLogger(__name__)


def _max_vector_dim() -> int:
    try:
        return max(1, int(os.getenv("ROBUST_VECTOR_MAX_DIM", "50000")))
    except Exception:
        return 50000


def _flatten_model_sampled(model: ODict, max_dim: int = None) -> torch.Tensor:
    """Flatten model state and deterministically subsample for robust scorers."""
    max_dim = _max_vector_dim() if max_dim is None else int(max_dim)
    parts = []
    for key in sorted(model.keys()):
        tensor = model[key]
        if not torch.is_tensor(tensor):
            continue
        vec = tensor.detach().float().cpu().reshape(-1)
        if not torch.isfinite(vec).all():
            vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(vec)
    flat = torch.cat(parts) if parts else torch.empty(0)
    if flat.numel() > max_dim:
        idx = torch.linspace(0, flat.numel() - 1, steps=max_dim).long()
        flat = flat[idx]
    return flat


def _flatten_delta_sampled(model: ODict, reference: ODict, max_dim: int = None) -> torch.Tensor:
    """Flatten floating-point model deltas against a reference model."""
    max_dim = _max_vector_dim() if max_dim is None else int(max_dim)
    parts = []
    for key in sorted(model.keys()):
        tensor = model[key]
        ref = reference.get(key) if isinstance(reference, dict) else None
        if not (torch.is_tensor(tensor) and torch.is_tensor(ref)):
            continue
        if not tensor.is_floating_point() or not ref.is_floating_point():
            continue
        if tensor.shape != ref.shape:
            continue
        vec = (tensor.detach().float().cpu() - ref.detach().float().cpu()).reshape(-1)
        if not torch.isfinite(vec).all():
            vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(vec)
    flat = torch.cat(parts) if parts else torch.empty(0)
    if flat.numel() > max_dim:
        idx = torch.linspace(0, flat.numel() - 1, steps=max_dim).long()
        flat = flat[idx]
    return flat


class BaselineAggregator:
    """Base class for baseline aggregation methods"""

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.selected_indices = []
        self.rejected_indices = []
        self.scores = []
        self.client_weights = []
        logger.info(f"Initialized {method_name}")

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Aggregate models

        Args:
            models: List of model state dicts
            weights: List of aggregation weights (default: equal weights)

        Returns:
            aggregated_model: Aggregated model state dict
        """
        raise NotImplementedError


class VanillaFLAggregator(BaselineAggregator):
    """
    Vanilla Federated Learning (FedAvg)

    Simple weighted average of client models.
    """

    def __init__(self):
        super().__init__("Vanilla_FL")

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        FedAvg: Weighted average of models

        Args:
            models: List of model state dicts
            weights: List of weights (default: equal weights)

        Returns:
            aggregated_model: Averaged model
        """
        if not models:
            raise ValueError("No models to aggregate")
        self.selected_indices = list(range(len(models)))
        self.rejected_indices = []
        self.scores = []

        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        self.client_weights = [float(w) for w in weights]

        # Initialize aggregated model
        aggregated_model = ODict()

        # Weighted average
        for key in models[0].keys():
            aggregated_model[key] = sum(
                w * model[key] for w, model in zip(weights, models)
            )

        logger.debug(f"Aggregated {len(models)} models using FedAvg")
        return aggregated_model


class KrumAggregator(BaselineAggregator):
    """
    Krum Aggregation

    Byzantine-robust aggregation that selects the model closest to others.
    Reference: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    """

    def __init__(
        self,
        num_byzantine: int = 0,
        multi_krum: bool = False,
        multi_krum_count: int = None,
    ):
        super().__init__("Krum")
        self.num_byzantine = num_byzantine
        self.multi_krum = bool(multi_krum)
        self.multi_krum_count = multi_krum_count

    def _compute_squared_distance(self, model1: ODict, model2: ODict) -> float:
        """Compute squared L2 distance between two model updates.

        Krum scores are defined as sums of squared Euclidean distances to the
        closest neighbors.  Taking the square root per pair before summing can
        change the selected client because sqrt is nonlinear.
        """
        distance = 0.0
        for key in model1.keys():
            tensor1 = model1[key]
            tensor2 = model2[key]
            if not torch.is_tensor(tensor1) or not torch.is_tensor(tensor2):
                continue
            if not tensor1.is_floating_point() or not tensor2.is_floating_point():
                continue
            diff = tensor1.detach().float().cpu() - tensor2.detach().float().cpu()
            if not torch.isfinite(diff).all():
                diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            distance += torch.sum(diff ** 2).item()
        return float(distance)

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Krum: Select model with smallest sum of distances to closest models

        Args:
            models: List of model state dicts
            weights: Not used in Krum

        Returns:
            selected_model: Selected model (not averaged)
        """
        if not models:
            raise ValueError("No models to aggregate")

        n = len(models)
        f = self.num_byzantine

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_squared_distance(models[i], models[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each model, compute sum of squared distances to n-f-2 closest models.
        # Krum is formally defined for n > 2f + 2; small smoke tests may violate
        # this, so clamp the neighbor count instead of silently producing all-zero
        # scores from an empty slice.
        neighbor_count = max(1, min(n - 1, n - f - 2))
        scores = []
        for i in range(n):
            # Sort distances for this model
            sorted_distances = np.sort(distances[i])
            # Sum of closest distances (excluding self)
            score = np.sum(sorted_distances[1:1 + neighbor_count])
            scores.append(score)

        # Select model with smallest score.  In Multi-Krum mode, aggregate the
        # lowest-scoring safe set instead of using a single client update; this
        # is the common stable FL variant of Krum for large client cohorts.
        selected_idx = int(np.argmin(scores))
        self.selected_index = int(selected_idx)
        self.scores = [float(score) for score in scores]

        if self.multi_krum:
            if self.multi_krum_count is None:
                keep_count = max(1, min(n - f - 2, n - 1))
            else:
                keep_count = max(1, min(int(self.multi_krum_count), n))
            selected = np.argsort(scores)[:keep_count].tolist()
            self.selected_indices = [int(i) for i in selected]
            selected_set = set(self.selected_indices)
            self.rejected_indices = [i for i in range(n) if i not in selected_set]
            if weights is not None:
                selected_weights = [float(weights[i]) for i in self.selected_indices]
                total = sum(selected_weights)
                if total <= 0.0:
                    selected_weights = [1.0 / len(self.selected_indices)] * len(self.selected_indices)
                else:
                    selected_weights = [w / total for w in selected_weights]
            else:
                selected_weights = [1.0 / len(self.selected_indices)] * len(self.selected_indices)
            self.client_weights = [0.0] * n
            for pos, idx in enumerate(self.selected_indices):
                self.client_weights[idx] = float(selected_weights[pos])

            aggregated_model = ODict()
            for key in models[0].keys():
                aggregated_model[key] = sum(
                    selected_weights[pos] * models[idx][key]
                    for pos, idx in enumerate(self.selected_indices)
                )
            logger.debug(
                f"Multi-Krum selected {len(self.selected_indices)}/{n} models "
                f"(best={selected_idx}, score={scores[selected_idx]:.4f})"
            )
            return aggregated_model

        self.selected_indices = [int(selected_idx)]
        self.rejected_indices = [i for i in range(n) if i != int(selected_idx)]
        self.client_weights = [0.0] * n
        self.client_weights[int(selected_idx)] = 1.0

        logger.debug(f"Krum selected model {selected_idx} (score={scores[selected_idx]:.4f})")
        return models[selected_idx]


class TrimmedMeanAggregator(BaselineAggregator):
    """
    Trimmed Mean Aggregation

    Byzantine-robust aggregation that removes extreme values before averaging.
    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
    """

    def __init__(self, trim_ratio: float = 0.1):
        super().__init__("Trimmed_Mean")
        self.trim_ratio = trim_ratio

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Trimmed Mean: Remove extreme values and average the rest

        Args:
            models: List of model state dicts
            trim_ratio: Ratio of values to trim from each end

        Returns:
            aggregated_model: Trimmed mean model
        """
        if not models:
            raise ValueError("No models to aggregate")
        self.selected_indices = list(range(len(models)))
        self.rejected_indices = []
        self.scores = []

        n = len(models)
        # 修正: 确保至少trim 1个值，避免trim_ratio太小时num_trim=0
        num_trim = max(1, int(n * self.trim_ratio))

        # 确保trim后至少剩下1个模型
        if num_trim >= n // 2:
            num_trim = max(0, n // 2 - 1)

        aggregated_model = ODict()

        for key in models[0].keys():
            # Stack parameters from all models
            params = torch.stack([model[key] for model in models])

            # Sort along the model dimension
            sorted_params, _ = torch.sort(params, dim=0)

            # Trim extreme values
            if num_trim > 0 and num_trim < n // 2:
                trimmed_params = sorted_params[num_trim:-num_trim]
            else:
                trimmed_params = sorted_params

            # Average trimmed parameters
            # Convert to float if needed to avoid dtype issues
            if trimmed_params.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                trimmed_params = trimmed_params.float()
            aggregated_model[key] = torch.mean(trimmed_params, dim=0)

        logger.debug(f"Aggregated {len(models)} models using Trimmed Mean (trim_ratio={self.trim_ratio}, num_trim={num_trim})")
        return aggregated_model


class MedianAggregator(BaselineAggregator):
    """
    Median Aggregation

    Byzantine-robust aggregation using coordinate-wise median.
    """

    def __init__(self):
        super().__init__("Median")

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Median: Coordinate-wise median of models

        Args:
            models: List of model state dicts

        Returns:
            aggregated_model: Median model
        """
        if not models:
            raise ValueError("No models to aggregate")
        self.selected_indices = list(range(len(models)))
        self.rejected_indices = []
        self.scores = []

        aggregated_model = ODict()

        for key in models[0].keys():
            # Stack parameters from all models
            params = torch.stack([model[key] for model in models])

            # Compute coordinate-wise median
            # torch.median has no deterministic CUDA implementation.
            # Move to CPU for deterministic behavior when needed
            original_device = params.device
            if params.is_cuda and torch.are_deterministic_algorithms_enabled():
                params_cpu = params.cpu()
                median_result = torch.median(params_cpu, dim=0)[0]
                aggregated_model[key] = median_result.to(original_device)
            else:
                aggregated_model[key] = torch.median(params, dim=0)[0]

        logger.debug(f"Aggregated {len(models)} models using Median")
        return aggregated_model


class BulyanAggregator(BaselineAggregator):
    """
    Bulyan Aggregation

    Combines Krum with trimmed mean for stronger Byzantine robustness.
    Reference: El Mhamdi et al., "The Hidden Vulnerability of Distributed Learning in Byzantium"
    """

    def __init__(self, num_byzantine: int = 0):
        super().__init__("Bulyan")
        self.num_byzantine = num_byzantine
        self.krum = KrumAggregator(num_byzantine)

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Bulyan: Apply Krum multiple times, then trimmed mean

        Args:
            models: List of model state dicts

        Returns:
            aggregated_model: Bulyan aggregated model
        """
        if not models:
            raise ValueError("No models to aggregate")
        self.selected_indices = list(range(len(models)))
        self.rejected_indices = []
        self.scores = []

        n = len(models)
        f = self.num_byzantine

        # Select n-2f models using Krum
        selected_models = []
        remaining_models = models.copy()

        for _ in range(n - 2 * f):
            # Run Krum on remaining models
            selected = self.krum.aggregate(remaining_models)
            selected_models.append(selected)

            # Remove selected model from remaining
            remaining_models = [m for m in remaining_models if m is not selected]

        # Apply trimmed mean on selected models
        trimmed_mean = TrimmedMeanAggregator(trim_ratio=0.1)
        aggregated_model = trimmed_mean.aggregate(selected_models)

        logger.debug(f"Aggregated {len(models)} models using Bulyan")
        return aggregated_model


class ShapleyFLAggregator(BaselineAggregator):
    """
    ShapleyFL Aggregator (Simplified for Experiments)

    Uses Shapley values to assess client contributions and filter malicious clients.
    Paper: "ShapleyFL: Robust Federated Learning Based on Shapley Value" (KDD 2023)

    This compatibility variant operates without validation data.
    It uses gradient similarity as a proxy for contribution.
    """

    def __init__(self, threshold_percentile: float = 0.0, num_byzantine: int = 0):
        super().__init__("ShapleyFL")
        self.threshold_percentile = threshold_percentile
        self.num_byzantine = max(0, int(num_byzantine))

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        ShapleyFL: Filter clients based on contribution similarity

        Args:
            models: List of model state dicts
            weights: Not used

        Returns:
            aggregated_model: Filtered and averaged model
        """
        if not models:
            raise ValueError("No models to aggregate")

        model_vectors = [_flatten_model_sampled(model) for model in models]
        if any(vec.numel() == 0 for vec in model_vectors):
            logger.warning("ShapleyFL: Empty model vector detected, falling back to FedAvg")
            self.selected_indices = list(range(len(models)))
            self.rejected_indices = []
            self.scores = [0.0] * len(models)
            self.client_weights = [1.0 / len(models)] * len(models)
            aggregated_model = ODict()
            for key in models[0].keys():
                aggregated_model[key] = sum(model[key] for model in models) / len(models)
            return aggregated_model
        model_matrix = torch.stack(model_vectors, dim=0)
        center = torch.median(model_matrix, dim=0).values
        distances = torch.norm(model_matrix - center.unsqueeze(0), p=2, dim=1)
        if not torch.isfinite(distances).all():
            logger.warning("ShapleyFL: NaN/Inf detected in contribution scores, falling back to FedAvg")
            self.selected_indices = list(range(len(models)))
            self.rejected_indices = []
            self.scores = [0.0] * len(models)
            self.client_weights = [1.0 / len(models)] * len(models)
            aggregated_model = ODict()
            for key in models[0].keys():
                aggregated_model[key] = sum(model[key] for model in models) / len(models)
            return aggregated_model
        # Higher score means closer to the robust center and therefore a more
        # plausible positive contribution under the current no-validation proxy.
        similarities = (-distances).detach().cpu().numpy()

        if self.num_byzantine > 0:
            keep = max(1, len(models) - min(self.num_byzantine, len(models) - 1))
            selected_indices = np.argsort(similarities)[-keep:].tolist()
        elif self.threshold_percentile > 0:
            threshold = np.percentile(similarities, self.threshold_percentile)
            selected_indices = [i for i in range(len(models)) if similarities[i] >= threshold]
        else:
            selected_indices = list(range(len(models)))

        if len(selected_indices) == 0:
            logger.warning("ShapleyFL: No clients selected, using all")
            selected_indices = list(range(len(models)))
        self.selected_indices = [int(i) for i in selected_indices]
        selected_set = set(self.selected_indices)
        self.rejected_indices = [i for i in range(len(models)) if i not in selected_set]
        self.scores = [float(score) for score in similarities]

        # Aggregate the accepted clients with the original FedAvg data weights
        # when available.  Contribution scores are used for filtering; using
        # negative distance values as aggregation weights is unstable.
        if weights is not None:
            selected_weights = np.array([float(weights[i]) for i in selected_indices], dtype=float)
            total = float(np.sum(selected_weights))
            if total > 0.0:
                selected_weights = selected_weights / total
            else:
                selected_weights = np.ones(len(selected_indices)) / len(selected_indices)
        else:
            selected_weights = np.ones(len(selected_indices)) / len(selected_indices)
        full_weights = np.zeros(len(models), dtype=float)
        for pos, idx in enumerate(selected_indices):
            full_weights[int(idx)] = float(selected_weights[pos])
        self.client_weights = full_weights.tolist()

        aggregated_model = ODict()
        for key in models[0].keys():
            aggregated_model[key] = sum(
                selected_weights[i] * models[selected_indices[i]][key]
                for i in range(len(selected_indices))
            )

        logger.debug(f"ShapleyFL: Selected {len(selected_indices)}/{len(models)} clients")
        return aggregated_model


class FoolsGoldAggregator(BaselineAggregator):
    """
    FoolsGold Aggregator (Simplified for Experiments)

    Detects Sybil attacks by analyzing gradient similarity.
    Paper: "Defending Against Sybils in Federated Learning" (RAID 2020)

    Note: This is a simplified version without cumulative history.
    """

    def __init__(self):
        super().__init__("FoolsGold")
        self.summed_deltas = None
        self.reference_model = None
        self.suspicion_scores = []

    def set_reference_model(self, reference_model: ODict) -> None:
        self.reference_model = reference_model

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        FoolsGold: Weight clients inversely to gradient similarity

        Args:
            models: List of model state dicts
            weights: Not used

        Returns:
            aggregated_model: FoolsGold weighted model
        """
        if not models:
            raise ValueError("No models to aggregate")

        from sklearn.metrics.pairwise import cosine_similarity

        # FoolsGold is defined on client gradients/updates, not full model
        # states.  Full state vectors are dominated by the shared global model
        # and can make benign clients look spuriously similar.
        model_vectors = []
        for model in models:
            if self.reference_model is not None:
                vec = _flatten_delta_sampled(model, self.reference_model).numpy()
            else:
                vec = _flatten_model_sampled(model).numpy()
            model_vectors.append(vec)

        model_matrix = np.array(model_vectors)
        if model_matrix.size == 0:
            logger.warning("FoolsGold: empty update vectors, falling back to FedAvg weights")
            if weights is None:
                wv = np.ones(len(models), dtype=float) / len(models)
            else:
                wv = np.array(weights, dtype=float)
                wv = wv / np.sum(wv) if np.sum(wv) > 0 else np.ones(len(models), dtype=float) / len(models)
            self.client_weights = [float(w) for w in wv]
            self.scores = self.client_weights.copy()
            self.selected_indices = list(range(len(models)))
            self.rejected_indices = []
            aggregated_model = ODict()
            for key in models[0].keys():
                if torch.is_tensor(models[0][key]) and models[0][key].is_floating_point():
                    aggregated_model[key] = sum(wv[i] * models[i][key] for i in range(len(models)))
                else:
                    aggregated_model[key] = models[0][key]
            return aggregated_model

        current_deltas = model_matrix
        center_delta = np.median(current_deltas, axis=0)
        center_distances = np.linalg.norm(current_deltas - center_delta, axis=1)
        max_center_distance = float(np.max(center_distances)) if center_distances.size else 0.0
        distance_rank_suspicion = (
            center_distances / max(max_center_distance, 1e-8)
            if max_center_distance > 0.0
            else np.zeros_like(center_distances)
        )
        median_distance = float(np.median(center_distances)) if center_distances.size else 0.0
        mad_distance = (
            float(np.median(np.abs(center_distances - median_distance)))
            if center_distances.size
            else 0.0
        )
        robust_scale = max(1e-8, 1.4826 * mad_distance)
        distance_suspicion = np.clip(
            (center_distances - median_distance) / (3.0 * robust_scale),
            0.0,
            1.0,
        )

        if self.summed_deltas is None or self.summed_deltas.shape != model_matrix.shape:
            self.summed_deltas = np.zeros_like(model_matrix, dtype=float)
        self.summed_deltas = self.summed_deltas + model_matrix
        model_matrix = self.summed_deltas

        # Compute FoolsGold weights
        n = len(models)
        epsilon = 1e-5

        # Cosine similarity
        cs = cosine_similarity(model_matrix) - np.eye(n)
        # Check for NaN in cosine similarity
        if not np.isfinite(cs).all():
            logger.warning("FoolsGold: Non-finite values in cosine similarity, replacing with zeros")
            cs = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)

        # Pardoning mechanism
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        # Compute weights
        wv = 1 - np.max(cs, axis=1)
        wv = np.clip(wv, 0, 1)

        # Normalize
        if np.max(wv) > 0:
            wv = wv / np.max(wv)
        wv[wv >= 0.99] = 0.99  # Avoid exactly 1.0 to prevent division issues

        # Logit function with numerical stability
        wv_safe = np.clip(wv, epsilon, 1 - epsilon)  # Ensure values are in valid range
        wv = np.log((wv_safe / (1 - wv_safe)) + epsilon) + 0.5
        wv = np.clip(wv, 0, 1)

        # Final NaN check
        if not np.isfinite(wv).all():
            logger.warning("FoolsGold: Non-finite weights detected, falling back to uniform weights")
            fg_trust = np.ones(n, dtype=float)
        else:
            fg_trust = np.clip(wv, 0.0, 1.0)

        if weights is not None:
            base_weights = np.array(weights, dtype=float)
            if not np.isfinite(base_weights).all() or np.sum(base_weights) <= 0:
                base_weights = np.ones(n, dtype=float) / n
            else:
                base_weights = base_weights / np.sum(base_weights)
        else:
            base_weights = np.ones(n, dtype=float) / n

        # FoolsGold's similarity signal is Sybil-specific. In IID non-Sybil
        # attacks, benign updates can be mutually similar while independent
        # random-noise attackers look unique. Add a robust distance-to-center
        # guard and blend with FedAvg so this baseline does not reverse-weight
        # outliers.
        similarity_suspicion = 1.0 - fg_trust
        combined_suspicion = distance_rank_suspicion + 0.25 * similarity_suspicion
        guarded_trust = fg_trust * (1.0 - np.maximum(distance_suspicion, distance_rank_suspicion))
        if np.sum(guarded_trust) > 0:
            fg_weights = guarded_trust / np.sum(guarded_trust)
        else:
            fg_weights = base_weights.copy()
        try:
            blend = float(os.getenv("FOOLSGOLD_BLEND", "0.5"))
        except Exception:
            blend = 0.5
        blend = float(np.clip(blend, 0.0, 1.0))
        wv = blend * fg_weights + (1.0 - blend) * base_weights
        wv = wv / np.sum(wv) if np.sum(wv) > 0 else base_weights

        # Aggregate
        self.client_weights = [float(w) for w in wv]
        self.suspicion_scores = [float(score) for score in combined_suspicion]
        self.scores = self.suspicion_scores.copy()
        self.selected_indices = [i for i, w in enumerate(self.client_weights) if w > 0.0]
        selected_set = set(self.selected_indices)
        self.rejected_indices = [i for i in range(n) if i not in selected_set]

        aggregated_model = ODict()
        for key in models[0].keys():
            if torch.is_tensor(models[0][key]) and models[0][key].is_floating_point():
                aggregated_model[key] = sum(wv[i] * models[i][key] for i in range(len(models)))
            else:
                aggregated_model[key] = models[0][key]

        logger.debug(f"FoolsGold: weights mean={np.mean(wv):.4f}, std={np.std(wv):.4f}")
        return aggregated_model


class SDEAAggregator(BaselineAggregator):
    """
    Clean-room SDEA-style robust aggregation.

    This implementation uses a distribution-estimation filter: estimate the
    central update by coordinate-wise median, score clients by distance to that
    center, drop the highest-scoring f clients, then average the selected set.
    """

    def __init__(self, num_byzantine: int = 0):
        super().__init__("SDEA")
        self.num_byzantine = max(0, int(num_byzantine))
        self.selected_indices = []

    def _flatten_model(self, model: ODict) -> torch.Tensor:
        return _flatten_model_sampled(model)

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        if not models:
            raise ValueError("No models to aggregate")
        if len(models) == 1:
            self.selected_indices = [0]
            self.rejected_indices = []
            self.scores = [0.0]
            self.client_weights = [1.0]
            return models[0]

        vectors = torch.stack([self._flatten_model(model) for model in models], dim=0)
        center = torch.median(vectors, dim=0).values
        distances = torch.norm(vectors - center.unsqueeze(0), p=2, dim=1)

        n = len(models)
        keep = max(1, n - min(self.num_byzantine, n - 1))
        selected = torch.argsort(distances)[:keep].tolist()
        self.selected_indices = [int(i) for i in selected]
        selected_set = set(self.selected_indices)
        self.rejected_indices = [i for i in range(n) if i not in selected_set]
        self.scores = [float(x) for x in distances.detach().cpu().tolist()]

        if weights is not None:
            selected_weights = [float(weights[i]) for i in self.selected_indices]
            total = sum(selected_weights)
            if total <= 0:
                selected_weights = [1.0 / len(self.selected_indices)] * len(self.selected_indices)
            else:
                selected_weights = [w / total for w in selected_weights]
        else:
            selected_weights = [1.0 / len(self.selected_indices)] * len(self.selected_indices)
        full_weights = np.zeros(n, dtype=float)
        for pos, idx in enumerate(self.selected_indices):
            full_weights[int(idx)] = float(selected_weights[pos])
        self.client_weights = full_weights.tolist()

        aggregated_model = ODict()
        for key in models[0].keys():
            aggregated_model[key] = sum(
                selected_weights[pos] * models[idx][key]
                for pos, idx in enumerate(self.selected_indices)
            )

        logger.debug(f"SDEA: selected {len(self.selected_indices)}/{len(models)} clients")
        return aggregated_model


# Factory function for creating aggregators
def create_aggregator(method: str, **kwargs) -> BaselineAggregator:
    """
    Factory function to create aggregator instances

    Args:
        method: Aggregation method name
        **kwargs: Method-specific parameters

    Returns:
        aggregator: Aggregator instance
    """
    aggregators = {
        'Vanilla_FL': VanillaFLAggregator,
        'Krum': KrumAggregator,
        'Trimmed_Mean': TrimmedMeanAggregator,
        'Median': MedianAggregator,
        'Bulyan': BulyanAggregator,
        'ShapleyFL': ShapleyFLAggregator,
        'FoolsGold': FoolsGoldAggregator,
        'SDEA': SDEAAggregator,
    }

    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}")

    return aggregators[method](**kwargs)
