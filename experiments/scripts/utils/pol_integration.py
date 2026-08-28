"""
PoL Integration Helper for Experiments

Provides utilities to integrate PoL verification into FL experiments.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import os
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from client.trainer.PoLTrainer import PoLTrainer
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from server.pol.PoLVerifier import PoLVerifier
from client.pol.PoLManager import PoLManager

logger = logging.getLogger(__name__)


class PoLExperimentHelper:
    """Helper class for integrating PoL into experiments"""

    @staticmethod
    def setup_pol_trainer(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        client_id: str,
        pol_config: Dict,
        device: str = 'cpu'
    ) -> PoLTrainer:
        """
        Setup PoL trainer for a client

        Args:
            model: PyTorch model
            dataloader: Training dataloader
            criterion: Loss function
            client_id: Client ID
            pol_config: PoL configuration dict
            device: Device to use

        Returns:
            trainer: PoLTrainer instance
        """
        args = {
            'client_id': client_id,
            'device': device,
            'optimizer': 'SGD',
            'lr': pol_config.get('learning_rate', pol_config.get('lr', 0.01)),
            'momentum': pol_config.get('momentum', 0.9),
            'weight_decay': pol_config.get('weight_decay', 1e-4),
            'enable_pol': pol_config.get('enable', True),
            'pol_save_freq': pol_config.get('save_freq', 10),
            'pol_save_dir': pol_config.get('save_dir', './pol_data'),
            'pol_compress': pol_config.get('compress', True),
        }

        trainer = PoLTrainer(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            args=args
        )

        logger.info(f"Setup PoL trainer for client {client_id}")
        return trainer

    @staticmethod
    def setup_pol_aggregator(
        model: nn.Module,
        pol_config: Dict,
        device: str = 'cpu',
        robust_aggregation: Optional[str] = None
    ) -> PoLVerifyAggregator:
        """
        Setup PoL aggregator

        Args:
            model: Global model
            pol_config: PoL configuration dict
            device: Device to use

        Returns:
            aggregator: PoLVerifyAggregator instance
        """
        # Allow env overrides for Top-Q
        try:
            _use_top_q_env = os.getenv('POL_USE_TOP_Q')
            use_top_q = (str(_use_top_q_env).lower() in ('1', 'true', 'yes')) if _use_top_q_env is not None else bool(pol_config.get('use_top_q', False))
        except Exception:
            use_top_q = bool(pol_config.get('use_top_q', False))
        try:
            _top_q_env = os.getenv('POL_TOP_Q')
            top_q = int(_top_q_env) if _top_q_env is not None else int(pol_config.get('top_q', 5))
        except Exception:
            top_q = int(pol_config.get('top_q', 5))

        aggregator_args = {
            'enable_pol': True,
            'verification_rate': pol_config.get('verification_rate', 0.3),
            'pol_delta': pol_config.get('delta', 10.0),
            'pol_distance_metric': pol_config.get('distance_metric', 'l2'),
            'device': device,
            'use_top_q': use_top_q,
            'top_q': top_q,
            # Root-cause improvements: acceptance & sampling
            'min_pair_success_rate': pol_config.get('min_pair_success_rate', 0.99),
            'always_verify_last_k': pol_config.get('always_verify_last_k', 2),
            'random_q': pol_config.get('random_q', 3),
            'enable_zkp': pol_config.get('enable_zkp', False),
            'zkp_use_simulation': (False if __import__('os').getenv('POL_INTEGRITY', '0') == '1' else pol_config.get('zkp_use_simulation', True)),
            'enable_incentives': pol_config.get('enable_incentives', True),
            'enable_sybil_detector': pol_config.get('enable_sybil_detector', True),
            'robust_aggregation': robust_aggregation or pol_config.get('robust_aggregation'),
            'robust_aggregation_args': pol_config.get('robust_aggregation_args', {}),
        }

        aggregator = PoLVerifyAggregator(model=model, args=aggregator_args)

        logger.info(f"Setup PoL aggregator with verification_rate={aggregator_args['verification_rate']}, min_pair_success_rate={aggregator_args.get('min_pair_success_rate')}, last_k={aggregator_args.get('always_verify_last_k')}, random_q={aggregator_args.get('random_q')}")
        return aggregator

    @staticmethod
    def compute_detection_metrics(
        verification_results: Dict[str, bool],
        malicious_clients: List[str],
        all_clients: List[str]
    ) -> Dict[str, float]:
        """
        Compute detection metrics from PoL verification results

        Args:
            verification_results: Dict {client_id: is_valid}
                                 is_valid=True means passed verification (honest)
                                 is_valid=False means failed verification (detected as malicious)
            malicious_clients: List of malicious client IDs (ground truth)
            all_clients: List of all client IDs

        Returns:
            metrics: Dict with TPR_e2e, TPR_conditional, FPR (e2e), Precision, Recall, F1, participation_rate
        """
        malicious_set = set(malicious_clients)

        # Participation among selected clients (those in all_clients)
        verified_ids = set(verification_results.keys())
        participation_rate = (len(verified_ids) / len(all_clients)) if all_clients else 0.0
        verify_pass_rate = 0.0
        if verified_ids:
            passes = sum(1 for cid in verified_ids if verification_results.get(cid, False) is True)
            verify_pass_rate = passes / len(verified_ids)

        # End-to-end confusion matrix (treat unverified as not detected => honest)
        tp_e2e = fp_e2e = fn_e2e = tn_e2e = 0
        for client_id in all_clients:
            is_malicious = client_id in malicious_set
            is_verified = client_id in verified_ids
            is_valid = verification_results.get(client_id, True)  # unverified => True (passes)
            detected_as_malicious = not is_valid if is_verified else False

            if is_malicious and detected_as_malicious:
                tp_e2e += 1
            elif is_malicious and not detected_as_malicious:
                fn_e2e += 1
            elif (not is_malicious) and detected_as_malicious:
                fp_e2e += 1
            else:
                tn_e2e += 1

        # Conditional confusion matrix (only among verified clients)
        tp_c = fp_c = fn_c = tn_c = 0
        verified_malicious = [cid for cid in all_clients if (cid in verified_ids and cid in malicious_set)]
        verified_honest = [cid for cid in all_clients if (cid in verified_ids and cid not in malicious_set)]
        for cid in verified_malicious:
            is_valid = verification_results.get(cid, True)
            if not is_valid:
                tp_c += 1
            else:
                fn_c += 1
        for cid in verified_honest:
            is_valid = verification_results.get(cid, True)
            if not is_valid:
                fp_c += 1
            else:
                tn_c += 1

        # Compute metrics
        tpr_e2e = tp_e2e / (tp_e2e + fn_e2e) if (tp_e2e + fn_e2e) > 0 else 0.0
        fpr_e2e = fp_e2e / (fp_e2e + tn_e2e) if (fp_e2e + tn_e2e) > 0 else 0.0
        precision_e2e = tp_e2e / (tp_e2e + fp_e2e) if (tp_e2e + fp_e2e) > 0 else 0.0
        recall_e2e = tpr_e2e
        f1_e2e = 2 * precision_e2e * recall_e2e / (precision_e2e + recall_e2e) if (precision_e2e + recall_e2e) > 0 else 0.0

        tpr_cond = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0

        return {
            # Backward-compat: keep 'TPR'/'FPR' as e2e
            'TPR': float(tpr_e2e),
            'FPR': float(fpr_e2e),
            'Precision': float(precision_e2e),
            'Recall': float(recall_e2e),
            'F1': float(f1_e2e),
            'TPR_e2e': float(tpr_e2e),
            'TPR_conditional': float(tpr_cond),
            'participation_rate': float(participation_rate),
            'verification_pass_rate': float(verify_pass_rate),
            'TP_e2e': int(tp_e2e),
            'FP_e2e': int(fp_e2e),
            'FN_e2e': int(fn_e2e),
            'TN_e2e': int(tn_e2e),
            'total_malicious': int(tp_e2e + fn_e2e),
            'total_honest': int(fp_e2e + tn_e2e),
        }

    @staticmethod
    def extract_verification_results(
        verification_results: Dict
    ) -> Dict[str, bool]:
        """
        Extract verification results from PoL aggregator

        Args:
            verification_results: Raw verification results from aggregator

        Returns:
            results: Dict {client_id: is_valid}
        """
        extracted = {}

        if isinstance(verification_results, dict):
            for client_id, result in verification_results.items():
                if isinstance(result, dict):
                    # Extract 'is_valid' field
                    extracted[client_id] = result.get('is_valid', True)
                elif isinstance(result, bool):
                    extracted[client_id] = result
                else:
                    extracted[client_id] = True

        return extracted
